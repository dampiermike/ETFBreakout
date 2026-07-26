#!/usr/bin/env python3
"""Backtest engine and signal-evaluation harness.

Two levels of evaluation:

1. `event_study` -- when a signal fires, what does the forward return
   distribution look like versus the unconditional baseline? This measures
   whether a rule has any predictive content at all, independent of sizing.

2. `run_backtest` -- turn a signal into target weights and simulate a
   portfolio, with costs, on the dynamic universe.

Timing convention throughout: signals are computed from data through the close
of day t, and the resulting position is held from the OPEN of day t+1 to the
open of day t+2. Nothing reads a price it could not have seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def metrics(returns: pd.Series, equity: Optional[pd.Series] = None) -> Dict[str, float]:
    """Standard performance stats from a series of periodic returns."""
    returns = returns.dropna()
    if len(returns) < 2:
        return {k: np.nan for k in
                ("cagr", "vol", "sharpe", "sortino", "max_dd", "calmar", "years", "final")}
    if equity is None:
        equity = (1.0 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    total = float(equity.iloc[-1])
    cagr = total ** (1.0 / years) - 1.0 if total > 0 else -1.0
    vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    dd = max_drawdown(equity)
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": cagr / vol if vol > 0 else np.nan,
        "sortino": cagr / downside if downside and downside > 0 else np.nan,
        "max_dd": dd,
        "calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "years": years,
        "final": total,
    }


# --------------------------------------------------------------------------
# Event study -- does the signal predict anything?
# --------------------------------------------------------------------------
def event_study(signal: pd.DataFrame, forward: Dict[int, pd.DataFrame],
                eligible: pd.DataFrame) -> pd.DataFrame:
    """Compare forward returns after a signal fires to the eligible baseline.

    `signal` is a boolean date x ticker frame. `forward[h]` holds the h-day
    forward return measured open-to-open starting the day after the signal.
    """
    rows = []
    fired = signal & eligible
    for horizon, fwd in sorted(forward.items()):
        hit = fwd.where(fired)
        base = fwd.where(eligible)
        hv = hit.to_numpy().ravel()
        bv = base.to_numpy().ravel()
        hv = hv[np.isfinite(hv)]
        bv = bv[np.isfinite(bv)]
        if len(hv) < 30:
            continue
        # Welch t-stat: signal mean vs baseline mean. Overlapping windows make
        # this optimistic, so treat it as a ranking device, not a p-value.
        se = np.sqrt(hv.var(ddof=1) / len(hv) + bv.var(ddof=1) / len(bv))
        # Geometric edge is what a compounding portfolio actually earns. On
        # these leveraged names the arithmetic mean is dominated by a handful of
        # enormous up-days, so the two can disagree in sign.
        log_hit = np.log1p(np.clip(hv, -0.99, None))
        log_base = np.log1p(np.clip(bv, -0.99, None))
        rows.append({
            "horizon": horizon,
            "n": len(hv),
            "mean": hv.mean(),
            "base_mean": bv.mean(),
            "edge": hv.mean() - bv.mean(),
            "log_edge": log_hit.mean() - log_base.mean(),
            "median": np.median(hv),
            "base_median": np.median(bv),
            "win_rate": (hv > 0).mean(),
            "base_win_rate": (bv > 0).mean(),
            "t_stat": (hv.mean() - bv.mean()) / se if se > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Portfolio backtest
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    stats: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def summary_row(self) -> Dict[str, float]:
        return {"strategy": self.name, **self.stats, **self.diagnostics}


def run_backtest(target_weights: pd.DataFrame, bar_returns: pd.DataFrame,
                 name: str = "strategy", cost_bps: float = 10.0,
                 start: Optional[str] = None) -> BacktestResult:
    """Simulate a portfolio from target weights.

    `target_weights` is a date x ticker frame of weights DECIDED on that date's
    close; they are applied to the following bar's return. `bar_returns` is
    open-to-open so that a decision at t's close is filled at t+1's open.
    """
    weights = target_weights.fillna(0.0)
    rets = bar_returns.reindex_like(weights).fillna(0.0)

    if start is not None:
        weights = weights.loc[start:]
        rets = rets.loc[start:]

    # Shift: weight decided at t earns the bar-return stamped at t+1.
    held = weights.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)

    # Turnover uses the drifted book, not the prior target: a position whose
    # weight moved because the asset moved is not a trade.
    #
    # Drift must divide by portfolio growth (1 + portfolio return), NOT by the
    # sum of the drifted legs. Dividing by the leg sum renormalizes gross
    # exposure to 1.0 every bar, so a book targeting 1.5x gross showed ~0.5 of
    # phantom turnover daily -- ~126x a year, charged only to levered variants.
    # Cash (positive or borrowed) is the implicit balance and needs no leg.
    drifted = held.mul(1.0 + rets)
    growth = 1.0 + gross
    drifted = drifted.div(growth.where(growth != 0, 1.0), axis=0)
    turnover = (weights - drifted).abs().sum(axis=1)
    costs = turnover * (cost_bps / 1e4)

    net = gross - costs
    equity = (1.0 + net).cumprod()

    result = BacktestResult(name=name, equity=equity, returns=net, weights=weights)
    result.stats = metrics(net, equity)
    exposure = held.abs().sum(axis=1)
    result.diagnostics = {
        "exposure": float(exposure.mean()),
        "days_in_mkt": float((exposure > 0.01).mean()),
        "turnover_yr": float(turnover.mean() * TRADING_DAYS),
        "cost_drag_yr": float(costs.mean() * TRADING_DAYS),
        "avg_names": float((held.abs() > 1e-6).sum(axis=1).mean()),
    }
    return result


def buy_and_hold(bar_returns: pd.DataFrame, ticker: str, start: Optional[str] = None,
                 cost_bps: float = 10.0) -> BacktestResult:
    weights = pd.DataFrame(0.0, index=bar_returns.index, columns=bar_returns.columns)
    valid = bar_returns[ticker].notna()
    weights.loc[valid, ticker] = 1.0
    return run_backtest(weights, bar_returns, name=f"B&H {ticker}", cost_bps=cost_bps, start=start)


def equal_weight_universe(eligible: pd.DataFrame, bar_returns: pd.DataFrame,
                          start: Optional[str] = None, cost_bps: float = 10.0) -> BacktestResult:
    counts = eligible.sum(axis=1)
    weights = eligible.astype(float).div(counts.where(counts > 0, 1.0), axis=0)
    return run_backtest(weights, bar_returns, name="EW universe", cost_bps=cost_bps, start=start)


# --------------------------------------------------------------------------
# Shared feature construction
# --------------------------------------------------------------------------
def bar_return_frame(panel: Dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Open-to-open returns stamped at the date the bar STARTS.

    A weight set at t's close is shifted onto t+1, where it meets the return
    from t+1's open to t+2's open -- an executable fill.
    """
    out = {}
    for ticker, df in panel.items():
        opens = df["open"]
        out[ticker] = (opens.shift(-1) / opens - 1.0).reindex(index)
    return pd.DataFrame(out)


def forward_returns(panel: Dict[str, pd.DataFrame], index: pd.DatetimeIndex,
                    horizons=(1, 3, 5, 10, 20)) -> Dict[int, pd.DataFrame]:
    """h-day forward open-to-open return, stamped at the signal date."""
    result = {}
    for h in horizons:
        out = {}
        for ticker, df in panel.items():
            opens = df["open"]
            out[ticker] = (opens.shift(-(h + 1)) / opens.shift(-1) - 1.0).reindex(index)
        result[h] = pd.DataFrame(out)
    return result


def adaptive_weights(score: pd.DataFrame, eligible: pd.DataFrame,
                     fraction: float = 0.30, min_n: int = 3, max_n: int = 12,
                     qualify: Optional[pd.DataFrame] = None,
                     rebalance_days: int = 20,
                     fill_cash: bool = True) -> pd.DataFrame:
    """Hold the best `fraction` of the LIVE universe, with unfilled slots in cash.

    Two departures from `rank_to_weights`, both aimed at the same flaw: with a
    fixed top-10 the early years hold 70% of a 14-name universe, so nothing is
    actually being selected.

    - The slot count tracks the eligible count, so selectivity is constant.
    - `qualify` is a per-name veto (e.g. price above its own trend). When
      `fill_cash` is set, vetoed slots stay in cash instead of being backfilled
      by the next-best name -- that is what lets the book de-risk on its own,
      without a market-wide gate.
    """
    masked = score.where(eligible)
    if qualify is not None:
        masked = masked.where(qualify.reindex_like(masked).fillna(False))

    live = eligible.sum(axis=1)
    slots = (live * fraction).round().clip(lower=min_n, upper=max_n)

    ranks = masked.rank(axis=1, ascending=False, method="first")
    picked = ranks.le(slots, axis=0) & masked.notna()

    if fill_cash:
        # Divide by the slot count, not the fill count: if only 4 of 10 slots
        # qualify, the book is 40% invested and 60% cash.
        denom = slots
    else:
        held = picked.sum(axis=1)
        denom = held.where(held > 0, 1.0)
    weights = picked.astype(float).div(denom.where(denom > 0, 1.0), axis=0)

    if rebalance_days > 1:
        stamps = np.arange(len(weights)) % rebalance_days == 0
        weights = weights.where(pd.Series(stamps, index=weights.index), np.nan).ffill()
    return weights.fillna(0.0)


def rank_to_weights(score: pd.DataFrame, eligible: pd.DataFrame, top_n: int,
                    rebalance_days: int = 1, min_score: Optional[float] = None,
                    rebalance_phase: int = 0) -> pd.DataFrame:
    """Equal-weight the top-N names by score among eligible tickers.

    `rebalance_phase` shifts which day-of-cycle the rebalance fires (0..rebalance
    _days-1). It exists only to measure timing luck; the locked strategy uses 0.
    """
    masked = score.where(eligible)
    if min_score is not None:
        masked = masked.where(masked >= min_score)
    ranks = masked.rank(axis=1, ascending=False, method="first")
    picked = (ranks <= top_n) & masked.notna()
    counts = picked.sum(axis=1)
    weights = picked.astype(float).div(counts.where(counts > 0, 1.0), axis=0)

    if rebalance_days > 1:
        stamps = (np.arange(len(weights)) - rebalance_phase) % rebalance_days == 0
        stamps[0] = True  # always invest from bar 0 so phases start identically
        weights = weights.where(pd.Series(stamps, index=weights.index), np.nan).ffill()
    return weights.fillna(0.0)
