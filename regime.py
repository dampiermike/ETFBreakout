#!/usr/bin/env python3
"""Market-regime construction.

The universe is entirely 2x/3x leveraged, so an always-invested book eats a
60%+ drawdown by construction. These filters decide when the book should be in
cash at all -- a separate question from which names to hold.

No index file ships with the data, so the market proxy is rebuilt by
de-levering SSO (2x S&P) and QLD (2x Nasdaq-100) daily returns. Both list in
2006, which covers the whole backtest.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def proxy_index(f: Dict[str, pd.DataFrame], ticker: str = "SSO", leverage: float = 2.0) -> pd.Series:
    """Reconstruct a 1x index level from a leveraged ETF's daily returns.

    Ignores the financing/decay wedge, which is second-order for a trend filter.
    """
    ret = f["close"][ticker].pct_change() / leverage
    return (1.0 + ret.fillna(0.0)).cumprod()


def trend_filter(level: pd.Series, window: int = 200) -> pd.Series:
    ma = level.rolling(window, min_periods=window // 2).mean()
    return level > ma


def dual_trend_filter(level: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    fast_ma = level.rolling(fast, min_periods=fast // 2).mean()
    slow_ma = level.rolling(slow, min_periods=slow // 2).mean()
    return (level > slow_ma) & (fast_ma > slow_ma)


def breadth_filter(f: Dict[str, pd.DataFrame], eligible: pd.DataFrame,
                   window: int = 100, threshold: float = 0.5) -> pd.Series:
    """Share of the eligible universe trading above its own moving average."""
    close = f["close"]
    ma = close.rolling(window, min_periods=window // 2).mean()
    above = (close > ma) & eligible
    count = eligible.sum(axis=1)
    breadth = above.sum(axis=1) / count.where(count > 0, np.nan)
    return breadth >= threshold


def volatility_filter(level: pd.Series, window: int = 20, lookback: int = 252,
                      max_pctile: float = 0.90) -> pd.Series:
    """Stand down when realized vol sits in its own top percentile.

    Superseded as the locked gate by graded_gross + asym_ewma_ratchet (round 21);
    kept because it is the benchmark every gate variant is measured against.
    """
    vol = level.pct_change().rolling(window, min_periods=window // 2).std()
    rank = vol.rolling(lookback, min_periods=lookback // 4).rank(pct=True)
    return rank <= max_pctile


# -- graded gross exposure (the locked gate, rounds 19-22) -------------------
#
# The binary filter above answers "in or out". It turned out that was the wrong
# question: its flaw was never the threshold (retuning window/percentile lost
# 3-10 CAGR points in round 19) but the DISCONTINUITY and the speed of re-entry.
# 30% of its defensive episodes lasted <= 3 bars -- it whipsawed back into 3x
# longs mid-crash. Scaling exposure smoothly, then forbidding fast re-leveraging,
# improves CAGR, Sharpe, drawdown, turnover and the loss tail simultaneously.

SQRT252 = np.sqrt(252.0)


def realized_vol(level: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close-to-close realized volatility."""
    return level.pct_change().rolling(
        window, min_periods=max(3, window // 2)).std() * SQRT252


def pctile_threshold(vol: pd.Series, pctile: float, lookback: int = 252) -> pd.Series:
    """The vol level corresponding to `pctile` of its own trailing year."""
    return vol.rolling(lookback, min_periods=lookback // 2).quantile(pctile)


def graded_gross(vol: pd.Series, p_lo: float = 0.80, p_hi: float = 0.95,
                 lookback: int = 252) -> pd.Series:
    """Gross exposure: 1.0 at/below the p_lo vol threshold, 0.0 at/above p_hi.

    Linear in between. Unlike a binary gate this never forces an all-or-nothing
    switch, so calm periods run at full exposure without whipsaw risk.
    """
    thr_lo = pctile_threshold(vol, p_lo, lookback)
    thr_hi = pctile_threshold(vol, p_hi, lookback)
    span = (thr_hi - thr_lo).replace(0, np.nan)
    return ((thr_hi - vol) / span).clip(lower=0.0, upper=1.0).fillna(1.0)


def asym_ewma_ratchet(gross: pd.Series, halflife: float = 10.0) -> pd.Series:
    """De-lever fast, re-lever slow.

    Exposure follows the raw graded level down immediately, but rebuilds toward
    it exponentially. This is what separates the locked gate from plain graded
    gross: it cuts maxDD by ~9 points and halves the count of trades costing the
    book >= 3%, because average size on losing trades falls from 10% to ~6.6%.
    """
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    raw = gross.to_numpy(dtype=float)
    out = np.empty_like(raw)
    prev = raw[0] if len(raw) and not np.isnan(raw[0]) else 1.0
    for i, v in enumerate(raw):
        if np.isnan(v):
            v = prev
        out[i] = prev = v if v < prev else prev + alpha * (v - prev)
    return pd.Series(out, index=gross.index)


def linear_ratchet(gross: pd.Series, step: float = 0.02) -> pd.Series:
    """As asym_ewma_ratchet, but exposure may rise at most `step` per bar.

    Round-21 runner-up (Sharpe 0.965 vs 0.953) -- kept for comparison.
    """
    raw = gross.to_numpy(dtype=float)
    out = np.empty_like(raw)
    prev = raw[0] if len(raw) and not np.isnan(raw[0]) else 1.0
    for i, v in enumerate(raw):
        if np.isnan(v):
            v = prev
        out[i] = prev = min(v, prev + step)
    return pd.Series(out, index=gross.index)


def apply_gross(weights: pd.DataFrame, gross: pd.Series) -> pd.DataFrame:
    """Scale the book by daily gross exposure (the graded analogue of apply_regime)."""
    g = gross.reindex(weights.index).ffill().fillna(1.0).clip(0.0, 1.0)
    return weights.mul(g, axis=0)


def drawdown_filter(level: pd.Series, threshold: float = 0.10) -> pd.Series:
    """Stand down while the index sits more than `threshold` below its peak."""
    return (level / level.cummax() - 1.0) > -threshold


def new_high_filter(level: pd.Series, window: int = 100) -> pd.Series:
    return level >= level.rolling(window, min_periods=window // 2).max() * 0.90


def apply_regime(weights: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """Zero the book on days the regime is off; the balance sits in cash."""
    gate = regime.reindex(weights.index).fillna(False).astype(float)
    return weights.mul(gate, axis=0)


def build_regimes(f: Dict[str, pd.DataFrame], eligible: pd.DataFrame) -> Dict[str, pd.Series]:
    spx = proxy_index(f, "SSO", 2.0)
    ndx = proxy_index(f, "QLD", 2.0)
    return {
        "none": pd.Series(True, index=spx.index),
        "spx_sma200": trend_filter(spx, 200),
        "spx_sma100": trend_filter(spx, 100),
        "ndx_sma200": trend_filter(ndx, 200),
        "ndx_sma100": trend_filter(ndx, 100),
        "spx_dual_50_200": dual_trend_filter(spx, 50, 200),
        "ndx_dual_50_200": dual_trend_filter(ndx, 50, 200),
        "breadth_50": breadth_filter(f, eligible, 100, 0.50),
        "breadth_40": breadth_filter(f, eligible, 100, 0.40),
        "spx_vol_90": volatility_filter(spx, 20, 252, 0.90),
        "spx_dd_10": drawdown_filter(spx, 0.10),
        "ndx_sma200_and_vol": trend_filter(ndx, 200) & volatility_filter(ndx, 20, 252, 0.90),
        "ndx_sma200_and_breadth": trend_filter(ndx, 200) & breadth_filter(f, eligible, 100, 0.40),
    }
