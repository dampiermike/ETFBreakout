#!/usr/bin/env python3
"""LOCKED STRATEGY -- the reference implementation.

Everything below is a decision that survived testing. Each is annotated with
what it beat, so a future change starts from evidence rather than taste.

    universe    dynamic; all leveraged ETFs in etfs.txt except TQQQ (benchmark)
    eligible    >= 60 bars of history AND 20d median dollar volume >= $1M
    score       breakout_proximity over 60 bars, measured in ATR(20) units
    positions   10, equal weight              [LOCKED -- see round 11]
    rebalance   every 20 trading days
    gate        graded gross exposure off SPX 20d realized vol: full size at or
                below the 80th percentile of the trailing year, zero at the 95th,
                linear between -- then ratcheted, so exposure falls immediately
                but rebuilds with a 10-day halflife    [round 21]
    defense     whatever gross the gate gives up (1 - gross) goes to the sleeve,
                ROUTED by a second dimension: GLD while 50-day TLT momentum is
                >= 0, the dollar (UUP) while bonds are falling      [round 25/26]
    execution   signal at close of t, filled at open of t+1, 10 bps per side

Headline is quoted on the 20-PHASE LADDER, not a single rebalance offset, because
round 13 showed phase 0 is the luckiest of 20 (it flattered Sharpe by ~0.14):

    CAGR 34.1%   Sharpe 0.994   maxDD -47.1%   Calmar 0.723
    (GLD-only sleeve:            32.7% / 0.943 / -48.1% / 0.680)
    (previous binary-gate lock:  32.6% / 0.857 / -57.3% / 0.570)

Read the sleeve Sharpe as ~0.970, not 0.994: the 50-day lookback is the peak of a
swept 40-70 plateau, and quoting a swept peak is the same mistake as quoting
rebalance phase 0. The plateau average is the honest expectation.

Re-locked 2026-07-28 on a full, non-incremental re-download plus the spliced-
history repair in data.py. The previous lock (34.8% / 1.008 / -47.1% / 0.739) was
priced off fabricated corporate-action bars -- see data.py for what they were.
Rounds 21, 25 and 26 were re-run on the corrected data and every adopted decision
survived unchanged, rankings and plateau shape included.

Why each choice, briefly:

  proximity, not breakout -- all nine classic breakout triggers (Donchian,
    squeeze, volume thrust, NR7, gap, pocket pivot) had negative edge in round
    1. Ranking names NEAR their high beats buying names THROUGH it.

  10 positions -- the Sharpe curve peaks cleanly at 10 (0.95) and falls away
    both directions (5: 0.72, 20: 0.83). These are 2-3x funds where the
    variance penalty is quadratic in leverage, so concentration costs more than
    a weak ranking signal can repay.

  equal weight, not inverse-vol -- inverse-vol posts a marginally better
    headline Sharpe but decays +0.13 across halves against equal weight's
    +0.02, and gives up 4 points of CAGR.

  graded gross, not a binary gate -- the binary gate's flaw was never its
    threshold. Retuning window and percentile (round 19, family A) lost 3-10
    CAGR points across nine cells. The problem was the discontinuity: 30% of its
    defensive episodes lasted <= 3 bars, i.e. it whipsawed back into 3x longs
    mid-crash. Scaling smoothly gains +0.09 ladder Sharpe and +1.9 CAGR points.

  ratcheted re-leveraging -- exposure falls freely and rebuilds slowly. This is
    the only change in the project that improved return, drawdown, turnover and
    the loss tail at once, and NOT by taking more risk: mean gross is 0.795
    against the binary gate's 0.858 (full size on only 43% of days, fully flat on
    8%, partial on the rest). It earns more while holding less on average, by
    distributing exposure better across time rather than by adding it.
    It also does what the failed stop-losses of rounds 12/15 were meant to do:
    average size on losing trades falls from a flat 10% to 6.6%, cutting trades
    that cost the book >= 3% from 26 to 12 and total loser drag by a quarter --
    without ever selling a name to achieve it.

  volatility, not trend -- every trend filter (200-SMA, dual-MA, breadth) cost
    12+ points of CAGR for comparable drawdown relief.

  gold, but only while bonds hold -- GLD returns 28.7% annualized while the gate
    is off, against 8.9% unconditionally, and that +19.8% is a genuine
    crisis-hedge effect. But it is a CRISIS hedge, not a RATE hedge: through the
    2021-22 drawdown GLD lost 9.9% and TLT 31.4% while the sleeve held most of
    the book, and the dollar gained 20.7%. Routing on bond trend lifts the sleeve
    from 7.9% to 9.0% standalone CAGR and turns its 2022 return from -9.9% to
    +18.3%. Two controls confirm the rates read is what earns it: inverting the
    signal scores 0.876 and holding UUP unconditionally scores 0.932, both well
    below the 0.953 GLD-only baseline. Static blends (50/50 GLD+BIL, 1/3 each)
    also fail, so this is conditioning rather than diversification.

  no per-name exits, no correlation cap, no universe trim -- all tested, all
    rejected (rounds 12, 15, 16, 17, 18). Every one of them changed WHICH names
    are held and merely relocated the loss tail, usually to a worse source.
    Only the gate, which changes HOW MUCH is held, ever helped.

Known open questions, deliberately not resolved here:
  - The residual loss tail is idiosyncratic and structurally invisible here: the
    worst remaining trades (SOXL Jun-2015, UCO Apr-2015) happened while MARKET
    vol was low, so the gate was correctly at full size. Round 18 showed that
    removing such names costs more than it saves -- they decorrelate the book.
  - The gate was explored over 37 cells across rounds 19 and 21. Both finalists
    cleared a both-halves test, but the cumulative selection risk is real and a
    clean out-of-sample re-validation has not been run.
  - The 20-day rebalance phase is arbitrary; the ladder (round 14) is the
    phase-independent reference. Single-phase results run ~0.11 Sharpe hot.
  - Gold's edge rests on ~12% of days concentrated in a handful of episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

import data
import engine
import macro
import regime
import research
import signals


@dataclass(frozen=True)
class Config:
    score_window: int = 60
    atr_window: int = 20
    positions: int = 10          # LOCKED
    rebalance_days: int = 20
    vol_window: int = 20
    vol_lookback: int = 252
    # graded gate: full size at/below p_lo, flat at zero at/above p_hi
    vol_band_lo: float = 0.80
    vol_band_hi: float = 0.95
    ratchet_halflife: float = 10.0   # bars to rebuild half the lost exposure
    graded_gate: bool = True         # False -> the legacy binary gate
    vol_max_pctile: float = 0.90     # binary gate only
    # second gate dimension: WHAT the defensive sleeve holds, by rates regime
    defensive: Optional[str] = "GLD"          # sleeve while bonds are stable
    defensive_stress: Optional[str] = "UUP"   # sleeve while bonds are falling
    rates_window: int = 50                    # TLT momentum lookback
    routed_defense: bool = True               # False -> always `defensive`
    use_credit_gate: bool = False   # the more defensive variant
    credit_window: int = 60
    cost_bps: float = 10.0
    start: str = research.START


DEFAULT = Config()
DEFENSIVE_VARIANT = Config(use_credit_gate=True)
# The pre-round-21 lock, for comparison: binary gate AND the unrouted GLD sleeve,
# so it still reproduces its own historical numbers rather than inheriting later work.
LEGACY_BINARY = Config(graded_gate=False, routed_defense=False)
GLD_ONLY_SLEEVE = Config(routed_defense=False)   # round-21 lock: graded gate, no routing


class Strategy:
    def __init__(self, cfg: Config = DEFAULT, ctx: Optional[research.Context] = None):
        self.cfg = cfg
        self.ctx = ctx or research.Context(start=cfg.start)
        self.macro = macro.load_macro()

    # -- components -------------------------------------------------------
    def score(self) -> pd.DataFrame:
        return signals.breakout_proximity(self.ctx.f, self.cfg.score_window)

    def gross(self) -> pd.Series:
        """Daily gross exposure in [0, 1]. Evaluated daily, independent of the
        rebalance. The binary gate is the degenerate case of this returning
        only 0.0 or 1.0."""
        cfg = self.cfg
        spx = regime.proxy_index(self.ctx.f, "SSO", 2.0)
        if cfg.graded_gate:
            vol = regime.realized_vol(spx, cfg.vol_window)
            raw = regime.graded_gross(vol, cfg.vol_band_lo, cfg.vol_band_hi,
                                      cfg.vol_lookback)
            g = regime.asym_ewma_ratchet(raw, cfg.ratchet_halflife)
        else:
            g = regime.volatility_filter(spx, cfg.vol_window, cfg.vol_lookback,
                                         cfg.vol_max_pctile).astype(float)
        g = g.reindex(self.ctx.index).ffill().fillna(1.0).clip(0.0, 1.0)
        if cfg.use_credit_gate:
            credit = macro.credit_momentum(self.macro, "HYG", cfg.credit_window)
            credit = credit.reindex(self.ctx.index).ffill().fillna(True).astype(float)
            g = g * credit
        return g

    def gate(self) -> pd.Series:
        """True where the book carries meaningful risk -- for reporting only.

        Prefer gross(); with the graded gate this is a lossy summary of it.
        """
        return self.gross() > 0.5

    def sleeve(self) -> pd.DataFrame:
        """Defensive-sleeve weights (rows sum to 1), the second gate dimension."""
        cfg = self.cfg
        if not cfg.routed_defense or not cfg.defensive_stress:
            out = pd.DataFrame(0.0, index=self.ctx.index, columns=macro.DEFENSIVE)
            if cfg.defensive:
                out[cfg.defensive] = 1.0
            return out
        calm = macro.bond_trend_ok(self.macro, self.ctx.index, cfg.rates_window)
        return macro.routed_sleeve(self.ctx.index, calm,
                                   cfg.defensive, cfg.defensive_stress)

    def weights(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Target weights plus the return frame they index into."""
        base = engine.rank_to_weights(self.score(), self.ctx.eligible,
                                      self.cfg.positions, self.cfg.rebalance_days)
        gross = self.gross()
        defensive = macro.defensive_bar_returns(self.ctx.index)
        combined = pd.concat([self.ctx.bar_returns, defensive.add_prefix("D_")], axis=1)

        weights = pd.DataFrame(0.0, index=self.ctx.index, columns=combined.columns)
        weights[self.ctx.bar_returns.columns] = regime.apply_gross(base, gross)
        # whatever gross the gate gives up is held in the defensive sleeve,
        # routed between the calm and stress assets by the rates read
        sleeve = self.sleeve()
        for asset in sleeve.columns:
            if f"D_{asset}" in weights.columns and (sleeve[asset] != 0).any():
                weights[f"D_{asset}"] = sleeve[asset] * (1.0 - gross)
        return weights, combined

    def run(self, name: str = "locked") -> engine.BacktestResult:
        weights, combined = self.weights()
        return engine.run_backtest(weights, combined, name=name,
                                   cost_bps=self.cfg.cost_bps, start=self.cfg.start)

    # -- live state -------------------------------------------------------
    def today(self) -> Dict:
        weights, _ = self.weights()
        gross = self.gross()
        asof = weights.index[-1]
        row = weights.loc[asof]
        held = row[row > 0].sort_values(ascending=False)
        g = float(gross.loc[asof])
        sleeve = self.sleeve().loc[asof]
        active = sleeve[sleeve > 0]
        return {
            "asof": asof,
            "gross": g,
            "state": "OFFENSE" if g >= 0.99 else ("DEFENSE" if g <= 0.01
                                                  else f"PARTIAL {g:.0%}"),
            "sleeve": active.idxmax() if len(active) else None,
            "holdings": held,
            "eligible": int(self.ctx.eligible.loc[asof].sum()),
        }
