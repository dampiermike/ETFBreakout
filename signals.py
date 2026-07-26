#!/usr/bin/env python3
"""Signal library for breakout research.

Every function takes the wide frames (open/high/low/close/volume, each date x
ticker) and returns a date x ticker frame computed only from data available at
that date's close. Boolean frames are events; float frames are scores for
cross-sectional ranking.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    return pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()


def atr(high, low, close, window: int = 14) -> pd.DataFrame:
    return true_range(high, low, close).rolling(window, min_periods=window // 2).mean()


def rolling_max(frame: pd.DataFrame, window: int, exclude_today: bool = True) -> pd.DataFrame:
    src = frame.shift(1) if exclude_today else frame
    return src.rolling(window, min_periods=max(2, window // 2)).max()


def rolling_min(frame: pd.DataFrame, window: int, exclude_today: bool = True) -> pd.DataFrame:
    src = frame.shift(1) if exclude_today else frame
    return src.rolling(window, min_periods=max(2, window // 2)).min()


def pct_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling percentile rank of the current value within its own history."""
    return frame.rolling(window, min_periods=max(10, window // 4)).rank(pct=True)


def zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=window // 2).mean()
    std = frame.rolling(window, min_periods=window // 2).std()
    return (frame - mean) / std.replace(0, np.nan)


def ols_slope(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized slope of a least-squares fit to log price."""
    log_px = np.log(frame.clip(lower=1e-9))
    x = np.arange(window)
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum()

    def _fit(values: np.ndarray) -> float:
        return float(np.dot(values - values.mean(), x_centered) / denom)

    return log_px.rolling(window, min_periods=window).apply(_fit, raw=True) * 252


# --------------------------------------------------------------------------
# Round 1 -- classic breakout family
# --------------------------------------------------------------------------
def donchian_breakout(f: Dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:
    """Close takes out the highest close of the prior `window` bars."""
    return f["close"] > rolling_max(f["close"], window)


def donchian_high_breakout(f: Dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:
    """Intraday high takes out the prior `window`-bar high, closing strong."""
    broke = f["high"] > rolling_max(f["high"], window)
    strong = f["close"] > (f["high"] + f["low"]) / 2
    return broke & strong


def squeeze_breakout(f: Dict[str, pd.DataFrame], bb_window: int = 20,
                     lookback: int = 120, pctile: float = 0.20) -> pd.DataFrame:
    """Volatility contraction resolving upward.

    Bollinger bandwidth in its own low percentile (the coil), then price closes
    above the upper band (the release).
    """
    close = f["close"]
    mid = close.rolling(bb_window, min_periods=bb_window // 2).mean()
    sd = close.rolling(bb_window, min_periods=bb_window // 2).std()
    width = (2 * sd) / mid.replace(0, np.nan)
    coiled = pct_rank(width, lookback).shift(1) <= pctile
    release = close > (mid + 2 * sd)
    return coiled & release


def volume_thrust(f: Dict[str, pd.DataFrame], window: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Breakout confirmed by a volume surge -- participation, not drift."""
    vol_ma = f["volume"].rolling(window, min_periods=window // 2).mean().shift(1)
    surge = f["volume"] > mult * vol_ma
    return donchian_breakout(f, window) & surge


def nr7_expansion(f: Dict[str, pd.DataFrame], window: int = 7) -> pd.DataFrame:
    """Narrowest range in `window` bars, then an up-close range expansion."""
    rng = f["high"] - f["low"]
    narrow = (rng.shift(1) == rng.shift(1).rolling(window, min_periods=window).min())
    expand = (rng > rng.shift(1)) & (f["close"] > f["close"].shift(1))
    return narrow & expand


def atr_thrust(f: Dict[str, pd.DataFrame], window: int = 14, mult: float = 1.5) -> pd.DataFrame:
    """A single bar advancing more than `mult` ATRs -- an impulse move."""
    a = atr(f["high"], f["low"], f["close"], window).shift(1)
    move = f["close"] - f["close"].shift(1)
    return move > mult * a


def gap_continuation(f: Dict[str, pd.DataFrame], gap_pct: float = 0.02) -> pd.DataFrame:
    """Gaps up through the prior high and holds it into the close."""
    gap = f["open"] / f["close"].shift(1) - 1.0
    return (gap > gap_pct) & (f["close"] > f["open"]) & (f["close"] > f["high"].shift(1))


def pocket_pivot(f: Dict[str, pd.DataFrame], window: int = 10) -> pd.DataFrame:
    """Up day whose volume exceeds the largest down-day volume of `window`."""
    down = f["close"] < f["close"].shift(1)
    down_vol = f["volume"].where(down)
    max_down_vol = down_vol.rolling(window, min_periods=3).max().shift(1)
    up = f["close"] > f["close"].shift(1)
    above_ma = f["close"] > f["close"].rolling(50, min_periods=25).mean()
    return up & (f["volume"] > max_down_vol) & above_ma


# --------------------------------------------------------------------------
# Score signals -- for cross-sectional ranking
# --------------------------------------------------------------------------
def momentum_score(f: Dict[str, pd.DataFrame], window: int = 60, skip: int = 5) -> pd.DataFrame:
    """Trailing return, skipping the most recent bars to dodge reversal."""
    close = f["close"]
    return close.shift(skip) / close.shift(window) - 1.0


def risk_adjusted_momentum(f: Dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """Return per unit of volatility -- prefers smooth trends to violent ones."""
    close = f["close"]
    ret = close / close.shift(window) - 1.0
    vol = close.pct_change().rolling(window, min_periods=window // 2).std() * np.sqrt(252)
    return ret / vol.replace(0, np.nan)


def slope_score(f: Dict[str, pd.DataFrame], window: int = 40) -> pd.DataFrame:
    return ols_slope(f["close"], window)


def breakout_proximity(f: Dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """How close price sits to its `window`-bar high, in ATR units.

    Scores the setup rather than the event: coiled just under resistance.
    """
    high_n = rolling_max(f["high"], window)
    a = atr(f["high"], f["low"], f["close"], 20)
    return -((high_n - f["close"]) / a.replace(0, np.nan))


def rs_rank(f: Dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """Cross-sectional relative strength: return rank among peers today."""
    ret = f["close"] / f["close"].shift(window) - 1.0
    return ret.rank(axis=1, pct=True)


def persistence_proximity(f: Dict[str, pd.DataFrame], block: int = 20,
                          n_blocks: int = 3, agg: str = "min") -> pd.DataFrame:
    """Multi-window version of breakout_proximity: near its high in EVERY block.

    Compute the `block`-bar proximity (distance below the block-bar high in ATR
    units), then snapshot it at t, t-block, t-2*block, ... and aggregate across
    the `n_blocks` snapshots. With agg='min' a name must have sat near its high
    in EACH block to score well -- a fund that only spiked in the most recent
    block was NOT near its high `block` bars ago, so its oldest snapshot is deep
    negative and drags the min down. This filters spike-then-crater names (the
    single-stock / commodity 3x funds that dominate the loss tail) while a steady
    climber, near its high throughout, scores high in every block.

    agg='mean' is the softer version (average strength, not strict AND).
    Covers block*n_blocks bars total -- with 20x3 that is the same 60-bar horizon
    as the locked breakout_proximity(60), only structured into sub-windows.
    """
    p = breakout_proximity(f, block)
    snaps = [p.shift(k * block) for k in range(n_blocks)]
    stack = pd.concat(snaps, axis=0).groupby(level=0)
    return stack.min() if agg == "min" else stack.mean()


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
EVENT_SIGNALS = {
    "donchian_20": lambda f: donchian_breakout(f, 20),
    "donchian_50": lambda f: donchian_breakout(f, 50),
    "donchian_high_20": lambda f: donchian_high_breakout(f, 20),
    "squeeze_20_120": lambda f: squeeze_breakout(f, 20, 120, 0.20),
    "volume_thrust_20": lambda f: volume_thrust(f, 20, 2.0),
    "nr7_expansion": nr7_expansion,
    "atr_thrust_1.5": lambda f: atr_thrust(f, 14, 1.5),
    "gap_continuation": lambda f: gap_continuation(f, 0.02),
    "pocket_pivot": pocket_pivot,
}

SCORE_SIGNALS = {
    "momentum_60": lambda f: momentum_score(f, 60, 5),
    "momentum_120": lambda f: momentum_score(f, 120, 5),
    "risk_adj_mom_60": lambda f: risk_adjusted_momentum(f, 60),
    "slope_40": lambda f: slope_score(f, 40),
    "breakout_proximity_60": lambda f: breakout_proximity(f, 60),
    "rs_rank_60": lambda f: rs_rank(f, 60),
}


# --------------------------------------------------------------------------
# Round 6 -- compounding-quality scores
#
# A leveraged fund's log return is roughly L*mu - 0.5*L^2*sigma^2. The variance
# term is what kills UCO/BOIL/AGQ over long holds, and it is quadratic in
# leverage. Signals that rank on return alone -- or even on return/vol -- do not
# penalize it hard enough. These score the quality of the compounding instead.
# --------------------------------------------------------------------------
def ols_fit(frame: pd.DataFrame, window: int):
    """Rolling slope and R-squared of a log-price regression."""
    log_px = np.log(frame.clip(lower=1e-9))
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()

    def _slope(v: np.ndarray) -> float:
        return float(np.dot(v - v.mean(), xc) / denom)

    def _r2(v: np.ndarray) -> float:
        b = np.dot(v - v.mean(), xc) / denom
        resid = (v - v.mean()) - b * xc
        ss_tot = float(((v - v.mean()) ** 2).sum())
        return float(1.0 - resid.dot(resid) / ss_tot) if ss_tot > 0 else 0.0

    slope = log_px.rolling(window, min_periods=window).apply(_slope, raw=True) * 252
    r2 = log_px.rolling(window, min_periods=window).apply(_r2, raw=True)
    return slope, r2


def trend_quality(f: Dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """Annualized log slope weighted by how straight the trend is.

    Rewards the smooth grinders and penalizes names that got to the same place
    through violent chop -- exactly the chop a leveraged fund pays for.
    """
    slope, r2 = ols_fit(f["close"], window)
    return slope * r2


def kelly_score(f: Dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """Drift over variance -- the Kelly ratio, and the natural decay penalty."""
    log_ret = np.log(f["close"].clip(lower=1e-9)).diff()
    mu = log_ret.rolling(window, min_periods=window // 2).mean() * 252
    var = log_ret.rolling(window, min_periods=window // 2).var() * 252
    return mu / var.replace(0, np.nan)


def drag_adjusted_momentum(f: Dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """Trailing log return net of an explicit variance-drag charge."""
    log_ret = np.log(f["close"].clip(lower=1e-9)).diff()
    total = log_ret.rolling(window, min_periods=window // 2).sum()
    var = log_ret.rolling(window, min_periods=window // 2).var() * window
    return total - 0.5 * var


def smoothness(f: Dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """Net move divided by the path walked to get there (efficiency ratio)."""
    close = f["close"]
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window // 2).sum()
    direction = np.sign(close - close.shift(window))
    return direction * net / path.replace(0, np.nan)


def composite_rank(f: Dict[str, pd.DataFrame], parts: Dict[str, float],
                   window_map: Dict[str, int] | None = None) -> pd.DataFrame:
    """Weighted average of cross-sectional percentile ranks.

    Ranking before combining keeps a single wild-scaled factor from dominating.
    """
    total = None
    for name, weight in parts.items():
        raw = SCORE_LIBRARY[name](f)
        ranked = raw.rank(axis=1, pct=True)
        total = ranked * weight if total is None else total + ranked * weight
    return total


SCORE_LIBRARY = {
    "momentum_60": lambda f: momentum_score(f, 60, 5),
    "momentum_120": lambda f: momentum_score(f, 120, 5),
    "risk_adj_mom_60": lambda f: risk_adjusted_momentum(f, 60),
    "slope_40": lambda f: slope_score(f, 40),
    "breakout_proximity_60": lambda f: breakout_proximity(f, 60),
    "trend_quality_60": lambda f: trend_quality(f, 60),
    "trend_quality_120": lambda f: trend_quality(f, 120),
    "kelly_60": lambda f: kelly_score(f, 60),
    "kelly_120": lambda f: kelly_score(f, 120),
    "drag_adj_mom_60": lambda f: drag_adjusted_momentum(f, 60),
    "drag_adj_mom_120": lambda f: drag_adjusted_momentum(f, 120),
    "smoothness_60": lambda f: smoothness(f, 60),
}
