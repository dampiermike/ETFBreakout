#!/usr/bin/env python3
"""Position sizing overlays.

Round 3 showed a binary volatility gate beating every trend gate on this
universe. A gate is a step function on a continuous quantity, so the natural
next move is to size continuously instead of switching on and off.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)


def blended_vol(returns: pd.Series, windows=(10, 20, 60)) -> pd.Series:
    """Average of several vol estimates -- less jumpy than any single window."""
    return pd.concat([realized_vol(returns, w) for w in windows], axis=1).mean(axis=1)


def vol_target_scale(market_returns: pd.Series, target_vol: float = 0.35,
                     windows=(10, 20, 60), max_leverage: float = 1.0,
                     min_leverage: float = 0.0) -> pd.Series:
    """Exposure multiplier that pushes portfolio vol toward `target_vol`.

    Uses only trailing returns, and is shifted a day so today's scale is set
    from data available at yesterday's close.
    """
    vol = blended_vol(market_returns, windows)
    scale = (target_vol / vol.replace(0, np.nan)).clip(min_leverage, max_leverage)
    return scale.shift(1).fillna(0.0)


def inverse_vol_weights(weights: pd.DataFrame, close: pd.DataFrame,
                        window: int = 20) -> pd.DataFrame:
    """Re-weight a selection inversely to each name's own volatility.

    A 3x semiconductor fund and a 2x bond fund do not belong at equal weight.
    """
    daily = close.pct_change(fill_method=None)
    vol = daily.rolling(window, min_periods=window // 2).std().shift(1)
    inv = (1.0 / vol.replace(0, np.nan)).reindex_like(weights)
    raw = weights.where(weights > 0).mul(inv)
    total = raw.sum(axis=1)
    return raw.div(total.where(total > 0, np.nan), axis=0).fillna(0.0)


def apply_scale(weights: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    return weights.mul(scale.reindex(weights.index).fillna(0.0), axis=0)


def trailing_stop(weights: pd.DataFrame, close: pd.DataFrame, high: pd.DataFrame,
                  atr_frame: pd.DataFrame, mult: float = 3.0) -> pd.DataFrame:
    """Drop a name once it closes more than `mult` ATRs below its held peak.

    Walked forward one row at a time because the peak depends on when the
    position was opened, which a vectorized rolling window cannot express.
    """
    out = weights.copy()
    peak: Dict[str, float] = {}
    cols = weights.columns
    w_vals = weights.to_numpy()
    c_vals = close.reindex_like(weights).to_numpy()
    a_vals = atr_frame.reindex_like(weights).to_numpy()
    stopped = np.zeros(len(cols), dtype=bool)
    peak_arr = np.full(len(cols), np.nan)

    for i in range(len(weights)):
        held = w_vals[i] > 0
        # A name that leaves the book clears its stop state.
        peak_arr = np.where(held, np.fmax(peak_arr, c_vals[i]), np.nan)
        stopped = np.where(held, stopped, False)
        if i > 0:
            trigger = held & np.isfinite(peak_arr) & np.isfinite(a_vals[i]) & (
                c_vals[i] < peak_arr - mult * a_vals[i])
            stopped = stopped | trigger
        out.iloc[i] = np.where(stopped, 0.0, w_vals[i])
    return out


# --------------------------------------------------------------------------
# Better volatility estimators
#
# Close-to-close throws away the whole bar. Range-based estimators use the
# high/low and are far more efficient per observation, so they register a
# volatility spike several days sooner -- which is when a 3x book needs it.
# --------------------------------------------------------------------------
def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    log_hl = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * log_hl.rolling(window, min_periods=window // 2).mean() * 252)


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, window: int = 20) -> pd.Series:
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(daily.rolling(window, min_periods=window // 2).mean().clip(lower=0) * 252)


def blended_gk_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                   windows=(10, 20, 60)) -> pd.Series:
    return pd.concat([garman_klass_vol(open_, high, low, close, w) for w in windows],
                     axis=1).mean(axis=1)


def vol_target_scale_gk(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                        target_vol: float = 0.35, windows=(10, 20, 60),
                        max_leverage: float = 1.0, min_leverage: float = 0.0) -> pd.Series:
    """Vol targeting on a Garman-Klass estimate, with leverage permitted.

    `max_leverage` above 1.0 means the book may gear up when realized vol sits
    below target -- the point of holding a higher-Sharpe portfolio than the
    benchmark. Financing is charged separately by the caller.
    """
    vol = blended_gk_vol(open_, high, low, close, windows)
    scale = (target_vol / vol.replace(0, np.nan)).clip(min_leverage, max_leverage)
    return scale.shift(1).fillna(0.0)


def drawdown_scale(returns: pd.Series, start_dd: float = 0.15, full_dd: float = 0.35,
                   floor: float = 0.25) -> pd.Series:
    """De-lever as the strategy's OWN equity curve falls away from its peak.

    Exposure runs at 1.0 until drawdown reaches `start_dd`, then falls linearly
    to `floor` by `full_dd`. Computed from realized returns only and lagged a
    day, so it never sees the bar it is sizing.
    """
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = (equity / equity.cummax() - 1.0).abs()
    span = max(full_dd - start_dd, 1e-9)
    scale = 1.0 - (dd - start_dd).clip(lower=0.0) / span * (1.0 - floor)
    return scale.clip(floor, 1.0).shift(1).fillna(1.0)


def apply_financing(returns: pd.Series, exposure: pd.Series, rate: float = 0.05) -> pd.Series:
    """Charge a borrow cost on gross exposure above 1.0."""
    borrow = (exposure.reindex(returns.index).fillna(0.0) - 1.0).clip(lower=0.0)
    return returns - borrow * rate / 252.0


def deadband(scale: pd.Series, tolerance: float = 0.15, quantum: float = 0.05) -> pd.Series:
    """Hold an exposure level until it drifts materially, then snap to a grid.

    Continuous vol targeting re-sizes every position every day. At 1.0x that
    costs a few percent of turnover; geared up it ran to 66x turnover a year and
    ate ~6.5%/yr in costs, which is what sank the levered variants in round 7.
    Only move when the target is more than `tolerance` away in relative terms,
    and snap to a `quantum` grid so small oscillations do not trade.
    """
    values = scale.to_numpy(dtype=float)
    out = np.empty_like(values)
    current = np.nan
    for i, target in enumerate(values):
        if not np.isfinite(target):
            out[i] = current if np.isfinite(current) else 0.0
            continue
        if not np.isfinite(current) or abs(target - current) > tolerance * max(current, 1e-6):
            current = round(target / quantum) * quantum
        out[i] = current
    return pd.Series(out, index=scale.index)
