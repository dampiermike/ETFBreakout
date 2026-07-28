#!/usr/bin/env python3
"""Macro timing signals: volatility term structure and credit.

These series are never held. They only decide whether the leveraged book is
allowed to be invested. Kept out of etfs.txt so they cannot leak into the
tradable universe.

Credit ETFs use adjusted_close -- HYG and JNK carry large distributions, and
price-only series understate credit risk appetite badly. Volatility indices
have no distributions, so adjusted_close equals close.

Every adjusted series goes through `data._continuous_close`, the spliced-history
repair (see data.py). None of these names currently carries a weld -- they are
mature funds whose splits predate the incremental downloader -- so it is a no-op
today. It is wired in anyway: the sleeve holds GLD and UUP with real money, and
a weld there would be a fabricated return in the defensive leg, which is exactly
where it would be least visible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import data

HISTORY_DIR = Path(__file__).resolve().parent / "json" / "history"


def _read_frame(symbol: str) -> pd.DataFrame:
    path = HISTORY_DIR / f"{symbol.replace('.', '_')}.json"
    frame = pd.DataFrame(json.load(path.open()))
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame[(frame["close"] > 0) & frame["close"].notna()]
    return data.drop_placeholder_rows(frame)


def load_series(symbol: str, field: str = "adjusted_close") -> pd.Series:
    frame = _read_frame(symbol)
    series = (data._continuous_close(frame, symbol) if field == "adjusted_close"
              else frame[field].astype(float))
    return series[series > 0]


def load_macro() -> Dict[str, pd.Series]:
    out = {}
    for name in ("VIX.INDX", "VIX9D.INDX", "VXN.INDX", "VVIX.INDX", "VIX3M.INDX"):
        try:
            out[name.split(".")[0]] = load_series(name, "close")
        except Exception:
            pass
    for name in ("HYG.US", "LQD.US", "JNK.US", "AGG.US", "TLT.US", "IEF.US", "SHY.US",
                 "SPY.US", "QQQ.US"):
        try:
            out[name.split(".")[0]] = load_series(name, "adjusted_close")
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------
# Volatility term structure
# --------------------------------------------------------------------------
def vix_term_structure(m: Dict[str, pd.Series], near: str = "VIX", far: str = "VIX3M",
                       threshold: float = 1.0) -> pd.Series:
    """Contango filter: near-dated vol below far-dated means a calm market.

    When VIX rises above VIX3M the curve inverts -- the classic stress read.
    """
    ratio = m[near] / m[far].reindex(m[near].index).ffill()
    return ratio < threshold


def vix9d_ratio(m: Dict[str, pd.Series], threshold: float = 1.0) -> pd.Series:
    """Very short-dated vol vs spot -- the fastest curve signal available."""
    ratio = m["VIX9D"] / m["VIX"].reindex(m["VIX9D"].index).ffill()
    return ratio < threshold


def vix_level(m: Dict[str, pd.Series], key: str = "VIX", lookback: int = 252,
              max_pctile: float = 0.90) -> pd.Series:
    """Stand down when spot vol sits in the top percentile of its own year."""
    series = m[key]
    rank = series.rolling(lookback, min_periods=lookback // 4).rank(pct=True)
    return rank <= max_pctile


def vix_below_ma(m: Dict[str, pd.Series], key: str = "VIX", window: int = 50) -> pd.Series:
    """Rising-volatility filter: spot above its own average is a warning."""
    series = m[key]
    return series < series.rolling(window, min_periods=window // 2).mean()


def vvix_filter(m: Dict[str, pd.Series], lookback: int = 252, max_pctile: float = 0.90) -> pd.Series:
    """Vol-of-vol -- often moves before spot vol does."""
    series = m["VVIX"]
    return series.rolling(lookback, min_periods=lookback // 4).rank(pct=True) <= max_pctile


# --------------------------------------------------------------------------
# Credit
# --------------------------------------------------------------------------
def credit_ratio_trend(m: Dict[str, pd.Series], risky: str = "HYG", safe: str = "LQD",
                       window: int = 50) -> pd.Series:
    """High-yield versus investment-grade, on total return.

    Credit reliably leads equities into trouble: the ratio rolls over while the
    index is still making highs.
    """
    ratio = (m[risky] / m[safe].reindex(m[risky].index).ffill()).dropna()
    return ratio > ratio.rolling(window, min_periods=window // 2).mean()


def credit_momentum(m: Dict[str, pd.Series], key: str = "HYG", window: int = 60) -> pd.Series:
    """Is the risky credit leg itself still advancing?"""
    series = m[key]
    return series / series.shift(window) - 1.0 > 0


def flight_to_quality(m: Dict[str, pd.Series], window: int = 50) -> pd.Series:
    """Equity vs long treasuries -- risk-on when stocks outrun bonds."""
    ratio = (m["SPY"] / m["TLT"].reindex(m["SPY"].index).ffill()).dropna()
    return ratio > ratio.rolling(window, min_periods=window // 2).mean()


def build_macro_regimes(m: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if "VIX3M" in m:
        out["vix_contango"] = vix_term_structure(m, "VIX", "VIX3M", 1.0)
        out["vix_contango_95"] = vix_term_structure(m, "VIX", "VIX3M", 0.95)
    if "VIX9D" in m:
        out["vix9d_ratio"] = vix9d_ratio(m, 1.0)
    out["vix_pctile_90"] = vix_level(m, "VIX", 252, 0.90)
    out["vix_pctile_80"] = vix_level(m, "VIX", 252, 0.80)
    out["vix_below_ma50"] = vix_below_ma(m, "VIX", 50)
    if "VXN" in m:
        out["vxn_below_ma50"] = vix_below_ma(m, "VXN", 50)
        out["vxn_pctile_90"] = vix_level(m, "VXN", 252, 0.90)
    if "VVIX" in m:
        out["vvix_pctile_90"] = vvix_filter(m, 252, 0.90)
    out["credit_hyg_lqd"] = credit_ratio_trend(m, "HYG", "LQD", 50)
    out["credit_hyg_lqd_20"] = credit_ratio_trend(m, "HYG", "LQD", 20)
    out["credit_jnk_agg"] = credit_ratio_trend(m, "JNK", "AGG", 50)
    out["credit_mom_hyg"] = credit_momentum(m, "HYG", 60)
    out["spy_vs_tlt"] = flight_to_quality(m, 50)
    return out


if __name__ == "__main__":
    m = load_macro()
    print("loaded series:")
    for k, v in sorted(m.items()):
        print(f"  {k:7s} {len(v):5d} rows  {v.index[0].date()} -> {v.index[-1].date()}")
    print("\nregimes (share of days risk-on, 2010+):")
    for k, v in sorted(build_macro_regimes(m).items()):
        w = v.loc["2010-03-11":]
        print(f"  {k:20s} {w.mean():.1%}  ({w.index[0].date()} -> {w.index[-1].date()})")


# --------------------------------------------------------------------------
# Defensive sleeve
# --------------------------------------------------------------------------
def load_ohlc(symbol: str) -> pd.DataFrame:
    """Adjusted OHLC for a non-universe asset, matching data.load_ticker.

    Needed so defensive holdings trade on the same open-to-open convention as
    the equity book rather than on closes.
    """
    frame = _read_frame(symbol)
    ratio = data._continuous_close(frame, symbol) / frame["close"]
    out = pd.DataFrame(index=frame.index)
    for col in ("open", "high", "low", "close"):
        out[col] = frame[col] * ratio
    return out.dropna(subset=["close"])


DEFENSIVE = ["BIL", "SHY", "IEF", "TLT", "AGG", "GLD", "UUP", "DBC", "LQD"]


# --------------------------------------------------------------------------
# Sleeve routing -- the second gate dimension (round 25/26)
# --------------------------------------------------------------------------
# The gross gate answers "how much exposure". It is silent on "defensive in
# WHAT", and that silence was expensive: gold is a crisis hedge, not a rate
# hedge. Through the 2021-22 drawdown GLD returned -9.9% and TLT -31.4% while
# the sleeve carried most of the book; the dollar returned +20.7%. Routing the
# sleeve on a bond-trend read is worth ~+0.03 ladder Sharpe, and the controls
# confirm it is the rates read doing the work: inverting the signal scores 0.876
# and holding UUP unconditionally scores 0.932, both below the 0.953 baseline.
def bond_trend_ok(m: Dict[str, pd.Series], index: pd.DatetimeIndex,
                  window: int = 50, key: str = "TLT") -> pd.Series:
    """True while bonds are not falling -- trailing momentum >= 0.

    Falling long bonds mean rising rates, the regime in which gold fails. Uses
    only closes through t, so it is safe against the open-to-open fill.
    """
    s = m[key].reindex(index).ffill()
    return (s / s.shift(window) - 1.0 >= 0).fillna(True)


def routed_sleeve(index: pd.DatetimeIndex, calm_ok: pd.Series,
                  calm: str = "GLD", stress: str = "UUP") -> pd.DataFrame:
    """Defensive-sleeve weights: `calm` asset when calm_ok, else `stress`.

    Rows sum to 1; multiply by (1 - gross) to get book weights.
    """
    cond = calm_ok.reindex(index).ffill().fillna(True).astype(bool)
    out = pd.DataFrame(0.0, index=index, columns=DEFENSIVE)
    out[calm] = cond.astype(float)
    out[stress] = (~cond).astype(float)
    return out


def defensive_bar_returns(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Open-to-open returns for the defensive candidates, stamped like the book."""
    out = {}
    for sym in DEFENSIVE:
        try:
            opens = load_ohlc(f"{sym}.US")["open"]
        except Exception:
            continue
        out[sym] = (opens.shift(-1) / opens - 1.0).reindex(index)
    return pd.DataFrame(out)
