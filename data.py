#!/usr/bin/env python3
"""Data layer for the ETF breakout research.

Loads per-ticker EODHD JSON history from json/history/ and returns a panel of
split/dividend-adjusted OHLCV frames.

The raw files carry unadjusted OHLC alongside an `adjusted_close`. These
leveraged ETFs split constantly (TQQQ's 2010 close of 83.06 corresponds to an
adjusted 0.2064), so every price column is scaled by adjusted_close/close.
Volume is scaled by the inverse so that dollar volume is preserved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HISTORY_DIR = ROOT / "json" / "history"
SYMBOLS_FILE = ROOT / "etfs.txt"          # the tradable universe, nothing else
BENCHMARKS_FILE = ROOT / "benchmarks.txt"  # priced for comparison, never tradable
CACHE_FILE = ROOT / "cache" / "panel.pkl"

# INTW's file carries a reused ticker's pre-2025 history; the leveraged Intel
# ETF listed alongside the DLLL/QCML cohort. FNGU's series restarts after its
# 2025 reorganization -- the earlier product is a different security.
INCEPTION_OVERRIDES = {
    "INTW": "2025-02-13",
}

MIN_HISTORY_BARS = 60  # bars a ticker needs before it may be traded


def _read_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_symbols() -> List[str]:
    """The tradable universe. Membership is the file's contents, full stop."""
    return _read_list(SYMBOLS_FILE)


def load_benchmarks() -> List[str]:
    """Names priced for comparison but never tradable (TQQQ).

    These live in their own file rather than in etfs.txt behind a hardcoded
    exclusion set, so the universe file means exactly one thing: what the
    strategy may hold.
    """
    return [t for t in _read_list(BENCHMARKS_FILE) if t not in set(load_symbols())]


def load_ticker(ticker: str) -> pd.DataFrame:
    """Return an adjusted OHLCV frame indexed by date for one ticker."""
    path = HISTORY_DIR / f"{ticker}_US.json"
    frame = pd.DataFrame(json.load(path.open()))
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    cutoff = INCEPTION_OVERRIDES.get(ticker)
    if cutoff is not None:
        frame = frame.loc[pd.Timestamp(cutoff) :]

    # Rows with a zero/NaN close cannot be adjusted; they are unusable anyway.
    frame = frame[(frame["close"] > 0) & frame["close"].notna()]

    ratio = frame["adjusted_close"] / frame["close"]
    out = pd.DataFrame(index=frame.index)
    for col in ("open", "high", "low", "close"):
        out[col] = frame[col] * ratio
    out["volume"] = frame["volume"] / ratio.replace(0, np.nan)
    out["dollar_volume"] = frame["close"] * frame["volume"]

    # A handful of rows carry a high/low that does not bracket open/close.
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    return out.dropna(subset=["close"])


def build_panel(tickers: List[str] | None = None) -> Dict[str, pd.DataFrame]:
    """Load the tradable universe plus the benchmarks.

    Benchmarks must be in the panel so they can be priced; `eligibility_mask`
    is what keeps them out of the strategy.
    """
    tickers = tickers or (load_symbols() + load_benchmarks())
    return {t: load_ticker(t) for t in tickers}


def to_wide(panel: Dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """Stack one field across tickers into a date x ticker frame."""
    return pd.DataFrame({t: df[field] for t, df in panel.items()}).sort_index()


def characterize(panel: Dict[str, pd.DataFrame], window: int = 252,
                 min_periods: int = 126, spx_ticker: str = "SSO",
                 ndx_ticker: str = "QLD") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing structural character of every name, price-derived, no look-ahead.

    Two frames, both computed on a rolling `window`:

      r2    fraction of the name's daily variance explained by the broad US
            equity market -- de-levered S&P (SSO/2) plus Nasdaq (QLD/2). High
            r2 = the risk is market-wide (the vol gate rotates to GLD in those
            events, and the name recovers). Low r2 = idiosyncratic single-name /
            commodity / crypto risk the gate is blind to and that may not recover.
      solo  rate of 'solo crash' days -- name fell > 8% while the market was
            flat-or-up (> -2%). Catches structural bleeders (UVIX) whose r2 looks
            fine only because of a large inverse beta, and spares bond leverage
            (TMF: low equity-r2 but low solo rate = systematic, not blowup).

    The 2-factor rolling R2 is built from rolling covariance moments rather than
    a per-window regression, so it is cheap:  R2 = Sxy' Sxx^-1 Sxy / Syy.
    """
    close = to_wide(panel, "close")
    rets = close.pct_change(fill_method=None)
    spx = rets[spx_ticker] / 2.0
    ndx = rets[ndx_ticker] / 2.0

    def roll(s):
        return s.rolling(window, min_periods=min_periods)

    v11 = roll(spx).var()
    v22 = roll(ndx).var()
    v12 = roll(spx).cov(ndx)
    det = (v11 * v22 - v12 ** 2).replace(0, np.nan)
    mkt = spx + ndx

    r2 = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    solo = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for t in close.columns:
        y = rets[t]
        c1 = roll(spx).cov(y)
        c2 = roll(ndx).cov(y)
        vy = roll(y).var().replace(0, np.nan)
        explained = c1 ** 2 * v22 - 2 * c1 * c2 * v12 + c2 ** 2 * v11
        r2[t] = (explained / (det * vy)).clip(lower=0.0, upper=1.0)
        crash = ((y < -0.08) & (mkt > -0.02)).astype(float)
        solo[t] = roll(crash).mean()
    return r2, solo


def eligibility_mask(panel: Dict[str, pd.DataFrame], min_bars: int = MIN_HISTORY_BARS,
                     min_dollar_volume: float = 1e6, min_r2: float | None = None,
                     max_solo_rate: float | None = None, r2_window: int = 252) -> pd.DataFrame:
    """True where a ticker is tradable: enough history and enough liquidity.

    This is what makes the universe dynamic -- names switch on as they list
    rather than requiring every ticker to exist on day one.

    With `min_r2` / `max_solo_rate` set, a structural character filter is layered
    on (see `characterize`): a name is tradable only while its TRAILING market-R2
    is high enough and its solo-crash rate low enough. Applied on trailing data,
    so a name is admitted only once it has demonstrated it is market-beta rather
    than idiosyncratic -- and drops out again if that stops being true. Names
    without enough history to estimate the character are excluded (conservative).
    """
    close = to_wide(panel, "close")
    tradable = set(load_symbols())
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    for ticker, df in panel.items():
        if ticker not in tradable:
            continue  # priced for benchmarking, never tradable
        bar_count = pd.Series(np.arange(len(df)) + 1, index=df.index)
        liquid = df["dollar_volume"].rolling(20, min_periods=5).median() >= min_dollar_volume
        ok = (bar_count >= min_bars) & liquid
        mask[ticker] = ok.reindex(close.index, fill_value=False)

    if min_r2 is not None or max_solo_rate is not None:
        r2, solo = characterize(panel, window=r2_window)
        if min_r2 is not None:
            mask &= (r2 >= min_r2).reindex_like(mask).fillna(False)
        if max_solo_rate is not None:
            mask &= (solo <= max_solo_rate).reindex_like(mask).fillna(False)
    return mask


def load_cached(rebuild: bool = False) -> Dict[str, pd.DataFrame]:
    if CACHE_FILE.exists() and not rebuild:
        return pd.read_pickle(CACHE_FILE)
    panel = build_panel()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(panel, CACHE_FILE)
    return panel


if __name__ == "__main__":
    panel = load_cached(rebuild=True)
    mask = eligibility_mask(panel)
    counts = mask.sum(axis=1)
    print(f"tickers: {len(panel)}  dates: {len(counts)}")
    print("\ntradable universe size over time:")
    for year in range(2010, 2027):
        window = counts[counts.index.year == year]
        if len(window):
            print(f"  {year}: mean {window.mean():5.1f}  max {window.max():3d}")
