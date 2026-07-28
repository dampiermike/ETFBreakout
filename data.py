#!/usr/bin/env python3
"""Data layer for the ETF breakout research.

Loads per-ticker EODHD JSON history from json/history/ and returns a panel of
split/dividend-adjusted OHLCV frames.

The raw files carry unadjusted OHLC alongside an `adjusted_close`. These
leveraged ETFs split constantly (TQQQ's 2010 close of 83.06 corresponds to an
adjusted 0.2064), so every price column is scaled by adjusted_close/close.
Volume is scaled by the inverse so that dollar volume is preserved.

SPLICED HISTORY -- why `_continuous_close` exists
-------------------------------------------------
The JSON files are maintained by an INCREMENTAL downloader: it appends only the
rows it is missing and never rewrites the rows it already has. EODHD, like every
vendor, restates its whole history when a corporate action lands. The two do not
compose. A split that happens today is applied by the vendor to every row, but
lands in our file only on the rows downloaded from that day forward -- so the
file carries two different price bases welded together at whatever date the last
incremental run happened to stop. Every such weld is a fabricated overnight
return, and on these names they are enormous:

    SMCL  2026-04-13   1:20 reverse split ->  +2119% in one open-to-open bar
    MSTX  2026-03-17   1:10 reverse split ->   +936%
    IONX  2026-03-17   1:3  reverse split ->   +201%
    MULL/KORU/INTW/DLLL/MVLL  2026-05-22  -> -85% to -96% on one day

They come in two shapes, and both are repaired here:

  factor step   `adjusted_close/close` jumps at the weld -- the vendor's
                adjustment factor is present on the new rows and absent on the
                old ones. Exactly one of the two series is continuous across the
                weld, so the smaller of the two moves is the real one.

  raw restate   both columns jump together (the vendor rebased the raw prices
                too, so the factor is 1.0 on both sides and nothing looks wrong
                column-wise). Detected as a move that sits on a clean split
                factor AND is matched by an inverse move in volume -- a share
                count changing, not a price. Real crashes fail that second test:
                SMCL's genuine -67% on 2026-03-20 came with 10x the volume, not
                a tenth of it.

The repair rebuilds one continuous return series and re-anchors it on the most
recent `adjusted_close`, so recent price levels are the real traded ones and
history is back-adjusted onto them. `python3 data.py` prints every repair it
made; anything it prints that is NOT a corporate action is a bug in the
detector, not in the vendor.

The durable fix is a full (non-incremental) re-download, which makes this layer
a no-op. It stays regardless: it is what makes a stale file fail loudly instead
of silently paying a 2119% return.
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

# -- spliced-history repair (see the module docstring) ---------------------
FACTOR_STEP_TOL = 0.05    # factor steps under this are distributions, not welds
IMPLAUSIBLE_MOVE = 1.5    # no fund moves this far on an adjustment factor change
SPLIT_MIN_MOVE = 2.0      # below this a weld cannot be told from a real bar
SPLIT_FACTOR_TOL = 0.15   # how near a clean split factor the move must sit
NORMAL_RANGE_MULT = 3.0   # own-day range allowed, in trailing-median multiples
NORMAL_RANGE_FLOOR = 0.20 # ...and never less than this in absolute terms
ACTIVITY_BAND = (0.2, 3.0)  # allowed dollar-volume change across a split

# Clean split factors, both directions. Deliberately integer-only: 3:2 and 4:3
# splits exist, but a 3x fund printing +50% is an ordinary Tuesday (SOXL did it
# on 2025-04-09), so those cannot be separated from real bars and are reported
# as suspects instead of repaired.
_SPLIT_FACTORS = sorted({f for n in range(2, 101) for f in (float(n), 1.0 / n)})

# Repairs made during this process, and bars that look like welds but did not
# clear the evidence bar: (ticker, date, kind, vendor move, repaired move).
REPAIRS: List[tuple] = []
SUSPECTS: List[tuple] = []


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


def drop_placeholder_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are not real sessions.

    EODHD ships two kinds of filler, and a full re-download brings MORE of them,
    not fewer: the sentinel `999999.9999` (DLLL's first row, and 803 BOIL rows in
    the 2026-07-28 refresh) and flat prints on token volume (SMCL opens at
    $0.0123 on 17 shares). Left in, each one is a fabricated return of thousands
    of percent at the edge of the block.

    The flat rule is skipped for series carrying no volume at all -- volatility
    indices print volume 0 on every row and are legitimately flat on quiet days.
    """
    bad = (frame["close"] >= 999999) | (frame["adjusted_close"] >= 999999)
    if (frame["volume"] > 0).any():
        bad |= ((frame["open"] == frame["high"]) & (frame["high"] == frame["low"])
                & (frame["low"] == frame["close"]) & (frame["volume"] < 100))
    return frame[~bad]


def _nearest_split_factor(move: float) -> float | None:
    """The clean split factor `move` sits on, or None if it sits on none.

    A 1:10 reverse split rarely prints exactly 10x -- the underlying moved
    overnight too -- so the match is on log distance within SPLIT_FACTOR_TOL.
    """
    best = min(_SPLIT_FACTORS, key=lambda f: abs(np.log(move / f)))
    return best if abs(np.log(move / best)) <= np.log1p(SPLIT_FACTOR_TOL) else None


def _continuous_close(frame: pd.DataFrame, ticker: str = "") -> pd.Series:
    """One continuous total-return close series, welds repaired.

    Works in gross returns (price relatives) rather than levels: the level is
    exactly what the vendor disagrees with itself about. `adjusted_close` is the
    default source because it carries the distributions; the raw column only
    wins where the factor steps, and the split branch only ever divides out a
    share-count change.
    """
    close = frame["close"].astype(float)
    adjusted = frame["adjusted_close"].astype(float)
    volume = frame["volume"].astype(float)

    factor = (adjusted / close).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    step = factor / factor.shift(1)
    raw_move = (close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    adj_move = (adjusted / adjusted.shift(1)).replace([np.inf, -np.inf], np.nan)

    # A share count change lands entirely in the overnight gap and leaves the
    # session itself ordinary. A real -55% bar does the opposite: it happens
    # DURING the day, with a range several times the name's normal one. That is
    # what separates SMCX's 1:2 reverse split (2026-03-19: +105% gap, 6.9%
    # range) from its genuine crash (2026-06-10: -55%, 81% range) -- volume
    # cannot, since both come with a collapse in share count or a spike in
    # activity.
    gap = (frame["open"].astype(float) / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    day_range = (frame["high"].astype(float) / frame["low"].astype(float) - 1.0)
    normal_range = day_range.shift(1).rolling(60, min_periods=10).median()
    med_before = volume.shift(1).rolling(5, min_periods=1).median()
    med_after = volume[::-1].rolling(5, min_periods=1).median()[::-1]
    lo, hi = ACTIVITY_BAND

    move = adj_move.copy()
    for i in range(1, len(close)):
        r_adj, r_raw = adj_move.iat[i], raw_move.iat[i]
        if not (np.isfinite(r_adj) and r_adj > 0):
            move.iat[i] = 1.0
            continue

        if abs(step.iat[i] - 1.0) > FACTOR_STEP_TOL:
            # The vendor's adjustment factor changes here -- either a corporate
            # action it handled (then `adjusted_close` is the continuous column,
            # which is already the default) or a weld (then the raw column is).
            # Only an implausible adjusted move indicts the adjusted column: a
            # fund CAN distribute 5%+ at year end, and there the adjusted return
            # is the right one.
            weld = (not 1 / IMPLAUSIBLE_MOVE < r_adj < IMPLAUSIBLE_MOVE
                    and np.isfinite(r_raw) and r_raw > 0
                    and abs(np.log(r_raw)) < abs(np.log(r_adj)))
            if weld:
                move.iat[i] = r_raw
                REPAIRS.append((ticker, close.index[i], "factor step",
                                r_adj - 1.0, r_raw - 1.0))
            continue

        # Factor unchanged, so both columns moved together and nothing looks
        # wrong column-wise. A weld shows up only as a bar that lands on a clean
        # split factor with the share count moving the other way.
        if 1 / SPLIT_MIN_MOVE < r_adj < SPLIT_MIN_MOVE:
            continue
        split = _nearest_split_factor(r_adj)
        if split is None:
            continue

        # 1. the jump is entirely overnight -- the gap itself carries the factor
        overnight = np.isfinite(gap.iat[i]) and _nearest_split_factor(gap.iat[i]) == split
        # 2. the session that follows it is an ordinary one for this name
        ceiling = max(NORMAL_RANGE_FLOOR, NORMAL_RANGE_MULT * normal_range.iat[i]) \
            if np.isfinite(normal_range.iat[i]) else NORMAL_RANGE_FLOOR
        ordinary = np.isfinite(day_range.iat[i]) and day_range.iat[i] <= ceiling
        # 3. dollar volume is roughly preserved -- a share count changed, not a
        #    price. Corroboration only, so the band is wide (a reverse split
        #    damps real activity too) and either the same-day or the 5-day
        #    median reading may carry it: SMCX's 2026-03-19 split is followed
        #    two days later by a genuine crash that floods the median window.
        activity = [r_adj * (volume.iat[i] / volume.iat[i - 1])
                    if volume.iat[i - 1] > 0 else np.nan,
                    r_adj * (med_after.iat[i] / med_before.iat[i])
                    if med_before.iat[i] > 0 else np.nan]
        corroborated = any(np.isfinite(a) and lo <= a <= hi for a in activity)

        if not (overnight and ordinary and corroborated):
            failed = ", ".join(k for k, ok in (("gap", overnight), ("range", ordinary),
                                               ("volume", corroborated)) if not ok)
            SUSPECTS.append((ticker, close.index[i], f"{split:g}:1 move, {failed} disagrees",
                             r_adj - 1.0, np.nan))
            continue
        move.iat[i] = r_adj / split      # keep the genuine overnight residual
        REPAIRS.append((ticker, close.index[i], f"unadjusted {split:g}:1 split",
                        r_adj - 1.0, r_adj / split - 1.0))

    move.iat[0] = 1.0
    level = move.fillna(1.0).cumprod()
    # Anchor on the newest row: today's prices are the ones actually traded, and
    # history is back-adjusted onto them.
    return level / level.iloc[-1] * adjusted.iloc[-1]


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
    frame = drop_placeholder_rows(frame)

    # Scale every column onto the repaired close rather than onto the vendor's
    # own adjustment factor, which is what the welds corrupt.
    ratio = _continuous_close(frame, ticker) / frame["close"]
    out = pd.DataFrame(index=frame.index)
    for col in ("open", "high", "low", "close"):
        out[col] = frame[col] * ratio
    out["volume"] = frame["volume"] / ratio.replace(0, np.nan)
    # Dollar volume stays on the vendor's own row (raw price x raw volume): both
    # come from the same snapshot, so it is unaffected by the weld even where
    # the price basis is stale.
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


def repair_report(records: List[tuple] | None = None) -> pd.DataFrame:
    """Spliced-history repairs made since import, newest first.

    Review this after any data refresh: every row should be a corporate action.
    Pass SUSPECTS for the bars that looked like welds but were left alone.
    """
    rows = REPAIRS if records is None else records
    frame = pd.DataFrame(rows, columns=["ticker", "date", "kind", "vendor", "repaired"])
    if frame.empty:
        return frame
    return frame.sort_values("date", ascending=False).reset_index(drop=True)


def _print_report(title: str, records: List[tuple]) -> None:
    print(f"\n{title}: {len(records)}")
    for _, r in repair_report(records).iterrows():
        repaired = "left alone" if not np.isfinite(r["repaired"]) else f"{r['repaired']:+8.2%}"
        print(f"  {r['ticker']:6s} {r['date'].date()}  {r['kind']:32s} "
              f"vendor {r['vendor']:+10.2%} -> {repaired}")


if __name__ == "__main__":
    panel = load_cached(rebuild=True)

    _print_report("spliced-history repairs", REPAIRS)
    _print_report("suspect bars (NOT repaired -- review by hand)", SUSPECTS)

    mask = eligibility_mask(panel)
    counts = mask.sum(axis=1)
    print(f"tickers: {len(panel)}  dates: {len(counts)}")
    print("\ntradable universe size over time:")
    for year in range(2010, 2027):
        window = counts[counts.index.year == year]
        if len(window):
            print(f"  {year}: mean {window.mean():5.1f}  max {window.max():3d}")
