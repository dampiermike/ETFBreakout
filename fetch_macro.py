#!/usr/bin/env python3
"""Refresh every price series the backtest needs.

Two groups, kept separate on purpose:

  MACRO/TIMING  volatility indices, credit, rates, and the defensive candidates.
                Never held by the strategy except as the defensive sleeve; kept
                out of etfs.txt so they cannot leak into the tradable universe.
                The volatility indices live on .INDX, not the .US the universe
                downloader assumes, which is why they are listed explicitly.

  UNIVERSE      whatever is currently in etfs.txt and benchmarks.txt, suffixed
                .US. Included so this repository can refresh its own data
                without depending on any external project.

Incremental: each symbol resumes from the last date already on disk.
Requires EODHD_API_TOKEN in the environment.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from download_eodhd_history import fetch_history, sanitize, to_params

HISTORY_DIR = Path(__file__).resolve().parent / "json" / "history"
START = date(2000, 1, 3)

SYMBOLS = [
    "VIX.INDX",    # spot volatility
    "VIX9D.INDX",  # 9-day -- term structure vs VIX
    "VXN.INDX",    # Nasdaq volatility
    "VVIX.INDX",   # vol-of-vol
    "VIX3M.INDX",  # 3-month -- the other end of the curve
    "HYG.US",      # high yield credit
    "LQD.US",      # investment grade
    "JNK.US",      # high yield, second read
    "AGG.US",      # aggregate bond
    "TLT.US",      # long treasuries
    "IEF.US",      # 7-10y treasuries
    "SHY.US",      # 1-3y treasuries
    "SPY.US",
    "QQQ.US",
    "BIL.US",      # 1-3 month T-bills -- the honest cash proxy
    "GLD.US",      # gold
    "IAU.US",      # gold, second read
    "UUP.US",      # dollar
    "DBC.US",      # broad commodities
]


def universe_symbols() -> list:
    """Tradable universe plus benchmarks, as EODHD .US symbols."""
    root = Path(__file__).resolve().parent
    out = []
    for name in ("etfs.txt", "benchmarks.txt"):
        path = root / name
        if path.exists():
            out += [f"{line.strip()}.US"
                    for line in path.read_text().splitlines() if line.strip()]
    return out


def main() -> None:
    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        print("set EODHD_API_TOKEN", file=sys.stderr)
        sys.exit(1)

    today = date.today()
    symbols = list(dict.fromkeys(SYMBOLS + universe_symbols()))
    print(f"refreshing {len(symbols)} symbols "
          f"({len(SYMBOLS)} macro/defensive + {len(symbols)-len(SYMBOLS)} universe)")
    for symbol in symbols:
        path = HISTORY_DIR / f"{sanitize(symbol)}.json"
        existing = []
        start = START
        if path.exists():
            try:
                existing = json.load(path.open())
                dates = [r["date"] for r in existing if r.get("date")]
                if dates:
                    start = date.fromisoformat(max(dates)) + timedelta(days=1)
            except Exception:
                existing = []
        if start > today:
            print(f"{symbol}: up to date ({len(existing)} rows)")
            continue
        try:
            rows = fetch_history(symbol, token, to_params(start, today), 60.0)
        except Exception as exc:
            print(f"{symbol}: ERROR {exc}")
            continue
        if not rows:
            print(f"{symbol}: no new data (have {len(existing)})")
            continue
        merged = {r["date"]: r for r in existing if r.get("date")}
        for r in rows:
            if r.get("date"):
                merged[r["date"]] = r
        out = [merged[d] for d in sorted(merged)]
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"{symbol}: +{len(out) - len(existing)} rows -> {len(out)} "
              f"({out[0]['date']} -> {out[-1]['date']})")


if __name__ == "__main__":
    main()
