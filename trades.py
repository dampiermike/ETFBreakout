#!/usr/bin/env python3
"""Reconstruct per-name trades from the locked strategy and rank the losers.

A TRADE is a contiguous spell in which one ticker carries nonzero weight in the
book -- from the bar its position is first held to the bar before it goes flat.
Two P&L views per trade:

  ret      the name's own compounded open-to-open return over the spell
           (what the position did, independent of size)
  pnl      the portfolio's gross return contribution = sum_t w_held[t]*r[t]
           over the spell (what it actually cost/made the book, ~1/10 of ret
           at a 10% weight -- this is the number that moves the equity curve)

Timing matches the engine: weight decided at close t is held on bar t+1, which
earns the open(t+1)->open(t+2) return. So held = weights.shift(1).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import strategy

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 400)


def extract_trades(weights: pd.DataFrame, bar_returns: pd.DataFrame,
                   start: str) -> pd.DataFrame:
    weights = weights.loc[start:].fillna(0.0)
    rets = bar_returns.reindex_like(weights).fillna(0.0)
    held = weights.shift(1).fillna(0.0)          # weight actually on risk each bar
    idx = held.index

    rows = []
    for ticker in held.columns:
        w = held[ticker].to_numpy()
        r = rets[ticker].to_numpy()
        on = w > 1e-9
        if not on.any():
            continue
        # boundaries of contiguous held spells
        edges = np.diff(on.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0])
        if on[0]:
            starts = [0] + starts
        if on[-1]:
            ends = ends + [len(on) - 1]
        for s, e in zip(starts, ends):
            span_r = r[s:e + 1]
            span_w = w[s:e + 1]
            trade_ret = np.prod(1.0 + span_r) - 1.0          # name's own return
            pnl = float(np.sum(span_w * span_r))             # gross book contribution
            rows.append({
                "ticker": ticker.replace("D_", "") + ("*" if ticker.startswith("D_") else ""),
                "entry": idx[s].date(),
                "exit": idx[e].date(),
                "bars": e - s + 1,
                "avg_w": float(span_w.mean()),
                "ret": trade_ret,
                "pnl": pnl,
            })
    return pd.DataFrame(rows)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    strat = strategy.Strategy()            # locked config, phase 0
    weights, combined = strat.weights()
    trades = extract_trades(weights, combined, strat.cfg.start)

    losers = trades[trades["ret"] < 0].copy()
    tot = len(trades)
    wins = tot - len(losers)
    print(f"Locked strategy (phase 0): {tot} trades  |  {wins} winners  {len(losers)} losers "
          f"({len(losers)/tot:.0%})   * = defensive GLD leg")
    print(f"Total loser drag on book (sum of pnl over losing trades): {losers['pnl'].sum():+.2%}")
    print(f"Worst single-name own-return: {losers['ret'].min():.2%}   "
          f"Worst single-name book P&L:   {losers['pnl'].min():+.2%}")

    print(f"\n===== {min(n, len(losers))} LARGEST LOSERS by the NAME'S OWN RETURN =====")
    by_ret = losers.sort_values("ret").head(n).reset_index(drop=True)
    show = by_ret.copy()
    show["ret"] = show["ret"].map(lambda v: f"{v:7.2%}")
    show["pnl"] = show["pnl"].map(lambda v: f"{v:+7.2%}")
    show["avg_w"] = show["avg_w"].map(lambda v: f"{v:.3f}")
    print(show.to_string(index=True))

    print(f"\n===== {min(n, len(losers))} LARGEST LOSERS by PORTFOLIO P&L (what actually hurt the book) =====")
    by_pnl = losers.sort_values("pnl").head(n).reset_index(drop=True)
    show2 = by_pnl.copy()
    show2["ret"] = show2["ret"].map(lambda v: f"{v:7.2%}")
    show2["pnl"] = show2["pnl"].map(lambda v: f"{v:+7.2%}")
    show2["avg_w"] = show2["avg_w"].map(lambda v: f"{v:.3f}")
    print(show2.to_string(index=True))

    trades.sort_values("pnl").to_csv("results/trades_locked.csv", index=False)
    print("\nwrote results/trades_locked.csv (all trades, sorted by book P&L)")


if __name__ == "__main__":
    main()
