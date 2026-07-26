#!/usr/bin/env python3
"""Final verification of the locked strategy: full stats, both variants,
benchmarks, year-by-year, drawdowns, and today's book."""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine
import research
import strategy
from diagnose import drawdown_table

pd.set_option("display.width", 200)


def main() -> None:
    ctx = research.Context()
    base = strategy.Strategy(strategy.DEFAULT, ctx)
    defensive = strategy.Strategy(strategy.DEFENSIVE_VARIANT, ctx)

    core = base.run("LOCKED (vol90 + GLD)")
    safe = defensive.run("LOCKED-defensive (vol90+credit + GLD)")
    tq = engine.buy_and_hold(ctx.bar_returns, "TQQQ", start=ctx.start)
    ew = engine.equal_weight_universe(ctx.eligible, ctx.bar_returns, start=ctx.start)

    print("=" * 104)
    print("LOCKED STRATEGY -- 10 positions, 2010-03-11 -> 2026-07-23, 10bps/side")
    print("=" * 104)
    table = pd.DataFrame([r.summary_row() for r in (core, safe, tq, ew)]).set_index("strategy")
    cols = ["cagr", "vol", "sharpe", "sortino", "max_dd", "calmar", "turnover_yr", "final"]
    print(table[cols].to_string(float_format=lambda v: f"{v:,.3f}"))

    for res in (core, safe):
        print()
        print("=" * 104)
        print(f"{res.name}")
        print("=" * 104)
        ret = res.returns
        mid = ret.index[len(ret) // 2]
        a, b = engine.metrics(ret.loc[:mid]), engine.metrics(ret.loc[mid:])
        print(f"  1st half ({ret.index[0].date()} -> {mid.date()})  "
              f"CAGR {a['cagr']:7.2%}  Sharpe {a['sharpe']:5.2f}  maxDD {a['max_dd']:7.2%}")
        print(f"  2nd half ({mid.date()} -> {ret.index[-1].date()})  "
              f"CAGR {b['cagr']:7.2%}  Sharpe {b['sharpe']:5.2f}  maxDD {b['max_dd']:7.2%}")
        print("\n  worst drawdowns:")
        dd = drawdown_table(res.equity, top=5)
        dd["depth"] = dd["depth"].map(lambda v: f"{v:.1%}")
        print(dd.to_string(index=False))

    print()
    print("=" * 104)
    print("CALENDAR YEARS")
    print("=" * 104)
    yearly = pd.DataFrame({
        "locked": core.returns.groupby(core.returns.index.year).apply(lambda r: (1 + r).prod() - 1),
        "defensive": safe.returns.groupby(safe.returns.index.year).apply(lambda r: (1 + r).prod() - 1),
        "TQQQ": tq.returns.groupby(tq.returns.index.year).apply(lambda r: (1 + r).prod() - 1),
    })
    print(yearly.to_string(float_format=lambda v: f"{v:+.1%}"))
    print(f"\n  years beating TQQQ: locked {int((yearly['locked'] > yearly['TQQQ']).sum())}/{len(yearly)}   "
          f"defensive {int((yearly['defensive'] > yearly['TQQQ']).sum())}/{len(yearly)}")
    print(f"  worst year:         locked {yearly['locked'].min():+.1%}   "
          f"defensive {yearly['defensive'].min():+.1%}   TQQQ {yearly['TQQQ'].min():+.1%}")

    print()
    print("=" * 104)
    print("TODAY")
    print("=" * 104)
    for label, strat in (("base", base), ("defensive", defensive)):
        state = strat.today()
        print(f"\n  {label}: {state['state']}   ({state['eligible']} eligible names, "
              f"as of {state['asof'].date()})")
        for ticker, weight in state["holdings"].items():
            print(f"      {ticker:8s} {weight:6.2%}")


if __name__ == "__main__":
    main()
