#!/usr/bin/env python3
"""Verify the adopted lock: strategy.py must reproduce the round-21 sweep result.

The sweep built its books inline; strategy.py builds them through Config/Strategy.
If those two paths disagree, the headline is an artifact of the harness. Checks
both the single-phase book and the 20-phase ladder, plus the legacy binary config
against its own known numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine
import macro
import regime
import research
import signals
import strategy
from gauntlet import N_PHASE

EXPECTED = {  # from results/round26_window_sweep.csv, "GLD/UUP mom50"
    "cagr": 0.3407, "sharpe": 0.994, "max_dd": -0.471, "calmar": 0.723,
}
# Re-locked 2026-07-28 on a full, non-incremental re-download. The previous lock
# (0.3480 / 1.008 / -0.471 / 0.739) was priced off a spliced history: two of the
# twenty rebalance phases held SMCL through a fabricated +2119% bar and seventeen
# held one of the 2026-05-22 welds through fabricated -85%/-96% bars. Repairing
# the splice in code recovered 0.3407/0.997; the refresh moved it a further
# 0.003 Sharpe (two more trading days, plus restated distributions). See data.py.


def ladder(cfg, ctx, gross, sleeve):
    """Run all 20 rebalance phases through the locked config and average."""
    score = signals.breakout_proximity(ctx.f, cfg.score_window)
    defensive = macro.defensive_bar_returns(ctx.index)
    combined = pd.concat([ctx.bar_returns, defensive.add_prefix("D_")], axis=1)
    rets = []
    for p in range(N_PHASE):
        base = engine.rank_to_weights(score, ctx.eligible, cfg.positions,
                                      cfg.rebalance_days, rebalance_phase=p)
        w = pd.DataFrame(0.0, index=ctx.index, columns=combined.columns)
        w[ctx.bar_returns.columns] = regime.apply_gross(base, gross)
        for asset in sleeve.columns:
            if f"D_{asset}" in w.columns and (sleeve[asset] != 0).any():
                w[f"D_{asset}"] = sleeve[asset] * (1.0 - gross)
        rets.append(engine.run_backtest(w, combined, name=f"p{p}",
                                        cost_bps=cfg.cost_bps, start=cfg.start).returns)
    r = sum(rets) / N_PHASE
    return engine.metrics(r, (1.0 + r).cumprod())


def main() -> None:
    ctx = research.Context(start=strategy.DEFAULT.start)

    strat = strategy.Strategy(strategy.DEFAULT, ctx)
    single = strat.run(name="locked-graded")
    lad = ladder(strategy.DEFAULT, ctx, strat.gross(), strat.sleeve())

    legacy = strategy.Strategy(strategy.LEGACY_BINARY, ctx).run(name="legacy-binary")

    print("=" * 92)
    print("VERIFY LOCK -- strategy.py vs the round-26 sweep")
    print("=" * 92)
    for label, s in (("graded lock, phase 0", single.stats),
                     ("graded lock, LADDER", lad),
                     ("legacy binary, phase 0", legacy.stats)):
        print(f"{label:24} CAGR {s['cagr']:7.2%}  Sharpe {s['sharpe']:5.3f}  "
              f"maxDD {s['max_dd']:7.2%}  Calmar {s['calmar']:5.3f}")

    print("\n--- ladder vs expected (round21 'E asymEWMA hl10 80-95') ---")
    ok = True
    for k, want in EXPECTED.items():
        got = lad[k]
        match = abs(got - want) <= 0.002
        ok &= match
        print(f"  {k:8} got {got:+.4f}  expected {want:+.4f}  "
              f"{'MATCH' if match else '*** MISMATCH ***'}")

    gross = strat.gross()
    print(f"\ngross exposure: mean {gross.mean():.3f}  "
          f"days at full {float((gross >= 0.999).mean()):.1%}  "
          f"days at zero {float((gross <= 0.001).mean()):.1%}")
    t = strat.today()
    print(f"today: {t['state']}  (gross {t['gross']:.1%}, sleeve {t['sleeve']})")
    print(f"\n{'PASS -- lock verified' if ok else 'FAIL -- strategy.py disagrees with the sweep'}")


if __name__ == "__main__":
    main()
