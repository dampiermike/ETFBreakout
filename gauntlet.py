#!/usr/bin/env python3
"""Reusable validation harness for gate variants (extracted from round 20).

Any candidate gate must clear the same bar before it is taken seriously:
  * 20-PHASE LADDER  -- phase 0 is the luckiest rebalance offset (round 13), so a
    single-phase headline proves nothing. Compare ladder-to-ladder.
  * HALF-SAMPLE      -- the improvement must appear in BOTH halves, not only in
    the full-sample aggregate (guards against sweep selection).
  * LOSS TAIL        -- worst trade, count of <=-30% / <=-40% trades, mean loser.
  * Q4-2018          -- the recurring stress event this project keeps failing.

A candidate is (label, kind, series) where kind is 'binary' (True = offense) or
'gross' (continuous exposure in [0, 1]); the remainder always goes to GLD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine
import macro
import regime
import research
import signals
import trades as tradelib

N_PHASE = 20
SPLIT = "2018-04-01"
Q_START, Q_TROUGH = "2018-11-08", "2018-12-24"
FLOAT = lambda v: f"{v:,.3f}"


class Harness:
    """Holds the fixed pieces (universe, score, 20 phase books) across candidates."""

    def __init__(self, top_n: int = 10, rebalance_days: int = 20, score_window: int = 60):
        self.ctx = ctx = research.Context()
        self.spx = regime.proxy_index(ctx.f, "SSO", 2.0)
        self.score = signals.breakout_proximity(ctx.f, score_window)
        defensive = macro.defensive_bar_returns(ctx.index)
        self.combined = pd.concat([ctx.bar_returns, defensive.add_prefix("D_")], axis=1)
        self.memberships = [
            engine.rank_to_weights(self.score, ctx.eligible, top_n,
                                   rebalance_days=rebalance_days, rebalance_phase=p)
            for p in range(N_PHASE)
        ]

    def align(self, s: pd.Series, fill):
        return s.reindex(self.ctx.index).ffill().fillna(fill)

    def book(self, membership: pd.DataFrame, kind: str, sig: pd.Series) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=self.ctx.index, columns=self.combined.columns)
        if kind == "binary":
            gate = self.align(sig, True).astype(bool)
            w[membership.columns] = regime.apply_regime(membership, gate)
            w.loc[~gate, "D_GLD"] = 1.0
            return w
        g = self.align(sig, 1.0).clip(0, 1)
        w[membership.columns] = membership.mul(g, axis=0)
        w["D_GLD"] = 1.0 - g
        return w

    def evaluate(self, label: str, kind: str, sig: pd.Series) -> dict:
        phase_res = [engine.run_backtest(self.book(self.memberships[p], kind, sig),
                                         self.combined, name=f"{label} p{p}",
                                         start=self.ctx.start)
                     for p in range(N_PHASE)]
        lad_ret = sum(r.returns for r in phase_res) / N_PHASE
        lad = engine.metrics(lad_ret, (1.0 + lad_ret).cumprod())

        def sub(lo, hi):
            seg = lad_ret.loc[lo:hi]
            return engine.metrics(seg, (1.0 + seg).cumprod())

        h1, h2 = sub(None, SPLIT), sub(SPLIT, None)
        sharpes = np.array([r.stats["sharpe"] for r in phase_res])
        cagrs = np.array([r.stats["cagr"] for r in phase_res])
        dds = np.array([r.stats["max_dd"] for r in phase_res])

        t = tradelib.extract_trades(self.book(self.memberships[0], kind, sig),
                                    self.combined, self.ctx.start)
        eq = t[~t["ticker"].str.endswith("*")]
        q = lad_ret.loc[Q_START:Q_TROUGH]

        return {
            "strategy": label,
            "lad_cagr": lad["cagr"], "lad_sharpe": lad["sharpe"],
            "lad_dd": lad["max_dd"], "lad_calmar": lad["calmar"],
            "ph_sharpe_mean": sharpes.mean(), "ph_sharpe_min": sharpes.min(),
            "ph_cagr_min": cagrs.min(), "ph_dd_worst": dds.min(),
            "h1_cagr": h1["cagr"], "h1_sharpe": h1["sharpe"],
            "h2_cagr": h2["cagr"], "h2_sharpe": h2["sharpe"],
            "n_trades": len(eq), "worst": eq["ret"].min(),
            "n_le_40": int((eq["ret"] <= -0.40).sum()),
            "n_le_30": int((eq["ret"] <= -0.30).sum()),
            "mean_loser": eq.loc[eq["ret"] < 0, "ret"].mean(),
            "q18": float((1.0 + q).prod() - 1.0),
            "turnover_yr": np.mean([r.diagnostics["turnover_yr"] for r in phase_res]),
        }

    def run(self, candidates) -> pd.DataFrame:
        return pd.DataFrame([self.evaluate(*c) for c in candidates]).set_index("strategy")


def report(table: pd.DataFrame, baseline: str, title: str) -> None:
    base = table.loc[baseline]
    print("=" * 132)
    print(title)
    print("=" * 132)

    print("\n--- LADDER (phase-independent, 20 offsets) ---")
    show = table[["lad_cagr", "lad_sharpe", "lad_dd", "lad_calmar", "turnover_yr"]].copy()
    show["d_sharpe"] = show["lad_sharpe"] - base["lad_sharpe"]
    show["d_cagr"] = show["lad_cagr"] - base["lad_cagr"]
    print(show.to_string(float_format=FLOAT))

    print("\n--- phase dispersion + half-sample ---")
    half = table[["ph_sharpe_mean", "ph_sharpe_min", "h1_sharpe", "h2_sharpe",
                  "h1_cagr", "h2_cagr"]].copy()
    half["h1_d"] = half["h1_sharpe"] - base["h1_sharpe"]
    half["h2_d"] = half["h2_sharpe"] - base["h2_sharpe"]
    print(half.to_string(float_format=FLOAT))

    print("\n--- loss tail (phase-0 trade log) + Q4-2018 (ladder) ---")
    print(table[["n_trades", "worst", "n_le_40", "n_le_30", "mean_loser", "q18"]]
          .to_string(float_format=FLOAT))

    print(f"\n--- VERDICT vs '{baseline}' (bar: ladder Sharpe AND both halves) ---")
    for label in table.index:
        if label == baseline:
            continue
        r = table.loc[label]
        lad_ok = r["lad_sharpe"] > base["lad_sharpe"]
        h1_ok = r["h1_sharpe"] > base["h1_sharpe"]
        h2_ok = r["h2_sharpe"] > base["h2_sharpe"]
        tail_ok = r["n_le_40"] <= base["n_le_40"] and r["n_le_30"] <= base["n_le_30"]
        verdict = "PASS" if (lad_ok and h1_ok and h2_ok) else "FAIL"
        print(f"  {label:26} {verdict:5} ladder {'+' if lad_ok else '-'} "
              f"H1 {'+' if h1_ok else '-'} H2 {'+' if h2_ok else '-'} "
              f"tail {'+' if tail_ok else '-'}   "
              f"(Sharpe {r['lad_sharpe']:.3f} vs {base['lad_sharpe']:.3f}, "
              f"CAGR {r['lad_cagr']:.2%} vs {base['lad_cagr']:.2%}, "
              f"n<=-40 {int(r['n_le_40'])} vs {int(base['n_le_40'])})")
