#!/usr/bin/env python3
"""Diagnostics for the leading strategy: when does it bleed, what does it hold,
and how correlated are its holdings really?"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine
import regime
import research
import signals
import sizing

pd.set_option("display.width", 220)


def drawdown_table(equity: pd.Series, top: int = 6) -> pd.DataFrame:
    dd = equity / equity.cummax() - 1.0
    underwater = dd < -0.01
    episodes = []
    start = None
    for date, flag in underwater.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            window = dd.loc[start:date]
            episodes.append((start, window.idxmin(), date, window.min()))
            start = None
    if start is not None:
        window = dd.loc[start:]
        episodes.append((start, window.idxmin(), dd.index[-1], window.min()))
    frame = pd.DataFrame(episodes, columns=["start", "trough", "end", "depth"])
    frame["days"] = (frame["end"] - frame["start"]).dt.days
    return frame.sort_values("depth").head(top)


def main() -> None:
    ctx = research.Context()
    ndx_ret = regime.proxy_index(ctx.f, "QLD", 2.0).pct_change(fill_method=None)
    score = signals.breakout_proximity(ctx.f, 60)
    base = engine.rank_to_weights(score, ctx.eligible, 10, rebalance_days=20)
    weights = sizing.inverse_vol_weights(base, ctx.f["close"])
    scale = sizing.vol_target_scale(ndx_ret, target_vol=0.25, max_leverage=1.0)
    res = engine.run_backtest(sizing.apply_scale(weights, scale), ctx.bar_returns,
                              name="leader", start=ctx.start)

    print("=" * 100)
    print("LEADER: prox60 top10 invvol vt25  --  worst drawdown episodes")
    print("=" * 100)
    dd = drawdown_table(res.equity)
    dd["depth"] = dd["depth"].map(lambda v: f"{v:.1%}")
    print(dd.to_string(index=False))

    print()
    print("=" * 100)
    print("CALENDAR-YEAR RETURNS vs TQQQ")
    print("=" * 100)
    tq = engine.buy_and_hold(ctx.bar_returns, "TQQQ", start=ctx.start)
    yearly = pd.DataFrame({
        "strategy": res.returns.groupby(res.returns.index.year).apply(lambda r: (1 + r).prod() - 1),
        "TQQQ": tq.returns.groupby(tq.returns.index.year).apply(lambda r: (1 + r).prod() - 1),
    })
    yearly["diff"] = yearly["strategy"] - yearly["TQQQ"]
    print(yearly.to_string(float_format=lambda v: f"{v:+.1%}"))

    print()
    print("=" * 100)
    print("HOW MUCH DIVERSIFICATION IS THERE? mean pairwise correlation of eligible names")
    print("=" * 100)
    daily = ctx.f["close"].pct_change(fill_method=None).loc[ctx.start:]
    for year in range(2011, 2027):
        window = daily[daily.index.year == year].dropna(axis=1, thresh=100)
        if window.shape[1] < 3:
            continue
        corr = window.corr().to_numpy()
        off = corr[~np.eye(len(corr), dtype=bool)]
        print(f"  {year}: n={window.shape[1]:2d}  mean pairwise corr = {np.nanmean(off):.3f}")


if __name__ == "__main__":
    main()
