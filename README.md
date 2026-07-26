# ETFBreakout

Dynamic-universe rotation over US-listed leveraged (2x/3x) ETFs with a
two-dimensional risk gate. Self-contained: price history is committed, so the
backtest runs on a fresh clone with no external data source.

**[SPEC.md](SPEC.md) is the authoritative description of the strategy.** Read it
before changing anything — it records what each decision beat, and eleven
approaches that were tested and rejected.

## Headline

20-phase rebalance ladder, 2010-03-11 → 2026-07-23, 10 bps per side:

| CAGR | Sharpe | Max drawdown | Calmar |
|---|---|---|---|
| 34.80% | 1.008 | −47.10% | 0.739 |

Two things to read carefully before quoting those numbers:

- **`sharpe` is CAGR ÷ vol with no risk-free subtraction** (see SPEC §7.3). It is
  not a textbook Sharpe ratio and is not comparable to externally quoted figures.
- **Quote Sharpe as ~0.985, not 1.008.** The 50-day sleeve lookback is the peak
  of a swept 40–70 plateau; the plateau average is the honest expectation.
- Buy-and-hold TQQQ produced more terminal wealth over this period (253x vs
  110x). The case here is the path — 0.66 Sharpe and −82% drawdown for TQQQ.

## Quick start

```bash
python3 data.py         # build cache/panel.pkl from json/history
python3 verify_lock.py  # must print PASS
python3 tearsheet.py    # per-year table and today's target book
```

`verify_lock.py` is the regression test. It must print:

```
cagr     got +0.3480  expected +0.3480  MATCH
sharpe   got +1.0083  expected +1.0080  MATCH
max_dd   got -0.4710  expected -0.4710  MATCH
calmar   got +0.7390  expected +0.7390  MATCH
PASS -- lock verified
```

Run it after any change to the gate, the sleeve, or the universe.

## Layout

| File | Role |
|---|---|
| `strategy.py` | `Config` + `Strategy` — the locked implementation |
| `data.py` | JSON loading, split adjustment, universe + eligibility |
| `research.py` | `Context` — loads the panel once, exposes wide OHLCV frames |
| `signals.py` | `breakout_proximity` and helpers |
| `engine.py` | `rank_to_weights`, `run_backtest`, `metrics` |
| `regime.py` | gate dimension 1 — graded gross exposure + ratchet |
| `macro.py` | gate dimension 2 — defensive sleeve routing; macro series |
| `gauntlet.py` | validation harness: 20-phase ladder + half-samples + tail |
| `verify_lock.py` | regression test against the locked headline |
| `trades.py` | per-trade log (entry/exit/own return/book P&L) |
| `tearsheet.py` | per-year returns and today's book |
| `diagnose.py` | drawdown table used by the tearsheet |
| `fetch_macro.py` | refresh all price series (needs `EODHD_API_TOKEN`) |
| `download_eodhd_history.py` | EODHD client used by `fetch_macro.py` |
| `etfs.txt` | the tradable universe — membership is this file, full stop |
| `benchmarks.txt` | priced for comparison, never tradable (TQQQ) |
| `json/history/` | committed price history, 70 symbols |

`cache/` and `results/` are generated and git-ignored.

## Adding a variant

Do not trust a single-phase backtest. Score every candidate through
`gauntlet.py`, which requires it to beat the baseline on the 20-phase ladder
**and** in both half-samples before it is believed. That bar is what rejected the
inverse-ETF universe expansion and the eight regime-conditioned variants that
followed it.

## Requirements

Python 3, `numpy`, `pandas`. `requests` only for `fetch_macro.py`.
