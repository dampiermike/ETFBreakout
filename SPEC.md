# ETFBreakout — Strategy Specification

A dynamic-universe rotation over US-listed leveraged (2x/3x) ETFs, with a
two-dimensional risk gate. This document specifies the locked strategy precisely
enough to reimplement it from scratch.

**Status:** locked as of 2026-07-26. `verify_lock.py` reproduces the headline
exactly on all four metrics.

---

## 1. Headline results

Quoted on the **20-phase rebalance ladder**, never a single rebalance offset —
see §7.1 for why this distinction is not cosmetic.

| Metric | Locked | Prior lock (GLD-only sleeve) | Prior lock (binary gate) | B&H TQQQ |
|---|---|---|---|---|
| CAGR | **34.80%** | 33.30% | 31.95% | 40.3% |
| Sharpe | **1.008** | 0.953 | 0.844 | 0.66 |
| Max drawdown | **−47.10%** | −48.4% | −55.4% | −81.6% |
| Calmar | **0.739** | 0.688 | 0.577 | — |
| Turnover | ~12x/yr | 12.1x/yr | 16.5x/yr | — |

Period: **2010-03-11 → 2026-07-23** (16.3 years). $100,000 → ~$11.0M at the
prior lock; the routed sleeve is modestly higher.

**Two honesty caveats, both load-bearing:**

1. **Quote Sharpe as ~0.985, not 1.008.** The 50-day sleeve lookback is the peak
   of a swept 40–70 plateau. Quoting a swept peak repeats the error round 13
   caught with rebalance phase 0. The plateau average is the honest expectation.
2. **Buy-and-hold TQQQ produced more terminal wealth** (253x vs 110x). The case
   for this strategy is the path — Sharpe 1.008 vs 0.66, and −47% vs −82%
   drawdown — not raw return.

---

## 2. Data layer

### 2.1 Source and adjustment

Per-ticker EODHD daily JSON at `json/history/{TICKER}_US.json`, each an array of
rows with `date, open, high, low, close, adjusted_close, volume`.

Leveraged ETFs reverse-split constantly, so **every price column is scaled by
`adjusted_close / close`**:

```
ratio            = adjusted_close / close
open/high/low/close   *= ratio          # → adjusted price series
volume           /= ratio               # preserves dollar volume
dollar_volume     = close * volume      # raw close × raw volume
```

Then:
- Drop rows where `close <= 0` or `close` is NaN (cannot be adjusted).
- Clamp the bracket: `high = max(open, high, close)`, `low = min(open, low, close)`
  — a handful of rows carry a high/low that does not bracket open/close.
- Drop duplicate dates, keeping the last.

### 2.2 Inception overrides

`data.INCEPTION_OVERRIDES` truncates tickers whose file carries a reused ticker's
earlier, unrelated history. Currently `{"INTW": "2025-02-13"}`.

### 2.3 Known data hazards

- Single corrupt rows exist in some files (FAZ prints `close = $0.0001` on
  2008-11-19, giving a 10,000x adjusted jump; EDZ similarly on 2008-12-30). Both
  predate the backtest start. **Any universe expansion must screen for
  `|daily return| > 80%` in-window before admitting a name.**
- `~/Documents` is iCloud-synced with ~66% of files evicted (dataless). Cold
  reads cost ~0.6s each and can raise `ETIMEDOUT`. Bulk scans must thread the
  I/O and retry. The panel cache (`cache/panel.pkl`) exists largely for this.

---

## 3. Universe

### 3.1 Membership

- **`etfs.txt`** — the tradable universe, 50 long leveraged ETFs. Membership is
  the file's contents, full stop; there is no hardcoded exclusion in code.
- **`benchmarks.txt`** — names priced for comparison but never tradable
  (currently `TQQQ`). Loaded into the panel, skipped by the eligibility mask.

The panel = `load_symbols() + load_benchmarks()`.

### 3.2 Eligibility (dynamic, evaluated daily)

A ticker is tradable on date *t* when **both** hold:

1. **History:** at least **60 bars** of its own history through *t*.
2. **Liquidity:** 20-day rolling median dollar volume ≥ **$1,000,000**
   (`min_periods=5`).

This makes the universe dynamic — names switch on as they list.

**Tradable count by year (mean):**

| 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 | 2024 | 2026 |
|---|---|---|---|---|---|---|---|---|
| 10.6 | 12.4 | 12.1 | 13.1 | 14.2 | 14.9 | 18.2 | 28.3 | 49.5 |

**This is a material limitation and must not be glossed.** The strategy picks the
top 10, but only ~10–15 names were eligible before 2021. For roughly the first
decade it therefore holds **most of everything available**, and the ranking signal
has little room to express a preference — those years approximate an equal-weight
leveraged-ETF portfolio under the risk gate, not a selective one. Genuine
selectivity only begins around 2023 (23 names) and is real by 2025–26 (43–50).

Two consequences worth carrying:
- Attribution differs by era. Pre-2021 performance is mostly *gate* (when to be
  invested and in what), not *selection*.
- It independently corroborates round 18, where trimming the universe to ~12
  tradable names collapsed Sharpe: below roughly 20 candidates, "top 10 of N" and
  "hold N" converge.

*Optional, not used in the lock:* `eligibility_mask` also accepts `min_r2` /
`max_solo_rate` for a rolling structural-character filter (see `data.characterize`).
Round 18 tested it and it **hurt**; it is retained but disabled.

---

## 4. Signal and selection

### 4.1 Score — `breakout_proximity(f, window=60)`

```
high_60 = rolling max of HIGH over 60 bars, EXCLUDING today (shift(1) first),
          min_periods = max(2, 30)
atr_20  = 20-bar mean of true range, min_periods = 10
score   = -( (high_60 - close) / atr_20 )
```

Higher is better: the score is the negative distance below the 60-bar high,
measured in ATR(20) units. It ranks names **near** their high, not names breaking
**through** it — all nine classic breakout *events* (Donchian, squeeze, volume
thrust, NR7, gap, pocket pivot) tested with negative edge.

`atr_20` divides with zeros replaced by NaN, so zero-range names score NaN and
drop out.

### 4.2 Selection — `rank_to_weights(score, eligible, top_n=10, rebalance_days=20)`

1. Mask the score to eligible names only.
2. Rank descending across tickers, `method="first"` (deterministic tie-break by
   column order).
3. Pick ranks ≤ 10 where the score is non-NaN.
4. Equal weight: `1 / count`. If fewer than 10 names are eligible, the surviving
   names still sum to 1.0.
5. **Rebalance stamping:** weights are held except on bars where
   `(i - phase) % 20 == 0`; other bars are NaN and forward-filled. Bar 0 is
   always stamped so every phase starts invested.

`rebalance_phase` exists **only** to measure timing luck. The locked
implementation reports the average over all 20 phases (§7.1).

**Why 10 positions:** the Sharpe curve peaks cleanly at 10 (0.95) and falls both
directions (5 → 0.72, 20 → 0.83). These are 2-3x funds where the variance penalty
is quadratic in leverage.

**Why equal weight:** inverse-vol posts a marginally better headline Sharpe but
decays +0.13 across halves versus equal weight's +0.02, and gives up 4 CAGR points.

---

## 5. Gate dimension 1 — how much exposure

Produces a daily **gross exposure** in [0, 1]. This replaced a binary in/out gate
in round 21.

### 5.1 Market proxy

No index file ships with the data, so the 1x S&P level is rebuilt by de-levering
SSO:

```
spx = cumprod( 1 + SSO.close.pct_change() / 2 )
```

Financing/decay drift is ignored — second-order for a filter.

### 5.2 Realized volatility

```
realized_vol(spx, 20) = std( spx.pct_change(), window=20,
                             min_periods = max(3, 10) ) * sqrt(252)
```

### 5.3 Graded gross

```
thr_lo = rolling 252-day 0.80 quantile of realized_vol  (min_periods=126)
thr_hi = rolling 252-day 0.95 quantile of realized_vol  (min_periods=126)
raw    = clip( (thr_hi - vol) / (thr_hi - thr_lo), 0, 1 )   # NaN span → NaN
raw    = raw.fillna(1.0)
```

Full size at or below the 80th percentile of the trailing year's own vol, zero at
or above the 95th, linear between.

### 5.4 Asymmetric ratchet — de-lever fast, re-lever slow

```
alpha = 1 - 0.5 ** (1 / 10)          # halflife = 10 bars
prev  = raw[0]
for each v in raw:                    # NaN → carry prev
    out = prev = v  if v < prev  else  prev + alpha * (v - prev)
```

Exposure follows the graded level **down immediately** but rebuilds toward it
exponentially. Finally: `reindex → ffill → fillna(1.0) → clip(0,1)`.

**This is the single most valuable component.** It improved return, drawdown,
turnover and the loss tail simultaneously — and *not* by taking more risk: mean
gross is **0.795** against the binary gate's 0.858 (full size on only 42.6% of
days, fully flat on 7.9%, partial the rest). Average position size on *losing*
trades falls from a flat 10% to 6.6%, cutting trades that cost the book ≥3% from
26 to 12 — which is what the failed stop-losses of rounds 12/15 were meant to do,
achieved without ever selling a name.

**Why not just retune the binary gate:** round 19 swept window {10,15,20} ×
percentile {0.85,0.90,0.95}; nine of ten cells lost 3–10 CAGR points. The
threshold was never the problem — the **discontinuity** was. The binary gate spent
30.4% of its defensive episodes lasting ≤3 bars, i.e. whipsawing back into 3x
longs mid-crash.

---

## 6. Gate dimension 2 — defensive in *what*

Dimension 1 is silent on what the defensive sleeve holds. That silence cost:
through the 2021-22 drawdown GLD returned **−9.9%** and TLT **−31.4%** while the
sleeve carried most of the book, while the dollar returned **+20.7%**. Gold is a
crisis hedge, not a rate hedge.

### 6.1 Rates read — `macro.bond_trend_ok(m, index, window=50, key="TLT")`

```
tlt = TLT adjusted_close, reindexed to the trading index, forward-filled
ok  = ( tlt / tlt.shift(50) - 1 >= 0 ).fillna(True)
```

Uses only closes through *t*, so it is safe against the open-to-open fill.

### 6.2 Sleeve routing — `macro.routed_sleeve(...)`

```
sleeve[GLD] = ok        # bonds stable or rising → gold
sleeve[UUP] = ~ok       # bonds falling (rates rising) → dollar
```

Rows sum to 1. Book weights are `sleeve[asset] * (1 - gross)`.

### 6.3 Evidence this is signal, not fitting

| Check | Sharpe | Reads as |
|---|---|---|
| GLD-only baseline | 0.953 | — |
| **GLD/UUP mom50 (locked)** | **1.008** | adopted |
| Lookback plateau 20→120 | .903 .943 .986 **1.008** .986 .961 .939 .954 .946 | smooth unimodal, not a spike |
| GLD/BIL, same sweep | peaks 0.979 | same shape, dollar > cash |
| **Inverted signal** | **0.876** | direction of the read matters |
| **Always UUP** | **0.932** | not "dollar beats gold" |
| **Static 50/50 GLD+BIL** | **0.934** | not diversification |
| Static ⅓ GLD+BIL+UUP | 0.938 | not diversification |
| TLT-only sleeve | 0.909, −58.7% DD | rejected |

Sleeve standalone: CAGR 7.9% → **9.0%**; its 2022 return −9.9% → **+18.3%**.

---

## 7. Execution and accounting

### 7.1 Timing convention

- **Bar returns are open-to-open, stamped at the bar that STARTS them:**
  `bar_returns[t] = open[t+1] / open[t] - 1`.
- `run_backtest` holds `weights.shift(1)`.
- Net effect: **a weight decided from close-of-*t* data is filled at *t+1*'s open**
  and earns *t+1*→*t+2*. No look-ahead. Any new signal must use only data through
  close of *t*.

### 7.2 Costs and turnover

- **10 bps per side**, charged as `turnover × 0.0010`.
- Turnover is measured against the **drifted** book, not the prior target — a
  position whose weight moved because the asset moved is not a trade.
- Drift divides by portfolio growth `(1 + portfolio return)`, **not** by the sum
  of drifted legs. Dividing by the leg sum renormalizes gross to 1.0 every bar and
  manufactures ~126x/yr of phantom turnover on any variant not targeting 1.0 gross.

### 7.3 Metrics — note the non-standard Sharpe

```
years   = len(returns) / 252
cagr    = final_equity ** (1/years) - 1
vol     = std(returns) * sqrt(252)
sharpe  = cagr / vol            # ← NO RISK-FREE RATE
sortino = cagr / downside_vol
max_dd  = min(equity / cummax(equity) - 1)
calmar  = cagr / |max_dd|
```

**`sharpe` here is CAGR-over-vol with no risk-free subtraction.** It is not a
textbook Sharpe ratio and is not comparable to externally quoted figures without
adjustment. Every number in this document uses this definition consistently.

### 7.4 The 20-phase ladder — the reporting standard

Round 13 established that phase 0 (rebalance anchored to the data start) is the
**single luckiest of 20 offsets**, flattering Sharpe by ~0.11–0.14. The reference
implementation therefore runs all 20 phases and averages the net-return series
(each 1/20 sleeve pays its own costs):

```
returns_ladder = mean over p in 0..19 of run_backtest(book(phase=p)).returns
```

Single-phase results run hot and must never be quoted as the headline.

---

## 8. Configuration reference

`strategy.Config` (frozen dataclass). The locked values:

| Field | Value | Meaning |
|---|---|---|
| `score_window` | 60 | proximity lookback (bars) |
| `atr_window` | 20 | ATR normalization |
| `positions` | 10 | equal-weight names |
| `rebalance_days` | 20 | rebalance cycle |
| `vol_window` | 20 | realized-vol window |
| `vol_lookback` | 252 | percentile lookback |
| `vol_band_lo` | 0.80 | full size at/below this vol percentile |
| `vol_band_hi` | 0.95 | zero size at/above |
| `ratchet_halflife` | 10.0 | bars to rebuild half the lost exposure |
| `graded_gate` | True | False → legacy binary gate |
| `vol_max_pctile` | 0.90 | binary gate only |
| `defensive` | "GLD" | sleeve while bonds hold |
| `defensive_stress` | "UUP" | sleeve while bonds fall |
| `rates_window` | 50 | TLT momentum lookback |
| `routed_defense` | True | False → always `defensive` |
| `use_credit_gate` | False | HYG-momentum AND-gate variant |
| `cost_bps` | 10.0 | per side |
| `start` | "2010-03-11" | backtest start |

Preset configs: `DEFAULT` (locked), `GLD_ONLY_SLEEVE` (round-21 lock),
`LEGACY_BINARY` (pre-round-21 lock), `DEFENSIVE_VARIANT` (credit gate on).

---

## 9. Negative results — do not re-run these

Each cost a full round; all are reproducible from the numbered scripts.

| Attempt | Round | Outcome |
|---|---|---|
| Breakout *events* (9 classic triggers) | 1 | negative edge; proximity beats them |
| Trend gates (200SMA, dual-MA, breadth) | — | −12+ CAGR points |
| Concentration (5 or 20 names) | 11 | Sharpe 0.72 / 0.83 vs 0.95 at 10 |
| Own-drawdown de-levering | 4 | halves CAGR (sells bottoms) |
| Stops / take-profits (full sweep) | 12, 15 | every level costs CAGR **and** raises loser-drag |
| Correlation caps / cluster caps | 16 | tail got **worse** (−49.7% → −86/−98%) |
| Multi-window persistence signal | 17 | deeper tail; selects stretched trends |
| Universe trim by market-R² | 18 | all risk metrics worse; idio names decorrelate |
| Annual walk-forward config re-selection | — | config-switching is itself the overfit |
| Adding 47 leveraged **inverse** ETFs | 23 | Sharpe 0.953 → 0.616; traded 42% of the time, bled −32% |
| Regime-**conditioned** inverse (8 variants) | 24 | monotonic — optimum inverse exposure is **zero** |

**The pattern across rounds 12–18:** the loss tail is intrinsic to holding
high-beta names near their highs. Every attempt to change *which names are held*
merely **relocates** the tail, usually to a worse source. Only changes to *how
much* is held, and *what the defense holds*, ever helped.

**Residual tail (irreducible):** the worst surviving trades (SOXL Jun-2015, UCO
Apr-2015) occurred while *market* vol was low, so the gate was correctly at full
size. A market-vol gate is structurally blind to idiosyncratic single-name
blowups, and round 18 showed removing such names costs more than it saves.

---

## 10. Open questions

- **Cumulative selection risk.** The gate was explored over 37 cells (rounds 19,
  21) and the sleeve over 22 more (25, 26). Every adopted change cleared a
  both-halves bar, but a clean out-of-sample re-validation has never been run.
- **Gold's conditional edge** rests on ~12% of days concentrated in a handful of
  episodes. The sleeve routing mitigates but does not remove this.
- **Survivorship.** The symbol snapshot is dated 2026-03-17; leveraged ETFs that
  delisted mid-period are absent from the universe and from any scan.
- **No hysteresis on universe entry/exit** — names enter the moment they clear
  60 bars and $1M.
- **Pre-2021 results are barely selective** (§3.2): with 10–15 eligible names and
  a top-10 book, the ranking signal has almost no room to choose. The strategy's
  first decade has never been separated into gate-attribution vs
  selection-attribution, and doing so would sharpen what the edge actually is.

---

## 11. Reproduction checklist

1. Place EODHD JSON histories in `json/history/`.
2. Confirm `etfs.txt` (50 names) and `benchmarks.txt` (TQQQ).
3. `python3 data.py` — rebuilds `cache/panel.pkl`, prints universe size by year.
4. `python3 verify_lock.py` — must print **PASS** with
   CAGR 0.3480 / Sharpe 1.008 / maxDD −0.471 / Calmar 0.739.
5. `python3 tearsheet.py` — per-year table and today's target book.

Any change to the gate or sleeve requires re-running step 4, and any new variant
should be scored through `gauntlet.py` (20-phase ladder + both half-samples +
loss tail) before it is believed.
