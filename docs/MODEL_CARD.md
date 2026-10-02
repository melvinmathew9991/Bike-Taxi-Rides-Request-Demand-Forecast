# Model card: bike-taxi demand forecast

## Overview

Two gradient-boosted tree models forecast ride-request demand per geographic
cluster per 30-minute interval.

| | Without lag | With lag |
|---|---|---|
| Features | cluster centroid (lat/lng), minute, hour, month, quarter, day-of-week | the above + `lag_1`, `lag_2`, `lag_3`, `lag_48` (yesterday), `lag_336` (last week), `rolling_mean` |
| Applied | directly, any horizon | recursively, one step at a time |
| Needs recent history | No | Yes — **7 days**, contiguous, per cluster (`max(lag)` = 336 intervals) |
| Use when | forecasting far ahead, or history is unavailable | forecasting the next few intervals |

Algorithm: XGBoost, `objective="count:poisson"`, tree count chosen by early
stopping on a chronological validation tail.

## Intended use

**In scope.** Short-horizon operational planning — rider positioning, surge
anticipation, shift planning — at the level of a geographic cluster.

**Out of scope.**
- Any decision about an identifiable individual (rider pay, penalties, ranking).
  The model is fitted on aggregate counts and says nothing about a person.
- Long-horizon strategic forecasting. Trained on a single year that includes the
  COVID-19 period; it has no basis for multi-year projection.
- Areas outside the training footprint. The cluster model was fitted on
  Bangalore; coordinates elsewhere are assigned to a nearest cluster that means
  nothing.

## Training data

Aggregated demand grid derived from ~8.38 M booking requests, Bangalore,
2020-03-26 to 2021-03-26. See `docs/DATA_GOVERNANCE.md` — the models are trained
**only on aggregated counts**, never on personal data.

Target: `request_count`, requests per cluster per 30 minutes.

### Measured target statistics

From the pipeline's own `Data_Prepared.csv.gz` on the reference dataset
(878,300 rows = 17,566 intervals x 50 clusters, 3,866,172 requests retained from
8,381,556 raw bookings — the business rules remove 53.5%):

| | value |
|---|---|
| mean | 4.402 |
| std | 7.320 |
| median | 2.0 |
| max | 141 |
| zero-demand intervals | 37.0% |

> **The retained count changed on 2026-10-02.** Rule 1 — "same rider rebooking
> the same pickup pin within an hour" — was evaluated as a different rule: it
> flagged every row whose (rider, pin) recurred *anywhere* in the dataset,
> measured the gap against the rider's previous booking from *any* pin, and
> dropped the first of each group as well. It also compared floored hours, so
> "within an hour" spanned anything under two. Corrected, the rules retain
> 3,866,172 requests instead of 3,708,240 — **157,932 more**, +4.3%.
>
> Rule 1 itself drops 548,437 fewer rows, but Rule 2 (retries under 8 minutes
> apart) independently catches about 390,000 of them, which is the right outcome
> for a burst of requests minutes apart. The rows actually recovered are
> commuters' legitimate repeat bookings on separate days — which matters for a
> model whose core signal is habitual travel.

The target is **over-dispersed and zero-inflated** — variance (53.6) is roughly
12x the mean, and over a third of all interval-cluster cells are empty. That is
what motivates `count:poisson`, the zero floor on predictions, and restricting
percentage error to non-zero actuals.

> An earlier revision of this card quoted mean 0.62 / std 1.08 / max 14. Those
> figures came from the notebook's own `Data_Prepared.csv`, which was produced by
> an earlier and far more aggressive cleaning pass that retained only ~6.5% of
> raw bookings. They do not describe the data this pipeline produces. The table
> above is measured from the current pipeline's output.

### Non-stationarity — the dominant property of this dataset

Mean demand per interval, by month:

| 2020-03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 2021-01 | 02 | 03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.91 | 1.71 | 1.76 | 1.99 | 2.14 | 2.59 | 3.13 | 3.25 | 4.40 | 5.13 | 7.10 | 9.30 | 9.83 |

**Demand grew 5.2x across the year** — COVID-19 lockdown through recovery. Under
the chronological split the test period carries **3.0x** the mean demand of the
training period (9.04 vs 3.02).

This single fact dominates everything below, and the original day-of-month split
concealed it entirely: interleaving test weeks between training weeks meant both
sides had the same demand level, so a model anchored to that level scored well.

## Evaluation

### How it is evaluated

- **Chronological split.** Fit on the earliest ~80% of the timeline, score on the
  most recent ~20%. No training interval postdates any test interval.
- **Chronological validation tail** inside the training period for early
  stopping — a random fold would leak future intervals into the stopping rule.
- **Recursive backtest** for the lag model: it forecasts a full horizon from its
  own predictions, compounding its own errors. Scoring one-step-ahead
  predictions against *observed* lags hands the model ground truth at every step
  and flatters it substantially.

### Metrics reported

`rmse`, `mae`, `r2`, `poisson_deviance`, and `mape` **computed only over
non-zero actuals** with its coverage reported alongside.

Percentage error is near-meaningless for this target — it is undefined wherever
the actual is 0, which is the majority of rows. A previous implementation
divided by `y_true + 1e-8` and reported values around 1e10 as a percentage.

### Baselines it must beat

`ModelEvaluator.compare_to_baselines` scores the model against:
- **seasonal naive** — same interval one week earlier;
- **cluster mean** — the cluster's historical average.

MASE below 1 means the model beats the seasonal-naive forecast. A demand model
that cannot beat "same time last week" should not be deployed; the baseline is
free, interpretable, and needs no retraining. `compare_to_baselines` logs a
warning when the model loses.

### Single-split performance — a worst case, not the verdict

Measured from the run of 2026-10-02 (`model_version 20261002_214312`), the
first with both the daily/weekly lags and the corrected Rule 1. Chronological
split: train 2020-04-02 → 2021-01-14 (689,200 rows), test 2021-01-14 →
2021-03-26 (172,300 rows). Test-period mean demand 9.54 against the training
period's 3.17.

Training starts a week later than the grid does, and both row counts are lower
than before this lag set landed: `lag_336` makes the first 336 intervals of each
cluster unusable, which costs 13,300 rows of 878,300.

**Read this section as a staleness stress test.** The test window runs up to ten
weeks past the training cut, so late test rows are scored against a badly stale
model. It is a useful bound on how bad things get if retraining stops; it is not
representative of a model retrained on a normal cadence.

**One step ahead** (model is given true observed lags):

| approach | RMSE | MAE | MASE | verdict |
|---|---|---|---|---|
| model **with lag** | **4.037** | **2.420** | **0.805** | beats naive by 18% |
| seasonal naive (same time last week) | 4.901 | 3.063 | 1.000 | — |
| model **without lag** | 9.308 | 5.127 | 1.780 | loses badly |
| cluster historical mean | 10.229 | 6.953 | 2.302 | loses badly |

All four rows are scored on the **same** window — the lag model's test split, so
they are directly comparable. Note this makes the without-lag figure here
(9.308) differ slightly from the 9.232 in `model_registry.json`, which scores it
on its own split: without a weekly lag it needs no 336-interval warm-up, so its
test window starts ~2 days earlier and is 3,350 rows longer. Same model, same
settings, different window.

**24 hours ahead** (recursive; the model consumes its own predictions):

| | value |
|---|---|
| RMSE | 3.811 |
| mean actual | 6.74 |
| mean predicted | 5.93 |
| level ratio | **0.88** |

> **What the seasonal lags changed.** Before them the lag set stopped at 90
> minutes, and the model was being asked to beat a baseline built from the value
> 336 intervals earlier — a signal it had never been given. It did not:
>
> | | before | after |
> |---|---|---|
> | one-step MASE | 0.999 — fails the gate | **0.805** — clears it |
> | recursive 24h level ratio | 0.21 (predicted 1.35 vs actual 6.37) | **0.88** |
>
> **Compare MASE, not RMSE.** The Rule 1 correction retains 157,932 more
> requests, which raises mean demand per interval from 4.222 to 4.402 — so
> absolute error rose with it even though the model improved. RMSE is not
> comparable across the two datasets; MASE and R² are, and both moved the right
> way (MASE 0.999 to 0.805, R² 0.807 to 0.884).
>
> The level ratio is the one to note. A recursive forecast anchored to its
> training-era level used to under-forecast demand roughly five-fold over a day.
> With a weekly lag carrying the current level into every step, it tracks.

### The gain depends on the forecast horizon

The two new lags are only *predictions* once the horizon reaches past them. Over
a 48-step horizon, `lag_48` and `lag_336` are always real observations, so the
model has two strong anchors at every step and only `lag_1/2/3` compound. Past 48
steps `lag_48` starts consuming the model's own output, and past 336 so does
`lag_336`.

Measured by recursive backtest on the reference dataset, against seasonal-naive
on the same rows:

| horizon | | share of steps where `lag_48` is a prediction | RMSE | MASE | level ratio |
|---|---|---|---|---|---|
| 48 | 1 day | 0% | 3.811 | **0.794** | 0.88 |
| 96 | 2 days | 50% | 4.065 | **0.815** | 1.06 |
| 192 | 4 days | 75% | 5.211 | 0.887 | 0.90 |
| 336 | 1 week | 86% | 5.902 | 0.986 | 0.84 |
| 672 | 2 weeks | 93% | 6.220 | **1.064** | 0.78 |

**The deploy gate is cleared comfortably out to about two days, marginally at
four, and not at all beyond a week.** At a two-week horizon the model is worse
than seasonal-naive and should not be used; the baseline is free.

This is a property of the lag set, not a regression — at every horizon measured
the current model beats what preceded it (recursive RMSE 8.261 at one day
before these lags). But it means the headline MASE of 0.805 is a *one-step*
figure, and the recursive figure of 0.794 is a *one-day* figure. Neither
generalises to an arbitrary horizon, and `run_pipeline.py --horizon-steps` will
happily accept one.

### Rolling-origin validation

> **Measured before the daily and weekly lags landed.** Everything in this
> section and in *Model staleness* below was produced by
> `scripts/compare_strategies.py` against the previous lag set `(1, 2, 3)`. The
> absolute numbers therefore no longer describe the shipped model, and the
> single-split result above suggests they understate it substantially.
>
> They are kept because the *relative* comparison between strategies is still
> the best evidence available, and because re-running the sweep is a job in its
> own right — four strategies at five origins in two modes. **Re-run it before
> relying on any figure below.** Until then, treat the retraining cadence in
> *Deployment verdict* as the conservative reading it is.

A single split on a series this non-stationary measures the fortnight you held
out as much as the model. `ML_Pipeline.validation` evaluates a strategy at five
successive origins, always training on the past. Test window one week
(one-step) or 24 hours (recursive); `MASE < 1` beats seasonal-naive.

**One step ahead** (true observed lags), 5 folds:

| strategy | RMSE | MASE | worst fold | folds beating naive |
|---|---|---|---|---|
| ratio target, last 8 weeks | 2.902 | **0.763** | 0.788 | 5/5 |
| ratio target, full history | 2.928 | 0.767 | 0.796 | 5/5 |
| level target, last 8 weeks | 2.983 | 0.784 | 0.810 | 5/5 |
| level target, full history (current) | 3.030 | 0.791 | 0.820 | 5/5 |

**Recursive, 24-hour horizon** (model consumes its own predictions):

| strategy | RMSE | MASE | folds beating naive | level ratio |
|---|---|---|---|---|
| ratio target, last 8 weeks | 2.725 | **0.831** | 5/5 | **0.99** |
| ratio target, full history | 2.851 | 0.841 | 5/5 | 0.92 |
| level target, last 8 weeks | 3.105 | 0.902 | 4/5 | 0.90 |
| level target, full history (current) | 3.400 | 0.947 | 3/5 | 0.80 |

`level ratio` is mean predicted over mean actual. A recursive forecast that
decays toward the training-era level shows up here well before RMSE makes it
obvious.

### Model staleness: the binding operational constraint

A model frozen at 2020-12-01 and scored on successive weeks with no retraining:

| weeks stale | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 13 |
|---|---|---|---|---|---|---|---|---|---|
| MASE | 0.84 | 0.72 | 0.78 | 0.76 | 0.98 | **1.23** | 1.38 | 1.55 | 1.75 |
| level ratio | 0.93 | 0.95 | 0.95 | 0.89 | 0.73 | 0.59 | 0.56 | 0.52 | 0.49 |

**The model beats seasonal-naive for about four weeks, reaches parity at five,
and is worse than naive from week six onward.** By week 13 it forecasts half the
actual demand. Demand grew 5.2x across the year and trees cannot extrapolate
past their training range, so a stale model is anchored to a level the city has
left behind.

### Deployment verdict

**Usable.** The model clears its own deploy gate on the single chronological
split — MASE 0.805 one step ahead, beating seasonal-naive by 18% — and holds the
right demand level across a 24-hour recursive horizon (level ratio 0.88). It
does this on the staleness stress test, which is the harshest configuration
measured here, so a model retrained on a normal cadence should do better.

Before the seasonal lags it did **not** clear the gate: MASE 0.999 with
`compare_to_baselines` logging "ship the baseline instead until it does".

Conditions for use:

1. **Retrain at least every four weeks.** Carried over unchanged, and
   deliberately conservative: the staleness measurement behind it predates the
   current lag set, and a weekly lag should slow decay by carrying the current
   level into the model's inputs. That has not been re-measured, so the old
   cadence stands until it has been.
2. **Monitor `level_ratio` in production.** It degrades earliest and most
   visibly, well before RMSE does. Nothing in the repository computes it on a
   schedule yet.
3. **Keep the horizon at or below two days.** The default is one day. The gain
   decays as the new lags start consuming the model's own predictions, and by a
   one-week horizon the model only ties the baseline — see the horizon table
   above. Nothing in the code enforces this; `--horizon-steps` accepts any value.
4. **Supply 7 days of contiguous history per cluster.** The weekly lag makes
   this a hard precondition of recursive serving, not a preference — the
   forecaster refuses rather than guesses if it is missing. Note that gaps are
   filled with zero and a warning, which matters more over a week than it did
   over 90 minutes.
5. **Do not use the without-lag model for anything but cold starts.** It has no
   channel carrying current demand level and loses to naive by 75%.
6. **Run the baseline comparison at every retrain.** It is not yet wired into
   the pipeline, so it has to be run deliberately. If the model stops beating
   seasonal-naive, ship the baseline.

> An earlier revision of this card concluded "NOT READY / not deployable", based
> on a single split whose test window ran up to ten weeks past the training cut.
> That measured a badly stale model, not the model's steady-state behaviour. The
> rolling-origin results above supersede it. The 4.7x under-forecast reported
> there is real but is a staleness artefact, and it is the reason for condition 1.

### What the ratio target does

Predicting `request_count / (rolling_mean + 1)` and multiplying back removes the
trend from the target, so the trees never have to extrapolate. Its gain was
modest one step ahead (MASE 0.763 vs 0.791) but clear in the recursive mode the
pipeline actually serves.

It is **not yet implemented**: it redefines the target, which is a modelling
decision rather than a bug fix.

> It was previously described here as "the only variant that holds the right
> demand level across a 24-hour horizon". That is no longer true as stated: the
> weekly lag now holds the level too (ratio 0.90 against the old 0.21), and it
> does so without redefining the target. The ratio target may still add
> something on top — both attack the same root cause, that trees cannot
> extrapolate a trend — but the case for it has to be re-made against the
> current lag set rather than the old one.


### Historical performance (pre-refactor, for reference)

From the committed `Notebook/Model_Training.ipynb` outputs, under the **old**
day-of-month split and hyperparameters:

| Model | R² | RMSE train | RMSE test |
|---|---|---|---|
| Without lag | 0.420 | 0.814 | 0.848 |
| With lag | 0.454 | 0.790 | 0.840 |

Read these carefully:

- Target std is ~1.08 and test RMSE is ~0.84 — the model explains **under half**
  the variance.
- The full lag block bought a **~1% RMSE improvement**, which is close to nothing
  for a large increase in serving complexity (the lag model needs recent history
  and must be applied recursively).
- Train and test RMSE are nearly identical, which is the signature of
  **under**-fitting, not overfitting. At 100 trees and learning rate 0.01 the
  effective learning budget was about 1.0.
- The split was on day-of-month, so these numbers measure interpolation between
  known weeks, not forecasting.

**These figures are not comparable to the measured results above.** They were
produced on a different (far more aggressively cleaned) aggregation, under a
day-of-month split that concealed the non-stationarity. They are retained only
as a record of the original work.

## Known limitations

1. **Variance explained is now 0.88 (one step), not under half.** The earlier
   figure of R² 0.39 was the without-lag model; the lag model reached 0.81 and,
   with the seasonal lags, 0.88. Substantial variation remains driven by factors
   absent from the features: weather, events, holidays, pricing, competitor
   supply, rider availability.
2. **No exogenous features.** Weather and a holiday calendar are the obvious
   first additions and are likely worth more than any further tuning. For a
   two-wheeler service in a monsoon city, rainfall is plausibly the largest
   single driver of demand variance still unrepresented.
3. **Recursive serving needs 7 days of contiguous history per cluster.** A
   consequence of the weekly lag. Gaps are filled with zero and a warning, which
   is more consequential over a week than over the 90 minutes it used to be.
4. **Trained through the COVID-19 period.** Demand patterns in 2020-21 are not a
   reliable guide to normal operation. Retrain before relying on it.
5. **Recursive error compounding, quantified.** MASE over a recursive horizon
   goes 0.80 at one day, 0.88 at four days, 0.99 at one week and 1.07 at two —
   see the horizon table above. `recursive_rmse` in the bundle records the
   one-day figure only. Note it is *not* comparable to `test_rmse`: the two are
   measured on different windows.
6. **Fulfilled requests, not latent demand.** The target counts logged booking
   requests. Demand that never materialised because no rider was nearby is
   invisible — see the feedback-loop discussion in `DATA_GOVERNANCE.md`.
7. **Cluster geometry is fixed at training time.** The city changes; the cluster
   model does not, until refitted.
8. **Aggressive cleaning.** The business rules remove 53.5% of raw bookings, on
   the assumption that rebookings and retries are duplicates of one intention.
   If a rider genuinely requests two rides nine minutes apart, that is counted
   once. The thresholds — one hour at the same pin, eight minutes anywhere — are
   domain judgements, now tested against their stated intent in
   `tests/test_cleaning_rules.py` but not validated against ground truth, which
   would need labelled data about which requests became trips.

## Ethical considerations

See `docs/DATA_GOVERNANCE.md` § 4 for feedback loops, geographic equity,
automation bias, and purpose limitation. In brief: this model influences where
service is supplied, so its errors are not evenly distributed in their
consequences, and under-served areas are structurally the most exposed.

## Maintenance

- **Retrain** when demand patterns shift materially, and at minimum when the
  recursive backtest RMSE degrades against the recorded baseline.
- **Monitor** forecast error per cluster, not just globally.
- **Compare to the seasonal-naive baseline at every retrain.** If the model
  stops beating it, ship the baseline.
- Every trained model is recorded in `output/model_registry.json` with its
  metrics, parameters, feature list and training row count.
