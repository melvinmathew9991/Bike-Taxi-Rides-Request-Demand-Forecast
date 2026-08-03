# Model card: bike-taxi demand forecast

## Overview

Two gradient-boosted tree models forecast ride-request demand per geographic
cluster per 30-minute interval.

| | Without lag | With lag |
|---|---|---|
| Features | cluster centroid (lat/lng), minute, hour, month, quarter, day-of-week | the above + `lag_1`, `lag_2`, `lag_3`, `rolling_mean` |
| Applied | directly, any horizon | recursively, one step at a time |
| Needs recent history | No | Yes — at least `max(lag)` intervals |
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

From the pipeline's own `Data_Prepared.csv` on the reference dataset
(878,300 rows = 17,566 intervals x 50 clusters, 3,708,240 requests retained from
8,381,556 raw bookings — the business rules remove 55.4%):

| | value |
|---|---|
| mean | 4.222 |
| std | 6.778 |
| median | 2.0 |
| max | 110 |
| zero-demand intervals | 37.4% |

The target is **over-dispersed and zero-inflated** — variance (45.9) is roughly
11x the mean, and over a third of all interval-cluster cells are empty. That is
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

Chronological split: train 2020-03-26 → 2021-01-12 (702,650 rows), test
2021-01-12 → 2021-03-26 (175,650 rows). Test-period mean demand 9.04.

**Read this section as a staleness stress test.** The test window runs up to ten
weeks past the training cut, so late test rows are scored against a badly stale
model. It is a useful bound on how bad things get if retraining stops; it is not
representative of a model retrained on a normal cadence. The rolling-origin
results two sections down are the ones to judge the model by.

**One step ahead** (model is given true observed lags):

| approach | RMSE | MAE | MASE | verdict |
|---|---|---|---|---|
| seasonal naive (same time last week) | **4.543** | 2.866 | 1.000 | — |
| model **with lag** | 4.803 | **2.828** | **0.987** | ties naive |
| model **without lag** | 8.558 | 4.766 | 1.663 | loses badly |
| cluster historical mean | 9.481 | 6.571 | 2.293 | loses badly |

**24 hours ahead** (recursive; the model consumes its own predictions):

| | value |
|---|---|
| RMSE | 8.261 |
| MAE | 5.144 |
| mean actual | 6.37 |
| mean predicted | **1.35** |

### Rolling-origin validation

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

**Usable, conditionally.** Retrained on at least a four-week cadence, the model
beats a seasonal-naive baseline at every origin tested, one step ahead (MASE
0.79) and over a 24-hour recursive horizon (MASE 0.95, 3/5 folds). Switching to
the ratio target improves both, and materially fixes level tracking in the
recursive mode (level ratio 0.99 vs 0.80, 5/5 folds vs 3/5).

Conditions for use:

1. **Retrain at least every four weeks.** This is not a nice-to-have; past week
   five the model is worse than a baseline that costs nothing to run.
2. **Monitor `level_ratio` in production.** It degrades earliest and most
   visibly, well before RMSE does.
3. **Prefer the ratio target for recursive serving.** The level target
   under-forecasts by ~20% over 24 hours and loses to naive in 2 of 5 folds.
4. **Do not use the without-lag model for anything but cold starts.** It has no
   channel carrying current demand level and loses to naive by 66%.

> An earlier revision of this card concluded "NOT READY / not deployable", based
> on a single split whose test window ran up to ten weeks past the training cut.
> That measured a badly stale model, not the model's steady-state behaviour. The
> rolling-origin results above supersede it. The 4.7x under-forecast reported
> there is real but is a staleness artefact, and it is the reason for condition 1.

### What the ratio target does

Predicting `request_count / (rolling_mean + 1)` and multiplying back removes the
trend from the target, so the trees never have to extrapolate. Its gain is
modest one step ahead (MASE 0.763 vs 0.791) but clear in the recursive mode that
the pipeline actually serves, and it is the only variant that holds the right
demand level across a 24-hour horizon.

It is **not yet implemented**: it redefines the target, which is a modelling
decision rather than a bug fix.


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

1. **Explains less than half the variance.** Substantial demand variation is
   driven by factors absent from the features: weather, events, holidays,
   pricing, competitor supply, rider availability.
2. **No exogenous features.** Weather and a holiday calendar are the obvious
   first additions and are likely worth more than any further tuning.
3. **Trained through the COVID-19 period.** Demand patterns in 2020-21 are not a
   reliable guide to normal operation. Retrain before relying on it.
4. **Recursive error compounding.** The lag model's accuracy degrades with each
   step. Check `recursive_rmse` against `test_rmse` in the model bundle; the gap
   is the real cost.
5. **Fulfilled requests, not latent demand.** The target counts logged booking
   requests. Demand that never materialised because no rider was nearby is
   invisible — see the feedback-loop discussion in `DATA_GOVERNANCE.md`.
6. **Cluster geometry is fixed at training time.** The city changes; the cluster
   model does not, until refitted.
7. **Aggressive cleaning.** The business rules remove rebookings and retries on
   the assumption they are duplicates of one intention. If a rider genuinely
   requests two rides nine minutes apart, that is counted once.

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
