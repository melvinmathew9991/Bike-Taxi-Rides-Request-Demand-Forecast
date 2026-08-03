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

### Measured performance on the reference dataset

Chronological split: train 2020-03-26 → 2021-01-12 (702,650 rows), test
2021-01-12 → 2021-03-26 (175,650 rows). Test-period mean demand 9.04.

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

### Deployment verdict: NOT READY

Three things follow from the table above, and none of them were visible under the
original evaluation.

1. **The lag features are the whole model.** Without them the model loses to
   "same time last week" by 66%. With them it merely ties it (MASE 0.987 — it
   wins narrowly on MAE, loses on RMSE). Under the original day-of-month split
   the lag block appeared to add ~1%; that measurement was an artefact of the
   split, not a property of the features.
2. **The recursive 24-hour forecast collapses.** Mean predicted demand is 1.35
   against an actual 6.37 — a 4.7x under-forecast. Each step feeds a slightly
   low prediction back in as the next step's lag, and over 48 steps the error
   compounds downward. **This is the mode the pipeline actually serves in**, and
   it is not fit for operational use.
3. **The cause is non-stationarity, not tuning.** Gradient-boosted trees cannot
   extrapolate beyond the target range they were trained on. With demand 3x
   higher in the test period than in training, the model is structurally
   incapable of reaching the right level; the lag features are the only channel
   carrying current level information, which is why removing them is fatal.

### What measurably fixes it

Measured on the same split (RMSE / MASE):

| variant | RMSE | MASE |
|---|---|---|
| seasonal naive | 4.543 | 1.000 |
| current: full-history, level target | 4.860 | 0.994 |
| train on last 8 weeks only | 4.184 | 0.880 |
| **ratio-to-rolling-mean target, last 8 weeks** | **3.892** | **0.848** |
| exponential recency weighting | 4.808 | 0.990 |

Predicting a *ratio to a recent baseline* rather than an absolute count removes
the trend from the target, so the trees no longer need to extrapolate. Combined
with a recent training window it beats the seasonal-naive baseline by 15%. This
is the recommended next change; it is not yet implemented, because it redefines
the target and that is a modelling decision, not a bug fix.

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
