# Model card: bike-taxi demand forecast

## Overview

Two gradient-boosted tree models forecast ride-request demand per geographic
cluster per 30-minute interval.

| | Without lag | With lag |
|---|---|---|
| Features | cluster centroid (lat/lng), minute, hour, month, quarter, day-of-week | the above + `lag_1`, `lag_2`, `lag_3`, `lag_48` (yesterday), `lag_336` (last week), `rolling_mean` |
| Applied | directly, any horizon | recursively, one step at a time |
| Needs recent history | No | Yes — **7 days**, contiguous, per cluster (`max(lag)` = 336 intervals) |
| Used for | only clusters with less than 7 days of history (cold starts) | every other cluster - the forecast |

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
- Areas outside the training footprint. Cleaning keeps only pickups inside
  12.70-13.30°N, 77.30-77.90°E (`BENGALURU_BBOX`), so the cluster model is
  fitted on Bangalore alone; coordinates elsewhere are assigned to a nearest
  cluster that means nothing.

## Training data

Aggregated demand grid derived from ~8.38 M booking requests, Bangalore,
2020-03-26 to 2021-03-26. See `docs/DATA_GOVERNANCE.md` — the models are trained
**only on aggregated counts**, never on personal data.

Target: `request_count`, requests per cluster per 30 minutes.

### Measured target statistics

From the pipeline's own `Data_Prepared.csv.gz` on the reference dataset
(878,300 rows = 17,566 intervals x 50 clusters, 3,709,432 requests retained from
8,381,556 raw bookings — the business rules remove 55.7%):

| | value |
|---|---|
| mean | 4.223 |
| std | 6.899 |
| median | 2.0 |
| max | 140 |
| zero-demand intervals | 35.8% |

> **Pickups outside Bengaluru are excluded since 2026-10-05** (Rule 6). The
> scope below always said Bangalore, but 156,740 cleaned pickups (4.05%) lay
> hundreds of km away — Hyderabad, Mysuru, Chennai, Odisha, Rajasthan — and
> took 8 of the 50 clusters, each centred between cities. They were the eight
> quiet clusters the monitor's replay drill kept flagging. Now all 50 centres
> are in the city. Accuracy is unchanged within noise: deploy-gate MASE 0.804
> against 0.806, 0 clusters losing to seasonal-naive either way, and R² 0.879
> against 0.884. RMSE fell (3.85 from 4.04) but is not comparable, because the
> clusters are different. The other figures on this card were measured before
> the change.

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
generalises to an arbitrary horizon, which is why `PipelineConfig` refuses a
horizon past two days, for the CLI and the API alike.

### Rolling-origin validation

Re-run on 2026-10-02 against the current lag set `(1, 2, 3, 48, 336)`. A single
split on a series this non-stationary measures the fortnight you held out as much
as the model, so `ML_Pipeline.modeling.validation` evaluates a *strategy* at five
successive origins, always training on the past. Test window one week (one-step)
or 24 hours (recursive); `MASE < 1` beats seasonal-naive.

**One step ahead** (true observed lags), 5 folds:

| strategy | RMSE | MASE | std | worst fold | beats naive |
|---|---|---|---|---|---|
| ratio target, full history | 3.006 | **0.761** | 0.031 | 0.791 | 5/5 |
| ratio target, last 8 weeks | 3.009 | 0.761 | 0.032 | 0.794 | 5/5 |
| level target, full history (current) | 3.063 | 0.771 | 0.025 | 0.795 | 5/5 |
| level target, last 8 weeks | 3.066 | 0.773 | 0.026 | 0.799 | 5/5 |

**Recursive, 24-hour horizon** (model consumes its own predictions):

| strategy | RMSE | MASE | std | beats naive | level ratio |
|---|---|---|---|---|---|
| ratio target, full history | 2.734 | **0.789** | 0.034 | 5/5 | 0.99 |
| level target, full history (current) | 2.767 | 0.791 | 0.022 | 5/5 | 0.94 |
| ratio target, last 8 weeks | 2.725 | 0.792 | 0.034 | 5/5 | 0.99 |
| level target, last 8 weeks | 2.763 | 0.795 | 0.024 | 5/5 | 0.95 |

`level ratio` is mean predicted over mean actual. A recursive forecast decaying
toward the training-era level shows up there well before RMSE makes it obvious.

#### The four strategies are now indistinguishable

| | before, lags (1,2,3) | after, lags (1,2,3,48,336) |
|---|---|---|
| spread across strategies, recursive MASE | 0.117 | **0.006** |
| fold-to-fold standard deviation | — | 0.029 |

**The spread between strategies is now five times smaller than the variation
between folds.** Choosing between them on this evidence would be choosing noise.
Before the seasonal lags the spread was four times the noise, and the ranking
meant something.

What changed, per strategy, in recursive mode:

| strategy | MASE before | after | change | beats naive |
|---|---|---|---|---|
| level, full history (shipped) | 0.947 | **0.791** | −0.157 | 3/5 → **5/5** |
| level, last 8 weeks | 0.902 | 0.795 | −0.107 | 4/5 → 5/5 |
| ratio, full history | 0.841 | 0.789 | −0.052 | 5/5 |
| ratio, last 8 weeks | 0.831 | 0.792 | −0.039 | 5/5 |

Two conclusions, and both correct a previous recommendation in this card.

**The ratio target no longer earns its place.** Its advantage in recursive mode
was 0.9470 − 0.8405 = **0.107** before; it is now 0.7905 − 0.7889 = **0.002**,
which is noise. That follows directly from *why* it worked: it removed the trend
from the target so the trees never had to extrapolate, and `lag_336` now carries
the current demand level into every step instead. There is nothing left for it to
fix, and it costs a redefined target and a wrapper at serving time. The level
target's level ratio rose from 0.80 to 0.94 without it.

**Restricting training to recent weeks has stopped helping, and now slightly
hurts.** "Last 8 weeks" improved the level target by 0.045 before (0.947 → 0.902);
it now costs 0.004 (0.791 → 0.795). Same reason: recency used to have to come from
the training window, and the weekly lag supplies it, so throwing away older data
only discards information. Both differences are inside the noise band, so the
honest statement is that it no longer helps — not that full history is now
provably better.

> The previous revision of this card recommended the ratio target, and reported
> the level target losing to naive in 2 of 5 folds. Both are superseded: every
> strategy now beats naive 5/5, and the ratio target's gain has gone. The
> recommendation was sound on the evidence available at the time; the lag change
> removed the problem it solved.

### Model staleness: the binding operational constraint

Re-measured on 2026-10-02 against the current lag set, by
`experiments/measure_staleness.py`. A model is frozen at an origin and scored on
successive weeks with no retraining, which is what a lapsed retraining schedule
produces: the model still sees observed demand arrive, it simply is not refitted.

**Three origins, not one.** The previous curve froze a single model at
2020-12-01. On a series where demand grew 5.2x in a year, one origin measures the
quarter you happened to pick as much as it measures decay — the same objection
this card raises against single train/test splits. Origins are 2020-09-01,
2020-10-15 and 2020-12-01.

| weeks stale | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 13 |
|---|---|---|---|---|---|---|---|---|---|
| one-step MASE, **mean** | 0.75 | 0.71 | 0.73 | 0.74 | 0.81 | 0.88 | 0.91 | 0.99 | 1.21 |
| one-step MASE, **worst origin** | 0.78 | 0.74 | 0.75 | 0.76 | 0.89 | **1.11** | 1.24 | 1.41 | 1.58 |
| recursive MASE, mean | 0.83 | 0.73 | 0.72 | 0.79 | 0.83 | **1.03** | 1.10 | 1.09 | 1.46 |
| level ratio, mean | 0.99 | 1.00 | 0.96 | 0.92 | 0.87 | 0.83 | 0.82 | 0.77 | 0.66 |

**The four-week cadence stands, and the reason is the worst case rather than the
average.** On the mean the model holds out to about ten weeks, which is far better
than the previous curve suggested. But the *worst* origin crosses MASE 1.0 at
**week six** — exactly where the old single-origin curve crossed it — and a
retraining cadence has to be set by the worst case, not the mean. Four weeks
leaves two weeks of margin against the earliest observed failure.

The spread across origins is the finding. At six weeks stale the three origins
score 0.71, 0.81 and 1.11. The 2020-12-01 origin is much the worst because it sits
where demand was accelerating hardest: mean demand over its probe window runs from
5.30 at one week to 10.86 at thirteen. Decay is not a property of the model alone
but of how fast the city is changing underneath it, so a cadence derived from a
calm quarter would be dangerous in a growing one.

Two things the weekly lag did and did not do:

- **It improved the average decay substantially.** Mean one-step MASE at six weeks
  is 0.88, against 1.23 on the previous lag set.
- **It did not fix the worst case.** 1.11 at six weeks, against 1.23. Better, but
  still beaten by a baseline that costs nothing.

That asymmetry is what justifies keeping a conservative cadence rather than
relaxing it on the strength of an improved mean.

**Level ratio degrades earlier than MASE**, which is why it is the thing to
monitor: it is already at 0.92 by week four and 0.83 by week six, while mean MASE
is still comfortably under 1. By thirteen weeks the model forecasts two-thirds of
actual demand.

> The previous revision reported a single-origin curve — 0.84, 0.72, 0.78, 0.76,
> 0.98, 1.23, 1.38, 1.55, 1.75 — measured against lags `(1, 2, 3)` by an ad-hoc
> run that was never committed, so it could not be reproduced or re-measured when
> the feature set changed. It happened to pick the worst of the three origins, so
> as a bound it was right; as a typical case it was pessimistic. The experiment is
> now a script.

### The promoted model was stale on the day it was trained

Found on 2026-10-03 by the dashboard's per-cluster view, which showed cluster 30
— the busiest, mean 25.8 requests per interval — losing to seasonal-naive over
the last eight weeks (MASE 1.003) while the deploy gate had reported 0 of 50
clusters losing.

**Cause: the model the pipeline saved was the one it scored.** Early stopping's
validation tail was folded back in, but the test window — the newest fifth of
the timeline — never was. Every model this pipeline has promoted was therefore
fitted on data ending at the chronological split: 2021-01-14, against history
running to 2021-03-26. **Ten weeks stale the day it was trained**, past the
six-week worst case in the table above. The four-week cadence could not have
helped; each retrain started ten weeks behind. The dashboard tile read "model
age: 0 days" throughout, because it measured the wall clock.

**Where it showed: the busiest cluster's evening peak.** The model beat the
baseline at every hour except 18:00–20:00, where actual demand averaged 66–78
and the model predicted 51–52. Cluster 30's demand doubled between December and
March; the model had never seen a count above 73 in that cluster and, across all
clusters, never predicts above about 67. Over the last eight weeks 6.3% of the
cluster's intervals exceeded its training maximum.

Measured by `experiments/measure_peak_error.py`, walking forward a week at a time over
the last eight weeks, one step ahead:

| strategy | MASE | clusters losing | cluster 30 MASE | peak level ratio | level ratio above training max | max prediction |
|---|---|---|---|---|---|---|
| frozen at the split (what was promoted) | 0.807 | 1 | **1.011** | 0.73 | 0.63 | 67 |
| level target, refitted weekly | 0.782 | 0 | **0.789** | 0.93 | 0.87 | 106 |
| ratio target over `rolling_mean`, weekly | 0.772 | 0 | 0.779 | 0.97 | 0.92 | 116 |
| ratio target over `lag_336`, weekly | 0.785 | 0 | 0.820 | 0.99 | 0.94 | 147 |

Peak hours are the busiest cluster's three busiest; "above training max" is the
170 intervals where it exceeded anything the frozen model was fitted on.

**Staleness was the cause, not the feature set.** Refitting weekly with nothing
else changed takes cluster 30 from 1.011 to 0.789 and clears every cluster. The
ratio targets add a little at the peak, discussed under the ratio target below.

**Fix.** Training now ends with a final refit on all data, at the tree count
early stopping chose (`PipelineConfig.refit_on_all_data`, on by default). Every
metric, the deploy gate included, still comes from the held-out fit — there is
nothing left to score the final model on, so the method is what is evaluated.
The bundle records `data_through`, and serving measures staleness as the gap
between that and the end of the observed history (`data_lag_days`), reported by
the API and the dashboard. Bundles saved before the fix recover `data_through`
from their split note, so the model promoted before this change now reports as
72 days stale, which it was. The dashboard scores only demand after
`data_through`, so a freshly refit model shows no accuracy figures until a week
of new demand has arrived, rather than in-sample ones.

### Deployment verdict

**Usable.** The model clears its own deploy gate on the single chronological
split — MASE 0.805 one step ahead, beating seasonal-naive by 18% — and holds the
right demand level across a 24-hour recursive horizon (level ratio 0.88). It
does this on the staleness stress test, which is the harshest configuration
measured here, so a model retrained on a normal cadence should do better.

Before the seasonal lags it did **not** clear the gate: MASE 0.999 with
`compare_to_baselines` logging "ship the baseline instead until it does".

Conditions for use:

1. **Retrain at least every four weeks.** Achievable only since 2026-10-03:
   before the final refit every model started ten weeks behind its data — see
   the section above. Now measured on the current lag set
   across three origins, not carried over. The weekly lag improved mean decay a
   great deal — MASE 0.88 at six weeks against 1.23 before — but the worst origin
   still loses to seasonal-naive at **week six**, and a cadence follows the worst
   case. Four weeks leaves two weeks of margin. Reproduce with
   `experiments/measure_staleness.py`.
2. **Monitor `level_ratio` in production.** It degrades earliest and most
   visibly, well before RMSE does. `biketaxi monitor` scores the serving
   model on the latest week of demand it was not fitted on. It exits 3 when the
   model is past the cadence in condition 1, loses to seasonal-naive, or has a
   level ratio outside 0.90-1.10. The 0.90 floor sits between weeks four (0.92)
   and five (0.87) of the staleness table above. Run it on a schedule; nothing
   in the repository schedules it.

   A replay drill on 2026-10-05 trained models with the final refit off, 14, 29
   and 43 days before the end of the data, and scored each on the last week
   (2021-03-20 to 26). All three held MASE 0.76 and a level ratio of 0.98-0.99:
   early 2021 showed no decay, unlike the 2020 origins of the table above. This
   is no reason to relax the cadence - it is set by the worst case, and the
   0.90 floor was not exercised. The drill did show that a per-cluster level
   band is noise on quiet clusters: it flagged the same eight every time, all
   beating the baseline, all under three requests per interval. The band now
   applies only to clusters at or above that volume; quieter ones are held to
   MASE alone.
3. **Keep the horizon at or below two days.** The default is one day. The gain
   decays as the new lags start consuming the model's own predictions, and by a
   one-week horizon the model only ties the baseline — see the horizon table
   above. Enforced: `PipelineConfig` refuses a `horizon_steps` past
   `MAX_HORIZON_DAYS` (two days, 96 intervals), so `biketaxi run` exits 2,
   and the API caps requests at the same limit.
4. **Supply 7 days of contiguous history per cluster.** The weekly lag makes
   this a hard precondition of recursive serving, not a preference — the
   forecaster refuses rather than guesses if it is missing. Note that gaps are
   filled with zero and a warning, which matters more over a week than it did
   over 90 minutes.
5. **Do not use the without-lag model for anything but cold starts.** It has no
   channel carrying current demand level and loses to naive by 75%. Enforced
   since 2026-10-09: the prediction stage forecasts a cluster with it only when
   that cluster lacks the 7 days of history the lag model reads, and the
   dashboard shows those clusters separately rather than offering the model as
   an alternative forecast. A history too short for every cluster is still
   refused, not handed to the weaker model.
6. **Run the baseline comparison at every retrain.** It runs as the deploy gate
   at the end of every training pass, and `biketaxi run` exits 3 when the
   model loses. If the model stops beating seasonal-naive, ship the baseline.

> An earlier revision of this card concluded "NOT READY / not deployable", based
> on a single split whose test window ran up to ten weeks past the training cut.
> That measured a badly stale model, not the model's steady-state behaviour. The
> rolling-origin results above supersede it. The 4.7x under-forecast reported
> there is real but is a staleness artefact, and it is the reason for condition 1.

### Prediction intervals

Every recursive forecast carries an 80% interval, `request_count_lower` to
`request_count_upper` (`lower`/`upper` in the API), in whole requests. Dispatch
costs are asymmetric - an under-served interval costs more than an idle rider -
so **the upper bound is the figure to plan supply from**, not the point
forecast.

**Poisson quantiles were the obvious interval and are wrong where it matters.**
The model's objective is Poisson, but demand is over-dispersed (one step ahead,
variance 1.4x the mean at moderate volume, 4.8x above 50 requests) and a growing
series is under-forecast. So the intervals are calibrated instead
(split-conformal): training backtests the held-out model recursively from 24
origins over the four weeks after its cut, the retraining cadence, and records
the 10th and 90th percentiles of the error scaled by `sqrt(prediction)`, per
horizon band (0-6 h, 6-24 h, 24-48 h) and predicted-volume band (<=1, 1-3, 3-10,
10-25, >25). They are stored on the bundle and applied to the refit model.

Measured by `experiments/measure_intervals.py`: fitted before the split,
calibrated on 12 origins in the first two weeks after it, scored on 12 origins
in weeks three and four - 57,600 forecasts it never saw, as stale as a deployed
model gets:

| | Poisson 80% | calibrated 80% |
|---|---|---|
| coverage | 0.782 | **0.797** |
| demand above the interval / below | 0.183 / 0.035 | 0.123 / 0.079 |
| coverage, predicted <= 1 | 0.903 | 0.828 |
| coverage, predicted 3-10 | 0.759 | 0.792 |
| coverage, predicted 10-25 | 0.708 | 0.787 |
| coverage, predicted > 25 | **0.555** | **0.732** |
| coverage, 24-48 h ahead | 0.767 | 0.801 |
| mean width (requests) | 5.7 | 6.5 |

Coverage is flat across the horizon and close to nominal at every volume except
the busiest band, which is the same weakness as the point forecast (limitation
9). Misses still lean above. Each training run records its own check as
`interval_holdout_coverage`: the later half of its origins scored against
quantiles from the earlier half. On the reference data that was 0.800.

Bounds are rounded inward (the lower up, the upper down), because the target is
a whole count; rounding outward pushed a nominal 80% to 90%. Both bounds are
widened if needed to include the rounded point forecast. Models trained before
intervals existed - including the hosted demo's - serve without them.

### What the ratio target does

Predicting `request_count / (rolling_mean + 1)` and multiplying back removes the
trend from the target, so the trees never have to extrapolate. On the previous lag
set that was worth 0.107 MASE in recursive mode and it was the recommended next
change.

**It is not implemented, and on current evidence it should not be.** The sweep was
re-run against lags `(1, 2, 3, 48, 336)` on 2026-10-02 and its advantage is now
0.002 MASE — noise, against a fold-to-fold standard deviation of 0.029.

The reason is that both changes attack the same root cause. Trees cannot
extrapolate a trend; the ratio target removed the trend from the target, and
`lag_336` instead hands the model the current level directly as a feature. Having
done the second, there is nothing left for the first to fix. The level target's
recursive level ratio is 0.94 without it, against 0.80 before.

**At the very top of the range it still helps a little.** Over the last eight
weeks, refitted weekly, the ratio target holds the busiest cluster's evening peak
at a level ratio of 0.97 against the level target's 0.93, and 0.92 against 0.87
on intervals above anything in the training data — see the stale-model section
above. A tree's prediction is bounded by its leaves, and `lag_336` cannot carry a
level the trees were never fitted on. That is one eight-week window and a
0.010 MASE gain overall, which is no stronger evidence than the sweep's, so the
target is unchanged.

**Re-tested on 2026-10-05, after the final refit and the Bengaluru geofence,
and still not adopted.** Peak under-forecasting persisted - the level target,
refitted weekly, holds the busiest cluster's peak hours at 0.93 of actual over
both the last 8 and the last 16 weeks (`experiments/measure_peak_error.py`). The
ratio target over `rolling_mean` raised that to 0.98 and 0.96, and won one step
ahead in every rolling-origin fold (`experiments/compare_strategies.py`, five
folds, 24-hour recursive horizon):

| fold | one step, level | one step, ratio | recursive, level | recursive, ratio | recursive level ratio, level / ratio |
|---|---|---|---|---|---|
| 1 | 0.768 | 0.761 | 0.813 | 0.799 | 0.85 / 0.92 |
| 2 | 0.795 | 0.791 | 0.780 | 0.787 | 1.01 / 1.06 |
| 3 | 0.741 | 0.720 | 0.771 | 0.755 | 0.85 / 0.93 |
| 4 | 0.790 | 0.783 | 0.803 | 0.833 | 1.03 / 1.06 |
| 5 | 0.780 | 0.768 | 0.812 | 0.806 | 0.92 / 0.97 |
| **mean** | **0.775** | **0.764** | **0.796** | **0.796** | **0.93 / 0.99** |

One step ahead the gain is real: 0.010 MASE, 5 of 5 folds, with a paired
standard deviation of 0.007. Over a 24-hour recursive horizon it is a tie: mean
difference +0.0002, 3 of 5 folds, paired standard deviation 0.019. The ratio
target does not correct the peaks specifically; it raises the level everywhere.
Where the level target under-forecast (folds 1, 3 and 5) that wins; where it
was already at or above 1.0 (folds 2 and 4) it over-forecasts to 1.06 and
loses. The pipeline and the API serve recursively, where there is nothing to
gain, and adopting it would mean a ratio wrapper in the bundle, serving, the
monitor and the dashboard.

**If the use changes, so does the answer.** For a next-interval use alone -
live dispatch, half an hour ahead - the ratio target over `rolling_mean` is the
better model.

Worth keeping in mind rather than discarding: if the lag set ever loses its weekly
component — a coarser interval, a shorter history requirement — the ratio target
becomes relevant again, because the problem it solves would come back.


### Historical performance (pre-refactor, for reference)

From the committed `notebooks/Model_Training.ipynb` outputs, under the **old**
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
   absent from the features: events, pricing, competitor supply, rider
   availability. Weather and holidays were measured and add nothing (item 2).
2. **No exogenous features, and weather would not help.** This card used to
   expect rainfall to be the largest missing driver. Measured on 2026-10-05 with
   `experiments/measure_weather.py`, it is not. Ten one-week folds from June 2020 to
   March 2021, chosen to cover the monsoon, scored one step ahead:

   | variant | MASE | vs base | better in | MASE, rainy intervals |
   |---|---|---|---|---|
   | base (shipped) | 0.7764 | — | — | 0.7638 |
   | + rain last 1h and 3h, temperature, humidity | 0.7782 | +0.0018 | 1/10 | 0.7632 |
   | + rain during the interval (perfect hindsight) | 0.7777 | +0.0013 | 2/10 | 0.7641 |
   | + Karnataka holiday flag | 0.7765 | +0.0001 | 4/10 | 0.7638 |

   Every difference is an order of magnitude inside the fold-to-fold standard
   deviation (0.014). The hindsight variant is the most weather could add, so a
   real weather forecast could only do worse. The reason is that demand barely
   responds: daytime city demand against the same slot a week earlier, when
   that slot was dry, has a median of 1.037 in dry intervals and 1.023 in rain
   of 3 mm or more. The base model already forecasts rainy intervals at a level
   ratio of 1.02.

   Holidays show a dip - median daily demand 0.95 of the week before, against
   1.03 on ordinary days - but it is inconsistent (0.73 at Dussehra, 1.45 at
   Buddha Purnima, the April 2020 ones under lockdown), only two fell in the
   test folds, and there are about 19 a year. Not worth a feature yet.

   Caveats: one weather point (12.97 N, 77.59 E, Open-Meteo reanalysis) stands
   in for the whole city, so local showers are averaged away; and only the
   one-step mode was tested, though with demand this unresponsive to rain a
   longer horizon has nothing to gain either. Weather data is fetched by
   `experiments/fetch_weather.py` into the git-ignored `data/`.
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
8. **Aggressive cleaning.** The business rules remove 55.7% of raw bookings,
   1.9 points of it pickups outside Bengaluru, and the rest on
   the assumption that rebookings and retries are duplicates of one intention.
   If a rider genuinely requests two rides nine minutes apart, that is counted
   once. The thresholds — one hour at the same pin, eight minutes anywhere — are
   domain judgements, now tested against their stated intent in
   `tests/test_cleaning_rules.py` but not validated against ground truth, which
   would need labelled data about which requests became trips.

9. **Peaks above the training range are under-forecast.** Trees cannot
   predict beyond the values they were fitted on. On a growing series the
   busiest clusters' peaks keep exceeding that range: refitted weekly, the model
   forecasts 0.87 of actual demand on intervals above anything it has seen. The
   final refit narrows the gap but cannot close it. The ratio target narrows it
   further one step ahead but ties over a 24-hour horizon, so it is not
   adopted; see the ratio target section.
10. **Intervals under-cover the busiest clusters.** The 80% interval covered
    73% of intervals predicted above 25 requests, against 79-83% elsewhere, and
    misses above outnumber misses below by about three to two. The upper bound
    is a better planning figure than the point forecast, but at the busiest
    clusters it is not yet a 90th percentile.

## Ethical considerations

See `docs/DATA_GOVERNANCE.md` § 4 for feedback loops, geographic equity,
automation bias, and purpose limitation. In brief: this model influences where
service is supplied, so its errors are not evenly distributed in their
consequences, and under-served areas are structurally the most exposed.

## Maintenance

- **Retrain** at least every four weeks, and sooner when
  `biketaxi monitor` exits 3 - stale, losing to seasonal-naive, or a
  level ratio outside 0.90-1.10. It scores the latest week the model was not
  fitted on; nothing schedules it yet, so run it weekly from cron or a
  scheduler wherever the output directory lives.
- **Monitor** forecast error per cluster, not just globally. The global number
  hid cluster 30 losing to the baseline; the per-cluster view found it. The
  monitor reports every cluster, and warns on any losing to the baseline or,
  above 3 requests per interval, outside the level band.
- **Measure staleness from the data, not the calendar.** `data_lag_days` — the
  history the model has not been fitted on — is what the decay curve is measured
  in. Wall-clock age reads zero for a model fitted today on old data.
- **Compare to the seasonal-naive baseline at every retrain.** If the model
  stops beating it, ship the baseline.
- Every trained model is recorded in `output/model_registry.json` with its
  metrics, parameters, feature list and training row count.
- **Promotion is what reaches production.** `ML_Pipeline.serving.api` serves the model
  marked `production` in the registry, never simply the newest. `promote_model`
  refuses a model that failed its deploy gate *or* that carries no verdict, and
  `rollback()` restores the previously promoted one. `biketaxi run --promote`
  does it as part of a training run; `biketaxi registry list | promote |
  rollback` does it from the command line, and `POST /reload` makes a running
  API serve the change without a restart.
- **The API caps the horizon at 96 intervals (two days)**, as does
  `PipelineConfig` for the CLI, and reports data lag and model age against the
  four-week cadence, so the two conditions of use above are enforced at the
  serving boundary rather than left to the caller.
- **Prediction intervals are recalibrated at every training run**, on the
  held-out fit. Check `interval_holdout_coverage` in the registry (also on
  `GET /model`) against the nominal `interval_level`; the reference run
  measured 0.800 for 0.80.
- **The input contract is checked whenever a booking file is loaded** (see
  `docs/DATA_SCHEMA.md`). A file that breaks it stops the run rather than being
  cleaned into a smaller dataset and trained on.
- **The hosted demo** (Cloud Run, see `deploy/README.md`) serves
  `xgb_with_lag_20261005_142518`, promoted 2026-10-05: geofenced data, refit on
  all data through 2021-03-26, deploy-gate MASE 0.804. It predates prediction
  intervals, so it serves `lower`/`upper` as null. It is frozen - the data ends
  there - so it reports itself stale 28 days after training. Retrain, promote
  and redeploy with `deploy/gcp_deploy.sh` to give it intervals.
