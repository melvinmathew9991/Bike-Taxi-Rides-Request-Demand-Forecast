# Bike-Taxi Ride-Request Demand Forecast

Forecasts ride-request demand per geographic cluster per 30-minute interval, from
booking logs, using XGBoost over calendar, geographic and lag features.

```
raw bookings
  -> clean & deduplicate            (data_prep_basic)
  -> business-rule filtering        (data_prep_advanced)
  -> cluster pickups & aggregate    (data_prep_geospatial)   <- personal data ends here
  -> train two models               (model_training)
  -> forecast a horizon             (prediction_pipeline)
```

## Live demo

The forecast API runs on Google Cloud Run:
**https://bike-taxi-forecast-mu5g6m6liq-el.a.run.app** — interactive docs at
[`/docs`](https://bike-taxi-forecast-mu5g6m6liq-el.a.run.app/docs).

- Every endpoint but `/health` currently needs an API key (`X-API-Key` header),
  available on request, until the dataset's terms are confirmed to allow
  public forecasts.
- The data ends on 2021-03-26, so forecasts are for 2021-03-27 onwards. It is a
  frozen demo, not a live service, and after 28 days it reports itself stale.
- It serves only aggregated files from a private bucket; no booking-level data
  is in the cloud. See [deploy/README.md](deploy/README.md).

```bash
curl https://bike-taxi-forecast-mu5g6m6liq-el.a.run.app/health
curl -H "X-API-Key: <key>" \
  "https://bike-taxi-forecast-mu5g6m6liq-el.a.run.app/forecast?steps=48&cluster=7"
```

## Quick start

```bash
pip install -r requirements-dev.txt        # or requirements.txt to run, not test
pytest                                    # 327 tests, no data needed
python run_pipeline.py --raw-data data/raw_data.csv --n-clusters 50
streamlit run streamlit_app.py            # dashboard, incl. model performance
```

The repository ships **no data** — it carries personal data and is git-ignored.
The test suite builds synthetic data in-memory, so a fresh clone can run `pytest`
immediately. To supply your own input see [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md) | What personal data this handles, where it stops, handling rules, ethical considerations. **Read before working with the data.** |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use, evaluation approach, known limitations. |
| [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) | Input and output formats. |
| [deploy/README.md](deploy/README.md) | Deploying the API to Google Cloud Run: what is uploaded, setup, cost, updating, teardown. |

## Usage

### Command line

```bash
python run_pipeline.py                          # full pipeline, defaults
python run_pipeline.py --stages data features   # subset of stages
python run_pipeline.py --n-clusters 100 --test-fraction 0.25 --horizon-steps 96
python run_pipeline.py --config output/pipeline_config_20240101_120000.json
python run_pipeline.py --config run.json --n-clusters 100   # flag wins
python run_pipeline.py --promote                 # also promote it for serving
python run_pipeline.py --allow-failed-gate       # exit 0 even if it loses
```

`run_pipeline.py` exits **3** when the trained model loses to its seasonal-naive
baseline — a distinct code, because the run itself succeeded and only the model
is inadequate.

Every flag reaches the code it names; `run_pipeline.py --help` lists them all.
`--config` and the flags compose: the snapshot sets the starting point, and any
flag you pass overrides it (and is logged as an override).

### Python

```python
from ML_Pipeline.config import PipelineConfig
from ML_Pipeline.pipeline import MLPipeline

config = PipelineConfig(raw_data_path="data/raw_data.csv", n_clusters=50)
results = MLPipeline(config=config).run_full_pipeline()

results["metrics"]      # {'without_lag': {...}, 'with_lag': {...}}
results["predictions"]  # {'without_lag': DataFrame, 'with_lag': DataFrame}
```

### Serving

```bash
pip install -e ".[serving]"
python run_pipeline.py --raw-data data/raw_data.csv --promote   # train, gate, promote
uvicorn ML_Pipeline.api:app --reload                            # serve
```

The dashboard's **Model performance** page scores the promoted model: the deploy
gate verdict and data lag, a recursive backtest of the last closed horizon against
what actually happened, MASE against the seasonal-naive baseline, and error per
cluster with the clusters that lose to the baseline called out by name. It scores
only demand the model was not fitted on, so a freshly refit model shows no
accuracy figures until a week of new demand has arrived. It shares
`ML_Pipeline.serving` with the API, so the two cannot disagree about which model is
live.

| Endpoint | Returns |
|---|---|
| `GET /health` | liveness, and whether a model **and its history** actually loaded |
| `GET /model` | what is serving: features, lags, gate verdict, data lag, age, staleness |
| `GET /clusters` | the clusters this model can forecast for |
| `GET /forecast?steps=48&cluster=7` | demand per cluster per interval, forward from the last observation |
| `POST /reload` | load a newly promoted model or refreshed grid without a restart (needs the API key) |

With `BIKETAXI_API_KEY` set, every endpoint but `/health` needs it in an
`X-API-Key` header. Without it the API is open, which suits a public demo, and
`/reload` is disabled.

Three behaviours are deliberate, and each is a measurement turned into code
rather than a preference:

- **It serves the *promoted* model, not the newest.** `ModelRegistry.promote_model`
  refuses a model that failed its deploy gate, or that carries no verdict at all —
  unverified is not the same as passing. `rollback()` restores the previous one.
  So training a model changes nothing in production until it is promoted.
- **The horizon is capped at two days** (`steps <= 96`). Past that the measured
  MASE stops beating seasonal-naive; a longer request is refused with that reason
  rather than served quietly. Anything past one day carries a warning in the
  response.
- **Every response says where its history ends** and whether the model is stale,
  because a recursive forecast's origin is the last observed interval and a model
  more than four weeks behind its data under-forecasts. Staleness is measured
  from where the model's training data ends (`data_lag_days`), not from when it
  was trained.

### Running it

The container is built and smoke-tested in CI: it starts on a synthetic output
directory, becomes ready, refuses a request without the key and serves a
forecast with it.

```bash
docker build -t bike-taxi-forecast .
docker run -d -p 8000:8000 -v "$PWD/output:/app/output:ro" \
  -e BIKETAXI_API_KEY=<key> bike-taxi-forecast
```

To host it, [deploy/README.md](deploy/README.md) deploys to Cloud Run with one
script, serving from a private bucket that holds only aggregated files.

`output/` is mounted, never baked in: it holds booking-level personal data. A
registry written on Windows loads in the Linux container - model paths that do
not exist as written are looked up by file name in the output directory.

### Operations runbook

| Situation | Do this |
|---|---|
| A new model is trained | `python scripts/registry.py promote <name>` (refused if it failed its gate), then `POST /reload` |
| The serving model misbehaves | `python scripts/registry.py rollback`, then `POST /reload` |
| What is registered, and what is serving? | `python scripts/registry.py list`, or `GET /model` |
| Is the serving model still healthy? | `python scripts/monitor_model.py` - exit 3 means retrain or ship the baseline |
| A reload is refused (409) | the running model keeps serving; the response says what could not be loaded |

Every request is logged with its status, duration and the serving model, so a
rollback is visible in the log as the model name changing.

### Forecasting from a saved model

```python
from ML_Pipeline.features import ModelBundle
from ML_Pipeline.forecast import forecast_recursive

bundle = ModelBundle.load_bundle("output/prediction_model_with_lag_<version>.joblib")
forecast = forecast_recursive(bundle, history_panel, horizon)
```

A `ModelBundle` carries the estimator **and its ordered feature list**, so
serving builds its design matrix from what the model was actually fitted on. A
mismatch raises rather than silently reordering columns.

## Layout

```
src/ML_Pipeline/
  features.py             canonical feature engineering (shared by train & serve)
  splitting.py            chronological train/test splitting
  forecast.py             direct and recursive multi-step forecasting
  config.py               PipelineConfig, ModelRegistry
  pipeline.py             orchestrator
  data_prep_basic.py      deduplication, type coercion, per-rider gaps
  advanced_cleanup.py     business-rule filters
  data_prep_advanced.py   cleaning stage + persistence
  data_prep_geospatial.py clustering + aggregation to the demand grid
  model_training.py       trains both model variants
  xgb_model.py            XGBoost fitting with early stopping
  evaluation.py           metrics, baselines, prediction validation
  clustering.py           offline cluster-count diagnostics
  api.py                  forecast serving API (FastAPI)
  monitoring.py           health checks on the serving model
run_pipeline.py           CLI entry point
streamlit_app.py          dashboard
scripts/smoke_run.py      manual full run against real data
scripts/compare_strategies.py  rolling-origin strategy sweep
scripts/measure_staleness.py   model decay by weeks since training
scripts/measure_peak_error.py  stale vs refitted model at the busiest cluster's peak
scripts/monitor_model.py       scheduled health check on the serving model
scripts/registry.py            list, promote and roll back models
scripts/build_smoke_output.py  synthetic output directory for the container smoke test
scripts/stage_demo_output.py   aggregated-only files for a hosted demo
deploy/gcp_deploy.sh           build, upload and deploy to Cloud Run
scripts/fetch_weather.py       hourly Bengaluru weather from Open-Meteo, for the experiment below
scripts/measure_weather.py     whether weather or holidays improve the forecast (they do not)
tests/                    pytest suite (synthetic data only)
Notebook/                 original exploratory notebooks (historical record)
```

## Modelling notes

Two variants are trained. **Without lag** uses calendar and geography only, so it
applies to any future interval — but it has no channel carrying the current
demand level, loses to a free baseline by 75%, and is useful only for cold
starts. **With lag** adds recent demand and must be applied recursively,
compounding its own errors; `recursive_rmse` in the model bundle measures that
honestly.

The lag set reaches back a week: `lag_1/2/3` (the last 90 minutes), `lag_48`
(same time yesterday) and `lag_336` (same time last week). The weekly lag is the
one that matters most — the model is judged against a seasonal-naive baseline
built from exactly that value, and until it was given the signal it could not
beat it. Adding the daily and weekly lags took one-step MASE from 0.999 — a
failed deploy gate — to **0.805**, and R² from 0.807 to 0.884. It also fixed
level tracking over a recursive horizon: the 24-hour forecast used to predict
1.35 against an actual of 6.37, and now tracks at a level ratio of 0.88.

The cost is a serving precondition: recursive forecasting needs **7 days of
contiguous history per cluster**, which the pipeline supplies from the demand
grid rather than from the test file.

The gain is also horizon-dependent. Within a 48-step horizon the new lags are
always real observations; past that they start consuming the model's own
predictions. Measured MASE against seasonal-naive: **0.79 at one day, 0.82 at
two, 0.89 at four, 0.99 at one week, 1.06 at two**. So `--horizon-steps` is
capped at 96, two days, and a larger value is refused; beyond a week the model
only ties a baseline that costs nothing.
See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the table.

Across five rolling origins on the current lag set, every strategy tried beats
seasonal-naive in **5 of 5 folds**, one step ahead and over a 24-hour recursive
horizon. The spread between them (0.006 MASE) is now five times smaller than the
fold-to-fold variation (0.029), so the ratio-target variant this project used to
recommend is no longer distinguishable from what ships —
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) explains why the weekly lag removed the
problem it solved.

Three properties of the target drive the design:

- **It is an over-dispersed count** (mean 4.22, variance ~46, 37% zeros). The objective is
  `count:poisson`, predictions are floored at zero, and percentage-error metrics
  are computed only over non-zero actuals.
- **It is a time series.** Splits are chronological, never random or
  day-of-month. Cross-validation uses `TimeSeriesSplit`.
- **Geography is nominal.** Cluster identity enters as centroid coordinates, not
  as a raw integer label — a tree splitting on `pickup_cluster < 23.5` is
  partitioning an arbitrary labelling, not the city.

Before deploying, check the model against the baselines in
`ModelEvaluator.compare_to_baselines`. A demand model that cannot beat "same time
last week" should not ship.

> **On the reference dataset it clears that bar, and the check now runs
> automatically.** MASE 0.805 one step ahead, beating seasonal-naive by 18%,
> measured on the frozen chronological split —
> the harshest configuration in the repository, where the test window runs up to
> ten weeks past the training cut.
>
> Demand grew 5.2x across the training year and trees cannot extrapolate, so
> staleness remains the binding constraint. **Retrain at least every four
> weeks**, and monitor the predicted-to-actual level ratio — it degrades earliest,
> reaching 0.92 by week four while MASE still looks fine. Measured across three
> freeze origins: the weekly lag improved mean decay a great deal, but the worst
> origin still loses to the baseline by week six, and a cadence follows the worst
> case. Reproduce with `scripts/measure_staleness.py`.
>
> Until 2026-10-03 no model could meet that cadence: the saved model was the
> scored one, which never saw the test window, so every model started ten weeks
> behind its data. Training now ends with a refit on all data. See "The promoted
> model was stale on the day it was trained" in the model card.
>
> The gate runs at the end of every training pass, records its verdict in the
> model registry, and `run_pipeline.py` exits **3** when the model loses, so
> automation can refuse to promote it. Error per cluster is reported alongside.
> See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the measured numbers.

## Development

```bash
pytest                    # everything
pytest -m "not slow"      # skip model fitting
ruff check .              # lint
```
