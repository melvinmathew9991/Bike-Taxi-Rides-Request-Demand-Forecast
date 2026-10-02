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

## Quick start

```bash
pip install -r requirements-dev.txt        # or requirements.txt to run, not test
pytest                                    # 170 tests, no data needed
python run_pipeline.py --raw-data data/raw_data.csv --n-clusters 50
streamlit run streamlit_app.py            # dashboard over pipeline output
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

## Usage

### Command line

```bash
python run_pipeline.py                          # full pipeline, defaults
python run_pipeline.py --stages data features   # subset of stages
python run_pipeline.py --n-clusters 100 --test-fraction 0.25 --horizon-steps 96
python run_pipeline.py --config output/pipeline_config_20240101_120000.json
python run_pipeline.py --config run.json --n-clusters 100   # flag wins
```

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
run_pipeline.py           CLI entry point
streamlit_app.py          dashboard
scripts/smoke_run.py      manual full run against real data
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
beat it. Adding the daily and weekly lags took one-step RMSE from 4.803 to
**3.751** and MASE from 0.999 to **0.809**. It also fixed level tracking over a
recursive horizon: the 24-hour forecast used to predict 1.35 against an actual
of 6.37, and now predicts 5.75.

The cost is a serving precondition: recursive forecasting needs **7 days of
contiguous history per cluster**, which the pipeline supplies from the demand
grid rather than from the test file.

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

> **On the reference dataset it clears that bar.** MASE 0.809 one step ahead,
> beating seasonal-naive by 19%, measured on the frozen chronological split —
> the harshest configuration in the repository, where the test window runs up to
> ten weeks past the training cut.
>
> Demand grew 5.2x across the training year and trees cannot extrapolate, so
> staleness remains the binding constraint. **Retrain at least every four
> weeks**, and monitor the predicted-to-actual level ratio — it degrades
> earliest. That cadence is deliberately conservative: it was measured under the
> previous lag set, and has not yet been re-measured with the weekly lag that
> should slow the decay.
>
> The gate is not yet run automatically — call `compare_to_baselines` at each
> retrain. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the measured numbers.

## Development

```bash
pytest                    # everything
pytest -m "not slow"      # skip model fitting
ruff check .              # lint
```
