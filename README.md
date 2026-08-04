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
pip install -r requirements.txt
pytest                                    # 135 tests, no data needed
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
```

Every flag reaches the code it names; `run_pipeline.py --help` lists them all.

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
applies to any future interval. **With lag** adds recent demand and is more
accurate one step out, but must be applied recursively, compounding its own
errors — `recursive_rmse` in the model bundle measures that honestly.

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

> **On the reference dataset it clears that bar — but only if retrained.**
> Across five rolling origins it beats seasonal-naive every time (MASE 0.79
> one-step, 0.95 over a 24-hour recursive horizon). Demand grew 5.2x across the
> training year, and a frozen model falls *below* the baseline by week six and
> forecasts half the actual demand by week thirteen.
>
> **Retrain at least every four weeks**, and monitor the predicted-to-actual
> level ratio — it degrades earliest. See
> [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the measured numbers.

## Development

```bash
pytest                    # everything
pytest -m "not slow"      # skip model fitting
ruff check .              # lint
```
