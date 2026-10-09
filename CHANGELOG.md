# Changelog

What changed in each merged pull request, newest first. The measured effect of
each change is in [docs/MODEL_CARD.md](docs/MODEL_CARD.md); the detail is in the
commit messages.

## Unreleased

### Added
- mypy in CI, over `src/ML_Pipeline`, with settings in `pyproject.toml`. pandas,
  scikit-learn, scipy and joblib are left unchecked; `pandas-stubs` was tried
  and not adopted (123 errors, nearly all overload strictness).
- `ServingState.loaded()` returns the model and its history, and raises if
  either is missing. The API, the health check and the dashboard use it where
  they had relied on an earlier `ready` check that a type checker cannot see.

### Fixed
- Found by mypy: `compare_models` wrote a model's name into its float metrics
  dict, and `train_xgb` read the tree count with `.get()`, so a missing value
  would have surfaced later as `float(None)`.

### Changed
- The package is split into subpackages. Module moves, old -> new:

  | Was `ML_Pipeline.` | Now `ML_Pipeline.` |
  |---|---|
  | `data_prep_basic`, `shift_time`, `data_prep_advanced`, `data_prep_geospatial`, `clustering` | `data.prep_basic`, `data.shift_time`, `data.prep_advanced`, `data.prep_geospatial`, `data.clustering` |
  | `advanced_cleanup` | `data.cleaning_rules` |
  | `features`, `splitting`, `xgb_model`, `forecast`, `evaluation`, `validation` | `modeling.` + same name |
  | `model_training`, `prediction_pipeline` | `modeling.training`, `modeling.prediction` |
  | `api`, `serving`, `monitoring` | `serving.api`, `serving.state`, `serving.monitoring` |

  Function and class names are unchanged. The API is now
  `uvicorn ML_Pipeline.serving.api:app`, and the Dockerfile uses that.
- The dashboard moved from one 969-line `streamlit_app.py` into
  `ML_Pipeline.dashboard`, one module per page. `streamlit run streamlit_app.py`
  is unchanged; the root file is now three lines.
- The personal-data columns and the grid's allow-list are defined once, in the
  new `ML_Pipeline.governance`, and read from there by the cleaning stage, the
  dashboard and demo staging. They used to be written out in each of the three.

### Kept working
- `ML_Pipeline/features.py` remains as a one-line shim so that models pickled
  before the split, which name `ML_Pipeline.features.ModelBundle`, still load -
  including the one the hosted demo serves. New models record the new path.

## 2026-10-06

- **#21** One `biketaxi` command (also `python -m ML_Pipeline`) with `run`,
  `registry`, `monitor` and `stage-demo`, replacing `run_pipeline.py` (kept as a
  wrapper) and three scripts. No entry point patches `sys.path`; the tests
  import the installed package. `ModelRegistry` moved to `registry.py`, artefact
  naming to `artifacts.py`.
- **#20** Experiment scripts moved from `scripts/` to `experiments/`, with their
  shared settings in `experiments/common.py`, read from `PipelineConfig`
  instead of a hand-written copy of the XGBoost parameters (values unchanged).
  `Notebook/` renamed to `notebooks/`. The version is written once, in
  `ML_Pipeline.__version__`. Two unused `utils` helpers and an unused re-export
  removed. Added an MIT licence (code only), this changelog and
  `experiments/README.md`.

## 2026-10-05

- **#19** Documents brought up to date after deployment: live-demo section,
  what is in Google Cloud and who can read it, and the open governance items.
- **#18** The demo deployed to Cloud Run from a private bucket that holds only
  aggregated files, with the API key in Secret Manager. Fixes what the first real
  deployment found: Git Bash path rewriting, a CRLF inside the API key, and IAM
  propagation delays.
- **#17** The serving API made deployable: the container built and smoke-tested
  in CI, registry paths that resolve inside a Linux container, and an optional
  API key on every endpoint but `/health`.
- **#16** Ratio target re-tested after the final refit: it wins one step ahead,
  ties over the 24-hour recursive horizon, and is still not adopted.
- **#15** Weather and holidays measured as features. Neither improves the
  forecast.
- **#14** Pickups outside Bengaluru dropped in cleaning (Rule 6). They had taken
  8 of the 50 clusters.
- **#13** The CLI refuses a forecast horizon past two days, as the API does.
- **#12** The monitor checks per-cluster level ratio only on clusters with real
  volume.
- **#11** Scheduled health check on the serving model (`scripts/monitor_model.py`).
- **#10** Dashboard tests give Streamlit AppTest an absolute path.

## 2026-10-03

- **#9** The promoted model is refit on all data. Every model promoted before this
  was ten weeks stale on the day it was trained. Staleness is now measured from
  the data, not the calendar.
- **#8** Dashboard model-performance page: gate verdict, recursive backtest, MASE
  against the baseline, and error per cluster.

## 2026-10-02

- **#7** Rolling-origin sweep re-run on the current lags; the ratio-target
  recommendation was withdrawn. Staleness measured across three origins by a
  committed script.
- **#6** Forecast serving API, model promotion and rollback, and a container.
- **#5** Cleaning Rule 1 evaluated as the rule it describes; cleaning layer
  tested; deploy gate run on every training pass; outputs written as `.csv.gz`.
- **#4** Daily and weekly lags, which take the model from failing the deploy gate
  to passing it. The dashboard reads the run that actually finished. Real
  dependencies declared, and rider identifiers no longer written.
- **#3** GitHub Actions CI on Python 3.10–3.13.
- **#2** The test-data path corrected (the forecasting stage had been skipped on
  every run), and flags allowed to override `--config`.

## 2026-08-04

- **#1** Audit and rebuild of the inherited pipeline: canonical features,
  chronological splitting, a forecaster that forecasts, config that is read,
  corrected methodology and metrics, a dashboard built on real output, the test
  suite, and the governance documents.
