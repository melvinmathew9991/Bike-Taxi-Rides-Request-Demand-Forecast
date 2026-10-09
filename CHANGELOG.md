# Changelog

What changed in each merged pull request, newest first. The measured effect of
each change is in [docs/MODEL_CARD.md](docs/MODEL_CARD.md); the detail is in the
commit messages.

## 2026-10-09

- **#30** Documents brought up to date at the end of the day: README (pipeline
  diagram, test count, mypy, branch protection, the demo's missing
  intervals), model card maintenance, data schema (bundle and registry
  fields), the deploy guide, and this changelog's dated sections.
- **#29** The booking input contract is checked when a file is loaded
  (`ML_Pipeline.data.contract`): stage 1 checks the raw file, the prediction
  stage the test file. A broken file stops the run with the reason and
  `biketaxi run` exits 2 - a missing column, no rows, more than 1% of rows
  failing a rule (timestamp format, rider id, coordinates), or most pickups
  outside Bengaluru, which is how swapped lat/lng show up. A few bad rows are
  logged and cleaning drops them as before. Written by hand, not with pandera.
- **#28** The lag-free model forecasts only cold-start clusters - those without
  the 7 days of history the lag model reads. It used to forecast every
  cluster, and the dashboard offered it as an alternative, although it loses
  to seasonal-naive by 75%. `data_without_lag` is now usually empty; the
  dashboard shows cold-start clusters separately when there are any. A history
  too short for every cluster is still refused.
- **#27** Prediction intervals. Every recursive forecast carries an 80%
  interval in whole requests - `request_count_lower`/`_upper` in the forecast
  files, `lower`/`upper` in the API - calibrated on the held-out model's
  recursive errors over the four weeks after its cut, by horizon and volume.
  Covered 79.7% on backtests the calibration did not use, and 73.2% at the
  busiest clusters, where Poisson quantiles covered 55.5%. Stored on the
  bundle as `intervals`; models trained before this serve without them.
  Training takes about 30 seconds longer. New settings `interval_level`,
  `interval_calibration_days`, `interval_origins`; new
  `experiments/measure_intervals.py`.
- **#26** mypy passes on Python 3.10, where CI resolves matplotlib 3.10.9, whose
  stubs mistype `Colorbar.outline`.
- **#25** mypy in CI over `src/ML_Pipeline`; pandas, scikit-learn, scipy and
  joblib left unchecked (`pandas-stubs` gave 123 errors, nearly all overload
  strictness). New `ServingState.loaded()`, used where callers relied on a
  `ready` check a type checker cannot see. Fixed two defects it found:
  `compare_models` wrote a model name into a float metrics dict, and
  `train_xgb` read the tree count with `.get()`, so a missing value would have
  surfaced later as `float(None)`.
- **#24** The local-only project report moved to `docs/`, still git-ignored.
- **#23** The pipeline diagrams in the README and the data documents use the
  post-split module names.

Also on 2026-10-09, outside the code: `main` is protected, so a pull request
merges only when all five CI jobs pass; the GitHub repository has a
description, website and topics.

## 2026-10-06

- **#22** The package is split into subpackages. Module moves, old -> new:

  | Was `ML_Pipeline.` | Now `ML_Pipeline.` |
  |---|---|
  | `data_prep_basic`, `shift_time`, `data_prep_advanced`, `data_prep_geospatial`, `clustering` | `data.prep_basic`, `data.shift_time`, `data.prep_advanced`, `data.prep_geospatial`, `data.clustering` |
  | `advanced_cleanup` | `data.cleaning_rules` |
  | `features`, `splitting`, `xgb_model`, `forecast`, `evaluation`, `validation` | `modeling.` + same name |
  | `model_training`, `prediction_pipeline` | `modeling.training`, `modeling.prediction` |
  | `api`, `serving`, `monitoring` | `serving.api`, `serving.state`, `serving.monitoring` |

  Function and class names are unchanged. The API is now
  `uvicorn ML_Pipeline.serving.api:app`, and the Dockerfile uses that. The
  dashboard moved from one 969-line `streamlit_app.py` into
  `ML_Pipeline.dashboard`, one module per page; `streamlit run
  streamlit_app.py` is unchanged. The personal-data columns and the grid's
  allow-list are defined once, in the new `ML_Pipeline.governance`.
  `ML_Pipeline/features.py` remains as a one-line shim so that models pickled
  before the split - including the one the hosted demo serves - still load.
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
