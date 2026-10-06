# Changelog

What changed in each merged pull request, newest first. The measured effect of
each change is in [docs/MODEL_CARD.md](docs/MODEL_CARD.md); the detail is in the
commit messages.

## Unreleased

### Added
- A `biketaxi` command (also `python -m ML_Pipeline`) with four subcommands:
  `run`, `registry`, `monitor` and `stage-demo`. Installed as a console script
  by `pip install -e .`.

### Changed
- `run_pipeline.py`, `scripts/registry.py`, `scripts/monitor_model.py` and
  `scripts/stage_demo_output.py` moved into `ML_Pipeline.cli` (history kept).
  `python run_pipeline.py` still works as a thin wrapper; the other three are
  now `biketaxi registry`, `biketaxi monitor` and `biketaxi stage-demo`, with
  the same options and exit codes.
- No entry point puts `src/` on `sys.path` by hand any more; the package must be
  installed, which both requirements files already do. The tests import it the
  same way: `pythonpath` is gone from the pytest settings, and the tests that
  loaded scripts by file path import modules instead.
- `ModelRegistry` moved from `config.py` to `registry.py`, and the artefact
  stems and `latest_artifact` / `latest_version` to `artifacts.py`. Saved models
  reference only `ML_Pipeline.features`, so every existing model still loads.
- `deploy/gcp_deploy.sh` stages through `python -m ML_Pipeline stage-demo`, and
  checks the package is importable before creating anything in the cloud.

## 2026-10-06

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
