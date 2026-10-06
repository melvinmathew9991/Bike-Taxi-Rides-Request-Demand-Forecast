# Changelog

What changed in each merged pull request, newest first. The measured effect of
each change is in [docs/MODEL_CARD.md](docs/MODEL_CARD.md); the detail is in the
commit messages.

## Unreleased

### Changed
- Experiment scripts moved from `scripts/` to `experiments/`, so `scripts/` holds
  only operational tools. Their shared settings live in `experiments/common.py`
  and are read from `PipelineConfig`. The XGBoost parameters used to be a
  hand-written second copy in `compare_strategies.py`; the values are unchanged.
- `Notebook/` renamed to `notebooks/`.
- The package version is written once, in `ML_Pipeline.__version__`;
  `pyproject.toml` and the API read it from there.

### Added
- MIT licence, covering the code only.
- This changelog, and `experiments/README.md`.

### Removed
- `utils.geodestic_distance` and `utils.round_timestamp_30interval`, which
  nothing called, and the `utils` re-export of two `features` functions that
  nothing imported from `utils`.

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
