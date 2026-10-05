# Data schema

The datasets are not in the repository — they carry personal data and are
git-ignored (see `DATA_GOVERNANCE.md`). This document is what you need to supply
your own input, or to understand the pipeline without access to the data.

## Input: `data/raw_data.csv`

Gzip-compressed CSV, booking-level, one row per ride request.

| Column | Type | Example | Notes |
|---|---|---|---|
| `ts` | string | `2020-03-26 07:07:17` | Request timestamp, `%Y-%m-%d %H:%M:%S`. |
| `number` | string | `14626` | Pseudonymous customer id. `-1` marks an unidentified rider and is dropped. |
| `pick_lat` | float | `12.313621` | Pickup latitude. |
| `pick_lng` | float | `76.658195` | Pickup longitude. |
| `drop_lat` | float | `12.287301` | Drop latitude. |
| `drop_lng` | float | `76.602280` | Drop longitude. |

Reference dataset: 8,381,556 rows, 2020-03-26 to 2021-03-26, Bangalore.

## Input: `data/cleaned_test_booking_data.csv`

Same schema. The serving window. `number` is not required for forecasting, only
`ts`, `pick_lat`, `pick_lng`.

## Intermediate: `output/clean_data_<version>.csv.gz`

**Contains personal data.** Booking-level, post-cleaning. Columns as
`data_prep_advanced.CLEANED_COLUMNS`, adding `geodesic_distance` (km),
calendar features, and per-rider booking gaps.

## The aggregation boundary: `output/Data_Prepared_<version>.csv.gz`

**No personal data from here on.** Rectangular grid, one row per
(interval x cluster).

| Column | Type | Notes |
|---|---|---|
| `ts` | datetime | Interval start, 30-minute boundaries. |
| `pickup_cluster` | int | Cluster label, `0 .. n_clusters-1`. |
| `request_count` | float | Requests in that cluster during that interval. |
| `mins`, `hour`, `month`, `quarter`, `dayofweek` | int | Calendar features. |

Rows = intervals x clusters, with zero-demand intervals present and equal to 0
(not missing). `features.validate_grid` reports whether this holds.

## Output: `output/data_{with,without}_lag_<version>.csv`

Forecasts. Adds:

| Column | Notes |
|---|---|
| `request_count_pred` | The forecast. Never negative. |
| `is_forecast` | `True` for every row — these files contain only predictions. |
| `cluster_lat`, `cluster_lng` | Cluster centroid, when centroid encoding is used. |
| `lag_1..lag_3`, `rolling_mean` | With-lag file only: the inputs each step used. |

## Models: `output/*.joblib`

`prediction_model_*.joblib` hold a `features.ModelBundle` — the estimator plus
its **ordered feature list**, lag settings, frequency, metrics and parameters.
Serving builds its design matrix from that list, so a train/serve mismatch
raises instead of silently reordering columns.

`pickup_cluster_model_<version>.joblib` is the fitted clustering model. It must
be paired with the demand models trained alongside it — cluster labels are only
meaningful relative to the model that produced them. As fitted, it also holds a
cluster label for every training booking (`labels_`, 3.7 M on the reference
data); serving reads only `cluster_centers_`.

`model_registry.json` records every trained model, its metrics and deploy-gate
verdict, and which one is `production`. Model paths are stored as the training
run wrote them; serving falls back to the same file name in its own output
directory, so a registry written on Windows loads in a Linux container.

## Monitoring report: `output/health.json`

Written by `scripts/monitor_model.py --json output/health.json`.

| Field | Notes |
|---|---|
| `healthy` | False when any check failed; the script then exits 3. |
| `checks` | `staleness`, `mase`, `level_ratio`, `clusters`, each `pass`, `warn`, `fail` or `skipped`, with a reason. |
| `mase`, `level_ratio` | Over the scored week, against seasonal-naive. `null` when skipped. |
| `window_start`, `window_end`, `rows_scored` | The week scored: the latest the model was not fitted on. |
| `clusters` | One row per cluster: `n`, `mean_actual`, `mean_pred`, `level_ratio`, `rmse`, `mae`, `mase`. |

## Hosted demo files: `deploy/.staging/`

Written by `scripts/stage_demo_output.py` and uploaded by `deploy/gcp_deploy.sh`.
Aggregated only, and git-ignored:

| File | Contents |
|---|---|
| `Data_Prepared_<version>.csv.gz` | the demand grid above, last 14 days (`--history-days`) |
| `prediction_model_with_lag_<version>.joblib` | the promoted model, as trained |
| `pickup_cluster_model_<version>.joblib` | an object holding `cluster_centers_` only — no per-booking labels |
| `model_registry.json` | the promoted entry only, with its path reduced to the file name |

## Generating synthetic data

`tests/conftest.py` builds a realistic grid in-memory (daily peaks, several
pickup hotspots). Reuse those fixtures to exercise the pipeline without the real
dataset.

## Reading the outputs

Every output is gzip-compressed and named `.csv.gz`, so nothing special is
needed:

```python
import pandas as pd
pd.read_csv("output/Data_Prepared_20260102_030405.csv.gz")
```

They were previously written as `.csv` while holding gzip bytes, which pandas
cannot infer from — a plain `read_csv` failed with
`UnicodeDecodeError: invalid start byte`. Input files may still be gzip under a
`.csv` name; `ML_Pipeline.utils.read_csv_any` detects that by content.

To resolve the newest run without hardcoding a version:

```python
from ML_Pipeline.config import latest_artifact, latest_version
latest_version("output")                   # '20260102_030405'
latest_artifact("output", "prepared")      # Path to the newest demand grid
```
