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

## Input: `data/test_dataset/cleaned_test_booking_data.csv`

Same schema. The serving window. `number` is not required for forecasting, only
`ts`, `pick_lat`, `pick_lng`.

## Intermediate: `output/clean_data_<version>.csv`

**Contains personal data.** Booking-level, post-cleaning. Columns as
`data_prep_advanced.CLEANED_COLUMNS`, adding `geodesic_distance` (km),
calendar features, and per-rider booking gaps.

## The aggregation boundary: `output/Data_Prepared_<version>.csv`

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
meaningful relative to the model that produced them.

## Generating synthetic data

`tests/conftest.py` builds a realistic grid in-memory (daily peaks, several
pickup hotspots). Reuse those fixtures to exercise the pipeline without the real
dataset.
