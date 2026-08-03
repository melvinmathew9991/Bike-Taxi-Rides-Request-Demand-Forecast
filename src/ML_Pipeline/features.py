"""
Canonical feature engineering for the demand-forecast panel.

This module is the *single* definition of every transform that both training and
serving depend on. Previously these transforms existed in two or three
near-identical copies that had already drifted apart (different timestamp
handling, different calendar columns), which is the classic setup for
train/serve skew: the paths look identical and are not.

Two rules keep them aligned:

1. Every transform lives here exactly once. Training and prediction both import
   from this module; neither defines its own copy.
2. The feature contract travels with the model. `ModelBundle` persists the exact
   ordered feature list used at fit time, and the prediction path builds its
   design matrix from `bundle.feature_names` rather than from a hardcoded list.
   A mismatch raises instead of silently reordering columns.

The panel is a rectangular grid of (timestamp x pickup_cluster) -> request_count
at a fixed frequency. "Rectangular" is load-bearing: lag features are only
meaningful when each cluster has an unbroken, evenly spaced series.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load

logger = logging.getLogger(__name__)

TS_COL = "ts"
CLUSTER_COL = "pickup_cluster"
TARGET_COL = "request_count"

#: Calendar features. `year` and `day` are deliberately excluded: `year` cannot
#: generalise beyond the training period, and `day` (day-of-month) encodes no
#: real demand signal while giving the model a way to memorise the old
#: day-of-month train/test split.
CALENDAR_FEATURES: tuple[str, ...] = ("mins", "hour", "month", "quarter", "dayofweek")

DEFAULT_FREQ = "30min"
DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3)
DEFAULT_ROLLING_WINDOW = 3


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------


def floor_to_interval(values, minutes: int = 30) -> pd.Series:
    """
    Floor timestamps down to the start of their interval.

    Vectorised replacement for the previous per-row `np.vectorize` helper, which
    was a Python loop over every booking and handled `str`, `np.datetime64` and
    `Timestamp` inconsistently across its two copies.

    Args:
        values: Anything `pd.to_datetime` accepts (Series, array, list).
        minutes: Interval width in minutes.

    Returns:
        Series of floored timestamps.
    """
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")
    return pd.to_datetime(pd.Series(values).reset_index(drop=True)).dt.floor(
        f"{minutes}min"
    )


def add_calendar_features(df: pd.DataFrame, ts_col: str = TS_COL) -> pd.DataFrame:
    """
    Add calendar features derived from `ts_col`.

    Always produces exactly `CALENDAR_FEATURES`, so training and serving cannot
    disagree about which columns exist.
    """
    if ts_col not in df.columns:
        raise KeyError(f"Column {ts_col!r} not found; have {list(df.columns)}")

    out = df.copy()
    stamps = pd.to_datetime(out[ts_col])
    out["mins"] = stamps.dt.minute.astype("int16")
    out["hour"] = stamps.dt.hour.astype("int16")
    out["month"] = stamps.dt.month.astype("int16")
    out["quarter"] = stamps.dt.quarter.astype("int16")
    out["dayofweek"] = stamps.dt.dayofweek.astype("int16")
    return out


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


def build_demand_grid(
    bookings: pd.DataFrame,
    *,
    ts_col: str = TS_COL,
    cluster_col: str = CLUSTER_COL,
    freq: str = DEFAULT_FREQ,
    interval_minutes: int = 30,
    clusters: Sequence[int] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Aggregate booking-level rows into a rectangular demand grid.

    Every (timestamp, cluster) pair in the requested range is present exactly
    once; intervals with no bookings get a count of 0.

    This replaces the previous approach, which appended a synthetic
    `pickup_cluster = -1` row for every interval of a hardcoded 365-day range to
    force `asfreq` to span the year, then dropped it again, then asserted the
    result was exactly 878,400 rows. That construction only worked for one
    dataset and one cluster count. Here the range is derived from the data (or
    passed explicitly) and the grid is built by reindexing onto the full
    MultiIndex product, which is both correct for any input and considerably
    faster.

    Args:
        bookings: Booking-level rows carrying `ts_col` and `cluster_col`.
        ts_col: Timestamp column name.
        cluster_col: Cluster label column name.
        freq: Pandas offset alias for the grid frequency.
        interval_minutes: Interval width used to floor booking timestamps.
        clusters: Cluster labels the grid must cover. Defaults to those observed.
            Pass explicitly when serving so the grid matches the trained model
            even if a cluster saw no bookings in the window.
        start: Grid start. Defaults to the earliest floored booking.
        end: Grid end (inclusive). Defaults to the latest floored booking.

    Returns:
        DataFrame with `[ts, pickup_cluster, request_count]`, sorted by
        cluster then timestamp.
    """
    for col in (ts_col, cluster_col):
        if col not in bookings.columns:
            raise KeyError(f"Column {col!r} not found; have {list(bookings.columns)}")

    working = pd.DataFrame(
        {
            ts_col: floor_to_interval(bookings[ts_col], interval_minutes),
            cluster_col: pd.Series(bookings[cluster_col]).reset_index(drop=True),
        }
    ).dropna(subset=[ts_col])

    if working.empty:
        raise ValueError("No usable rows after flooring timestamps.")

    counts = (
        working.groupby([ts_col, cluster_col]).size().rename(TARGET_COL).reset_index()
    )

    grid_start = pd.Timestamp(start) if start is not None else counts[ts_col].min()
    grid_end = pd.Timestamp(end) if end is not None else counts[ts_col].max()
    if grid_end < grid_start:
        raise ValueError(f"end ({grid_end}) precedes start ({grid_start})")

    stamps = pd.date_range(grid_start, grid_end, freq=freq)
    labels = (
        np.sort(np.asarray(clusters))
        if clusters is not None
        else np.sort(counts[cluster_col].unique())
    )

    index = pd.MultiIndex.from_product([stamps, labels], names=[ts_col, cluster_col])
    grid = (
        counts.set_index([ts_col, cluster_col])[TARGET_COL]
        .reindex(index, fill_value=0)
        .astype("float64")
        .reset_index()
    )

    dropped = len(counts) - int(counts.set_index([ts_col, cluster_col]).index.isin(index).sum())
    if dropped:
        logger.warning(
            "%d aggregated (ts, cluster) pairs fell outside the requested grid "
            "and were dropped.", dropped,
        )

    logger.info(
        "Demand grid: %d intervals x %d clusters = %d rows (%s to %s, freq=%s)",
        len(stamps), len(labels), len(grid), grid_start, grid_end, freq,
    )
    return grid.sort_values([cluster_col, ts_col]).reset_index(drop=True)


def validate_grid(
    df: pd.DataFrame,
    *,
    ts_col: str = TS_COL,
    cluster_col: str = CLUSTER_COL,
    freq: str = DEFAULT_FREQ,
) -> dict[str, Any]:
    """
    Check that a panel is rectangular and evenly spaced.

    Returns a report rather than asserting, so callers decide how to react. The
    previous code used a bare `assert len(data) == 878400`, which crashed on any
    dataset other than the original one.
    """
    stamps = pd.to_datetime(df[ts_col]).drop_duplicates().sort_values()
    labels = df[cluster_col].unique()
    expected_stamps = (
        pd.date_range(stamps.min(), stamps.max(), freq=freq) if len(stamps) else []
    )
    report = {
        "rows": len(df),
        "clusters": len(labels),
        "timestamps": len(stamps),
        "expected_timestamps": len(expected_stamps),
        "expected_rows": len(expected_stamps) * len(labels),
        "duplicate_keys": int(df.duplicated(subset=[ts_col, cluster_col]).sum()),
        "missing_timestamps": int(len(expected_stamps) - len(stamps)),
    }
    report["is_rectangular"] = (
        report["rows"] == report["expected_rows"] and report["duplicate_keys"] == 0
    )
    if not report["is_rectangular"]:
        logger.warning("Demand grid is not rectangular: %s", report)
    return report


# ---------------------------------------------------------------------------
# Lag / rolling features
# ---------------------------------------------------------------------------


def add_lag_features(
    df: pd.DataFrame,
    *,
    target: str = TARGET_COL,
    cluster_col: str = CLUSTER_COL,
    ts_col: str = TS_COL,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Add per-cluster lag and rolling-mean features.

    All features are strictly backward-looking: `rolling_mean` is shifted by one
    interval *within each cluster* so the current value never enters its own
    feature.

    The previous implementation applied `.shift(1)` to the result of a
    `groupby(...).transform(...)`. Because that shift runs over the whole frame
    rather than within groups, the first row of each cluster received the last
    rolling value of the *previous* cluster. Measured, that contaminates exactly
    one row per cluster boundary - and in the shipped configuration those rows
    were always removed by the `dropna` on `lag_3` that followed, so no training
    row was ever affected. It was a latent hazard, not an active corruption: it
    would have surfaced the moment `dropna` was relaxed or the lag set changed.
    Computing the shift inside the group removes the hazard outright.

    Args:
        dropna: Drop rows whose lags are undefined (the first `max(lag)` and
            `rolling_window` intervals of each cluster). Set False to keep the
            panel rectangular and inspect the NaNs yourself.

    Returns:
        Copy of `df` with `lag_<k>` and `rolling_mean` columns added.
    """
    lags = tuple(int(lag) for lag in lags)
    if any(lag < 1 for lag in lags):
        raise ValueError(f"lags must all be >= 1, got {lags}")
    if rolling_window < 1:
        raise ValueError(f"rolling_window must be >= 1, got {rolling_window}")

    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out = (
        out.sort_values([cluster_col, ts_col])
        .drop_duplicates(subset=[ts_col, cluster_col], keep="last")
        .reset_index(drop=True)
    )

    grouped = out.groupby(cluster_col, sort=False)[target]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)

    # shift(1) INSIDE the per-group lambda - not after the transform - so the
    # window never spans two clusters.
    out["rolling_mean"] = out.groupby(cluster_col, sort=False)[target].transform(
        lambda s: s.shift(1).rolling(window=rolling_window, min_periods=1).mean()
    )

    if dropna:
        feature_cols = [f"lag_{lag}" for lag in lags] + ["rolling_mean"]
        before = len(out)
        out = out.dropna(subset=feature_cols).reset_index(drop=True)
        logger.debug(
            "add_lag_features dropped %d warm-up rows (%d -> %d)",
            before - len(out), before, len(out),
        )
    return out


def lag_feature_names(
    lags: Iterable[int] = DEFAULT_LAGS, rolling_window: int | None = None
) -> list[str]:
    """Ordered names of the lag block, matching `add_lag_features` output."""
    names = [f"lag_{int(lag)}" for lag in lags]
    if rolling_window is None or rolling_window >= 1:
        names.append("rolling_mean")
    return names


def build_feature_names(
    *,
    use_lags: bool,
    lags: Iterable[int] = DEFAULT_LAGS,
    cluster_features: Sequence[str] = (CLUSTER_COL,),
) -> list[str]:
    """
    Assemble the ordered feature list for a model variant.

    Order is fixed and shared by both paths; the result is persisted in the
    `ModelBundle` so serving never has to guess.
    """
    names = list(cluster_features) + list(CALENDAR_FEATURES)
    if use_lags:
        names += lag_feature_names(lags)
    return names


def attach_cluster_centroids(
    df: pd.DataFrame,
    centroids: np.ndarray,
    *,
    cluster_col: str = CLUSTER_COL,
) -> pd.DataFrame:
    """
    Attach each cluster's centroid latitude/longitude.

    A K-Means label is nominal: cluster 23 is not "between" 22 and 24, and it is
    not "more" than cluster 4. Feeding the raw integer to a tree model invites
    splits like `pickup_cluster < 23.5`, which partition on an arbitrary
    labelling rather than on geography. The centroid coordinates restore the
    real spatial relationship, so a split genuinely separates areas of the city.

    Args:
        centroids: `(n_clusters, 2)` array of (lat, lng), indexed by label.
    """
    centroids = np.asarray(centroids)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"centroids must be (n_clusters, 2), got {centroids.shape}")

    labels = df[cluster_col].to_numpy()
    valid = (labels >= 0) & (labels < len(centroids))
    if not valid.all():
        raise ValueError(
            f"{(~valid).sum()} rows carry cluster labels outside "
            f"[0, {len(centroids)}); the clustering model does not match this data."
        )

    out = df.copy()
    out["cluster_lat"] = centroids[labels, 0]
    out["cluster_lng"] = centroids[labels, 1]
    return out


# ---------------------------------------------------------------------------
# Model bundle: the feature contract, persisted with the model
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """
    A fitted model plus everything needed to reproduce its input.

    Persisting the ordered feature list alongside the estimator is what makes
    train/serve skew detectable instead of silent: `design_matrix()` raises on a
    missing or misordered column rather than handing the model a differently
    shaped frame.
    """

    model: Any
    feature_names: list[str]
    uses_lags: bool
    lags: tuple[int, ...] = DEFAULT_LAGS
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    freq: str = DEFAULT_FREQ
    target: str = TARGET_COL
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat())
    training_rows: int = 0
    notes: str = ""

    def design_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select this model's features, in fit order, validating presence."""
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise KeyError(
                f"Frame is missing feature(s) required by this model: {missing}. "
                f"Model expects, in order: {self.feature_names}"
            )
        return df.loc[:, self.feature_names]

    def predict(self, df: pd.DataFrame, clip_min: float | None = 0.0) -> np.ndarray:
        """
        Predict, optionally clipping at a floor.

        The target is a non-negative count; a squared-error objective will
        happily emit negative demand. Clipping at zero is applied by default so
        no negative forecast reaches a consumer.
        """
        preds = np.asarray(self.model.predict(self.design_matrix(df)), dtype="float64")
        return np.clip(preds, clip_min, None) if clip_min is not None else preds

    def save(self, path: str | Path, compress: int = 3) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self, path, compress=compress)
        logger.info("Saved model bundle to %s", path)
        return str(path)

    @staticmethod
    def load_bundle(path: str | Path) -> ModelBundle:
        """
        Load a bundle, tolerating legacy bare-estimator artefacts.

        Older runs saved the estimator alone with no feature contract. Those are
        wrapped with the historical feature order and a loud warning, because
        there is no way to verify what they were actually fitted on.
        """
        obj = load(path)
        if isinstance(obj, ModelBundle):
            return obj
        logger.warning(
            "%s holds a bare estimator with no feature contract (pre-bundle "
            "artefact). Assuming the legacy feature order; retrain to remove "
            "this ambiguity.", path,
        )
        return ModelBundle(
            model=obj,
            feature_names=build_feature_names(use_lags=False),
            uses_lags=False,
            notes="Legacy artefact; feature order assumed, not verified.",
        )
