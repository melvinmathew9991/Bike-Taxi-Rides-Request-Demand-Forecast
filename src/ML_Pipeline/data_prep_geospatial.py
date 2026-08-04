"""
Geospatial stage: cluster pickup locations, then aggregate to a demand grid.

Changes from the previous implementation:

* The cluster count comes from configuration instead of being hardcoded to 50
  while `PipelineConfig.n_clusters` (default 300) and the `--n-clusters` CLI flag
  were silently ignored.
* The cluster-spacing diagnostic no longer runs on every execution. It fitted
  nine extra K-Means models and ran an O(k^2) Python distance loop, then threw
  the result away - it only ever printed. It is now opt-in.
* The grid is built by `features.build_demand_grid`, which reindexes onto the
  full (interval x cluster) product. The old approach appended a synthetic
  `pickup_cluster = -1` row for each interval of a hardcoded 365-day range to
  coerce `asfreq` into spanning the year, dropped it again, and then asserted the
  result was exactly 878,400 rows - a number that only holds for one dataset at
  one cluster count. The dummy range (48*365) and the assert (366 days) did not
  even agree with each other.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans, MiniBatchKMeans

from ML_Pipeline.features import (
    CLUSTER_COL,
    add_calendar_features,
    build_demand_grid,
    validate_grid,
)

logger = logging.getLogger(__name__)

DEFAULT_N_CLUSTERS = 50
DEFAULT_FREQ = "30min"
DEFAULT_INTERVAL_MINUTES = 30


def fit_cluster_model(
    coords,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    algorithm: str = "minibatch",
    random_state: int = 5,
    batch_size: int = 10_000,
):
    """
    Fit a clustering model over pickup coordinates.

    Args:
        coords: `(n_samples, 2)` array of (latitude, longitude).
        n_clusters: Number of geographic regions.
        algorithm: `"minibatch"` (fast, default) or `"kmeans"` (exact).
        random_state: Seed, fixed so runs are reproducible.
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")

    logger.info("Fitting %s with %d clusters on %d points", algorithm, n_clusters, len(coords))
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    elif algorithm == "minibatch":
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            n_init=10,
        )
    else:
        raise ValueError(
            f"Unknown clustering algorithm {algorithm!r}; expected "
            "'minibatch' or 'kmeans'."
        )
    return model.fit(coords)


def data_prep_geospatial(
    df: pd.DataFrame,
    model_path: str,
    data_path: str,
    *,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    algorithm: str = "minibatch",
    freq: str = DEFAULT_FREQ,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    random_state: int = 5,
    run_cluster_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Assign pickup clusters and aggregate bookings into a demand grid.

    Args:
        df: Cleaned booking-level data with `ts`, `pick_lat`, `pick_lng`.
        model_path: Where to persist the fitted clustering model.
        data_path: Where to write the resulting grid (gzip CSV).
        n_clusters: Number of geographic regions.
        algorithm: Clustering algorithm.
        freq: Grid frequency.
        interval_minutes: Interval width used to floor booking timestamps.
        random_state: Seed for reproducibility.
        run_cluster_diagnostics: Fit a sweep of cluster counts and report
            inter-cluster spacing. Expensive; off by default.

    Returns:
        The demand grid with calendar features.
    """
    start_time = datetime.now()

    required = {"ts", "pick_lat", "pick_lng"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    coords = df[["pick_lat", "pick_lng"]].to_numpy()

    if run_cluster_diagnostics:
        from ML_Pipeline.clustering import optimal_cluster

        logger.info("Running cluster-spacing diagnostics (expensive)...")
        optimal_cluster(coords)

    model = fit_cluster_model(
        coords, n_clusters=n_clusters, algorithm=algorithm, random_state=random_state
    )
    df = df.copy()
    df[CLUSTER_COL] = model.predict(coords)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path, compress=3)
    logger.info("Clustering model saved to %s", model_path)

    grid = build_demand_grid(
        df,
        freq=freq,
        interval_minutes=interval_minutes,
        clusters=list(range(n_clusters)),
    )

    # Report rather than assert: the old `assert len(data) == 878400` crashed on
    # any dataset that was not the original one.
    report = validate_grid(grid, freq=freq)
    if report["is_rectangular"]:
        logger.info(
            "Demand grid OK: %(timestamps)d intervals x %(clusters)d clusters "
            "= %(rows)d rows", report,
        )
    else:
        logger.warning("Demand grid is not rectangular: %s", report)

    grid = add_calendar_features(grid, "ts")

    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(data_path, index=False, compression="gzip")
    logger.info("Prepared data written to %s", data_path)
    logger.info("Geospatial preparation took %s", datetime.now() - start_time)
    return grid
