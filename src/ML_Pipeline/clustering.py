"""
Offline diagnostics for choosing a cluster count.

This is analysis support, not a pipeline stage. It used to be called
unconditionally from `data_prep_geospatial`, where it fitted nine additional
K-Means models and ran an O(k^2) pure-Python distance loop on every run - and
then discarded the return value, because it only prints.

`min_distance` is now vectorised (a full pairwise haversine matrix via numpy
rather than a nested loop) and `optimal_cluster` no longer takes the unused
DataFrame argument it previously required.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

EARTH_RADIUS_MILES = 3958.7613


def pairwise_haversine_miles(coords: np.ndarray) -> np.ndarray:
    """
    Pairwise great-circle distances, in miles.

    Args:
        coords: `(n, 2)` array of (latitude, longitude) in degrees.

    Returns:
        `(n, n)` symmetric distance matrix with a zero diagonal.
    """
    lat, lng = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlng = lng[:, None] - lng[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlng / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def min_distance(
    region_centers: np.ndarray, threshold_miles: float = 2.0
) -> dict[str, float]:
    """
    Summarise how tightly packed a set of cluster centres is.

    Returns:
        Mean neighbours within / outside `threshold_miles`, and the smallest
        distance between any two distinct centres.
    """
    centers = np.asarray(region_centers)
    n = len(centers)
    if n < 2:
        return {
            "clusters": n, "mean_within": 0.0,
            "mean_outside": 0.0, "min_distance": float("nan"),
        }

    distances = pairwise_haversine_miles(centers)
    off_diagonal = ~np.eye(n, dtype=bool)
    within = (distances < threshold_miles) & off_diagonal

    summary = {
        "clusters": n,
        "mean_within": float(within.sum(axis=1).mean()),
        "mean_outside": float((off_diagonal.sum(axis=1) - within.sum(axis=1)).mean()),
        "min_distance": float(distances[off_diagonal].min()),
    }
    logger.info(
        "k=%(clusters)d: mean neighbours <%.1f mi = %(mean_within).1f, "
        "closest pair = %(min_distance).3f mi",
        summary, threshold_miles, summary,
    )
    return summary


def making_regions(n_regions: int, coords, random_state: int = 0) -> np.ndarray:
    """Fit K-Means and return its cluster centres."""
    model = MiniBatchKMeans(
        n_clusters=n_regions, batch_size=10_000, random_state=random_state, n_init=10
    ).fit(coords)
    return model.cluster_centers_


def optimal_cluster(
    coords,
    candidates: Iterable[int] = range(10, 100, 10),
    threshold_miles: float = 2.0,
) -> pd.DataFrame:
    """
    Sweep candidate cluster counts and report inter-cluster spacing.

    Args:
        coords: `(n_samples, 2)` array of (latitude, longitude).
        candidates: Cluster counts to evaluate.
        threshold_miles: "Nearby" radius used in the summary.

    Returns:
        One row per candidate. Returning the sweep (rather than only printing
        it, as before) lets a caller pick a `k` programmatically or record the
        evidence behind the choice.
    """
    started = datetime.now()
    rows = [
        min_distance(making_regions(k, coords), threshold_miles=threshold_miles)
        for k in candidates
    ]
    logger.info("Cluster sweep took %s", datetime.now() - started)
    return pd.DataFrame(rows)
