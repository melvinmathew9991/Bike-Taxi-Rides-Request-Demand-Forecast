"""
Tests for choosing which clusters the lag-free model forecasts.

The lag-free model has no channel carrying current demand and loses to
seasonal-naive by 75%, yet every run used to emit it for every cluster, and the
dashboard offered it as an alternative forecast. It is now the fallback for
clusters too new to have a week of history, and nothing else.
"""

from __future__ import annotations

import pandas as pd

from ML_Pipeline.modeling.prediction import cold_start_clusters

FREQ = "30min"
START = pd.Timestamp("2021-01-08")
NEEDED = 48


def _history(cluster_intervals: dict[int, int]) -> pd.DataFrame:
    """For each cluster, its last `n` intervals before START."""
    rows = []
    for cluster, n in cluster_intervals.items():
        for t in pd.date_range(end=START - pd.Timedelta(FREQ), periods=n, freq=FREQ):
            rows.append((t, cluster, 1.0))
    return pd.DataFrame(rows, columns=["ts", "pickup_cluster", "request_count"])


def _cold(history, clusters=(0, 1, 2)):
    return cold_start_clusters(history, list(clusters), START, freq=FREQ, needed=NEEDED)


class TestColdStartClusters:
    def test_a_cluster_with_enough_history_is_not_cold(self):
        assert _cold(_history({0: 48, 1: 100, 2: 48})) == []

    def test_a_cluster_short_of_history_is_cold(self):
        assert _cold(_history({0: 48, 1: 47, 2: 48})) == [1]

    def test_a_cluster_absent_from_the_history_is_cold(self):
        assert _cold(_history({0: 48, 1: 48})) == [2]

    def test_history_after_the_horizon_does_not_count(self):
        late = _history({0: 48, 1: 48, 2: 48})
        late.loc[late["pickup_cluster"] == 2, "ts"] += pd.Timedelta(days=30)
        assert _cold(pd.concat([_history({0: 48, 1: 48}), late])) == [2]

    def test_a_history_short_for_every_cluster_calls_none_cold(self):
        """
        A missing history source, not a set of new clusters. The recursive
        forecaster refuses it by name; quietly handing every cluster to the
        weaker model would hide it.
        """
        assert _cold(_history({0: 10, 1: 10, 2: 10})) == []
