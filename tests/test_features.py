"""Unit tests for the canonical feature-engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.features import (
    CALENDAR_FEATURES,
    ModelBundle,
    add_calendar_features,
    add_lag_features,
    attach_cluster_centroids,
    build_demand_grid,
    build_feature_names,
    floor_to_interval,
    lag_feature_names,
    validate_grid,
)


class TestFloorToInterval:
    def test_floors_to_half_hour(self):
        got = floor_to_interval(
            ["2021-03-27 00:29:59", "2021-03-27 00:30:00", "2021-03-27 23:59:59"], 30
        )
        assert list(got) == [
            pd.Timestamp("2021-03-27 00:00"),
            pd.Timestamp("2021-03-27 00:30"),
            pd.Timestamp("2021-03-27 23:30"),
        ]

    def test_accepts_mixed_input_types(self):
        """The two old copies disagreed on str / datetime64 / Timestamp handling."""
        mixed = pd.Series(
            [
                "2021-03-27 00:45:10",
                np.datetime64("2021-03-27T01:20:00"),
                pd.Timestamp("2021-03-27 02:59:59"),
            ]
        )
        got = floor_to_interval(mixed, 30)
        assert list(got) == [
            pd.Timestamp("2021-03-27 00:30"),
            pd.Timestamp("2021-03-27 01:00"),
            pd.Timestamp("2021-03-27 02:30"),
        ]

    def test_rejects_non_positive_interval(self):
        with pytest.raises(ValueError, match="must be positive"):
            floor_to_interval(["2021-03-27 00:00:00"], 0)


class TestCalendarFeatures:
    def test_produces_exactly_the_declared_features(self, panel):
        assert set(CALENDAR_FEATURES).issubset(panel.columns)

    def test_values_are_correct(self):
        df = pd.DataFrame({"ts": [pd.Timestamp("2021-03-27 14:30")]})  # a Saturday
        got = add_calendar_features(df, "ts").iloc[0]
        assert (got.hour, got.mins, got.month, got.quarter, got.dayofweek) == (
            14, 30, 3, 1, 5,
        )

    def test_excludes_year_and_day_of_month(self):
        """Neither generalises; `day` also let the model memorise the old split."""
        assert "year" not in CALENDAR_FEATURES
        assert "day" not in CALENDAR_FEATURES

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="not found"):
            add_calendar_features(pd.DataFrame({"other": [1]}), "ts")


class TestBuildDemandGrid:
    def test_grid_is_rectangular(self, bookings):
        grid = build_demand_grid(bookings, freq="30min", clusters=list(range(4)))
        report = validate_grid(grid, freq="30min")
        assert report["is_rectangular"]
        assert report["duplicate_keys"] == 0
        assert report["rows"] == report["expected_rows"]

    def test_counts_match_source_bookings(self, bookings):
        grid = build_demand_grid(bookings, freq="30min", clusters=list(range(4)))
        assert grid["request_count"].sum() == len(bookings)

    def test_empty_intervals_are_zero_not_missing(self):
        sparse = pd.DataFrame(
            {
                "ts": ["2021-03-27 00:00:00", "2021-03-27 02:00:00"],
                "pickup_cluster": [0, 0],
            }
        )
        grid = build_demand_grid(sparse, freq="30min", clusters=[0])
        assert len(grid) == 5  # 00:00 .. 02:00 inclusive
        assert grid["request_count"].notna().all()
        assert grid["request_count"].tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]

    def test_covers_requested_clusters_even_with_no_bookings(self):
        """A cluster silent in the serving window must still appear in the grid."""
        one = pd.DataFrame({"ts": ["2021-03-27 00:00:00"], "pickup_cluster": [0]})
        grid = build_demand_grid(one, freq="30min", clusters=[0, 1, 2])
        assert sorted(grid["pickup_cluster"].unique()) == [0, 1, 2]
        assert grid.loc[grid.pickup_cluster == 2, "request_count"].sum() == 0

    def test_range_is_derived_from_data_not_hardcoded(self):
        """The old code only produced a grid for one hardcoded 2020-2021 year."""
        for year in (2019, 2023, 2030):
            df = pd.DataFrame(
                {"ts": [f"{year}-06-01 00:00:00", f"{year}-06-01 01:00:00"],
                 "pickup_cluster": [0, 0]}
            )
            grid = build_demand_grid(df, freq="30min", clusters=[0])
            assert len(grid) == 3
            assert pd.to_datetime(grid["ts"]).dt.year.eq(year).all()

    def test_explicit_range_is_honoured(self):
        df = pd.DataFrame({"ts": ["2021-03-27 12:00:00"], "pickup_cluster": [0]})
        grid = build_demand_grid(
            df, freq="30min", clusters=[0],
            start=pd.Timestamp("2021-03-27 11:00"), end=pd.Timestamp("2021-03-27 13:00"),
        )
        assert len(grid) == 5
        assert grid["request_count"].sum() == 1

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError, match="pickup_cluster"):
            build_demand_grid(pd.DataFrame({"ts": ["2021-03-27"]}))

    def test_end_before_start_raises(self):
        df = pd.DataFrame({"ts": ["2021-03-27 00:00:00"], "pickup_cluster": [0]})
        with pytest.raises(ValueError, match="precedes start"):
            build_demand_grid(
                df, clusters=[0],
                start=pd.Timestamp("2021-03-28"), end=pd.Timestamp("2021-03-27"),
            )


class TestAddLagFeatures:
    def test_lags_are_exact(self, small_panel):
        out = add_lag_features(small_panel, lags=(1, 2, 3), rolling_window=3)
        c0 = out[out.pickup_cluster == 0].sort_values("ts")
        # cluster 0 counts are 0..9; warm-up drops the first 3 intervals.
        assert c0["request_count"].tolist() == [3, 4, 5, 6, 7, 8, 9]
        assert c0["lag_1"].tolist() == [2, 3, 4, 5, 6, 7, 8]
        assert c0["lag_2"].tolist() == [1, 2, 3, 4, 5, 6, 7]
        assert c0["lag_3"].tolist() == [0, 1, 2, 3, 4, 5, 6]

    def test_rolling_mean_is_backward_looking(self, small_panel):
        """rolling_mean must never include the row's own target."""
        out = add_lag_features(small_panel, lags=(1,), rolling_window=3)
        c0 = out[out.pickup_cluster == 0].sort_values("ts").reset_index(drop=True)
        row = c0[c0.request_count == 5].iloc[0]
        assert row["rolling_mean"] == pytest.approx(np.mean([2, 3, 4]))

    def test_no_leakage_across_cluster_boundary(self):
        """
        Regression test for the frame-wide `.shift(1)` in the old implementation,
        which handed the first row of each cluster the previous cluster's value.
        """
        stamps = pd.date_range("2021-03-27", periods=6, freq="30min")
        df = pd.DataFrame(
            {
                "ts": list(stamps) * 2,
                "pickup_cluster": [0] * 6 + [1] * 6,
                "request_count": [1000.0] * 6 + [1.0] * 6,
            }
        )
        out = add_lag_features(df, lags=(1,), rolling_window=3, dropna=False)
        cluster1 = out[out.pickup_cluster == 1].sort_values("ts")
        # Cluster 1's first row has no history of its own -> NaN, never 1000.
        assert pd.isna(cluster1.iloc[0]["rolling_mean"])
        assert pd.isna(cluster1.iloc[0]["lag_1"])
        finite = cluster1["rolling_mean"].dropna()
        assert (finite <= 1.0).all(), "cluster 0's magnitude leaked into cluster 1"

    def test_warm_up_rows_dropped_per_cluster(self, small_panel):
        out = add_lag_features(small_panel, lags=(1, 2, 3), rolling_window=3)
        # 10 intervals per cluster, first 3 lack a full lag set.
        assert len(out) == (10 - 3) * 2
        assert out[["lag_1", "lag_2", "lag_3", "rolling_mean"]].notna().all().all()

    def test_dropna_false_keeps_panel_rectangular(self, small_panel):
        out = add_lag_features(small_panel, lags=(1, 2, 3), dropna=False)
        assert len(out) == len(small_panel)
        assert out["lag_3"].isna().sum() == 6  # 3 per cluster

    def test_invalid_parameters_raise(self, small_panel):
        with pytest.raises(ValueError, match="lags must all be"):
            add_lag_features(small_panel, lags=(0,))
        with pytest.raises(ValueError, match="rolling_window"):
            add_lag_features(small_panel, rolling_window=0)


class TestClusterCentroids:
    def test_attaches_coordinates(self):
        centroids = np.array([[12.9, 77.5], [13.0, 77.6]])
        df = pd.DataFrame({"pickup_cluster": [0, 1, 0]})
        out = attach_cluster_centroids(df, centroids)
        assert out["cluster_lat"].tolist() == [12.9, 13.0, 12.9]
        assert out["cluster_lng"].tolist() == [77.5, 77.6, 77.5]

    def test_out_of_range_label_raises(self):
        with pytest.raises(ValueError, match="outside"):
            attach_cluster_centroids(
                pd.DataFrame({"pickup_cluster": [5]}), np.array([[12.9, 77.5]])
            )

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(n_clusters, 2\)"):
            attach_cluster_centroids(
                pd.DataFrame({"pickup_cluster": [0]}), np.array([1.0, 2.0])
            )


class TestFeatureNames:
    def test_lag_block_order_is_stable(self):
        assert lag_feature_names((1, 2, 3)) == ["lag_1", "lag_2", "lag_3", "rolling_mean"]

    def test_nolag_and_lag_share_a_prefix(self):
        """The lag model must extend the base features, not reorder them."""
        base = build_feature_names(use_lags=False)
        full = build_feature_names(use_lags=True, lags=(1, 2, 3))
        assert full[: len(base)] == base

    def test_centroid_features_replace_raw_label(self):
        names = build_feature_names(
            use_lags=False, cluster_features=("cluster_lat", "cluster_lng")
        )
        assert "pickup_cluster" not in names
        assert names[:2] == ["cluster_lat", "cluster_lng"]


class NegativeStub:
    """Always predicts a negative value. Module-level so joblib can pickle it."""

    def predict(self, X):
        return np.full(len(X), -5.0)


class TestModelBundle:
    def _bundle(self):
        return ModelBundle(
            model=NegativeStub(), feature_names=["hour", "dayofweek"], uses_lags=False
        )

    def test_design_matrix_enforces_order(self):
        bundle = self._bundle()
        df = pd.DataFrame({"dayofweek": [1], "hour": [9], "extra": [0]})
        assert list(bundle.design_matrix(df).columns) == ["hour", "dayofweek"]

    def test_missing_feature_raises_rather_than_silently_reordering(self):
        bundle = self._bundle()
        with pytest.raises(KeyError, match="missing feature"):
            bundle.design_matrix(pd.DataFrame({"hour": [9]}))

    def test_predictions_are_clipped_at_zero(self):
        """The target is a non-negative count; negative demand must not ship."""
        bundle = self._bundle()
        preds = bundle.predict(pd.DataFrame({"hour": [9], "dayofweek": [1]}))
        assert (preds >= 0).all()

    def test_clipping_can_be_disabled(self):
        bundle = self._bundle()
        preds = bundle.predict(
            pd.DataFrame({"hour": [9], "dayofweek": [1]}), clip_min=None
        )
        assert preds[0] == -5.0

    def test_roundtrip_preserves_contract(self, tmp_path):
        bundle = self._bundle()
        bundle.metrics = {"rmse": 1.5}
        path = bundle.save(tmp_path / "m.joblib")
        loaded = ModelBundle.load_bundle(path)
        assert loaded.feature_names == bundle.feature_names
        assert loaded.metrics == {"rmse": 1.5}
        assert loaded.uses_lags is False

    def test_legacy_bare_estimator_is_wrapped(self, tmp_path):
        from joblib import dump

        path = tmp_path / "legacy.joblib"
        dump(NegativeStub(), path)
        loaded = ModelBundle.load_bundle(path)
        assert isinstance(loaded, ModelBundle)
        assert "Legacy" in loaded.notes


class TestValidateGrid:
    def test_reports_instead_of_asserting(self):
        """
        The old code used `assert len(data) == 878400`, which crashed on any
        other dataset. Validation now returns a report the caller can act on.
        """
        ragged = pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    ["2021-03-27 00:00", "2021-03-27 00:30", "2021-03-27 00:00"]
                ),
                "pickup_cluster": [0, 0, 1],
                "request_count": [1.0, 2.0, 3.0],
            }
        )
        report = validate_grid(ragged, freq="30min")
        assert report["is_rectangular"] is False
        assert report["rows"] == 3
        assert report["expected_rows"] == 4

    def test_detects_duplicate_keys(self, panel):
        doubled = pd.concat([panel, panel.head(1)], ignore_index=True)
        assert validate_grid(doubled, freq="30min")["duplicate_keys"] == 1
