"""
Scheduled health checks on the model that is serving.

The model card sets conditions for using the model, and two of them depended on
someone remembering: retrain at least every four weeks, and monitor
`level_ratio`, which degrades earliest of any metric as the model goes stale.
`check_health` turns those into checks a scheduler can run and act on; the CLI
wrapper is `biketaxi monitor` (`ML_Pipeline.cli.monitor`).

It scores the most recent `days` of observed demand, one step ahead, which is how
the deploy gate and the model card's staleness curve are measured, so the numbers
are comparable to both. Only demand after the model's `data_through` is scored:
a model refit on all data has seen everything before it, and scoring it there is
in-sample.

Thresholds, from the staleness table in docs/MODEL_CARD.md (mean level ratio by
weeks stale: 0.92 at four, 0.87 at five, 0.83 at six, where the worst origin
first loses to seasonal-naive):

* level ratio below 0.90 fails - past the four-week cadence, before the measured
  failure. Above 1.10 also fails; over-forecasting has not been observed, but it
  would send supply where demand is not, and nothing else here would notice.
* MASE at or above 1 fails - the model loses to a baseline that costs nothing.
* staleness past `STALE_AFTER_DAYS` fails, by the same rule the API reports.

Per-cluster results warn rather than fail. One week of one cluster's sparse
half-hourly counts is noisy enough that a single cluster crossing 1.0 is not, on
its own, evidence that the model should be pulled - but it is the view that found
cluster 30 losing to the baseline when the global number looked fine, so it is
always reported.

The level band applies per cluster only above `MIN_CLUSTER_VOLUME`. A replay
drill on the real data - models trained 2, 4 and 6 weeks before the last week
and scored on it - flagged the same 8 clusters at every staleness, all beating
the baseline (MASE 0.69-0.88) and all among the quietest, at 0.3-2.8 requests
per interval against a median of 10. At that volume over-forecasting by a
fraction of a request doubles the ratio, so the band measures noise and would
warn every week. Quiet clusters are still held to MASE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ML_Pipeline.evaluation import ModelEvaluator
from ML_Pipeline.features import (
    CLUSTER_COL,
    TARGET_COL,
    TS_COL,
    add_calendar_features,
    add_lag_features,
    attach_cluster_centroids,
)
from ML_Pipeline.serving import STALE_AFTER_DAYS, ServingState

#: Level-ratio band outside which the model fails its health check.
MIN_LEVEL_RATIO = 0.90
MAX_LEVEL_RATIO = 1.10

#: Mean requests per interval below which a cluster's level ratio is not
#: checked: a cluster this quiet has Poisson noise on the scale of its mean.
MIN_CLUSTER_VOLUME = 3.0

#: Days of unseen demand scored. A whole week, so every weekday is represented
#: once and the level ratio is not skewed by a weekend.
DEFAULT_WINDOW_DAYS = 7

PASS, WARN, FAIL, SKIPPED = "pass", "warn", "fail", "skipped"


class MissingFeaturesError(ValueError):
    """The model needs features the demand grid cannot supply."""


def scoring_frame(state: ServingState) -> pd.DataFrame:
    """
    Rebuild every feature the serving model expects from the bare demand grid.

    The serving history is `[ts, cluster, count]` only, so lags, calendar
    features and centroids all have to be rebuilt. A feature that still cannot be
    supplied is a train/serve mismatch, and is raised by name rather than left to
    surface as an opaque `KeyError` from `predict`.
    """
    bundle = state.bundle
    frame = add_lag_features(
        state.history, lags=bundle.lags, rolling_window=bundle.rolling_window
    )
    frame = add_calendar_features(frame, TS_COL)
    if state.centroids is not None and "cluster_lat" in bundle.feature_names:
        frame = attach_cluster_centroids(frame, state.centroids)

    missing = [c for c in bundle.feature_names if c not in frame.columns]
    if missing:
        raise MissingFeaturesError(
            "This model needs feature(s) the demand grid cannot supply: "
            + ", ".join(missing)
        )
    return frame


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class HealthReport:
    model_name: str | None
    data_through: str | None = None
    history_ends_at: str | None = None
    data_lag_days: float | None = None
    window_start: str | None = None
    window_end: str | None = None
    rows_scored: int = 0
    mase: float | None = None
    level_ratio: float | None = None
    checks: list[Check] = field(default_factory=list)
    clusters: list[dict[str, Any]] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("data_lag_days", "mase", "level_ratio"):
            out[key] = _plain(out[key])
        out["healthy"] = self.healthy
        return out


def _iso(ts) -> str | None:
    return None if ts is None else pd.Timestamp(ts).isoformat()


def _plain(value):
    """A JSON-safe scalar: numpy unwrapped, NaN as None."""
    value = value.item() if hasattr(value, "item") else value
    return None if isinstance(value, float) and not np.isfinite(value) else value


def check_health(
    state: ServingState,
    *,
    days: int = DEFAULT_WINDOW_DAYS,
    min_level_ratio: float = MIN_LEVEL_RATIO,
    max_level_ratio: float = MAX_LEVEL_RATIO,
    min_cluster_volume: float = MIN_CLUSTER_VOLUME,
) -> HealthReport:
    """
    Score the serving model on its most recent unseen demand.

    Raises:
        ValueError: if no model is serving or it has no history - there is
            nothing to check, which is a different failure from an unhealthy
            model. `MissingFeaturesError` is a subclass.
    """
    if not state.ready:
        raise ValueError(
            "No model is ready to serve: none is promoted, its file is missing, "
            "or there is no demand grid."
        )

    report = HealthReport(
        model_name=state.model_name,
        data_through=_iso(state.data_through),
        history_ends_at=_iso(state.history_ends_at),
        data_lag_days=state.data_lag_days,
    )

    staleness = state.staleness_warning()
    if staleness:
        report.add("staleness", FAIL, staleness)
    else:
        lag = state.data_lag_days
        report.add(
            "staleness", PASS,
            f"Training data ends {lag:.1f} days before the latest demand, within "
            f"the {STALE_AFTER_DAYS}-day cadence." if lag is not None
            else "Training data end unknown; model age is within the cadence.",
        )

    frame = scoring_frame(state)
    bundle = state.bundle
    season = int(pd.Timedelta(days=7) / pd.Timedelta(bundle.freq))

    # The seasonal-naive forecast comes from the full history, so the window can
    # be exactly the span scored. Computed inside the window, its first week
    # would have no baseline.
    history = state.history.sort_values([CLUSTER_COL, TS_COL])
    naive = history[[TS_COL, CLUSTER_COL]].assign(
        _naive=ModelEvaluator.seasonal_naive_baseline(history, season_length=season)
    )

    end = frame[TS_COL].max()
    start = end - pd.Timedelta(days=days) + pd.Timedelta(bundle.freq)
    window = frame[frame[TS_COL] >= start]
    if state.data_through is not None:
        window = window[window[TS_COL] > state.data_through]
    window = (
        window.merge(naive, on=[TS_COL, CLUSTER_COL], how="left")
        .sort_values([CLUSTER_COL, TS_COL])
        .reset_index(drop=True)
    )

    unseen_days = (
        window[TS_COL].nunique() * pd.Timedelta(bundle.freq) / pd.Timedelta(days=1)
    )
    if unseen_days < days:
        reason = (
            f"Only {unseen_days:.1f} days of demand arrived after the model's "
            f"training data; {days} are needed for a full-week score. Until then "
            "the deploy gate, measured on a held-out fit, is the out-of-sample "
            "measurement."
        )
        report.add("mase", SKIPPED, reason)
        report.add("level_ratio", SKIPPED, reason)
        return report

    report.window_start = _iso(window[TS_COL].min())
    report.window_end = _iso(window[TS_COL].max())
    report.rows_scored = len(window)

    pred = np.asarray(bundle.predict(window), dtype="float64")
    actual = window[TARGET_COL].to_numpy(dtype="float64")
    naive_pred = window["_naive"].to_numpy(dtype="float64")

    report.mase = ModelEvaluator.mase(actual, pred, naive_pred)
    report.level_ratio = (
        float(pred.mean() / actual.mean()) if actual.mean() else float("nan")
    )

    if not np.isfinite(report.mase):
        report.add("mase", SKIPPED, "No seasonal-naive baseline over the window.")
    elif report.mase >= 1.0:
        report.add(
            "mase", FAIL,
            f"MASE {report.mase:.3f}: the model loses to a seasonal-naive "
            "forecast. Ship the baseline until a retrained model beats it.",
        )
    else:
        report.add("mase", PASS, f"MASE {report.mase:.3f} against seasonal-naive.")

    band = f"{min_level_ratio:.2f}-{max_level_ratio:.2f}"
    if not np.isfinite(report.level_ratio):
        report.add("level_ratio", SKIPPED, "No demand in the window.")
    elif not min_level_ratio <= report.level_ratio <= max_level_ratio:
        direction = (
            "under-forecasting" if report.level_ratio < min_level_ratio
            else "over-forecasting"
        )
        report.add(
            "level_ratio", FAIL,
            f"Level ratio {report.level_ratio:.3f} is outside {band}: the model is "
            f"{direction} demand. Retrain.",
        )
    else:
        report.add(
            "level_ratio", PASS,
            f"Level ratio {report.level_ratio:.3f}, within {band}.",
        )

    per_cluster = ModelEvaluator.per_cluster_error(
        window, pred, season_length=season, naive=naive_pred
    )
    # The level band only where volume makes the ratio meaningful. A cluster
    # with no demand all week has no ratio at all; neither is a fault.
    ratio = per_cluster["level_ratio"]
    busy = per_cluster["mean_actual"] >= min_cluster_volume
    flagged = per_cluster[
        (per_cluster["mase"] >= 1.0)
        | (busy & ratio.notna() & ~ratio.between(min_level_ratio, max_level_ratio))
    ]
    report.clusters = [
        {k: _plain(v) for k, v in row.items()}
        for row in per_cluster.to_dict("records")
    ]
    quiet = int((~busy).sum())
    scope = (
        f" ({quiet} below {min_cluster_volume:g} requests per interval held to "
        "the baseline only)" if quiet else ""
    )
    if len(flagged):
        report.add(
            "clusters", WARN,
            f"{len(flagged)} of {len(per_cluster)} clusters lose to the baseline "
            f"or fall outside the {band} level band{scope}: "
            + ", ".join(f"#{int(c)}" for c in flagged[CLUSTER_COL].head(20)),
        )
    else:
        report.add(
            "clusters", PASS,
            f"All {len(per_cluster)} clusters beat the baseline within the "
            f"{band} level band{scope}.",
        )
    return report
