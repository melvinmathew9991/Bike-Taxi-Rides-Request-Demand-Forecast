"""
Prediction intervals for the recursive forecast, calibrated on held-out errors.

Dispatch decisions are asymmetric - an under-served peak costs more than an idle
rider - so a point forecast alone is not enough to plan supply from. The model
is fitted with a Poisson objective, which makes Poisson quantiles around the
prediction the obvious cheap interval. Measured on the reference dataset, they
are wrong exactly where it matters (`experiments/measure_intervals.py`):

* demand is over-dispersed - one step ahead, variance is 1.4x the mean at
  moderate volume and 4.8x above 50 requests per interval - so a nominal 80%
  Poisson interval covered 56% of intervals predicted above 25 requests, and
  90% of those predicted below 1;
* misses are one-sided: demand landed above the interval five times as often
  as below (18% against 4%), because a growing series is under-forecast;
* error compounds across a recursive horizon, which a per-step Poisson
  interval knows nothing about.

The calibrated intervals below covered 79.7% for a nominal 80% on the same
backtests, and 73% above 25 requests - see "Prediction intervals" in
docs/MODEL_CARD.md.

So the intervals here are empirical (split-conformal). Training backtests the
held-out model recursively from origins spread over the weeks after its
training cut, records each error scaled by `sqrt(max(prediction, 1))`, and keeps
the lower and upper quantiles of that scaled error per (horizon, volume) cell.
Serving adds them back: `prediction + q * sqrt(max(prediction, 1))`, rounded
inward to whole requests and floored at zero.

The calibration window is the retraining cadence - four weeks - so the errors
are from a model as stale as a deployed one gets. The quantiles come from the
held-out fit and are applied to the model refit on all data, which is the
standard approximation: once the test window is in the fit, nothing is left to
calibrate on.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ML_Pipeline.modeling.features import CLUSTER_COL, TARGET_COL, TS_COL, ModelBundle

logger = logging.getLogger(__name__)

LOWER_COL = "request_count_lower"
UPPER_COL = "request_count_upper"

#: Upper edges of the horizon buckets, in hours ahead. The last bucket also
#: takes anything beyond it.
HORIZON_EDGES_HOURS: tuple[float, ...] = (6.0, 24.0, 48.0)

#: Upper edges of the volume buckets, in predicted requests per interval. The
#: last bucket is open-ended. Over-dispersion grows with volume, so one set of
#: quantiles for every cluster over-covers the quiet ones and under-covers the
#: busy ones.
VOLUME_EDGES: tuple[float, ...] = (1.0, 3.0, 10.0, 25.0)

#: Fewer scaled errors than this in a cell, and the cell falls back to its
#: volume bucket pooled over horizons, then to every error pooled.
MIN_CELL_ROWS = 100


@dataclass(frozen=True)
class IntervalCalibration:
    """
    Quantiles of scaled forecast error per (horizon, volume) cell.

    Stored on the `ModelBundle`, so a model and its intervals cannot drift
    apart. Plain tuples throughout, so it pickles with the bundle and stays
    readable without this module's internals.
    """

    level: float
    horizon_edges: tuple[int, ...]  # in steps, upper edge of each bucket
    volume_edges: tuple[float, ...]
    lower_z: tuple[tuple[float, ...], ...]  # [horizon][volume]
    upper_z: tuple[tuple[float, ...], ...]
    cell_rows: tuple[tuple[int, ...], ...]
    origins: int
    window_start: str
    window_end: str
    #: Coverage measured on the later half of the origins, using quantiles from
    #: the earlier half only. None when there were too few origins to split.
    holdout_coverage: float | None = None

    def bounds(
        self, predicted: np.ndarray, steps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Lower and upper bounds, in whole requests, for predictions `steps` ahead.

        Rounded inward - the lower bound up, the upper down - because the target
        is an integer count: an interval [2.3, 7.8] covers exactly the counts
        3..7. Rounding outward instead was measured to push a nominal 80% to
        90%. Both bounds are widened if needed to include the rounded point
        forecast, so a consumer never sees a prediction of 0.6 with an interval
        of [0, 0].
        """
        mu = np.clip(np.asarray(predicted, dtype="float64"), 0.0, None)
        h = np.minimum(
            np.searchsorted(self.horizon_edges, np.asarray(steps), side="left"),
            len(self.horizon_edges) - 1,
        )
        v = np.searchsorted(self.volume_edges, mu, side="left")
        lower_z = np.asarray(self.lower_z)[h, v]
        upper_z = np.asarray(self.upper_z)[h, v]

        scale = np.sqrt(np.maximum(mu, 1.0))
        point = np.round(mu)
        lower = np.minimum(np.maximum(np.ceil(mu + lower_z * scale), 0.0), point)
        upper = np.maximum(np.floor(mu + upper_z * scale), point)
        return lower, upper

    def summary(self) -> dict[str, float]:
        """The figures worth recording in the registry."""
        out = {"interval_level": self.level, "interval_origins": float(self.origins)}
        if self.holdout_coverage is not None:
            out["interval_holdout_coverage"] = self.holdout_coverage
        return out


def add_intervals(
    forecast: pd.DataFrame,
    calibration: IntervalCalibration,
    *,
    prediction_col: str,
    horizon_start: pd.Timestamp,
    freq: str,
) -> pd.DataFrame:
    """Add `request_count_lower` and `request_count_upper` to a forecast frame."""
    step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    steps = ((pd.to_datetime(forecast[TS_COL]) - horizon_start) / step).to_numpy()
    lower, upper = calibration.bounds(
        forecast[prediction_col].to_numpy(), steps.astype("int64") + 1
    )
    return forecast.assign(**{LOWER_COL: lower, UPPER_COL: upper})


def horizon_edges_in_steps(freq: str) -> tuple[int, ...]:
    """`HORIZON_EDGES_HOURS` as step counts at `freq`."""
    step_hours = pd.Timedelta(pd.tseries.frequencies.to_offset(freq)) / pd.Timedelta("1h")
    return tuple(max(1, math.ceil(hours / step_hours)) for hours in HORIZON_EDGES_HOURS)


def backtest_errors(
    bundle: ModelBundle,
    panel: pd.DataFrame,
    origins: Sequence[pd.Timestamp],
    *,
    steps: int,
    centroids: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Recursive forecasts from each origin, paired with what happened.

    Returns:
        `[origin, step, predicted, actual]`, one row per (origin, step, cluster).
    """
    # Imported here: forecast imports this module to attach intervals.
    from ML_Pipeline.modeling.forecast import PREDICTION_COL, forecast_recursive

    observed = panel[[TS_COL, CLUSTER_COL, TARGET_COL]].copy()
    observed[TS_COL] = pd.to_datetime(observed[TS_COL])
    frames = []
    for number, origin in enumerate(origins):
        horizon = pd.date_range(origin, periods=steps, freq=bundle.freq)
        predicted = forecast_recursive(
            bundle, observed[observed[TS_COL] < origin], horizon,
            centroids=centroids, with_intervals=False,
        )
        paired = observed[observed[TS_COL].isin(horizon)].merge(
            predicted[[TS_COL, CLUSTER_COL, PREDICTION_COL]],
            on=[TS_COL, CLUSTER_COL], how="inner", validate="one_to_one",
        )
        step_len = pd.Timedelta(pd.tseries.frequencies.to_offset(bundle.freq))
        frames.append(
            pd.DataFrame({
                "origin": number,
                "step": ((paired[TS_COL] - origin) / step_len).astype("int64") + 1,
                "predicted": paired[PREDICTION_COL].clip(lower=0).to_numpy(),
                "actual": paired[TARGET_COL].to_numpy(dtype="float64"),
            })
        )
    return pd.concat(frames, ignore_index=True)


def _quantile_table(
    errors: pd.DataFrame,
    level: float,
    horizon_edges: tuple[int, ...],
    volume_edges: tuple[float, ...],
) -> tuple[list[list[float]], list[list[float]], list[list[int]]]:
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    mu = errors["predicted"].to_numpy()
    z = (errors["actual"].to_numpy() - mu) / np.sqrt(np.maximum(mu, 1.0))
    h = np.minimum(
        np.searchsorted(horizon_edges, errors["step"].to_numpy(), side="left"),
        len(horizon_edges) - 1,
    )
    v = np.searchsorted(volume_edges, mu, side="left")

    pooled = (float(np.quantile(z, lo_q)), float(np.quantile(z, hi_q)))
    lower, upper, rows = [], [], []
    for hi in range(len(horizon_edges)):
        lower_row, upper_row, rows_row = [], [], []
        for vi in range(len(volume_edges) + 1):
            cell = z[(h == hi) & (v == vi)]
            if len(cell) < MIN_CELL_ROWS:
                cell = z[v == vi]  # this volume, every horizon
            q = (
                (float(np.quantile(cell, lo_q)), float(np.quantile(cell, hi_q)))
                if len(cell) >= MIN_CELL_ROWS
                else pooled
            )
            lower_row.append(q[0])
            upper_row.append(q[1])
            rows_row.append(int(((h == hi) & (v == vi)).sum()))
        lower.append(lower_row)
        upper.append(upper_row)
        rows.append(rows_row)
    return lower, upper, rows


def coverage(
    errors: pd.DataFrame, calibration: IntervalCalibration
) -> float:
    """Share of `errors` whose actual falls inside `calibration`'s interval."""
    lower, upper = calibration.bounds(
        errors["predicted"].to_numpy(), errors["step"].to_numpy()
    )
    actual = errors["actual"].to_numpy()
    return float(np.mean((actual >= lower) & (actual <= upper)))


def _calibration(
    errors: pd.DataFrame, level: float, freq: str, origins: Sequence[pd.Timestamp],
    holdout_coverage: float | None = None,
) -> IntervalCalibration:
    horizon_edges = horizon_edges_in_steps(freq)
    lower, upper, rows = _quantile_table(errors, level, horizon_edges, VOLUME_EDGES)
    return IntervalCalibration(
        level=level,
        horizon_edges=horizon_edges,
        volume_edges=VOLUME_EDGES,
        lower_z=tuple(tuple(r) for r in lower),
        upper_z=tuple(tuple(r) for r in upper),
        cell_rows=tuple(tuple(r) for r in rows),
        origins=len(origins),
        window_start=pd.Timestamp(min(origins)).isoformat(),
        window_end=pd.Timestamp(max(origins)).isoformat(),
        holdout_coverage=holdout_coverage,
    )


def calibrate_intervals(
    bundle: ModelBundle,
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp,
    days: int,
    n_origins: int,
    steps: int,
    level: float,
    centroids: np.ndarray | None = None,
) -> IntervalCalibration | None:
    """
    Calibrate intervals for a held-out lag model.

    Args:
        bundle: The lag model fitted on data before `start` only.
        panel: Observed demand `[ts, pickup_cluster, request_count]`, running
            past `start`.
        start: First interval the model was not fitted on.
        days: Width of the window origins are spread over - the retraining
            cadence, so the errors are from a model as stale as a deployed one.
        n_origins: Forecast origins in that window.
        steps: Horizon forecast from each origin - the longest the API serves.
        level: Nominal coverage, e.g. 0.8 for the 10th to 90th percentile.
        centroids: Cluster centroids, if the model uses them.

    Returns:
        The calibration, or None when the panel runs too short past `start` for
        two full-horizon origins.
    """
    observed_end = pd.to_datetime(panel[TS_COL]).max()
    step = pd.tseries.frequencies.to_offset(bundle.freq)
    last_origin = min(
        start + pd.Timedelta(days=days) - steps * step,
        observed_end - (steps - 1) * step,
    )
    candidates = pd.date_range(start, last_origin, freq=bundle.freq)
    if len(candidates) < 2:
        logger.warning(
            "Prediction intervals not calibrated: the data runs only to %s, too "
            "short past %s for two %d-step backtests.", observed_end, start, steps,
        )
        return None

    picks = np.unique(np.linspace(0, len(candidates) - 1, n_origins).astype(int))
    origins = [candidates[i] for i in picks]
    errors = backtest_errors(bundle, panel, origins, steps=steps, centroids=centroids)

    holdout = None
    if len(origins) >= 4:
        half = len(origins) // 2
        early = errors[errors["origin"] < half]
        late = errors[errors["origin"] >= half]
        holdout = coverage(late, _calibration(early, level, bundle.freq, origins[:half]))

    calibration = _calibration(errors, level, bundle.freq, origins, holdout)
    logger.info(
        "Prediction intervals (%.0f%%) calibrated on %d recursive backtests, "
        "%s to %s: %d errors. Coverage on the later half, calibrated on the "
        "earlier half only: %s.",
        level * 100, len(origins), origins[0], origins[-1], len(errors),
        "n/a" if holdout is None else f"{holdout:.3f}",
    )
    return calibration
