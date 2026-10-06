"""
Model performance page: is the promoted model fit to serve?

Gate verdict and data lag, a recursive backtest of the last closed horizon,
accuracy against the baselines over a wider window, and error per cluster. It
scores only demand the model was not fitted on.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ML_Pipeline.dashboard.common import (
    INK_MUTED,
    SEQUENTIAL_BLUE,
    SERIES_BLUE,
    SERIES_ORANGE,
    STATUS_BAD,
    style_axes,
)
from ML_Pipeline.modeling.evaluation import ModelEvaluator
from ML_Pipeline.modeling.forecast import PREDICTION_COL, backtest_recursive
from ML_Pipeline.serving.monitoring import MissingFeaturesError, scoring_frame
from ML_Pipeline.serving.state import ServingState

#: One week of 30-minute intervals: the seasonal period for the naive baseline.
SEASON_INTERVALS = 336


@st.cache_resource(show_spinner="Loading the promoted model...")
def load_serving_state() -> ServingState:
    """
    The model the API would serve, and the history its lags read from.

    Shared with `ML_Pipeline.serving.api` rather than reimplemented, so the dashboard
    cannot disagree with production about which model is live.
    """
    return ServingState()


@st.cache_data(show_spinner="Backtesting the last closed horizon...")
def backtest_last_horizon(model_name: str, steps: int):
    """
    Forecast the most recent `steps` intervals and pair with what happened.

    The forecast files the pipeline writes cover a horizon *after* the observed
    data ends, so they carry no actuals to compare against - correct for serving,
    useless for evaluation. The honest view of accuracy is a backtest: hold out
    the last closed horizon and let the model forecast it recursively, compounding
    its own errors as it does in production.

    `model_name` is in the signature only so Streamlit invalidates this when the
    promoted model changes.
    """
    state = load_serving_state()
    if not state.ready:
        return None
    try:
        return backtest_recursive(
            state.bundle, state.history, horizon_steps=steps,
            centroids=state.centroids,
        )
    except ValueError:
        return None


@st.cache_data(show_spinner="Scoring the recent window...")
def evaluate_recent_window(model_name: str, weeks: int):
    """
    Score one-step predictions over the last `weeks` weeks of observed demand.

    This exists because a seasonal-naive baseline is the value 336 intervals
    earlier, so MASE can only be computed over a frame that contains those
    intervals. The recursive backtest below is at most 96 intervals wide, which is
    why its baseline column came out empty - the comparison needs a wider window,
    not a different metric.

    One step ahead means true observed lags, which is the optimistic mode. It is
    also exactly how the deploy gate and `per_cluster_error` are measured during
    training, so the numbers here are comparable to the verdict in the registry.

    Returns:
        `(scored_frame, predictions)`, or `(None, None)` if the model cannot be
        scored on this history.
    """
    state = load_serving_state()
    if not state.ready:
        return None, None

    # Shared with the scheduled health check, so the two cannot disagree about
    # what the model is scored on. A feature the grid cannot supply is a real
    # train/serve mismatch, and is named rather than swallowed.
    try:
        frame = scoring_frame(state)
    except MissingFeaturesError as exc:
        st.error(str(exc))
        return None, None

    cutoff = frame["ts"].max() - pd.Timedelta(weeks=weeks)
    window = frame[frame["ts"] >= cutoff]
    # Only demand the model has not been fitted on. A model refit on all data
    # has seen every row of this window, and scoring it there is in-sample.
    if state.data_through is not None:
        window = window[window["ts"] > state.data_through]
    window = window.reset_index(drop=True)
    if len(window) == 0:
        return None, None
    return window, state.bundle.predict(window)


def _gate_tiles(state: ServingState) -> None:
    """Headline: is this model fit to serve, and how old is it?"""
    metrics = state.info.get("metrics", {})
    beats = metrics.get("beats_seasonal_naive")
    mase = metrics.get("baseline_mase")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Deploy gate",
        "not measured" if beats is None else ("PASS" if bool(beats) else "FAIL"),
    )
    col2.metric("MASE vs naive", f"{mase:.3f}" if mase is not None else "n/a")
    # Data lag, not wall-clock age. The model promoted on 2026-10-02 showed
    # "0 days" here while its training data ended ten weeks before the history
    # did - and that gap, not the calendar, is what made it lose on cluster 30.
    lag, age = state.data_lag_days, state.age_days
    col3.metric(
        "Data lag",
        f"{lag:.0f} days" if lag is not None else "unknown",
        help="Days of observed demand the model was not fitted on. Staleness is "
             "measured from this; the training run itself was "
             + (f"{age:.0f} days ago." if age is not None else "at an unknown time."),
    )
    losing = metrics.get("clusters_losing_to_naive")
    col4.metric(
        "Clusters losing to naive",
        f"{int(losing)}" if losing is not None else "n/a",
    )

    if beats is not None and not bool(beats):
        st.error(
            "This model does not beat a seasonal-naive baseline. It should not be "
            "serving: the baseline is free, interpretable and needs no retraining."
        )
    staleness = state.staleness_warning()
    if staleness:
        st.warning(
            f"{staleness} The measured failure point is six weeks in the worst case."
        )


def _render_backtest(state: ServingState, lag_days: float | None) -> None:
    """Recursive backtest of the last closed horizon, if the model has not seen it."""
    steps = st.slider(
        "Backtest horizon (30-minute intervals)", 24, 96, 48, step=24,
        help="Recursive backtest over the last closed horizon. Capped at 96 - two "
             "days - because past that the model's measured advantage over the "
             "baseline runs out.",
    )
    if lag_days is not None and steps / 48 > lag_days:
        st.info(
            f"A {steps}-interval backtest reaches back before "
            f"{state.data_through:%Y-%m-%d %H:%M}, into data this model was fitted "
            "on. Shorten the horizon, or wait for more new demand."
        )
        return
    backtest = backtest_last_horizon(state.model_name or "", steps)
    if backtest is None or len(backtest) == 0:
        st.error("Could not backtest: not enough history for this model's lags.")
        return

    actual = backtest["request_count"].to_numpy(dtype="float64")
    predicted = backtest[PREDICTION_COL].to_numpy(dtype="float64")

    # Job: two series over time -> line chart. Two series, so a legend is present.
    st.subheader("Forecast against actual, recursively")
    st.caption(
        "The last closed horizon, forecast from the model's own predictions - "
        "the mode the pipeline serves. Accuracy against the baseline is scored "
        "over a wider window further down, because a same-time-last-week "
        "baseline needs a week of history to exist."
    )
    totals = (
        backtest.groupby("ts", as_index=False)[["request_count", PREDICTION_COL]]
        .sum()
        .sort_values("ts")
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(totals["ts"], totals["request_count"], color=SERIES_BLUE,
            linewidth=2, label="Actual")
    ax.plot(totals["ts"], totals[PREDICTION_COL], color=SERIES_ORANGE,
            linewidth=2, linestyle="--", label="Forecast")
    ax.set_ylabel("Requests, all clusters")
    style_axes(ax)
    legend = ax.legend(frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_MUTED)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    level_ratio = predicted.mean() / actual.mean() if actual.mean() else float("nan")
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual over the horizon", f"{actual.sum():,.0f}")
    col2.metric("Forecast over the horizon", f"{predicted.sum():,.0f}")
    col3.metric("Level ratio", f"{level_ratio:.2f}")
    st.caption(
        "Level ratio is mean forecast over mean actual. It degrades earliest of "
        "any metric as a model goes stale - 0.92 by week four, while error still "
        "looks fine - which is why it is the thing to watch."
    )


def render(df: pd.DataFrame | None = None) -> None:
    st.header("Model performance")

    state = load_serving_state()
    if not state.ready:
        st.warning("No model is promoted to production, so there is nothing to score.")
        st.markdown(
            """
            Train and promote one:

            ```bash
            biketaxi run --raw-data data/raw_data.csv --promote
            ```

            Promotion refuses a model that failed its deploy gate, or that carries
            no verdict at all, so whatever reaches this page has evidence behind it.
            """
        )
        return

    st.caption(
        f"Serving `{state.model_name}` - {len(state.bundle.feature_names)} features "
        f"- lags {list(state.bundle.lags)} - history to "
        f"{state.history_ends_at:%Y-%m-%d %H:%M}"
    )
    _gate_tiles(state)

    lag = state.data_lag_days
    if lag is not None and lag < 7:
        st.info(
            f"The serving model was fitted on all observed demand through "
            f"{state.data_through:%Y-%m-%d %H:%M}, so only {lag:.1f} days of "
            "demand exist that it has not seen. Scoring it on data it was trained "
            "on would flatter it, so the backtest and baseline comparison appear "
            "once a week of new demand has arrived. Until then, the deploy gate "
            "above is the out-of-sample measurement: it was taken on a held-out "
            "fit of the same model before the final refit."
        )
        return

    _render_backtest(state, lag)

    # ---------------- accuracy against the baseline ----------------
    #
    # Measured over a wider window than the backtest above, because a
    # same-time-last-week baseline needs a week of history inside the frame to
    # exist at all. One step ahead, matching how the deploy gate is measured.
    st.subheader("Against the baselines it must beat")
    weeks = st.select_slider(
        "Scoring window", options=[4, 6, 8, 12], value=8,
        help="Weeks of observed demand, scored one step ahead. At least 1 week is "
             "needed for a same-time-last-week baseline to exist; 4+ makes it "
             "stable.",
    )
    window, window_pred = evaluate_recent_window(state.model_name or "", weeks)
    if window is None:
        st.error("Could not score the recent window with this model's features.")
        return
    # The window stops at the model's training data, so it can be shorter than
    # the slider says. Name the span actually scored.
    span_days = (window["ts"].max() - window["ts"].min()) / pd.Timedelta(days=1)
    over = (
        f"the last {weeks} weeks" if span_days >= weeks * 7 - 1
        else f"the {span_days:.0f} days since the model's training data ends"
    )

    comparison = ModelEvaluator.compare_to_baselines(
        window, window_pred, season_length=SEASON_INTERVALS
    )
    model_mase = float(comparison.loc["model", "mase"])
    beats = (
        "seasonal_naive" in comparison.index
        and comparison.loc["model", "rmse"] < comparison.loc["seasonal_naive", "rmse"]
    )
    if "seasonal_naive" in comparison.index:
        if beats:
            st.success(
                f"Beats the baseline over {over}: MASE "
                f"{model_mase:.3f}."
            )
        else:
            st.error(
                f"Loses to a seasonal-naive baseline over {over} "
                f"(MASE {model_mase:.3f}). Ship the baseline instead until it does "
                "not."
            )
    st.dataframe(
        comparison.rename(
            index={
                "model": "This model",
                "seasonal_naive": "Seasonal naive (same time last week)",
                "cluster_mean": "Cluster historical mean",
            }
        ).style.format(
            {"rmse": "{:.3f}", "mae": "{:.3f}", "mase": "{:.3f}", "n": "{:,.0f}"}
        ),
        use_container_width=True,
    )
    st.caption(
        f"{len(window):,} interval-cluster rows from "
        f"{window['ts'].min():%Y-%m-%d} to {window['ts'].max():%Y-%m-%d}. "
        "MASE below 1 beats the baseline; a model that cannot should not ship."
    )

    # ---------------- error per cluster ----------------
    #
    # Job: magnitude across many categories -> sorted bar, one sequential hue, with
    # a reference line where the meaning changes.
    st.subheader("Error by cluster")
    st.caption(
        "A single global error hides which areas get under-served. This model "
        "decides where supply goes, so its errors are not evenly consequential."
    )
    per_cluster = ModelEvaluator.per_cluster_error(
        window, window_pred, season_length=SEASON_INTERVALS
    )
    scored = per_cluster.dropna(subset=["mase"])
    if len(scored) == 0:
        st.info("No cluster has a week of history in this window.")
        return

    losing = scored[scored["mase"] >= 1.0]
    if len(losing):
        st.error(
            f"{len(losing)} of {len(scored)} clusters lose to the baseline: "
            + ", ".join(f"#{int(c)}" for c in losing["pickup_cluster"].head(20))
            + ". Those areas would be systematically under-served."
        )
    else:
        st.success(f"All {len(scored)} clusters beat the seasonal-naive baseline.")

    shown = scored.head(25)
    fig, ax = plt.subplots(figsize=(12, max(3.2, 0.26 * len(shown))))
    span = max(float(shown["mase"].max()), 1e-9)
    colors = [
        STATUS_BAD if m >= 1.0 else SEQUENTIAL_BLUE(0.30 + 0.6 * (m / span))
        for m in shown["mase"]
    ]
    ax.barh([f"#{int(c)}" for c in shown["pickup_cluster"]], shown["mase"],
            color=colors, height=0.7)
    ax.axvline(1.0, color=STATUS_BAD, linestyle="--", linewidth=1.2)
    ax.annotate("1.0 = baseline", xy=(1.0, len(shown) - 0.5), xytext=(5, 0),
                textcoords="offset points", color=STATUS_BAD, fontsize=9,
                va="center")
    ax.invert_yaxis()
    ax.set_xlabel("MASE (lower is better)")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Worst {len(shown)} of {len(scored)} clusters, worst first.")

    st.dataframe(
        scored.rename(columns={
            "pickup_cluster": "Cluster", "n": "Intervals",
            "mean_actual": "Mean actual", "mean_pred": "Mean forecast",
            "level_ratio": "Level ratio", "rmse": "RMSE", "mae": "MAE",
            "mase": "MASE",
        }).style.format({
            "Intervals": "{:,.0f}", "Mean actual": "{:.2f}",
            "Mean forecast": "{:.2f}", "Level ratio": "{:.2f}",
            "RMSE": "{:.3f}", "MAE": "{:.3f}", "MASE": "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )

