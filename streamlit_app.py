"""
Bike-Taxi Demand Forecast - Analysis Dashboard.

Every figure on every page is computed from the aggregated demand grid produced
by the pipeline (`output/Data_Prepared.csv`). Nothing is hardcoded, sampled from
a random generator, or otherwise invented: if the data is not present, the app
says so and renders nothing rather than showing placeholder numbers.

Data governance
---------------
This dashboard is restricted by design to the *aggregated* demand grid
(timestamp x pickup_cluster -> request_count). That grid carries no personal
data. The upstream booking-level tables DO carry personal data - a pseudonymous
customer identifier (`number`) joined to pickup/drop coordinates at ~0.1 m
precision, from which home and workplace locations are trivially inferable.

`assert_no_personal_data()` enforces this: if the app is ever pointed at a
booking-level file, it refuses to render instead of leaking identifiers or
coordinates into a browser session. See docs/DATA_GOVERNANCE.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ML_Pipeline.config import latest_artifact, latest_version  # noqa: E402
from ML_Pipeline.evaluation import ModelEvaluator  # noqa: E402
from ML_Pipeline.forecast import PREDICTION_COL, backtest_recursive  # noqa: E402
from ML_Pipeline.monitoring import MissingFeaturesError, scoring_frame  # noqa: E402
from ML_Pipeline.serving import ServingState  # noqa: E402
from ML_Pipeline.utils import read_csv_any  # noqa: E402

# --------------------------------------------------------------------------
# Data governance: columns that must never reach this dashboard
# --------------------------------------------------------------------------

#: Booking-level columns carrying personal data. Presence of any of these means
#: the file is not an aggregated grid and must not be rendered.
RESTRICTED_COLUMNS: frozenset[str] = frozenset(
    {"number", "pick_lat", "pick_lng", "drop_lat", "drop_lng"}
)

#: Columns this dashboard needs in order to do anything at all.
REQUIRED_COLUMNS: frozenset[str] = frozenset({"ts", "pickup_cluster", "request_count"})

#: Where the pipeline writes. Overridable so a dashboard can point at another
#: run's output directory.
OUTPUT_DIR = os.environ.get("BIKETAXI_OUTPUT_DIR", "output")


def resolve_data_path(data_type: str = "prepared") -> str:
    """
    Path the dashboard should read for a given artefact type.

    Precedence: an explicit `BIKETAXI_PREPARED_DATA` override, then the newest
    versioned file in the output directory, then the unversioned legacy name.

    This function exists because the previous constant was the literal
    `"output/Data_Prepared.csv"`, which the pipeline has never written - it
    writes `Data_Prepared_<version>.csv`. The two never agreed, so a completely
    successful run still rendered "No prepared demand data found".
    """
    override = os.environ.get("BIKETAXI_PREPARED_DATA")
    if override and data_type == "prepared":
        return override

    found = latest_artifact(OUTPUT_DIR, data_type)
    if found is not None:
        return str(found)

    from ML_Pipeline.config import DATA_STEMS

    return str(Path(OUTPUT_DIR) / f"{DATA_STEMS.get(data_type, data_type)}.csv")


DEFAULT_DATA_PATH = resolve_data_path("prepared")

# --------------------------------------------------------------------------
# Palette (validated categorical/sequential tokens; see dataviz reference)
# --------------------------------------------------------------------------

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
INK_MUTED = "#898781"  # identical in light and dark by design
GRID_LIGHT = "#e1e0d9"
GRID_DARK = "#2c2c2a"
BASELINE_LIGHT = "#c3c2b7"
BASELINE_DARK = "#383835"

#: Reserved status hue, used only for "this loses to the baseline" and never
#: as a series colour. Always paired with a label, never meaning alone.
STATUS_BAD = "#c0372c"

# Sequential blue ramp, light -> dark (steps 100..700 of the reference ramp).
SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", SEQUENTIAL_BLUE_STEPS
)

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _is_dark_theme() -> bool:
    """Best-effort detection of the active Streamlit theme."""
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False


def _style_axes(ax: plt.Axes) -> plt.Axes:
    """Apply recessive chrome: hairline grid, muted ticks, no top/right spines."""
    dark = _is_dark_theme()
    grid = GRID_DARK if dark else GRID_LIGHT
    baseline = BASELINE_DARK if dark else BASELINE_LIGHT

    ax.figure.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.grid(True, color=grid, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(baseline)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    return ax


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


class DataGovernanceError(RuntimeError):
    """Raised when a file would expose personal data to the dashboard."""


def assert_no_personal_data(df: pd.DataFrame) -> None:
    """
    Refuse to proceed if the frame carries booking-level personal data.

    Raises:
        DataGovernanceError: if any restricted column is present.
    """
    present = sorted(RESTRICTED_COLUMNS.intersection(df.columns))
    if present:
        raise DataGovernanceError(
            "Refusing to display this file: it contains booking-level personal "
            f"data ({', '.join(present)}). This dashboard renders only the "
            "aggregated demand grid. Point it at output/Data_Prepared.csv."
        )


def _read_any_csv(source) -> pd.DataFrame:
    """Read a CSV that may or may not be gzip-compressed, path or upload."""
    return read_csv_any(source)


@st.cache_data(show_spinner="Loading demand grid...")
def load_prepared_data(path: str) -> pd.DataFrame:
    """Load and validate the aggregated demand grid from disk."""
    return _prepare(_read_any_csv(path))


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate governance + schema, then derive calendar features from `ts`."""
    assert_no_personal_data(df)

    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            f"File is missing required column(s): {', '.join(missing)}. "
            f"Expected the aggregated grid with {sorted(REQUIRED_COLUMNS)}."
        )

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])

    # Derive calendar features here rather than trusting upstream columns, so
    # the dashboard cannot silently disagree with the timestamps it displays.
    df["hour"] = df["ts"].dt.hour
    df["mins"] = df["ts"].dt.minute
    df["day"] = df["ts"].dt.day
    df["month"] = df["ts"].dt.month
    df["year"] = df["ts"].dt.year
    df["dayofweek"] = df["ts"].dt.dayofweek
    df["quarter"] = df["ts"].dt.quarter
    df["request_count"] = pd.to_numeric(df["request_count"], errors="coerce")
    return df


def render_empty_state(path: str) -> None:
    """Explain how to produce the data instead of inventing it."""
    st.warning(f"No prepared demand data found at `{path}`.")
    st.markdown(
        """
        This dashboard reports **only** on real pipeline output. Nothing is
        rendered until that output exists.

        **To generate it**

        ```bash
        python run_pipeline.py --stages data features
        ```

        That writes `output/Data_Prepared.csv` - the aggregated
        `timestamp x pickup_cluster -> request_count` grid this app reads.

        Alternatively, set `BIKETAXI_PREPARED_DATA` to an existing grid, or
        upload one in the sidebar.
        """
    )


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def page_overview(df: pd.DataFrame) -> None:
    st.header("Dataset overview")

    span_days = (df["ts"].max() - df["ts"].min()).days + 1
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Grid rows", f"{len(df):,}")
    col2.metric("Pickup clusters", f"{df['pickup_cluster'].nunique():,}")
    col3.metric("Days covered", f"{span_days:,}")
    col4.metric("Total requests", f"{df['request_count'].sum():,.0f}")

    st.caption(
        f"Observed period: {df['ts'].min():%Y-%m-%d %H:%M} to "
        f"{df['ts'].max():%Y-%m-%d %H:%M}"
    )

    st.subheader("Schema")
    schema = pd.DataFrame(
        {
            "Column": df.columns,
            "Dtype": [str(t) for t in df.dtypes],
            "Non-null": [int(df[c].notna().sum()) for c in df.columns],
            "Nulls": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.subheader("Demand distribution")
    counts = df["request_count"].dropna()
    desc = counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(
            desc.rename("request_count").to_frame().style.format("{:.3f}"),
            use_container_width=True,
        )
    with right:
        zero_share = float((counts == 0).mean() * 100)
        st.metric("Intervals with zero demand", f"{zero_share:.1f}%")
        st.caption(
            "A high zero share is the defining property of this target and "
            "drives model choice - squared-error regression on a sparse count "
            "is a poor fit. See docs/MODEL_CARD.md."
        )
        fig, ax = plt.subplots(figsize=(8, 3.4))
        upper = int(np.nanpercentile(counts, 99.5)) if len(counts) else 1
        ax.hist(
            counts.clip(upper=upper),
            bins=range(0, max(upper, 1) + 2),
            color=SERIES_BLUE,
            edgecolor="none",
        )
        ax.set_xlabel("Requests per 30-min interval")
        ax.set_ylabel("Frequency")
        _style_axes(ax)
        st.pyplot(fig)
        plt.close(fig)


def page_quality(df: pd.DataFrame) -> None:
    st.header("Data quality")
    st.caption("All figures below are measured from the loaded file.")

    nulls = df.isna().sum()
    total_cells = int(df.size)
    total_nulls = int(nulls.sum())
    completeness = (1 - total_nulls / total_cells) * 100 if total_cells else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Cell completeness", f"{completeness:.4f}%")
    col2.metric("Null cells", f"{total_nulls:,}")
    col3.metric("Columns with nulls", f"{int((nulls > 0).sum())} of {df.shape[1]}")

    quality = pd.DataFrame(
        {
            "Column": nulls.index,
            "Nulls": nulls.to_numpy(),
            "Null %": (nulls / max(len(df), 1) * 100).to_numpy().round(4),
            "Dtype": [str(t) for t in df.dtypes],
        }
    )
    st.dataframe(quality, use_container_width=True, hide_index=True)

    st.subheader("Time-grid integrity")
    n_clusters = df["pickup_cluster"].nunique()
    stamps = df["ts"].drop_duplicates().sort_values()
    expected = 0
    if len(stamps) > 1:
        step = stamps.diff().dropna().mode()
        if len(step):
            expected = int((stamps.max() - stamps.min()) / step.iloc[0]) + 1

    dup = int(df.duplicated(subset=["ts", "pickup_cluster"]).sum())
    observed_stamps = len(stamps)
    expected_rows = expected * n_clusters

    grid = pd.DataFrame(
        {
            "Check": [
                "Distinct timestamps observed",
                "Timestamps expected at modal interval",
                "Missing timestamps",
                "Duplicate (ts, cluster) pairs",
                "Rows observed",
                "Rows expected (timestamps x clusters)",
            ],
            "Value": [
                f"{observed_stamps:,}",
                f"{expected:,}" if expected else "n/a",
                f"{max(expected - observed_stamps, 0):,}" if expected else "n/a",
                f"{dup:,}",
                f"{len(df):,}",
                f"{expected_rows:,}" if expected else "n/a",
            ],
        }
    )
    st.dataframe(grid, use_container_width=True, hide_index=True)

    if dup:
        st.error(f"{dup:,} duplicate (timestamp, cluster) pairs found.")
    elif expected and expected_rows != len(df):
        st.warning(
            f"Grid is not rectangular: {len(df):,} rows vs {expected_rows:,} "
            "expected. Some cluster/interval combinations are absent."
        )
    else:
        st.success("Grid is complete and free of duplicate keys.")


def page_demand_patterns(df: pd.DataFrame) -> None:
    st.header("Demand patterns")
    st.caption("Aggregated from the loaded grid.")

    # Job: trend across the day, one series -> line, no legend needed.
    hourly = df.groupby("hour", as_index=False)["request_count"].mean()
    st.subheader("Mean requests by hour of day")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        hourly["hour"], hourly["request_count"],
        color=SERIES_BLUE, linewidth=2, marker="o", markersize=5,
    )
    ax.fill_between(hourly["hour"], hourly["request_count"], color=SERIES_BLUE, alpha=0.12)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean requests per interval")
    ax.set_xticks(range(0, 24, 2))
    _style_axes(ax)
    st.pyplot(fig)
    plt.close(fig)

    # Job: compare magnitude across a small ordered set -> bar, sequential hue.
    st.subheader("Mean requests by day of week")
    dow = df.groupby("dayofweek", as_index=False)["request_count"].mean()
    dow["label"] = dow["dayofweek"].map(lambda d: DAY_LABELS[int(d)])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    norm = (dow["request_count"] - dow["request_count"].min()) / (
        np.ptp(dow["request_count"]) or 1
    )
    ax.bar(
        dow["label"], dow["request_count"],
        color=[SEQUENTIAL_BLUE(0.35 + 0.5 * v) for v in norm], width=0.62,
    )
    ax.set_ylabel("Mean requests per interval")
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    st.pyplot(fig)
    plt.close(fig)

    # Job: magnitude over a 2-D grid -> heatmap, sequential single hue.
    st.subheader("Demand by hour and day of week")
    pivot = (
        df.pivot_table(
            index="dayofweek", columns="hour", values="request_count", aggfunc="mean"
        )
        .reindex(index=range(7), columns=range(24))
    )
    fig, ax = plt.subplots(figsize=(12, 3.6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=SEQUENTIAL_BLUE, origin="upper")
    ax.set_yticks(range(7), DAY_LABELS)
    ax.set_xticks(range(0, 24, 2), [str(h) for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Mean requests", color=INK_MUTED, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    cbar.outline.set_visible(False)
    _style_axes(ax)
    ax.grid(False)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Total requests over time")
    period = df.groupby(df["ts"].dt.to_period("M"))["request_count"].sum()
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(
        [p.to_timestamp() for p in period.index], period.to_numpy(),
        color=SERIES_BLUE, linewidth=2, marker="s", markersize=6,
    )
    ax.set_ylabel("Total requests")
    _style_axes(ax)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)


def page_clusters(df: pd.DataFrame) -> None:
    st.header("Geographic clusters")

    by_cluster = (
        df.groupby("pickup_cluster", as_index=False)["request_count"]
        .agg(total="sum", mean="mean")
        .sort_values("total", ascending=False)
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", f"{len(by_cluster):,}")
    col2.metric("Busiest cluster", f"#{int(by_cluster.iloc[0]['pickup_cluster'])}")
    share = by_cluster["total"].head(10).sum() / max(by_cluster["total"].sum(), 1) * 100
    col3.metric("Top-10 share of demand", f"{share:.1f}%")

    st.caption(
        "Cluster demand is strongly skewed; this concentration is why a raw "
        "integer cluster id is a poor model feature (see docs/MODEL_CARD.md)."
    )

    # Job: compare magnitude across many categories -> sorted bar, sequential.
    st.subheader("Total requests by cluster")
    top_n = st.slider("Clusters shown", 10, max(len(by_cluster), 10), min(30, len(by_cluster)))
    shown = by_cluster.head(top_n)
    fig, ax = plt.subplots(figsize=(12, max(3.2, 0.22 * len(shown))))
    norm = shown["total"] / max(shown["total"].max(), 1)
    ax.barh(
        [f"#{int(c)}" for c in shown["pickup_cluster"]], shown["total"],
        color=[SEQUENTIAL_BLUE(0.30 + 0.6 * v) for v in norm], height=0.7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Total requests")
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        by_cluster.rename(
            columns={
                "pickup_cluster": "Cluster",
                "total": "Total requests",
                "mean": "Mean per interval",
            }
        ).style.format({"Total requests": "{:,.0f}", "Mean per interval": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )


def page_forecasts() -> None:
    st.header("Forecasts")

    candidates = {
        "With lag features": resolve_data_path("with_lag"),
        "Without lag features": resolve_data_path("without_lag"),
    }
    available = {k: v for k, v in candidates.items() if Path(v).exists()}

    if not available:
        st.warning("No forecast output found in `output/`.")
        st.markdown(
            "Generate it with:\n\n```bash\npython run_pipeline.py --stages predict\n```"
        )
        return

    choice = st.selectbox("Forecast file", list(available))
    fc = _read_any_csv(available[choice])
    assert_no_personal_data(fc)
    fc["ts"] = pd.to_datetime(fc["ts"], errors="coerce")

    value_col = "request_count_pred" if "request_count_pred" in fc else "request_count"

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast rows", f"{len(fc):,}")
    col2.metric("Clusters", f"{fc['pickup_cluster'].nunique():,}")
    col3.metric("Total forecast demand", f"{fc[value_col].sum():,.0f}")
    st.caption(
        f"Horizon: {fc['ts'].min():%Y-%m-%d %H:%M} to {fc['ts'].max():%Y-%m-%d %H:%M}"
    )

    totals = fc.groupby("ts", as_index=False)[value_col].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(totals["ts"], totals[value_col], color=SERIES_ORANGE, linewidth=2)
    ax.set_ylabel("Forecast requests (all clusters)")
    _style_axes(ax)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(fc.head(500), use_container_width=True, hide_index=True)
    st.caption("First 500 rows.")


# --------------------------------------------------------------------------
# Model performance
# --------------------------------------------------------------------------

#: One week of 30-minute intervals: the seasonal period for the naive baseline.
SEASON_INTERVALS = 336


@st.cache_resource(show_spinner="Loading the promoted model...")
def load_serving_state() -> ServingState:
    """
    The model the API would serve, and the history its lags read from.

    Shared with `ML_Pipeline.api` rather than reimplemented, so the dashboard
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
    _style_axes(ax)
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


def page_model_performance() -> None:
    st.header("Model performance")

    state = load_serving_state()
    if not state.ready:
        st.warning("No model is promoted to production, so there is nothing to score.")
        st.markdown(
            """
            Train and promote one:

            ```bash
            python run_pipeline.py --raw-data data/raw_data.csv --promote
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
    _style_axes(ax)
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


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Bike Taxi Demand Forecast", layout="wide")
    st.title("Bike Taxi Demand Forecast")
    st.markdown(
        "Analysis of aggregated ride-request demand. Every figure is computed "
        "from pipeline output - no illustrative or placeholder values."
    )

    st.sidebar.header("Data source")
    path = st.sidebar.text_input("Prepared data path", value=DEFAULT_DATA_PATH)
    upload = st.sidebar.file_uploader("or upload a demand grid (CSV)", type=["csv", "gz"])

    try:
        if upload is not None:
            df = _prepare(_read_any_csv(upload))
            st.sidebar.success("Using uploaded file.")
        elif Path(path).exists():
            df = load_prepared_data(path)
            st.sidebar.success(f"Loaded `{Path(path).name}`")
            # Which run produced this, so a stale output directory is visible
            # rather than silently assumed to be the latest.
            version = latest_version(OUTPUT_DIR)
            if version:
                st.sidebar.caption(f"Pipeline run `{version}`")
        else:
            render_empty_state(path)
            st.stop()
    except DataGovernanceError as exc:
        st.error(str(exc))
        st.stop()
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read that file: {exc}")
        st.stop()

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Model performance", "Data quality", "Demand patterns",
         "Clusters", "Forecasts"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "**Data governance** - this dashboard reads only aggregated demand "
        "counts. Customer identifiers and raw coordinates are blocked at load "
        "time and never rendered."
    )

    if page == "Overview":
        page_overview(df)
    elif page == "Model performance":
        page_model_performance()
    elif page == "Data quality":
        page_quality(df)
    elif page == "Demand patterns":
        page_demand_patterns(df)
    elif page == "Clusters":
        page_clusters(df)
    else:
        page_forecasts()

    st.divider()
    st.caption(
        f"Source: `{path if upload is None else upload.name}` - "
        f"{len(df):,} rows - {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}"
    )


if __name__ == "__main__":
    main()
