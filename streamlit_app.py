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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

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

DEFAULT_DATA_PATH = os.environ.get(
    "BIKETAXI_PREPARED_DATA", "output/Data_Prepared.csv"
)

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
    """Read a CSV that may or may not be gzip-compressed."""
    try:
        return pd.read_csv(source, compression="gzip", low_memory=False)
    except (OSError, EOFError, ValueError):
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, compression=None, low_memory=False)


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
        "With lag features": "output/data_with_lag.csv",
        "Without lag features": "output/data_without_lag.csv",
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
            st.sidebar.success(f"Loaded `{path}`")
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
        ["Overview", "Data quality", "Demand patterns", "Clusters", "Forecasts"],
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
