"""
What every dashboard page shares: where the data is, how it is loaded and
checked, and the chart styling.

The output directory is read from `BIKETAXI_OUTPUT_DIR` on every call rather
than once at import. As a package module this is imported once per process and
then cached, so a value captured at import would outlive the run that set it.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from ML_Pipeline.artifacts import DATA_STEMS, latest_artifact
from ML_Pipeline.governance import assert_no_personal_data
from ML_Pipeline.utils import read_csv_any

#: Columns this dashboard needs in order to do anything at all.
REQUIRED_COLUMNS: frozenset[str] = frozenset({"ts", "pickup_cluster", "request_count"})


def output_dir() -> str:
    """Where the pipeline writes. Overridable to point at another run's output."""
    return os.environ.get("BIKETAXI_OUTPUT_DIR", "output")


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

    found = latest_artifact(output_dir(), data_type)
    if found is not None:
        return str(found)
    return str(Path(output_dir()) / f"{DATA_STEMS.get(data_type, data_type)}.csv")


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


def is_dark_theme() -> bool:
    """Best-effort detection of the active Streamlit theme."""
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False


def style_axes(ax: plt.Axes) -> plt.Axes:
    """Apply recessive chrome: hairline grid, muted ticks, no top/right spines."""
    dark = is_dark_theme()
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


@st.cache_data(show_spinner="Loading demand grid...")
def load_prepared_data(path: str) -> pd.DataFrame:
    """Load and validate the aggregated demand grid from disk."""
    return prepare(read_csv_any(path))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
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
        biketaxi run --stages data features
        ```

        That writes `output/Data_Prepared.csv` - the aggregated
        `timestamp x pickup_cluster -> request_count` grid this app reads.

        Alternatively, set `BIKETAXI_PREPARED_DATA` to an existing grid, or
        upload one in the sidebar.
        """
    )

