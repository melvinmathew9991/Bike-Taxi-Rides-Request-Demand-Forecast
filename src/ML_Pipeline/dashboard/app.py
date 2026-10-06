"""
The dashboard's layout: data source, navigation, and which page renders.

Each page is a module in this package with a `render(df)` function. Adding a
page is a new module and one line in `PAGES`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st

from ML_Pipeline.artifacts import latest_version
from ML_Pipeline.dashboard import (
    clusters,
    forecasts,
    overview,
    patterns,
    performance,
    quality,
)
from ML_Pipeline.dashboard.common import (
    load_prepared_data,
    output_dir,
    prepare,
    render_empty_state,
    resolve_data_path,
)
from ML_Pipeline.governance import DataGovernanceError
from ML_Pipeline.utils import read_csv_any

#: Sidebar label -> page renderer, in navigation order.
PAGES: dict[str, Callable[[pd.DataFrame], None]] = {
    "Overview": overview.render,
    "Model performance": performance.render,
    "Data quality": quality.render,
    "Demand patterns": patterns.render,
    "Clusters": clusters.render,
    "Forecasts": forecasts.render,
}


def main() -> None:
    st.set_page_config(page_title="Bike Taxi Demand Forecast", layout="wide")
    st.title("Bike Taxi Demand Forecast")
    st.markdown(
        "Analysis of aggregated ride-request demand. Every figure is computed "
        "from pipeline output - no illustrative or placeholder values."
    )

    st.sidebar.header("Data source")
    path = st.sidebar.text_input("Prepared data path", value=resolve_data_path("prepared"))
    upload = st.sidebar.file_uploader("or upload a demand grid (CSV)", type=["csv", "gz"])

    try:
        if upload is not None:
            df = prepare(read_csv_any(upload))
            st.sidebar.success("Using uploaded file.")
        elif Path(path).exists():
            df = load_prepared_data(path)
            st.sidebar.success(f"Loaded `{Path(path).name}`")
            # Which run produced this, so a stale output directory is visible
            # rather than silently assumed to be the latest.
            version = latest_version(output_dir())
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
    page = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.caption(
        "**Data governance** - this dashboard reads only aggregated demand "
        "counts. Customer identifiers and raw coordinates are blocked at load "
        "time and never rendered."
    )

    PAGES[page](df)

    st.divider()
    st.caption(
        f"Source: `{path if upload is None else upload.name}` - "
        f"{len(df):,} rows - {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}"
    )
