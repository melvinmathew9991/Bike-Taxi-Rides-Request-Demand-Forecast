"""Forecasts page: the forecast files the pipeline's last run wrote."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ML_Pipeline.dashboard.common import SERIES_ORANGE, resolve_data_path, style_axes
from ML_Pipeline.governance import assert_no_personal_data
from ML_Pipeline.utils import read_csv_any


def render(df: pd.DataFrame | None = None) -> None:
    st.header("Forecasts")

    path = Path(resolve_data_path("with_lag"))
    if not path.exists():
        st.warning("No forecast output found in `output/`.")
        st.markdown(
            "Generate it with:\n\n```bash\nbiketaxi run --stages predict\n```"
        )
        return

    fc = read_csv_any(path)
    assert_no_personal_data(fc)
    fc["ts"] = pd.to_datetime(fc["ts"], errors="coerce")

    value_col = "request_count_pred" if "request_count_pred" in fc else "request_count"

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast rows", f"{len(fc):,}")
    col2.metric("Clusters", f"{fc['pickup_cluster'].nunique():,}")
    col3.metric("Total forecast demand", f"{fc[value_col].sum():,.0f}")
    if fc.empty:
        st.warning("The forecast file has no rows: no cluster had enough history.")
    else:
        st.caption(
            f"Horizon: {fc['ts'].min():%Y-%m-%d %H:%M} to {fc['ts'].max():%Y-%m-%d %H:%M}"
        )

        totals = fc.groupby("ts", as_index=False)[value_col].sum()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(totals["ts"], totals[value_col], color=SERIES_ORANGE, linewidth=2)
        ax.set_ylabel("Forecast requests (all clusters)")
        style_axes(ax)
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(fc.head(500), use_container_width=True, hide_index=True)
        st.caption("First 500 rows.")

    # The lag-free model is not offered as an alternative forecast: it loses to
    # seasonal-naive by 75%. It forecasts only clusters too new to have a week
    # of history, and those are shown here, separately, when there are any.
    cold_path = Path(resolve_data_path("without_lag"))
    if not cold_path.exists():
        return
    cold = read_csv_any(cold_path)
    if cold.empty:
        return
    assert_no_personal_data(cold)
    st.subheader("Cold-start clusters")
    st.warning(
        f"{cold['pickup_cluster'].nunique()} cluster(s) had less than a week of "
        "history, so they are forecast by the lag-free model, which loses to a "
        "same-time-last-week baseline by 75%. Treat these as rough: "
        + ", ".join(str(c) for c in sorted(cold["pickup_cluster"].unique()))
    )
    st.dataframe(cold.head(500), use_container_width=True, hide_index=True)
