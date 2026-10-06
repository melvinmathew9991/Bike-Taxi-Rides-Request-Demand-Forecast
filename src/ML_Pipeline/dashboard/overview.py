"""Overview page: size, span, schema and the shape of the target."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ML_Pipeline.dashboard.common import SERIES_BLUE, style_axes


def render(df: pd.DataFrame) -> None:
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
        style_axes(ax)
        st.pyplot(fig)
        plt.close(fig)

