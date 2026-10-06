"""Clusters page: how demand is distributed across geographic clusters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ML_Pipeline.dashboard.common import SEQUENTIAL_BLUE, style_axes


def render(df: pd.DataFrame) -> None:
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
    style_axes(ax)
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

