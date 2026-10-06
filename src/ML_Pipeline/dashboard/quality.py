"""Data quality page: completeness and time-grid integrity, measured from the file."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(df: pd.DataFrame) -> None:
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

