"""Demand patterns page: by hour, by day of week, and over time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ML_Pipeline.dashboard.common import (
    DAY_LABELS,
    INK_MUTED,
    SEQUENTIAL_BLUE,
    SERIES_BLUE,
    style_axes,
)


def render(df: pd.DataFrame) -> None:
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
    style_axes(ax)
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
    style_axes(ax)
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
    # The same object as `cbar.outline`. matplotlib 3.10's stubs, the last
    # release for Python 3.10, type `outline` as the whole spine collection, so
    # mypy rejects calling a method on it; the mapping entry is typed correctly.
    cbar.ax.spines["outline"].set_visible(False)
    style_axes(ax)
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
    style_axes(ax)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

