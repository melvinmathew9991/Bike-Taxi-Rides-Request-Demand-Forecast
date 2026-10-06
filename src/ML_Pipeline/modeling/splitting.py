"""
Chronological splitting for time-series panels.

The previous split assigned day-of-month <= 23 to train and > 23 to test, across
all twelve months. That interleaves test weeks throughout the training period, so
for any test point the model has already seen data from later dates. It measures
interpolation, not forecasting, and cannot surface drift - and because
`day`-of-month was also a model feature, the split boundary was partially
learnable.

A forecasting model must be evaluated the way it will be used: fit on the past,
scored on a future it has not seen. Every split here cuts on a single point in
time, so no training row postdates any test row.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TS_COL = "ts"


class TimeSplit(NamedTuple):
    """A chronological train/test partition and the timestamp dividing them."""

    train: pd.DataFrame
    test: pd.DataFrame
    split_at: pd.Timestamp


def chronological_split(
    df: pd.DataFrame, test_fraction: float = 0.2, ts_col: str = TS_COL
) -> TimeSplit:
    """
    Split a panel at a single point in time.

    The cut is placed on the *timeline* (distinct timestamps), not on row count,
    so every cluster contributes the same period to each side and the partition
    stays balanced across clusters.

    Args:
        df: Panel carrying `ts_col`.
        test_fraction: Share of the distinct timestamps held out at the end.

    Returns:
        `TimeSplit(train, test, split_at)`, where every `train` timestamp is
        strictly before every `test` timestamp.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    working = df.copy()
    working[ts_col] = pd.to_datetime(working[ts_col])
    stamps = np.sort(working[ts_col].unique())
    if len(stamps) < 2:
        raise ValueError(
            f"Need at least 2 distinct timestamps to split, got {len(stamps)}"
        )

    cut_index = int(round(len(stamps) * (1.0 - test_fraction)))
    cut_index = min(max(cut_index, 1), len(stamps) - 1)
    split_at = pd.Timestamp(stamps[cut_index])

    train = working[working[ts_col] < split_at].reset_index(drop=True)
    test = working[working[ts_col] >= split_at].reset_index(drop=True)

    logger.info(
        "Chronological split at %s: train %d rows (%s to %s), test %d rows (%s to %s)",
        split_at, len(train), train[ts_col].min(), train[ts_col].max(),
        len(test), test[ts_col].min(), test[ts_col].max(),
    )
    return TimeSplit(train=train, test=test, split_at=split_at)


def train_validation_split(
    df: pd.DataFrame, validation_fraction: float = 0.1, ts_col: str = TS_COL
) -> TimeSplit:
    """
    Carve a chronological validation tail off a training set.

    Used for early stopping. It must also be chronological: a random validation
    fold would leak future intervals into the stopping decision and choose a tree
    count that is optimistic for genuine forecasting.
    """
    if validation_fraction <= 0:
        return TimeSplit(train=df, test=df.iloc[0:0], split_at=pd.Timestamp.max)
    return chronological_split(df, test_fraction=validation_fraction, ts_col=ts_col)
