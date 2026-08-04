"""Per-rider gaps between consecutive bookings."""

from __future__ import annotations

import pandas as pd


def shift_time(df: pd.DataFrame, rider_col: str = "number") -> pd.DataFrame:
    """
    Add the time since each rider's previous booking.

    A rider's first booking has no predecessor. The previous implementation
    filled the missing prior timestamp with `0`, which made the resulting gap
    roughly 27 million minutes - so first bookings sailed through the downstream
    "at least 8 minutes apart" filter. That behaviour is correct (a first booking
    is real demand), but it depended on an accidental sentinel rather than an
    expressed intent, and it would have broken silently had the fill value
    changed.

    `is_first_booking` now states it explicitly, so the cleaning rule can keep
    first bookings on purpose. The numeric fill is retained for compatibility.

    Returns:
        Copy with `shift_booking_ts`, `booking_time_diff_hr`,
        `booking_time_diff_min` and `is_first_booking`.
    """
    if "booking_timestamp" not in df.columns:
        raise KeyError("shift_time requires a 'booking_timestamp' column")

    out = df.copy()
    previous = out.groupby(rider_col)["booking_timestamp"].shift(1)
    out["is_first_booking"] = previous.isna()
    out["shift_booking_ts"] = previous.fillna(0).astype("int64")

    elapsed = out["booking_timestamp"] - out["shift_booking_ts"]
    out["booking_time_diff_hr"] = elapsed // 3600
    out["booking_time_diff_min"] = elapsed // 60
    return out
