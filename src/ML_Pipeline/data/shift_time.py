"""Per-rider gaps between consecutive bookings, overall and per pickup pin."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def shift_time(
    df: pd.DataFrame,
    rider_col: str = "number",
    pin_cols: tuple[str, ...] = ("pick_lat", "pick_lng"),
) -> pd.DataFrame:
    """
    Add the time since each rider's previous booking, overall and per pickup pin.

    A rider's first booking has no predecessor. The previous implementation
    filled the missing prior timestamp with `0`, which made the resulting gap
    roughly 27 million minutes - so first bookings sailed through the downstream
    "at least 8 minutes apart" filter. That behaviour is correct (a first booking
    is real demand), but it depended on an accidental sentinel rather than an
    expressed intent, and it would have broken silently had the fill value
    changed.

    `is_first_booking` now states it explicitly, so the cleaning rule can keep
    first bookings on purpose. The numeric fill is retained for compatibility.

    **Per-pin gaps.** `pin_time_diff_hr` is the gap to the same rider's previous
    booking *from the same pickup pin*, which is what a "rebooking the same pin"
    rule actually needs. Without it, `advanced_cleanup` Rule 1 combined
    `duplicated(..., keep=False)` - true for every row whose (rider, pin) recurs
    anywhere in the data - with the gap to the rider's previous booking from
    *any* pin, and so deleted 548,510 rows that no rebooking rule describes.

    Rows are expected in `(rider, timestamp)` order, as `data_prep_basic` sorts
    them; the per-group shift then reads chronologically within each pin.

    Args:
        rider_col: Rider identifier column.
        pin_cols: Columns identifying a pickup pin. When absent from `df` the
            per-pin columns are not produced, and `advanced_cleanup` will say so
            rather than fall back to a different rule.

    Returns:
        Copy with `shift_booking_ts`, `booking_time_diff_hr`,
        `booking_time_diff_min`, `is_first_booking`, and - when `pin_cols` are
        present - `pin_time_diff_hr`, `pin_time_diff_min` and `is_first_at_pin`.
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

    missing_pins = [c for c in pin_cols if c not in out.columns]
    if missing_pins:
        logger.warning(
            "Per-pin booking gaps not derived: column(s) %s absent. Rule 1 in "
            "advanced_cleanup needs them.", missing_pins,
        )
        return out

    prev_at_pin = out.groupby([rider_col, *pin_cols], sort=False)[
        "booking_timestamp"
    ].shift(1)
    # NaN for a rider's first booking at a pin, which by definition cannot be a
    # rebooking. Kept as NaN rather than filled, so the rule states that intent
    # instead of relying on a sentinel being large enough.
    at_pin_elapsed = out["booking_timestamp"] - prev_at_pin
    out["is_first_at_pin"] = prev_at_pin.isna()
    out["pin_time_diff_hr"] = at_pin_elapsed // 3600
    out["pin_time_diff_min"] = at_pin_elapsed // 60
    return out
