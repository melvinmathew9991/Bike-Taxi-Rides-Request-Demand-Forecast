"""
Tests for the business-rule cleaning layer.

This module makes the largest single data decision in the pipeline — it removes
roughly 55% of all rows — and had no tests at all. The audit found a defect
hiding in exactly that gap: Rule 1 dropped 548,510 rows that its own docstring
does not describe.

Every test here states the rule's *intent* on a handful of hand-built rows, so
the next change to a threshold or a mask has something to answer to.

Rows carry the columns `advanced_cleanup` requires: the rider id, pickup and
drop coordinates, and the per-rider gaps that `shift_time` derives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ML_Pipeline.advanced_cleanup import (
    INDIA_BBOX,
    KARNATAKA_BBOX,
    MAX_PLAUSIBLE_TRIP_KM,
    MIN_MINUTES_BETWEEN_BOOKINGS,
    MIN_TRIP_DISTANCE_KM,
    advanced_cleanup,
)
from ML_Pipeline.shift_time import shift_time
from ML_Pipeline.utils import haversine_km

# Two points ~1.4 km apart in Bangalore, comfortably inside every bounding box.
PICK = (12.9716, 77.5946)
DROP = (12.9850, 77.5990)


def rows(**overrides) -> pd.DataFrame:
    """One clean booking per row, with columns overridable per-column."""
    n = len(next(iter(overrides.values()))) if overrides else 1
    base = {
        "number": [1] * n,
        "pick_lat": [PICK[0]] * n,
        "pick_lng": [PICK[1]] * n,
        "drop_lat": [DROP[0]] * n,
        "drop_lng": [DROP[1]] * n,
        # Large gaps, so nothing is dropped unless a test asks for it.
        "booking_time_diff_hr": [99] * n,
        "booking_time_diff_min": [99 * 60] * n,
        "pin_time_diff_hr": [99] * n,
        "pin_time_diff_min": [99 * 60] * n,
        "is_first_booking": [False] * n,
        "is_first_at_pin": [False] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def bookings(spec: list[tuple[int, int, tuple[float, float]]]) -> pd.DataFrame:
    """
    Build bookings and derive the gap columns the way the pipeline does.

    Driving the rules through `shift_time` rather than hand-setting gap columns
    means these tests exercise the real derivation chain - which is where the
    Rule 1 defect lived.

    Args:
        spec: `(rider, minutes_from_epoch, (pick_lat, pick_lng))` per booking.
    """
    df = pd.DataFrame(
        {
            "number": [r for r, _, _ in spec],
            "booking_timestamp": [m * 60 for _, m, _ in spec],
            "pick_lat": [p[0] for _, _, p in spec],
            "pick_lng": [p[1] for _, _, p in spec],
            "drop_lat": [DROP[0]] * len(spec),
            "drop_lng": [DROP[1]] * len(spec),
        }
    ).sort_values(["number", "booking_timestamp"]).reset_index(drop=True)
    return shift_time(df)


class TestRule1RebookingTheSamePin:
    """
    "Same rider rebooking the same pickup pin within an hour."

    These drive the real chain: timestamps -> shift_time -> advanced_cleanup.

    The old mask combined `duplicated(subset=[number, pick_lat, pick_lng],
    keep=False)` - true for every row whose (rider, pin) recurs *anywhere* in the
    data - with `booking_time_diff_hr`, the gap to that rider's previous booking
    from *any* pin. So a row was dropped when a commuter had ever used that pin
    twice and happened to book something else in the preceding hour, and
    `keep=False` dropped the first of the group as well.
    """

    OTHER = (12.9000, 77.6500)

    def test_a_rapid_rebooking_at_the_same_pin_is_dropped(self):
        out = advanced_cleanup(bookings([(1, 0, PICK), (1, 10, PICK)]))
        assert len(out) == 1, "the second request 10 min later at the same pin is a retry"

    def test_the_first_of_a_burst_is_kept(self):
        """`keep=False` used to delete the original request too."""
        out = advanced_cleanup(
            bookings([(1, 0, PICK), (1, 5, PICK), (1, 12, PICK)])
        )
        assert len(out) == 1
        assert out["booking_timestamp"].iloc[0] == 0, "the original request must survive"

    def test_the_same_pin_beyond_an_hour_apart_is_kept(self):
        out = advanced_cleanup(bookings([(1, 0, PICK), (1, 61, PICK)]))
        assert len(out) == 2

    def test_exactly_an_hour_apart_is_treated_as_a_rebooking(self):
        """The boundary is inclusive, as `<= REBOOK_SAME_LOCATION_HOURS` says."""
        out = advanced_cleanup(bookings([(1, 0, PICK), (1, 60, PICK)]))
        assert len(out) == 1

    def test_the_threshold_is_an_hour_not_two(self):
        """
        `pin_time_diff_hr` is floored, so comparing hours would make 61-119
        minutes read as "1 hour" and drop them. The rule compares minutes.
        """
        out = advanced_cleanup(bookings([(1, 0, PICK), (1, 90, PICK)]))
        assert len(out) == 2, "90 minutes apart is not 'within an hour'"

    def test_a_commuter_booking_the_same_pin_daily_is_kept_entirely(self):
        """
        The defect that mattered. Five daily bookings from one pin, with other
        bookings elsewhere shortly before some of them. Under the old rule the
        unrelated nearby activity made the per-rider gap small, and every one of
        these was deleted.
        """
        spec = []
        for day in range(5):
            base = day * 24 * 60
            spec.append((1, base, PICK))             # the daily commute
            spec.append((1, base + 20, self.OTHER))  # something else, 20 min later
        out = advanced_cleanup(bookings(spec))
        at_pin = out[(out["pick_lat"] == PICK[0]) & (out["pick_lng"] == PICK[1])]
        assert len(at_pin) == 5, (
            f"kept {len(at_pin)} of 5 daily commutes; repeat use of a pin on "
            "separate days is real demand, not duplicate requests"
        )

    def test_a_different_pin_within_the_hour_is_not_a_rebooking(self):
        out = advanced_cleanup(bookings([(1, 0, PICK), (1, 10, self.OTHER)]))
        assert len(out) == 2, "two different pins are two intentions"

    def test_different_riders_at_the_same_pin_are_independent(self):
        out = advanced_cleanup(bookings([(1, 0, PICK), (2, 10, PICK)]))
        assert len(out) == 2, "a shared pickup point is not one rider rebooking"

    def test_the_rule_needs_the_per_pin_gap_and_says_so(self):
        """Evaluating Rule 1 without the per-pin gap must fail loudly."""
        df = rows().drop(columns=["pin_time_diff_min"])
        with pytest.raises(KeyError, match="pin_time_diff_min"):
            advanced_cleanup(df)


class TestRule2Retries:
    def test_bookings_closer_than_the_threshold_are_dropped(self):
        out = advanced_cleanup(
            rows(booking_time_diff_min=[99 * 60, MIN_MINUTES_BETWEEN_BOOKINGS - 1])
        )
        assert len(out) == 1

    def test_the_threshold_itself_is_kept(self):
        out = advanced_cleanup(
            rows(booking_time_diff_min=[99 * 60, MIN_MINUTES_BETWEEN_BOOKINGS])
        )
        assert len(out) == 2, f">= {MIN_MINUTES_BETWEEN_BOOKINGS} min is new demand"

    def test_a_first_booking_survives_regardless_of_its_gap(self):
        """
        A rider's first booking has no predecessor. This used to survive only
        because the missing timestamp was filled with 0, making the gap
        enormous; `is_first_booking` states it on purpose.
        """
        out = advanced_cleanup(
            rows(booking_time_diff_min=[0], is_first_booking=[True])
        )
        assert len(out) == 1


class TestRule3DegenerateTrips:
    def test_a_pickup_and_drop_at_the_same_point_is_dropped(self):
        out = advanced_cleanup(
            rows(drop_lat=[PICK[0]], drop_lng=[PICK[1]])
        )
        assert out.empty

    def test_a_trip_longer_than_the_floor_is_kept(self):
        out = advanced_cleanup(rows())
        assert len(out) == 1
        assert out["geodesic_distance"].iloc[0] > MIN_TRIP_DISTANCE_KM


class TestRule4OutsideIndia:
    @pytest.mark.parametrize(
        "lat,lng",
        [(51.5074, -0.1278), (0.0, 0.0), (-33.8688, 151.2093)],
        ids=["london", "null-island", "sydney"],
    )
    def test_coordinates_outside_india_are_dropped(self, lat, lng):
        out = advanced_cleanup(rows(pick_lat=[lat], pick_lng=[lng]))
        assert out.empty

    def test_a_drop_outside_india_also_disqualifies_the_row(self):
        out = advanced_cleanup(rows(drop_lat=[51.5074], drop_lng=[-0.1278]))
        assert out.empty

    def test_the_bounding_box_is_exclusive_at_its_edges(self):
        """`_outside` uses <= and >=, so a point exactly on the edge is outside."""
        min_lat, _, min_lng, _ = INDIA_BBOX
        out = advanced_cleanup(rows(pick_lat=[min_lat], pick_lng=[min_lng]))
        assert out.empty


class TestRule5OutsideKarnatakaAndImplausiblyLong:
    def test_a_long_trip_inside_karnataka_is_kept(self):
        """Either condition alone is allowed; only the combination is bad data."""
        min_lat, max_lat, min_lng, max_lng = KARNATAKA_BBOX
        out = advanced_cleanup(
            rows(
                pick_lat=[min_lat + 0.1], pick_lng=[min_lng + 0.1],
                drop_lat=[max_lat - 0.1], drop_lng=[max_lng - 0.1],
            )
        )
        assert len(out) == 1

    def test_a_short_trip_outside_karnataka_is_kept(self):
        # Mumbai: inside India, outside Karnataka, a short hop.
        out = advanced_cleanup(
            rows(
                pick_lat=[19.0760], pick_lng=[72.8777],
                drop_lat=[19.0900], drop_lng=[72.8800],
            )
        )
        assert len(out) == 1

    def test_outside_karnataka_and_implausibly_long_is_dropped(self):
        # Delhi to Chennai: ~1750 km, both endpoints outside Karnataka.
        out = advanced_cleanup(
            rows(
                pick_lat=[28.6139], pick_lng=[77.2090],
                drop_lat=[13.0827], drop_lng=[80.2707],
            )
        )
        assert out.empty
        assert (
            haversine_km(
                np.array([28.6139]), np.array([77.2090]),
                np.array([13.0827]), np.array([80.2707]),
            )[0]
            > MAX_PLAUSIBLE_TRIP_KM
        )


class TestContract:
    def test_missing_columns_are_named(self):
        with pytest.raises(KeyError, match="booking_time_diff_hr"):
            advanced_cleanup(pd.DataFrame({"number": [1]}))

    def test_the_input_frame_is_not_mutated(self):
        df = rows(booking_time_diff_hr=[99, 0])
        before = df.copy()
        advanced_cleanup(df)
        pd.testing.assert_frame_equal(df, before)

    def test_a_distance_column_is_added(self):
        out = advanced_cleanup(rows())
        assert "geodesic_distance" in out.columns

    def test_the_index_is_reset(self):
        out = advanced_cleanup(rows(booking_time_diff_min=[99 * 60, 0, 99 * 60]))
        assert out.index.tolist() == list(range(len(out)))


class TestHaversine:
    """`haversine_km` replaced a per-row geodesic call; the filters depend on it."""

    def test_zero_distance_for_identical_points(self):
        d = haversine_km(np.array([12.97]), np.array([77.59]),
                         np.array([12.97]), np.array([77.59]))
        assert d[0] == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_of_latitude_is_about_111_km(self):
        d = haversine_km(np.array([0.0]), np.array([0.0]),
                         np.array([1.0]), np.array([0.0]))
        assert d[0] == pytest.approx(111.19, rel=1e-3)

    def test_it_is_symmetric(self):
        a = haversine_km(np.array([12.97]), np.array([77.59]),
                         np.array([13.08]), np.array([80.27]))
        b = haversine_km(np.array([13.08]), np.array([80.27]),
                         np.array([12.97]), np.array([77.59]))
        assert a[0] == pytest.approx(b[0])

    def test_it_is_vectorised_over_arrays(self):
        d = haversine_km(
            np.array([12.97, 0.0]), np.array([77.59, 0.0]),
            np.array([12.97, 1.0]), np.array([77.59, 0.0]),
        )
        assert d.shape == (2,)
        assert d[0] == pytest.approx(0.0, abs=1e-9)
        assert d[1] == pytest.approx(111.19, rel=1e-3)

    def test_agrees_with_the_50_m_cleaning_threshold(self):
        """The only decision this function feeds is a 50 m cut."""
        # ~0.0005 degrees of latitude is ~55 m: just over the threshold.
        d = haversine_km(np.array([12.9716]), np.array([77.5946]),
                         np.array([12.9721]), np.array([77.5946]))
        assert d[0] > MIN_TRIP_DISTANCE_KM


class TestShiftTime:
    def test_the_first_booking_per_rider_is_flagged(self):
        df = pd.DataFrame(
            {
                "number": [1, 1, 2],
                "booking_timestamp": [1_000, 2_000, 5_000],
            }
        )
        out = shift_time(df)
        assert out["is_first_booking"].tolist() == [True, False, True]

    def test_gaps_are_measured_against_the_riders_own_previous_booking(self):
        df = pd.DataFrame(
            {"number": [1, 1], "booking_timestamp": [0, 3_600]}
        )
        out = shift_time(df)
        assert out["booking_time_diff_hr"].tolist()[1] == 1
        assert out["booking_time_diff_min"].tolist()[1] == 60

    def test_a_missing_booking_timestamp_column_is_reported(self):
        with pytest.raises(KeyError, match="booking_timestamp"):
            shift_time(pd.DataFrame({"number": [1]}))
