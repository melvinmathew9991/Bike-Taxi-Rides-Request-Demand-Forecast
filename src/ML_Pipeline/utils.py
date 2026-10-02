"""
Small shared helpers.

Timestamp flooring and calendar features previously lived here *and* in
`prediction_pipeline`, in copies that had already drifted (the serving copy
handled fewer input types). Both now live once in `ML_Pipeline.features`; this
module re-exports them so existing imports keep working.

The module-level `Nominatim(user_agent="OLABikes")` geocoder that used to be
instantiated here on import - and in two other modules - has been removed. It
was never called, and constructing a network client at import time is a side
effect no importer asked for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ML_Pipeline.features import (  # noqa: F401  (re-exported for compatibility)
    add_calendar_features,
    floor_to_interval,
)

EARTH_RADIUS_KM = 6371.0088

#: Magic bytes at the start of a gzip member.
GZIP_MAGIC = b"\x1f\x8b"


def read_csv_any(source, **kwargs) -> pd.DataFrame:
    """
    Read a CSV whose compression is not reliably indicated by its name.

    Four near-identical copies of this logic existed - in `pipeline`,
    `prediction_pipeline`, `streamlit_app` and `scripts/compare_strategies` -
    each a `try` on gzip with a fallback to plain. They existed because the
    pipeline wrote gzip-compressed files under a bare `.csv` extension, which
    `pandas` cannot infer from, so a plain `pd.read_csv` on one failed with
    `UnicodeDecodeError: invalid start byte`.

    Pipeline outputs are now written as `.csv.gz` and need none of this - pandas
    infers from the extension. This remains for *input* files, which are supplied
    by whoever runs the pipeline and in the reference dataset are gzip under a
    `.csv` name.

    Detection is by content rather than by exception: a gzip member starts with
    `1f 8b`, so the file says what it is. The previous approach caught
    `(OSError, EOFError, ValueError)` around a full parse, which also swallows
    genuine malformed-CSV errors and reports them as "not gzip".

    Args:
        source: Path, or an open binary file object (a Streamlit upload).
        **kwargs: Passed through to `pd.read_csv`.

    Returns:
        The parsed frame.
    """
    kwargs.setdefault("low_memory", False)

    if hasattr(source, "read"):
        head = source.read(2)
        source.seek(0)
        compression = "gzip" if head == GZIP_MAGIC else None
        return pd.read_csv(source, compression=compression, **kwargs)

    path = Path(source)
    with open(path, "rb") as fh:
        head = fh.read(2)
    if head == GZIP_MAGIC:
        return pd.read_csv(path, compression="gzip", **kwargs)
    # Let pandas infer, so `.csv.gz`, `.zip` and friends keep working.
    return pd.read_csv(path, **kwargs)


def remove_duplicates(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Drop duplicate rows, returning a new frame.

    Note the previous version mutated its argument in place (`inplace=True`) and
    also returned it, so callers could not tell whether their own frame had been
    modified. It kept `keep='last'`, whereas the exploratory notebook this stage
    was derived from kept `'first'`; `keep` is now explicit at the call site.
    """
    return df.drop_duplicates(subset=cols, keep="last").reset_index(drop=True)


def convert_into_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse `col` as datetime, coercing unparseable values to NaT."""
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def convert_into_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse `col` as numeric, coercing unparseable values to NaN."""
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def haversine_km(
    lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray
) -> np.ndarray:
    """
    Great-circle distance in kilometres, vectorised over numpy arrays.

    This replaces a `np.vectorize`-wrapped `geopy.distance.geodesic` call, which
    was a Python-level loop constructing a geodesic object per row and running
    Karney's algorithm on each. Over millions of bookings that dominated the
    cleaning stage's runtime. The filter it feeds only needs a 50 m threshold, so
    the sub-metre accuracy of the geodesic method bought nothing.

    Accuracy, measured over 3,000 Bangalore-scale trips against
    `geopy.distance.geodesic`: max relative difference 0.51%, mean absolute
    difference 29 m, max 154 m. Crucially, the two methods disagreed on the 50 m
    cleaning threshold for **zero** of those trips, and `geodesic_distance` is
    rounded to 10 m before use.

    Speed on the same benchmark: ~249x faster, which on the reference 8.38 M-row
    dataset is ~8 seconds instead of ~32 minutes.
    """
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(v, dtype="float64"))
                              for v in (lat1, lng1, lat2, lng2))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def geodestic_distance(pick_lat, pick_lng, drop_lat, drop_lng) -> float:
    """
    Scalar great-circle distance in kilometres.

    Retained for compatibility; prefer `haversine_km` on arrays.
    """
    return float(np.round(haversine_km(pick_lat, pick_lng, drop_lat, drop_lng), 2))


def round_timestamp_30interval(x):
    """Deprecated: use `features.floor_to_interval`, which is vectorised."""
    return floor_to_interval([x], 30).iloc[0]
