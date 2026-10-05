#!/usr/bin/env python
"""
Download hourly historical weather for Bengaluru from Open-Meteo.

Saved once to a CSV so that nothing in the pipeline calls an external service:
runs stay reproducible and work offline. Only the coordinates and the date
range are sent; nothing from the booking data.

Timestamps are requested in Asia/Kolkata, matching the booking data (demand
peaks at 09:00-10:00 and 18:00-19:00 local, so the grid is in IST, not UTC).

Open-Meteo's hourly `precipitation` and `rain` are the total over the
*preceding* hour: the value stamped 15:00 fell between 14:00 and 15:00.

Usage:
    python scripts/fetch_weather.py --out data/weather_bengaluru.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
#: Bengaluru city centre. The service area is ~65 km across; one point stands
#: in for all of it, which smooths out rain that falls on part of the city.
LATITUDE, LONGITUDE = 12.97, 77.59
HOURLY = ("precipitation", "rain", "temperature_2m", "relative_humidity_2m")


def fetch(start: str, end: str) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(HOURLY),
            "timezone": "Asia/Kolkata",
        }
    )
    with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=60) as response:
        payload = json.load(response)
    hourly = payload["hourly"]
    frame = pd.DataFrame(hourly).rename(columns={"time": "ts"})
    frame["ts"] = pd.to_datetime(frame["ts"])
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # A week before the data starts, so trailing-rain features have history.
    parser.add_argument("--start", default="2020-03-19")
    parser.add_argument("--end", default="2021-03-27")
    parser.add_argument("--out", type=Path, default=Path("data/weather_bengaluru.csv"))
    args = parser.parse_args(argv)

    frame = fetch(args.start, args.end)
    missing = frame[list(HOURLY)].isna().sum()
    if missing.any():
        print(f"Missing values per column:\n{missing[missing > 0]}", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        f"Wrote {len(frame):,} hours, {frame['ts'].min()} to {frame['ts'].max()}, "
        f"to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
