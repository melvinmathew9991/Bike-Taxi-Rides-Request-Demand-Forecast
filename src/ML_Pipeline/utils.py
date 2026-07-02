import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta

geolocator = Nominatim(user_agent="OLABikes")


def remove_duplicates(df, cols=None):
    df.drop_duplicates(subset=cols, inplace=True, keep='last')
    return df


def round_timestamp_30interval(x):
    if type(x) == str:
        x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # Convert numpy.datetime64 to pandas Timestamp
    elif isinstance(x, np.datetime64):
        x = pd.Timestamp(x)
    # Handle pandas Timestamp
    elif not isinstance(x, datetime):
        x = pd.Timestamp(x)
    return x - timedelta(minutes=x.minute % 30, seconds=x.second, microseconds=x.microsecond)


def convert_into_datetime(df, col):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def convert_into_numeric(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
    return df


def geodestic_distance(pick_lat, pick_lng, drop_lat, drop_lng) -> float:
    return round(geodesic((pick_lat, pick_lng), (drop_lat, drop_lng)).miles * 1.60934, 2)
