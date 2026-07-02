import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from datetime import datetime


def add_time_features(df, dt_col):
    df['hour'] = df[dt_col].dt.hour
    df['mins'] = df[dt_col].dt.minute
    df['day'] = df[dt_col].dt.day
    df['month'] = df[dt_col].dt.month
    df['year'] = df[dt_col].dt.year
    df['dayofweek'] = df[dt_col].dt.dayofweek
    df['quarter'] = df[dt_col].dt.quarter
    return df
