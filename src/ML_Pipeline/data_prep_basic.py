import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from datetime import datetime

from ML_Pipeline.utils import remove_duplicates
from ML_Pipeline.utils import convert_into_datetime
from ML_Pipeline.utils import convert_into_numeric
from ML_Pipeline.add_time_features import add_time_features
from ML_Pipeline.shift_time import shift_time

geolocator = Nominatim(user_agent="OLABikes")


def data_prep_basic(df):
    start_time = datetime.now()
    df = df.copy()
    df = remove_duplicates(df, ['ts', 'number'])
    df.reset_index(inplace=True, drop=True)
    df = convert_into_numeric(df, 'number')
    df = df.dropna(subset=['number'])  # Drop rows where number is NaN after conversion
    df.reset_index(inplace=True, drop=True)  # Reset index after dropna

    df['number'] = df['number'].astype(float).astype('int32')
    df = df[df['number'] != -1].reset_index(drop=True)

    df = convert_into_datetime(df, 'ts')
    df = df.dropna(subset=['ts'])  # Drop rows where ts is NaN after conversion
    df.reset_index(inplace=True, drop=True)
    df = add_time_features(df, 'ts')
    df.sort_values(by=['number', 'ts'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['booking_timestamp'] = df.ts.values.astype(np.int64) // 10 ** 9
    df = shift_time(df)
    print("Basic pre processing done")
    print("Time Taken for basic pre Preprocessing: {}".format(datetime.now()-start_time))
    return df
