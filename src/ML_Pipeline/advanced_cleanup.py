
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from datetime import datetime
from ML_Pipeline.utils import geodestic_distance


def advanced_cleanup(df) -> pd.DataFrame:
    # remove duplicate booking within 1hour from same user at same pickup lat-long
    df = df[~((df.duplicated(subset=['number', 'pick_lat', 'pick_lng'],
                             keep=False)) & (df.booking_time_diff_hr <= 1))]

    # remove demand count / repeat booking by same user within 4mins (consider multiple retry/error booking)
    df = df[(df.booking_time_diff_min >= 8)]

    # Geodesic Distance calculate
    df['geodesic_distance'] = np.vectorize(geodestic_distance)(df['pick_lat'], df['pick_lng'], df['drop_lat'],
                                                               df['drop_lng'])

    # remove ride where pickup and drop location distance is less than 50meters
    df = df[df.geodesic_distance > 0.05]

    # remove rides outside India Bounding Box
    df.reset_index(inplace=True, drop=True)
    outside_India = df[(df.pick_lat <= 6.2325274) | (df.pick_lat >= 35.6745457) | (df.pick_lng <= 68.1113787) | (
        df.pick_lng >= 97.395561) | (df.drop_lat <= 6.2325274) | (df.drop_lat >= 35.6745457) | (
        df.drop_lng <= 68.1113787) | (df.drop_lng >= 97.395561)]
    df = df[~df.index.isin(outside_India.index)].reset_index(drop=True)

    # remove rides outside KA and pickup/drop distance > 500kms
    total_ride_outside_KA = df[
        (df.pick_lat <= 11.5945587) | (df.pick_lat >= 18.4767308) | (df.pick_lng <= 74.0543908) | (
            df.pick_lng >= 78.588083) | (df.drop_lat <= 11.5945587) | (df.drop_lat >= 18.4767308) | (
            df.drop_lng <= 74.0543908) | (df.drop_lng >= 78.588083)]
    suspected_bad_rides = total_ride_outside_KA[total_ride_outside_KA.geodesic_distance > 500]

    df = df[~df.index.isin(suspected_bad_rides.index)].reset_index(drop=True)
    return df
