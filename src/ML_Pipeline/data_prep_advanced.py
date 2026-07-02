import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from datetime import datetime
from ML_Pipeline.advanced_cleanup import advanced_cleanup


def data_prep_advanced(df, path):
    start_time = datetime.now()
    df = advanced_cleanup(df)
    print("Advanced pre processing done")
    print("After Advance CleanUp Row Count: {}".format(len(df)))
    print("Time Taken for Data Preprocessing: {}".format(
        datetime.now()-start_time))
    dataset = df[
        ['ts', 'number', 'pick_lat', 'pick_lng', 'drop_lat', 'drop_lng', 'geodesic_distance', 'hour', 'mins', 'day',
            'month', 'year', 'dayofweek', 'booking_timestamp', 'booking_time_diff_hr', 'booking_time_diff_min']]
    dataset.to_csv(path,
                   index=False, compression='gzip')
    print(f"Checkpoint -- data is stored as clean_data.csv in {str(path)}")
    return dataset
