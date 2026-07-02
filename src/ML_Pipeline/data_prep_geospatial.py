import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.cluster import MiniBatchKMeans, KMeans
import gpxpy.geo
from datetime import datetime, timedelta
from joblib import dump, load

from ML_Pipeline.utils import round_timestamp_30interval
from ML_Pipeline.add_time_features import add_time_features
# from ML_Pipeline.clustering import min_distance
# from ML_Pipeline.clustering import makingRegions
from ML_Pipeline.clustering import optimal_cluster


def data_prep_geospatial(df, model_path, data_path):
    start_time = datetime.now()
    coord = df[["pick_lat", "pick_lng"]].values
    optimal_cluster(df, coord)
    regions = MiniBatchKMeans(
        n_clusters=50, batch_size=10000, random_state=5).fit(coord)
    df["pickup_cluster"] = regions.predict(df[["pick_lat", "pick_lng"]])
    # Model to Define pickup cluster, given latitude and longitude
    dump(regions, model_path, compress=3)
    print(f"Model stored in {model_path}")
    df['ts'] = np.vectorize(round_timestamp_30interval)(df['ts'])
    # Ensure ts is datetime for groupby operations
    df['ts'] = pd.to_datetime(df['ts'])
    dataset = deepcopy(df)
    dataset = dataset[['ts', 'number', 'pickup_cluster']]
    dataset = dataset.groupby(
        by=['ts', 'pickup_cluster']).count().reset_index()
    dataset.columns = ['ts', 'pickup_cluster', 'request_count']

    l = [datetime(2020, 3, 26, 00, 00, 00) + timedelta(minutes=30 * i)
         for i in range(0, 48 * 365)]
    lt = []
    for x in l:
        lt.append([x, -1, 0])
    temp = pd.DataFrame(lt, columns=['ts', 'pickup_cluster', 'request_count'])
    dataset = pd.concat([dataset, temp], ignore_index=True)
    data = dataset.set_index(['ts', 'pickup_cluster']).unstack().fillna(value=0).asfreq(
        freq='30Min').stack().sort_index(level=1).reset_index()
    # Removing Dummy Cluster
    data = data[data.pickup_cluster >= 0]
    # 366days x 48 (30 mins intervals) x 50 regions
    assert len(data) == 878400
    data = add_time_features(data, 'ts')

    data.to_csv(data_path, index=False, compression='gzip')
    print(f'Checkpoint -- data stored in {str(data_path)}')
    print("Time Taken for Data Preparation: {}".format(
        datetime.now() - start_time))
    return data
