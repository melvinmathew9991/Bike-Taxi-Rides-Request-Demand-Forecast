import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime, timedelta


def round_timestamp_30interval(x):
    if type(x) == str:
        x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return x - timedelta(minutes=x.minute % 30, seconds=x.second, microseconds=x.microsecond)


def time_features(data):
    data['mins'] = data.ts.dt.minute
    data['hour'] = data.ts.dt.hour
    data['day'] = data.ts.dt.day
    data['month'] = data.ts.dt.month
    data['dayofweek'] = data.ts.dt.dayofweek
    data['quarter'] = data.ts.dt.quarter
    return data


def prediction_without_lag(df, predict_without_lag):
    return predict_without_lag.predict(df[['pickup_cluster', 'mins', 'hour', 'month', 'quarter', 'dayofweek']])


def prediction_with_lag(df, predict_with_lag):
    return predict_with_lag.predict(df[['pickup_cluster', 'mins', 'hour', 'month', 'quarter',
                                        'dayofweek', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean']])


def shift_with_lag_and_rollingmean(df):
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(
        subset=['ts', 'pickup_cluster'], keep='last').reset_index(drop=True)

    df['lag_1'] = df.groupby('pickup_cluster')['request_count'].shift(1)
    df['lag_2'] = df.groupby('pickup_cluster')['request_count'].shift(2)
    df['lag_3'] = df.groupby('pickup_cluster')['request_count'].shift(3)
    df['rolling_mean'] = df.groupby('pickup_cluster')['request_count'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()).shift(1)

    df = df.dropna()
    required_cols = ['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter',
                     'dayofweek', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'request_count']
    available_cols = [col for col in required_cols if col in df.columns]
    return df[available_cols]


def prediction_pipeline(cleaned_data_path, cluster_model_path, predict_without_lag_path, predict_with_lag_path, data_without_lag_path, data_with_lag_path):
    start_time = datetime.now()
    # Loading Test Dataset and Model Files
    df = pd.read_csv(cleaned_data_path,
                     compression='gzip', low_memory=False)
    cluster_model = load(cluster_model_path)
    predict_without_lag = load(predict_without_lag_path)
    predict_with_lag = load(predict_with_lag_path)

    # Geospacial Feature to Pickup Cluster - using Clustering Kmeans Algorithm
    df['pickup_cluster'] = cluster_model.predict(df[['pick_lat', 'pick_lng']])

    # Data Processing
    df['ts'] = np.vectorize(round_timestamp_30interval)(df['ts'])
    df['ts'] = pd.to_datetime(df['ts'])

    df = df[['ts', 'number', 'pickup_cluster']]
    df = df.groupby(by=['ts', 'pickup_cluster']).count().reset_index()
    df.columns = ['ts', 'pickup_cluster', 'request_count']

    # Adding Dummy pickup cluster -1

    # Change this Data based on your data
    l = [datetime(2021, 3, 26, 00, 00, 00) + timedelta(minutes=30*i)
         for i in range(0, 51)]
    lt = []
    for x in l:
        lt.append([x, -1, 0])
    temp = pd.DataFrame(lt, columns=['ts', 'pickup_cluster', 'request_count'])
    df = pd.concat([df, temp], ignore_index=True)

    data = df.set_index(['ts', 'pickup_cluster']).unstack().fillna(
        value=0).asfreq(freq='30Min').stack().sort_index(level=1).reset_index()

    # Removing Dummy Cluster
    data = data[data.pickup_cluster >= 0]

    df = time_features(data)

    # Prediction without Lag Features
    data_without_lag = df[df['ts'] >= datetime(
        2021, 3, 27, 00, 00, 00)].copy()
    data_without_lag['request_count'] = prediction_without_lag(
        data_without_lag, predict_without_lag)
    data_without_lag.to_csv(
        data_without_lag_path, index=False, compression='gzip')

    # Prediction with Lag and RollingMean Features
    start_date = datetime(2021, 3, 27, 00, 00, 00)
    for x in range(3):
        df = shift_with_lag_and_rollingmean(df)
        df.loc[df[df['ts'] == start_date+timedelta(minutes=30*x)].index, 'request_count'] = prediction_with_lag(
            df[df['ts'] == start_date+timedelta(minutes=30*x)], predict_with_lag)
    data_with_lag = df[df['ts'] >= datetime(
        2021, 3, 27, 00, 00, 00)].__copy__()

    data_with_lag.to_csv(
        data_with_lag_path, index=False, compression='gzip')
    print("Time Taken by Prediction Pipeline: {}".format(
        datetime.now() - start_time))


