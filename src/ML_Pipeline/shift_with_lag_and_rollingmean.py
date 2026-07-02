import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime, timedelta


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
