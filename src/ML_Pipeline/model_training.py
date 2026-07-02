import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime
from joblib import dump, load
from ML_Pipeline.train_test_data_prep import train_test_data_prep
from ML_Pipeline.xgb_model import xgb_model
from ML_Pipeline.shift_with_lag_and_rollingmean import shift_with_lag_and_rollingmean


def model_training(df, without_lag_model_path, with_lag_model_path):
    start_time = datetime.now()
    
    df['request_count'] = pd.to_numeric(
        df['request_count'], downcast='integer')
    
    df.ts = pd.to_datetime(df.ts)
    # First 24days of every month in Train and last 7 days of every month in Test
    df_train = df[df.ts.dt.day <= 23]
    df_test = df[df.ts.dt.day > 23]
    X, y, X_test, y_test = train_test_data_prep(df_train, df_test)
    xgb_model(X, y, X_test, y_test, without_lag_model_path)
    print("Time Taken for Model Training without lag: {}".format(datetime.now() - start_time))

    df = shift_with_lag_and_rollingmean(df)
    train1 = df[df.ts.dt.day <=23]
    test1 = df[df.ts.dt.day >23]
    X = train1.iloc[:, 1:-1]
    y = train1.iloc[:, -1]
    X_test = test1.iloc[:, 1:-1]
    y_test = test1.iloc[:, -1]
    xgb_model(X, y, X_test, y_test, with_lag_model_path)
    print("Time Taken for Model Training without lag: {}".format(datetime.now() - start_time))
