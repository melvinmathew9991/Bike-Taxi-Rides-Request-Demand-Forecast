import pandas as pd


def train_test_data_prep(df_train, df_test):
    # Ensure datetime
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train['ts'] = pd.to_datetime(df_train['ts'])
    df_test['ts'] = pd.to_datetime(df_test['ts'])
    
    # Remove duplicates
    df_test = df_test.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(
        subset=['ts', 'pickup_cluster'], keep='last')
    
    # Combine datasets
    temp = pd.concat([df_train, df_test], ignore_index=True)
    temp = temp.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(
        subset=['ts', 'pickup_cluster'], keep='last').reset_index(drop=True)
    
    # Create lag features using simple groupby (no MultiIndex)
    temp['lag_1'] = temp.groupby('pickup_cluster')['request_count'].shift(1)
    temp['lag_2'] = temp.groupby('pickup_cluster')['request_count'].shift(2)
    temp['lag_3'] = temp.groupby('pickup_cluster')['request_count'].shift(3)
    
    # Rolling mean
    temp['rolling_mean'] = temp.groupby('pickup_cluster')['request_count'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()).shift(1)
    
    # Drop NaN values
    temp = temp.dropna()
    
    # Select features (exclude ts and pickup_cluster from X)
    feature_cols = ['pickup_cluster', 'mins', 'hour', 'month', 'quarter', 'dayofweek']
    
    # Ensure all feature columns exist
    feature_cols = [col for col in feature_cols if col in temp.columns]
    
    # Split by day
    train1 = temp[temp.ts.dt.day <= 23].copy()
    test1 = temp[temp.ts.dt.day > 23].copy()

    X = train1[feature_cols].copy()
    y = train1['request_count'].copy()
    X_test = test1[feature_cols].copy()
    y_test = test1['request_count'].copy()
    
    return X, y, X_test, y_test
