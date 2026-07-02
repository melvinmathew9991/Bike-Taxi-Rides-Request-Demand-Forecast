
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from datetime import datetime
from copy import deepcopy
from sklearn.cluster import MiniBatchKMeans, KMeans
import gpxpy.geo
from datetime import datetime, timedelta
from joblib import dump, load

from ML_Pipeline.data_prep_basic import data_prep_basic
from ML_Pipeline.data_prep_advanced import data_prep_advanced
from ML_Pipeline.data_prep_geospatial import data_prep_geospatial
from ML_Pipeline.model_training import model_training
from ML_Pipeline.prediction_pipeline import prediction_pipeline


geolocator = Nominatim(user_agent="OLABikes")

df = pd.read_csv('C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/data/raw_data.csv', low_memory=False, compression='gzip')


df = data_prep_basic(df)


df = data_prep_advanced(df, 'C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/data/clean_data.csv')


df = pd.read_csv('C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/data/clean_data.csv', compression='gzip')


df = data_prep_geospatial(
    df,
    'C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/output/pickup_cluster_model.joblib',
    'C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/output/Data_Prepared.csv'
)


df = pd.read_csv('C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/data/clean_data.csv', compression='gzip')


model_training(df,
    'C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/output/prediction_model_without_lag.joblib',
    'C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/output/prediction_model_with_lag.joblib'
)


prediction_pipeline(
    cleaned_data_path='C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/data/test_dataset/cleaned_test_booking_data.csv',
    cluster_model_path='C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/output/pickup_cluster_model.joblib',
    predict_without_lag_path='C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/output/prediction_model_without_lag.joblib',
    predict_with_lag_path='C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/output/prediction_model_with_lag.joblib',
    data_with_lag_path='C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/data/prediction_with_lag_model.csv',
    data_without_lag_path='C:/Sachin/project/Bike-Taxi-Rides-Request-Demand-Forecast/data/prediction_without_lag_model.csv'
)