import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime
from joblib import dump, load
import xgboost as xgb
from ML_Pipeline.regressor_evaluate import regressor_evaluate


def xgb_model(X, y, X_test, y_test, model_path):

    model = xgb.XGBRegressor(learning_rate=0.01, random_state=0, n_estimators=100, max_depth=8,
                             objective="reg:squarederror")

    model.fit(X, y, verbose=False)
    print("XGBOOST Regressor")
    print("Model Score: ", model.score(X, y))
    print(
        "RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y, model.predict(X))), regressor_evaluate(model, X_test, y_test)))
    dump(model, model_path, compress=3)
    print(f'Model save in {model_path}')
