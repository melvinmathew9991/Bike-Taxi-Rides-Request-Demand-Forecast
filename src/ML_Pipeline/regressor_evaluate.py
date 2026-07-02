import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime
from joblib import dump, load
def regressor_evaluate(regressor, X_test, y_test):
    y_pred = regressor.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    return rms
