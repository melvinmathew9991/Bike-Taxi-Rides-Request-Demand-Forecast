"""
Features, training, forecasting and evaluation.

    features.py     canonical feature engineering, shared by training and
                    serving, and `ModelBundle`
    splitting.py    chronological train/test splitting
    xgb_model.py    XGBoost fitting with early stopping, and the final refit
    training.py     the training stage: both model variants and the deploy gate
    forecast.py     direct and recursive multi-step forecasting
    prediction.py   the forecasting stage of the pipeline
    evaluation.py   metrics, baselines, per-cluster error
    validation.py   rolling-origin validation
"""
