import pandas as pd

from src.ML_Pipeline.prediction_pipeline import shift_with_lag_and_rollingmean


def test_shift_with_lag_and_rollingmean_handles_grouped_data():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime([
                "2021-03-27 00:00:00",
                "2021-03-27 00:30:00",
                "2021-03-27 01:00:00",
                "2021-03-27 00:00:00",
                "2021-03-27 00:30:00",
                "2021-03-27 01:00:00",
            ]),
            "pickup_cluster": [0, 0, 0, 1, 1, 1],
            "mins": [0, 30, 0, 0, 30, 0],
            "hour": [0, 0, 1, 0, 0, 1],
            "month": [3, 3, 3, 3, 3, 3],
            "quarter": [1, 1, 1, 1, 1, 1],
            "dayofweek": [4, 4, 4, 4, 4, 4],
            "request_count": [10, 12, 14, 8, 10, 12],
        }
    )

    result = shift_with_lag_and_rollingmean(df)

    assert {"lag_1", "lag_2", "lag_3", "rolling_mean"}.issubset(result.columns)
    assert len(result) == 4
    assert result["lag_1"].isna().sum() == 0
