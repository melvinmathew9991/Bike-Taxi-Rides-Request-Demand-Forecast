import pandas as pd
import numpy as np


def shift_time(df):
    # get difference of time booking b/w two consecutive rides booked by user
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['shift_booking_ts'] = df.groupby('number')['booking_timestamp'].shift(1)
    df['shift_booking_ts'] = df['shift_booking_ts'].fillna(0).astype('int64')
    df['booking_time_diff_hr'] = round(
        (df['booking_timestamp'] - df['shift_booking_ts']) // 3600)
    df['booking_time_diff_min'] = round(
        (df['booking_timestamp'] - df['shift_booking_ts']) // 60)
    return df
