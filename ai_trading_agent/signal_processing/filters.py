"""
signal_processing/filters.py

Provides noise filtering utilities for price and sentiment signals.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def exponential_moving_average(series: pd.Series, span: int = 10) -> pd.Series:
    """
    Simple exponential moving average filter.
    """
    return series.ewm(span=span, adjust=False).mean()


def savitzky_golay_filter(series: pd.Series, window_length: int = 11, polyorder: int = 2) -> pd.Series:
    """
    Savitzky-Golay filter for smoothing signals (noise reduction).
    """
    if len(series) < window_length:
        window_length = len(series) if len(series) % 2 == 1 else len(series) - 1
        if window_length < 3:
            return series  # Not enough data to filter
    return pd.Series(savgol_filter(series, window_length=window_length, polyorder=polyorder), index=series.index)


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling z-score normalization (can help with outlier/noise suppression).
    """
    return (series - series.rolling(window).mean()) / series.rolling(window).std()
