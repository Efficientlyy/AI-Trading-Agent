"""
signal_processing/indicators.py

Provides functions for calculating common technical trading indicators.
"""

import pandas as pd

def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA) for a given series.

    Args:
        series: The input pandas Series (e.g., closing prices).
        window: The number of periods to include in the moving average.

    Returns:
        A pandas Series containing the SMA values.
    """
    return series.rolling(window=window).mean()


def relative_strength_index(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a given series.

    Args:
        series: The input pandas Series (e.g., closing prices).
        window: The number of periods to include in the RSI calculation.

    Returns:
        A pandas Series containing the RSI values.
    """
    # Ensure the series is of a numeric type
    series = series.astype(float)
    delta = series.diff()
    # Ensure delta is numeric to avoid type issues with comparisons
    delta = pd.to_numeric(delta, errors='coerce')

    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)

    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def moving_average_convergence_divergence(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD) and Signal Line.

    Args:
        series: The input pandas Series (e.g., closing prices).
        fast_period: The number of periods for the fast EMA.
        slow_period: The number of periods for the slow EMA.
        signal_period: The number of periods for the signal line EMA of the MACD.

    Returns:
        A pandas DataFrame with 'MACD' and 'Signal_Line' columns.
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()

    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()

    return pd.DataFrame({'MACD': macd, 'Signal_Line': signal_line})


def bollinger_bands(series: pd.Series, window: int = 20, num_std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculates Bollinger Bands for a given series.

    Args:
        series: The input pandas Series (e.g., closing prices).
        window: The number of periods for the moving average and standard deviation.
        num_std_dev: The number of standard deviations to use for the bands.

    Returns:
        A pandas DataFrame with 'Middle_Band', 'Upper_Band', and 'Lower_Band' columns.
    """
    middle_band = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    upper_band = middle_band + (rolling_std * num_std_dev)
    lower_band = middle_band - (rolling_std * num_std_dev)

    return pd.DataFrame({
        'Middle_Band': middle_band,
        'Upper_Band': upper_band,
        'Lower_Band': lower_band
    })
