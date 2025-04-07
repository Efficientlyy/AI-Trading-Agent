"""
Functions and classes for feature engineering based on market data and indicators.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.common import logger
from .indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd

def add_lagged_features(df: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
    """
    Adds lagged versions of specified columns to the DataFrame.

    Args:
        df: Input DataFrame (should have a DatetimeIndex).
        columns: List of column names to create lagged features for.
        lags: List of integers representing the lag periods (e.g., [1, 2, 3]).

    Returns:
        DataFrame with added lagged features. Original columns are preserved.
    """
    df_out = df.copy()
    for col in columns:
        if col not in df_out.columns:
            logger.warning(f"Column '{col}' not found in DataFrame. Skipping lags for it.")
            continue
        for lag in lags:
            if lag <= 0:
                logger.warning(f"Lag value {lag} is not positive. Skipping.")
                continue
            df_out[f'{col}_lag_{lag}'] = df_out[col].shift(lag)
    return df_out

def add_technical_indicators(df: pd.DataFrame, close_col: str = 'close', indicators_config: dict = None) -> pd.DataFrame:
    """
    Adds various technical indicators to the DataFrame.

    Args:
        df: Input DataFrame with OHLCV data.
        close_col: Name of the column containing close prices (default: 'close').
        indicators_config: Dictionary specifying which indicators to add and their params.
            Example: {
                'sma': {'windows': [10, 20]},
                'ema': {'windows': [10, 20]},
                'rsi': {'window': 14},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9}
            }

    Returns:
        DataFrame with added indicator columns.
    """
    if indicators_config is None:
        indicators_config = {
            'sma': {'windows': [10, 50]},
            'ema': {'windows': [10, 50]},
            'rsi': {'window': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9}
        }

    df_out = df.copy()
    if close_col not in df_out.columns:
        logger.error(f"Close column '{close_col}' not found in DataFrame. Cannot calculate indicators.")
        return df_out

    close_prices = df_out[close_col]

    # SMA
    if 'sma' in indicators_config and 'windows' in indicators_config['sma']:
        for window in indicators_config['sma']['windows']:
            df_out[f'SMA_{window}'] = calculate_sma(close_prices, window)

    # EMA
    if 'ema' in indicators_config and 'windows' in indicators_config['ema']:
        for window in indicators_config['ema']['windows']:
            df_out[f'EMA_{window}'] = calculate_ema(close_prices, window)

    # RSI
    if 'rsi' in indicators_config and 'window' in indicators_config['rsi']:
        window = indicators_config['rsi']['window']
        df_out[f'RSI_{window}'] = calculate_rsi(close_prices, window)

    # MACD
    if 'macd' in indicators_config:
        params = indicators_config['macd']
        macd_df = calculate_macd(close_prices,
                                 fast_period=params.get('fast', 12),
                                 slow_period=params.get('slow', 26),
                                 signal_period=params.get('signal', 9))
        df_out[['MACD', 'Signal', 'Histogram']] = macd_df

    return df_out

def scale_features(df: pd.DataFrame, columns_to_scale: list, scaler_type: str = 'minmax') -> (pd.DataFrame, object):
    """
    Scales specified features using MinMaxScaler or StandardScaler.

    Args:
        df: Input DataFrame.
        columns_to_scale: List of column names to scale.
        scaler_type: Type of scaler ('minmax' or 'standard'). Default is 'minmax'.

    Returns:
        Tuple containing:
        - DataFrame with scaled features (original columns replaced).
        - Fitted scaler object (can be used to inverse transform later).
    """
    df_out = df.copy()
    valid_columns = [col for col in columns_to_scale if col in df_out.columns]

    if not valid_columns:
        logger.warning("No valid columns found to scale.")
        return df_out, None

    data_to_scale = df_out[valid_columns].values

    # Handle potential NaNs before scaling (e.g., fill or use imputation)
    # Simple approach: check for NaNs
    if np.isnan(data_to_scale).any():
        logger.warning(f"NaN values detected in columns {valid_columns} before scaling. Consider handling NaNs (e.g., imputation or dropping). Returning unscaled data.")
        # Returning original df and no scaler as scaling would fail/propagate NaNs
        return df, None 

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler_type. Choose 'minmax' or 'standard'.")

    scaled_data = scaler.fit_transform(data_to_scale)
    df_out[valid_columns] = scaled_data

    return df_out, scaler


# --- Example Usage --- 
if __name__ == '__main__':
    # Create sample data (reuse from indicators.py example)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close_prices = pd.Series(np.random.randn(100).cumsum() + 50, index=dates)
    ohlcv_data = pd.DataFrame({
        'open': close_prices - np.random.rand(100) * 0.5,
        'high': close_prices + np.random.rand(100) * 0.5,
        'low': close_prices - np.random.rand(100) * 0.5,
        'close': close_prices,
        'volume': np.random.randint(1000, 5000, 100).astype(float) # Ensure volume is float
    }, index=dates)

    print("--- Original OHLCV ---")
    print(ohlcv_data.head())

    # 1. Add indicators
    df_with_indicators = add_technical_indicators(ohlcv_data)
    print("\n--- With Indicators ---")
    print(df_with_indicators.tail())

    # 2. Add lagged features (e.g., lag close and volume)
    features_to_lag = ['close', 'volume', 'RSI_14']
    lags = [1, 2, 3]
    df_with_lags = add_lagged_features(df_with_indicators, features_to_lag, lags)
    print("\n--- With Lags ---")
    print(df_with_lags.tail())

    # 3. Scale some features
    # First, drop NaNs resulting from indicators/lags for scaling example
    df_cleaned = df_with_lags.dropna()
    features_to_scale = ['close', 'volume', 'RSI_14', 'MACD', 'Histogram', 'close_lag_1']
    if not df_cleaned.empty:
        df_scaled, scaler_obj = scale_features(df_cleaned, features_to_scale, scaler_type='minmax')
        print("\n--- Scaled Features (MinMax) ---")
        print(df_scaled[features_to_scale].tail())

        # Example inverse transform (only works if scaler_obj is not None)
        # inverse_scaled = scaler_obj.inverse_transform(df_scaled[features_to_scale])
        # print("\n--- Inverse Scaled ---")
        # print(pd.DataFrame(inverse_scaled, columns=features_to_scale, index=df_scaled.index).tail())
    else:
        print("\n--- DataFrame empty after dropping NaNs, skipping scaling example ---")
