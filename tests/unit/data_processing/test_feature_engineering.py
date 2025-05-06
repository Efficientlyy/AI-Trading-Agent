"""
Unit tests for feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Assuming indicators are tested separately, we can mock them if needed,
# but for simplicity, let's use the real ones here.
from ai_trading_agent.data_processing.feature_engineering import (
    add_lagged_features,
    add_technical_indicators,
    scale_features
)
from ai_trading_agent.data_processing.indicators import calculate_sma # Import one for simple testing

# Sample DataFrame for testing
@pytest.fixture
def sample_ohlcv_data():
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = {
        'open': np.linspace(50, 55, 20),
        'high': np.linspace(51, 57, 20),
        'low': np.linspace(49, 54, 20),
        'close': np.linspace(50.5, 56, 20),
        'volume': np.linspace(1000, 2000, 20).astype(int)
    }
    return pd.DataFrame(data, index=dates)

# --- Test add_lagged_features --- 
def test_add_lagged_features_single(sample_ohlcv_data):
    df_lagged = add_lagged_features(sample_ohlcv_data, columns=['close'], lags=[1, 3])
    assert 'close_lag_1' in df_lagged.columns
    assert 'close_lag_3' in df_lagged.columns
    assert df_lagged['close_lag_1'].isnull().sum() == 1
    assert df_lagged['close_lag_3'].isnull().sum() == 3
    # Check values
    assert_series_equal(df_lagged['close_lag_1'].iloc[1:], sample_ohlcv_data['close'].iloc[:-1], check_names=False, check_index=False)
    assert_series_equal(df_lagged['close_lag_3'].iloc[3:], sample_ohlcv_data['close'].iloc[:-3], check_names=False, check_index=False)

def test_add_lagged_features_multiple(sample_ohlcv_data):
    df_lagged = add_lagged_features(sample_ohlcv_data, columns=['close', 'volume'], lags=[1])
    assert 'close_lag_1' in df_lagged.columns
    assert 'volume_lag_1' in df_lagged.columns
    assert_series_equal(df_lagged['close_lag_1'].iloc[1:], sample_ohlcv_data['close'].iloc[:-1], check_names=False, check_index=False)
    # Convert volume to float for comparison if necessary, as shift might change dtype if NaNs are introduced
    assert_series_equal(df_lagged['volume_lag_1'].iloc[1:], sample_ohlcv_data['volume'].astype(float).iloc[:-1], check_names=False, check_index=False)

def test_add_lagged_features_edge_cases(sample_ohlcv_data):
    # Empty lags list
    df_lagged_empty_lags = add_lagged_features(sample_ohlcv_data, columns=['close'], lags=[])
    assert_frame_equal(df_lagged_empty_lags, sample_ohlcv_data)
    # Non-positive lags
    df_lagged_bad_lags = add_lagged_features(sample_ohlcv_data, columns=['close'], lags=[1, 0, -1])
    assert 'close_lag_1' in df_lagged_bad_lags.columns
    assert 'close_lag_0' not in df_lagged_bad_lags.columns
    assert 'close_lag_-1' not in df_lagged_bad_lags.columns
    # Column not found
    df_lagged_bad_col = add_lagged_features(sample_ohlcv_data, columns=['nonexistent'], lags=[1])
    assert 'nonexistent_lag_1' not in df_lagged_bad_col.columns
    assert_frame_equal(df_lagged_bad_col, sample_ohlcv_data)
    # Empty DataFrame
    empty_df = pd.DataFrame()
    df_lagged_empty_df = add_lagged_features(empty_df, columns=['close'], lags=[1])
    assert_frame_equal(df_lagged_empty_df, empty_df)

# --- Test add_technical_indicators --- 
def test_add_technical_indicators_default(sample_ohlcv_data):
    df_indic = add_technical_indicators(sample_ohlcv_data)
    # Check if default indicators are added
    assert 'SMA_10' in df_indic.columns
    assert 'SMA_50' in df_indic.columns # Will be all NaN as window > data length
    assert 'EMA_10' in df_indic.columns
    assert 'EMA_50' in df_indic.columns # Will be all NaN
    assert 'RSI_14' in df_indic.columns
    assert 'MACD' in df_indic.columns
    assert 'Signal' in df_indic.columns
    assert 'Histogram' in df_indic.columns
    assert df_indic['SMA_50'].isnull().all()

def test_add_technical_indicators_custom_config(sample_ohlcv_data):
    config = {
        'sma': {'windows': [5]},
        'rsi': {'window': 7}
    }
    df_indic = add_technical_indicators(sample_ohlcv_data, indicators_config=config)
    assert 'SMA_5' in df_indic.columns
    assert 'RSI_7' in df_indic.columns
    # Check defaults were NOT added
    assert 'SMA_10' not in df_indic.columns
    assert 'EMA_10' not in df_indic.columns
    assert 'MACD' not in df_indic.columns

def test_add_technical_indicators_missing_close(sample_ohlcv_data):
    df_no_close = sample_ohlcv_data.drop(columns=['close'])
    df_indic = add_technical_indicators(df_no_close)
    # Should return original df without indicators
    assert_frame_equal(df_indic, df_no_close)

def test_add_technical_indicators_custom_close_col(sample_ohlcv_data):
    df_renamed = sample_ohlcv_data.rename(columns={'close': 'price'})
    config = {'sma': {'windows': [5]}}
    df_indic = add_technical_indicators(df_renamed, close_col='price', indicators_config=config)
    assert 'SMA_5' in df_indic.columns
    assert not df_indic['SMA_5'].isnull().all()

# --- Test scale_features --- 
def test_scale_features_minmax(sample_ohlcv_data):
    df_to_scale = sample_ohlcv_data.copy()
    # Add a simple indicator column without NaNs for scaling
    df_to_scale['SMA_5'] = calculate_sma(df_to_scale['close'], 5)
    df_to_scale = df_to_scale.dropna() # Drop initial NaNs
    cols = ['close', 'volume', 'SMA_5']
    
    df_scaled, scaler = scale_features(df_to_scale, cols, scaler_type='minmax')

    assert isinstance(scaler, MinMaxScaler)
    assert 'close' in df_scaled.columns
    assert 'volume' in df_scaled.columns
    assert 'SMA_5' in df_scaled.columns
    # Check bounds (MinMax should be between 0 and 1)
    for col in cols:
        assert df_scaled[col].min() >= 0.0
        assert np.isclose(df_scaled[col].max(), 1.0)
        # Check one value is exactly 0 and one is 1 (unless data is constant)
        if df_to_scale[col].nunique() > 1:
            assert np.isclose(df_scaled[col].min(), 0.0)
            assert np.isclose(df_scaled[col].max(), 1.0)

def test_scale_features_standard(sample_ohlcv_data):
    df_to_scale = sample_ohlcv_data.copy()
    df_to_scale['SMA_5'] = calculate_sma(df_to_scale['close'], 5)
    df_to_scale = df_to_scale.dropna()
    cols = ['close', 'volume', 'SMA_5']

    df_scaled, scaler = scale_features(df_to_scale, cols, scaler_type='standard')

    assert isinstance(scaler, StandardScaler)
    # Check mean approx 0 and std dev approx 1
    for col in cols:
        assert np.isclose(df_scaled[col].mean(), 0.0, atol=1e-8)
        assert np.isclose(df_scaled[col].std(ddof=0), 1.0, atol=1e-8)

def test_scale_features_edge_cases(sample_ohlcv_data):
    df_to_scale = sample_ohlcv_data.copy()
    # Non-existent column
    df_scaled_bad_col, scaler_bad_col = scale_features(df_to_scale, ['nonexistent'])
    assert scaler_bad_col is None
    assert_frame_equal(df_scaled_bad_col, df_to_scale)
    # Empty column list
    df_scaled_empty_cols, scaler_empty_cols = scale_features(df_to_scale, [])
    assert scaler_empty_cols is None
    assert_frame_equal(df_scaled_empty_cols, df_to_scale)
    # Invalid scaler type
    with pytest.raises(ValueError):
        scale_features(df_to_scale, ['close'], scaler_type='invalid')
    # Data with NaNs
    df_with_nan = df_to_scale.copy()
    df_with_nan.loc[df_with_nan.index[0], 'close'] = np.nan
    df_scaled_nan, scaler_nan = scale_features(df_with_nan, ['close'])
    assert scaler_nan is None # Scaler should not be fitted
    assert_frame_equal(df_scaled_nan, df_with_nan) # Should return original data
