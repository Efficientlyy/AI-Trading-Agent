"""
Rust-accelerated feature engineering module.

This module provides Python wrappers for feature engineering functions implemented in Rust.
"""
import numpy as np
from typing import List, Union, Optional, Dict
from decimal import Decimal

# Import our lag features implementation
from ..features.lag_features import (
    lag_feature,
    diff_feature,
    pct_change_feature,
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    RUST_AVAILABLE
)
import pandas as pd
import logging
import traceback
import sys
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

# Try to import Rust extensions, but provide robust fallback if not available
_create_lag_features_rs_rust = None
_create_rolling_window_features_rs_rust = None

RUST_FEATURES_AVAILABLE = False # This module's flag for Rust feature availability

try:
    from ai_trading_agent_rs import create_lag_features_rs as _create_lag_features_rs_rust
    from ai_trading_agent_rs import create_rolling_window_features_rs as _create_rolling_window_features_rs_rust

    # Check if the primary functions intended for use by this module are callable
    if callable(_create_lag_features_rs_rust) and callable(_create_rolling_window_features_rs_rust):
        RUST_FEATURES_AVAILABLE = True
        logger.info("Successfully imported callable Rust functions (create_lag_features_rs, create_rolling_window_features_rs) from ai_trading_agent_rs.")
    else:
        logger.warning("ai_trading_agent_rs module imported, but required Rust functions are not callable. Falling back to Python for feature engineering.")
        # Ensure placeholders are None if they weren't properly imported or aren't callable
        _create_lag_features_rs_rust = None
        _create_rolling_window_features_rs_rust = None

except ImportError as e:
    logger.warning(f"Failed to import from ai_trading_agent_rs. Rust extensions not available for feature engineering. Error: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred while importing from ai_trading_agent_rs: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    logger.warning("Falling back to Python implementations for Rust feature engineering.")


def create_lag_feature(series: Union[List[float], np.ndarray], lag: int) -> List[float]:
    """
    Create lag feature from a time series.
    
    Args:
        series: Input time series
        lag: Lag period
        
    Returns:
        List of lagged values
    """
    try:
        # Convert to list if numpy array
        if isinstance(series, np.ndarray):
            series = series.tolist()
        
        # Use our lag_feature implementation
        return lag_feature(series, lag)
    except Exception as e:
        logging.error(f"Error in create_lag_feature: {e}")
        logging.error(traceback.format_exc())
        return [np.nan] * len(series)


def create_lag_features(series: Union[List[Union[float, Decimal]], np.ndarray, pd.Series], lags: List[int]) -> np.ndarray:
    """
    Create lag features from a time series.
    
    Args:
        series: Input time series as a list, numpy array, or pandas Series
        lags: List of lag periods
        
    Returns:
        numpy.ndarray: 2D array with each column representing a lag feature
        
    Raises:
        ValueError: If series is empty or lags is empty
    """
    if not isinstance(series, (list, np.ndarray, pd.Series)):
        raise TypeError("series must be a list, numpy array, or pandas Series")
    # Input validation
    if not isinstance(lags, list) or len(lags) == 0:
        raise ValueError("lags must be a non-empty list of integers")
    
    # Convert series to numpy array if it's not already
    if isinstance(series, pd.Series):
        series_array = series.values
    elif isinstance(series, list):
        # Convert any Decimal values to float for numpy compatibility
        float_series = [float(x) if isinstance(x, Decimal) else x for x in series]
        series_array = np.array(float_series)
    else:
        series_array = series
        
    if len(series_array) == 0:
        raise ValueError("series must be non-empty")
    
    # Try to use Rust implementation
    if RUST_FEATURES_AVAILABLE:
        try:
            return _create_lag_features_rs_rust(series_array, lags)
        except Exception as e:
            logger.warning(f"Rust extension failed for lag features: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            logger.warning("Using Python implementation as fallback.")
    
    # Python fallback implementation
    n_samples = len(series_array)
    n_features = len(lags)
    result = np.full((n_samples, n_features), np.nan)
    
    for i, lag in enumerate(lags):
        if lag <= 0:
            raise ValueError(f"Lag periods must be positive integers, got {lag}")
        
        # For each lag, shift the series and store in the result array
        for j in range(n_samples):
            if j >= lag:
                result[j, i] = series_array[j - lag]
    
    return result


def create_diff_feature(series: Union[List[float], np.ndarray], period: int) -> List[float]:
    """
    Create difference feature from a time series.
    
    Args:
        series: Input time series
        period: Period for calculating difference
        
    Returns:
        List of difference values
    """
    try:
        # Convert to list if numpy array
        if isinstance(series, np.ndarray):
            series = series.tolist()
        
        # Use our diff_feature implementation
        return diff_feature(series, period)
    except Exception as e:
        logging.error(f"Error in create_diff_feature: {e}")
        logging.error(traceback.format_exc())
        return [np.nan] * len(series)


def create_diff_features(series: Union[List[Union[float, Decimal]], np.ndarray, pd.Series], periods: List[int]) -> np.ndarray:
    """
    Create difference features from a time series.
    
    Args:
        series: Input time series as a list, numpy array, or pandas Series
        periods: List of periods for calculating differences
        
    Returns:
        numpy.ndarray: 2D array with each column representing a difference feature
        
    Raises:
        ValueError: If series is empty or periods is empty
    """
    if not isinstance(series, (list, np.ndarray, pd.Series)):
        raise TypeError("series must be a list, numpy array, or pandas Series")
    # Input validation
    if not isinstance(periods, list) or len(periods) == 0:
        raise ValueError("periods must be a non-empty list of integers")
    
    # Convert series to numpy array if it's not already
    if isinstance(series, pd.Series):
        series_array = series.values
    elif isinstance(series, list):
        # Convert any Decimal values to float for numpy compatibility
        float_series = [float(x) if isinstance(x, Decimal) else x for x in series]
        series_array = np.array(float_series)
    else:
        series_array = series
        
    if len(series_array) == 0:
        raise ValueError("series must be non-empty")
    
    # Convert to list for our implementation
    series_list = series_array.tolist()
    
    # Create features using our implementation
    features_list = []
    for period in periods:
        if period <= 0:
            raise ValueError(f"Periods must be positive integers, got {period}")
        features_list.append(diff_feature(series_list, period))
    
    # Convert to numpy array
    n_samples = len(series_array)
    n_features = len(periods)
    result = np.full((n_samples, n_features), np.nan)
    
    for i, feature in enumerate(features_list):
        for j in range(n_samples):
            if j >= periods[i]:
                result[j, i] = feature[j]
    
    return result


def create_pct_change_feature(series: Union[List[float], np.ndarray], period: int) -> List[float]:
    """
    Create percentage change feature from a time series.
    
    Args:
        series: Input time series
        period: Period for calculating percentage change
        
    Returns:
        List of percentage change values
    """
    try:
        # Convert to list if numpy array
        if isinstance(series, np.ndarray):
            series = series.tolist()
        
        # Use our pct_change_feature implementation
        return pct_change_feature(series, period)
    except Exception as e:
        logging.error(f"Error in create_pct_change_feature: {e}")
        logging.error(traceback.format_exc())
        return [np.nan] * len(series)


def create_pct_change_features(series: Union[List[Union[float, Decimal]], np.ndarray, pd.Series], periods: List[int]) -> np.ndarray:
    """
    Create percentage change features from a time series.
    
    Args:
        series: Input time series as a list, numpy array, or pandas Series
        periods: List of periods for calculating percentage changes
        
    Returns:
        numpy.ndarray: 2D array with each column representing a percentage change feature
        
    Raises:
        ValueError: If series is empty or periods is empty
    """
    if not isinstance(series, (list, np.ndarray, pd.Series)):
        raise TypeError("series must be a list, numpy array, or pandas Series")
    # Input validation
    if not isinstance(periods, list) or len(periods) == 0:
        raise ValueError("periods must be a non-empty list of integers")
    
    # Convert series to numpy array if it's not already
    if isinstance(series, pd.Series):
        series_array = series.values
    elif isinstance(series, list):
        # Convert any Decimal values to float for numpy compatibility
        float_series = [float(x) if isinstance(x, Decimal) else x for x in series]
        series_array = np.array(float_series)
    else:
        series_array = series
        
    if len(series_array) == 0:
        raise ValueError("series must be non-empty")
    
    # Convert to list for our implementation
    series_list = series_array.tolist()
    
    # Create features using our implementation
    features_list = []
    for period in periods:
        if period <= 0:
            raise ValueError(f"Periods must be positive integers, got {period}")
        features_list.append(pct_change_feature(series_list, period))
    
    # Convert to numpy array
    n_samples = len(series_array)
    n_features = len(periods)
    result = np.full((n_samples, n_features), np.nan)
    
    for i, feature in enumerate(features_list):
        for j in range(n_samples):
            if j >= periods[i]:
                result[j, i] = feature[j]
    
    return result


def create_rolling_window_features(
    series: Union[List[Union[float, Decimal]], np.ndarray, pd.Series], 
    window_sizes: List[int],
    feature_type: str = 'mean'
) -> np.ndarray:
    """
    Create rolling window features from a time series.
    
    Args:
        series: Input time series as a list, numpy array, or pandas Series
        window_sizes: List of window sizes for rolling calculations
        feature_type: Type of feature to calculate ('min', 'max', 'mean', 'std', 'sum')
        
    Returns:
        numpy.ndarray: 2D array with each column representing a rolling window feature
        
    Raises:
        ValueError: If series is empty, window_sizes is empty, or feature_type is invalid
    """
    # Input validation
    if not isinstance(window_sizes, list) or len(window_sizes) == 0:
        raise ValueError("window_sizes must be a non-empty list of integers")
    
    valid_feature_types = ['min', 'max', 'mean', 'std', 'sum']
    if feature_type not in valid_feature_types:
        raise ValueError(f"feature_type must be one of {valid_feature_types}, got {feature_type}")
    
    # Convert series to numpy array if it's not already
    if isinstance(series, pd.Series):
        series_array = series.values
    elif isinstance(series, list):
        # Convert any Decimal values to float for numpy compatibility
        float_series = [float(x) if isinstance(x, Decimal) else x for x in series]
        series_array = np.array(float_series)
    else:
        series_array = series
        
    if len(series_array) == 0:
        raise ValueError("series must be non-empty")
    
    # Try to use Rust implementation
    if RUST_FEATURES_AVAILABLE:
        try:
            return _create_rolling_window_features_rs_rust(series_array, window_sizes, feature_type)
        except Exception as e:
            logger.warning(f"Rust extension failed for rolling {feature_type} features: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            logger.warning("Using Python implementation as fallback.")
    
    # Python fallback implementation
    n_samples = len(series_array)
    n_features = len(window_sizes)
    result = np.full((n_samples, n_features), np.nan)
    
    for i, window in enumerate(window_sizes):
        if window <= 0:
            raise ValueError(f"Window sizes must be positive integers, got {window}")
        
        # For each window size, calculate the rolling feature and store in the result array
        for j in range(n_samples):
            if j >= window - 1:
                window_values = series_array[j - window + 1:j + 1]
                
                if feature_type == 'min':
                    result[j, i] = np.min(window_values)
                elif feature_type == 'max':
                    result[j, i] = np.max(window_values)
                elif feature_type == 'mean':
                    result[j, i] = np.mean(window_values)
                elif feature_type == 'sum':
                    result[j, i] = np.sum(window_values)
                elif feature_type == 'std':
                    result[j, i] = np.std(window_values, ddof=0)  # Population standard deviation
    
    return result


def create_feature_matrix(
    series: Union[List[Union[float, Decimal]], np.ndarray, pd.Series],
    lag_periods: Optional[List[int]] = None,
    diff_periods: Optional[List[int]] = None,
    pct_change_periods: Optional[List[int]] = None,
    rolling_windows: Optional[Dict[str, List[int]]] = None
) -> np.ndarray:
    """
    Create a comprehensive feature matrix from a time series.
    
    Args:
        series: Input time series as a list, numpy array, or pandas Series
        lag_periods: List of lag periods for creating lag features
        diff_periods: List of periods for calculating differences
        pct_change_periods: List of periods for calculating percentage changes
        rolling_windows: Dictionary mapping feature types to window sizes
            e.g., {'mean': [5, 10], 'std': [5, 10]}
        
    Returns:
        numpy.ndarray: 2D array with each column representing a feature
        
    Raises:
        ValueError: If series is empty or no feature types are specified
    """
    # Convert series to numpy array if it's not already
    if isinstance(series, pd.Series):
        series_array = series.values
    elif isinstance(series, list):
        # Convert any Decimal values to float for numpy compatibility
        float_series = [float(x) if isinstance(x, Decimal) else x for x in series]
        series_array = np.array(float_series)
    else:
        series_array = series
        
    if len(series_array) == 0:
        raise ValueError("series must be non-empty")
    
    # Check if at least one feature type is specified
    if not any([lag_periods, diff_periods, pct_change_periods, rolling_windows]):
        raise ValueError("At least one feature type must be specified")
    
    # Initialize list to store feature matrices
    feature_matrices = []
    
    # Add lag features if specified
    if lag_periods:
        lag_features = create_lag_features(series_array, lag_periods)
        feature_matrices.append(lag_features)
    
    # Add difference features if specified
    if diff_periods:
        diff_features = create_diff_features(series_array, diff_periods)
        feature_matrices.append(diff_features)
    
    # Add percentage change features if specified
    if pct_change_periods:
        pct_change_features = create_pct_change_features(series_array, pct_change_periods)
        feature_matrices.append(pct_change_features)
    
    # Add rolling window features if specified
    if rolling_windows:
        for feature_type, window_sizes in rolling_windows.items():
            rolling_features = create_rolling_window_features(series_array, window_sizes, feature_type)
            feature_matrices.append(rolling_features)
    
    # Concatenate all feature matrices horizontally
    if len(feature_matrices) == 1:
        return feature_matrices[0]
    else:
        return np.hstack(feature_matrices)


def create_lag_features_df(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for specified columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of column names to create lags for
        lags: List of lag periods
        
    Returns:
        pandas.DataFrame: DataFrame with original columns and lag features
        
    Raises:
        ValueError: If df is empty, columns is empty, or lags is empty
        KeyError: If any column in columns is not in df
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
        
    if not columns:
        raise ValueError("columns list cannot be empty")
        
    if not lags:
        raise ValueError("lags list cannot be empty")
    
    # Check if all columns exist in df
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
    
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Create lag features for each column
    for col in columns:
        lag_features = create_lag_features(df[col].values, lags)
        
        # Add lag features to result DataFrame
        for i, lag in enumerate(lags):
            result_df[f"{col}_lag_{lag}"] = lag_features[:, i]
    
    return result_df
