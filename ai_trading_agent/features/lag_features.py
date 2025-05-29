"""
Lag features for time series analysis.

This module provides functions for creating lag, difference, and percentage change features
for time series data. It attempts to use the Rust implementation for better performance,
but falls back to pure Python implementations if the Rust extension is not available.
"""

import logging
import numpy as np
from typing import List, Union, Optional

# This module now provides pure Python implementations.
# Rust-accelerated versions are handled by ai_trading_agent.rust_integration.features
RUST_AVAILABLE = False 
logging.info("ai_trading_agent.features.lag_features is using Python implementations. For Rust acceleration, use functions from ai_trading_agent.rust_integration.features.")

# Python fallback implementations
def py_create_lag_feature(series: List[float], lag: int) -> List[float]:
    """
    Create lag feature from a time series (Python implementation).
    
    Args:
        series: Input time series as a list of floats
        lag: Lag period
        
    Returns:
        List of lagged values
    """
    if not series:
        raise ValueError("series must be non-empty")
    if lag <= 0:
        raise ValueError("lag must be greater than 0")
    
    result = [float('nan')] * len(series)
    for i in range(lag, len(series)):
        result[i] = series[i - lag]
    
    return result

def py_create_diff_feature(series: List[float], period: int) -> List[float]:
    """
    Create difference feature from a time series (Python implementation).
    
    Args:
        series: Input time series as a list of floats
        period: Period for calculating difference
        
    Returns:
        List of difference values
    """
    if not series:
        raise ValueError("series must be non-empty")
    if period <= 0:
        raise ValueError("period must be greater than 0")
    
    result = [float('nan')] * len(series)
    for i in range(period, len(series)):
        result[i] = series[i] - series[i - period]
    
    return result

def py_create_pct_change_feature(series: List[float], period: int) -> List[float]:
    """
    Create percentage change feature from a time series (Python implementation).
    
    Args:
        series: Input time series as a list of floats
        period: Period for calculating percentage change
        
    Returns:
        List of percentage change values
    """
    if not series:
        raise ValueError("series must be non-empty")
    if period <= 0:
        raise ValueError("period must be greater than 0")
    
    result = [float('nan')] * len(series)
    for i in range(period, len(series)):
        if series[i - period] != 0:
            result[i] = (series[i] - series[i - period]) / series[i - period]
    
    return result

# Use Rust implementation if available, otherwise use Python fallback
if RUST_AVAILABLE:
    # This block should ideally not be reached if RUST_AVAILABLE is hardcoded to False.
    # Keeping for structural consistency but it's effectively dead code now.
    try:
        # This import would fail if RUST_AVAILABLE was True and rust_test_extension didn't exist
        from rust_test_extension import create_lag_feature as rust_create_lag_feature
        from rust_test_extension import create_diff_feature as rust_create_diff_feature
        from rust_test_extension import create_pct_change_feature as rust_create_pct_change_feature
        lag_feature = rust_create_lag_feature
        diff_feature = rust_create_diff_feature
        pct_change_feature = rust_create_pct_change_feature
        logging.info("This line in lag_features.py should not be logged if RUST_AVAILABLE is False.")
    except ImportError:
        # Fallback to Python if the specific import fails even if RUST_AVAILABLE was hypothetically True
        lag_feature = py_create_lag_feature
        diff_feature = py_create_diff_feature
        pct_change_feature = py_create_pct_change_feature
else:
    lag_feature = py_create_lag_feature
    diff_feature = py_create_diff_feature
    pct_change_feature = py_create_pct_change_feature

# Convenience functions that handle numpy arrays and pandas Series
def create_lag_features(series: Union[List[float], np.ndarray], lags: List[int]) -> List[List[float]]:
    """
    Create multiple lag features from a time series.
    
    Args:
        series: Input time series
        lags: List of lag periods
        
    Returns:
        List of lists, where each inner list is a lag feature
    """
    if isinstance(series, np.ndarray):
        series = series.tolist()
    
    return [lag_feature(series, lag) for lag in lags]

def create_diff_features(series: Union[List[float], np.ndarray], periods: List[int]) -> List[List[float]]:
    """
    Create multiple difference features from a time series.
    
    Args:
        series: Input time series
        periods: List of periods for calculating differences
        
    Returns:
        List of lists, where each inner list is a difference feature
    """
    if isinstance(series, np.ndarray):
        series = series.tolist()
    
    return [diff_feature(series, period) for period in periods]

def create_pct_change_features(series: Union[List[float], np.ndarray], periods: List[int]) -> List[List[float]]:
    """
    Create multiple percentage change features from a time series.
    
    Args:
        series: Input time series
        periods: List of periods for calculating percentage changes
        
    Returns:
        List of lists, where each inner list is a percentage change feature
    """
    if isinstance(series, np.ndarray):
        series = series.tolist()
    
    return [pct_change_feature(series, period) for period in periods]
