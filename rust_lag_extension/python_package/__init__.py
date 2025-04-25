"""
Rust-accelerated lag features for time series analysis.

This module provides high-performance implementations of time series feature engineering
functions using Rust.
"""

# Import the Rust extension module
try:
    from rust_lag_extension import (
        create_lag_feature,
        create_diff_feature,
        create_pct_change_feature
    )
    RUST_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"Failed to import Rust extension: {e}")
    RUST_AVAILABLE = False

# Define Python fallback functions
import numpy as np

def py_create_lag_feature(series, lag):
    """
    Create lag feature from a time series (Python fallback).
    
    Args:
        series: Input time series as a list or numpy array of floats
        lag: Lag period
        
    Returns:
        List of lagged values
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series)
    
    result = np.full_like(series, np.nan)
    if lag > 0 and len(series) > lag:
        result[lag:] = series[:-lag]
    
    return result.tolist()

def py_create_diff_feature(series, period):
    """
    Create difference feature from a time series (Python fallback).
    
    Args:
        series: Input time series as a list or numpy array of floats
        period: Period for calculating difference
        
    Returns:
        List of difference values
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series)
    
    result = np.full_like(series, np.nan)
    if period > 0 and len(series) > period:
        result[period:] = series[period:] - series[:-period]
    
    return result.tolist()

def py_create_pct_change_feature(series, period):
    """
    Create percentage change feature from a time series (Python fallback).
    
    Args:
        series: Input time series as a list or numpy array of floats
        period: Period for calculating percentage change
        
    Returns:
        List of percentage change values
    """
    if not isinstance(series, np.ndarray):
        series = np.array(series)
    
    result = np.full_like(series, np.nan)
    if period > 0 and len(series) > period:
        prev_values = series[:-period]
        # Avoid division by zero
        mask = prev_values != 0
        if np.any(mask):
            indices = np.arange(period, len(series))[mask[period-len(series):]]
            result[indices] = (series[indices] - series[indices - period]) / series[indices - period]
    
    return result.tolist()

# Use Rust functions if available, otherwise use Python fallbacks
if RUST_AVAILABLE:
    lag_feature = create_lag_feature
    diff_feature = create_diff_feature
    pct_change_feature = create_pct_change_feature
else:
    lag_feature = py_create_lag_feature
    diff_feature = py_create_diff_feature
    pct_change_feature = py_create_pct_change_feature
