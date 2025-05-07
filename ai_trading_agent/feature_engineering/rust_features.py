"""
Rust-accelerated Feature Engineering Module

This module provides high-performance feature engineering functions implemented in Rust.
These functions are significantly faster than their Python equivalents, especially for
large datasets.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional
import logging

# Import Rust extension
try:
    from ai_trading_agent.rust_extensions import (
        create_lag_features_rs,
        create_diff_features_rs,
        create_pct_change_features_rs,
        create_rolling_window_features_rs,
        create_ema_features_rs
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust extensions not available. Using slower Python implementations.")

logger = logging.getLogger(__name__)


def create_lag_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    lags: List[int],
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create lag features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        lags: List of lag periods (e.g., [1, 2, 5] for 1-period, 2-period, and 5-period lags)
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with lag features as columns
    """
    # Input validation
    if not lags:
        raise ValueError("lags must be a non-empty list of integers")
    
    # Convert input to list if needed
    if isinstance(series, pd.Series):
        series_values = series.values.tolist()
        index = series.index
    elif isinstance(series, np.ndarray):
        series_values = series.tolist()
        index = pd.RangeIndex(len(series))
    else:
        series_values = series
        index = pd.RangeIndex(len(series))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_lag_features_rs(series_values, lags)
            
            # Convert result to DataFrame
            df = pd.DataFrame(result, index=index)
            df.columns = [f'lag_{lag}' for lag in lags]
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        
        # Calculate lag features
        for lag in lags:
            if lag <= 0:
                raise ValueError(f"Lag periods must be positive integers, got {lag}")
            df[f'lag_{lag}'] = series.shift(lag)
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create lag features")


def create_diff_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    periods: List[int],
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create difference features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        periods: List of periods for calculating differences
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with difference features as columns
    """
    # Input validation
    if not periods:
        raise ValueError("periods must be a non-empty list of integers")
    
    # Convert input to list if needed
    if isinstance(series, pd.Series):
        series_values = series.values.tolist()
        index = series.index
    elif isinstance(series, np.ndarray):
        series_values = series.tolist()
        index = pd.RangeIndex(len(series))
    else:
        series_values = series
        index = pd.RangeIndex(len(series))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_diff_features_rs(series_values, periods)
            
            # Convert result to DataFrame
            df = pd.DataFrame(result, index=index)
            df.columns = [f'diff_{period}' for period in periods]
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        
        # Calculate difference features
        for period in periods:
            if period <= 0:
                raise ValueError(f"Periods must be positive integers, got {period}")
            df[f'diff_{period}'] = series.diff(period)
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create difference features")


def create_pct_change_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    periods: List[int],
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create percentage change features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        periods: List of periods for calculating percentage changes
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with percentage change features as columns
    """
    # Input validation
    if not periods:
        raise ValueError("periods must be a non-empty list of integers")
    
    # Convert input to list if needed
    if isinstance(series, pd.Series):
        series_values = series.values.tolist()
        index = series.index
    elif isinstance(series, np.ndarray):
        series_values = series.tolist()
        index = pd.RangeIndex(len(series))
    else:
        series_values = series
        index = pd.RangeIndex(len(series))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_pct_change_features_rs(series_values, periods)
            
            # Convert result to DataFrame
            df = pd.DataFrame(result, index=index)
            df.columns = [f'pct_change_{period}' for period in periods]
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        
        # Calculate percentage change features
        for period in periods:
            if period <= 0:
                raise ValueError(f"Periods must be positive integers, got {period}")
            df[f'pct_change_{period}'] = series.pct_change(period)
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create percentage change features")


def create_rolling_window_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    windows: List[int],
    feature_type: str = "mean",
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create rolling window features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        windows: List of window sizes
        feature_type: Type of feature to calculate (mean, std, min, max, sum)
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with rolling window features as columns
    """
    # Input validation
    if not windows:
        raise ValueError("windows must be a non-empty list of integers")
    
    # Map feature type to integer code
    feature_type_map = {
        "mean": 0,
        "std": 1,
        "min": 2,
        "max": 3,
        "sum": 4
    }
    
    if feature_type not in feature_type_map:
        raise ValueError(f"feature_type must be one of {list(feature_type_map.keys())}, got {feature_type}")
    
    feature_type_code = feature_type_map[feature_type]
    
    # Convert input to list if needed
    if isinstance(series, pd.Series):
        series_values = series.values.tolist()
        index = series.index
    elif isinstance(series, np.ndarray):
        series_values = series.tolist()
        index = pd.RangeIndex(len(series))
    else:
        series_values = series
        index = pd.RangeIndex(len(series))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_rolling_window_features_rs(series_values, windows, feature_type_code)
            
            # Convert result to DataFrame
            df = pd.DataFrame(result, index=index)
            df.columns = [f'rolling_{window}_{feature_type}' for window in windows]
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        
        # Calculate rolling window features
        for window in windows:
            if window <= 0:
                raise ValueError(f"Window sizes must be positive integers, got {window}")
            
            rolling = series.rolling(window=window)
            
            if feature_type == "mean":
                df[f'rolling_{window}_mean'] = rolling.mean()
            elif feature_type == "std":
                df[f'rolling_{window}_std'] = rolling.std()
            elif feature_type == "min":
                df[f'rolling_{window}_min'] = rolling.min()
            elif feature_type == "max":
                df[f'rolling_{window}_max'] = rolling.max()
            elif feature_type == "sum":
                df[f'rolling_{window}_sum'] = rolling.sum()
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create rolling window features")


def create_ema_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    spans: List[int],
    alpha: Optional[float] = None,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create exponential moving average (EMA) features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        spans: List of EMA spans (periods)
        alpha: Optional smoothing factor (if None, alpha = 2/(span+1))
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with EMA features as columns
    """
    # Input validation
    if not spans:
        raise ValueError("spans must be a non-empty list of integers")
    
    # Convert input to list if needed
    if isinstance(series, pd.Series):
        series_values = series.values.tolist()
        index = series.index
    elif isinstance(series, np.ndarray):
        series_values = series.tolist()
        index = pd.RangeIndex(len(series))
    else:
        series_values = series
        index = pd.RangeIndex(len(series))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_ema_features_rs(series_values, spans, alpha)
            
            # Convert result to DataFrame
            df = pd.DataFrame(result, index=index)
            df.columns = [f'ema_{span}' for span in spans]
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        
        # Calculate EMA features
        for span in spans:
            if span <= 0:
                raise ValueError(f"Span periods must be positive integers, got {span}")
            
            alpha_value = alpha if alpha is not None else 2.0 / (span + 1.0)
            
            if alpha_value <= 0.0 or alpha_value > 1.0:
                raise ValueError(f"Alpha must be in (0, 1], got {alpha_value}")
            
            df[f'ema_{span}'] = series.ewm(span=span, adjust=False).mean()
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create EMA features")


def create_all_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    lags: Optional[List[int]] = None,
    diff_periods: Optional[List[int]] = None,
    pct_change_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    rolling_feature_types: Optional[List[str]] = None,
    ema_spans: Optional[List[int]] = None,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create all types of features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        lags: List of lag periods
        diff_periods: List of periods for calculating differences
        pct_change_periods: List of periods for calculating percentage changes
        rolling_windows: List of window sizes for rolling window features
        rolling_feature_types: List of feature types for rolling window features
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with all features as columns
    """
    # Convert input to pandas Series if needed
    if isinstance(series, np.ndarray) or isinstance(series, list):
        series = pd.Series(series)
    
    # Create DataFrame to store results
    result_df = pd.DataFrame(index=series.index)
    
    # Add original series as a column
    result_df['original'] = series
    
    # Create lag features
    if lags:
        lag_df = create_lag_features(series, lags, fallback_to_python)
        result_df = pd.concat([result_df, lag_df], axis=1)
    
    # Create difference features
    if diff_periods:
        diff_df = create_diff_features(series, diff_periods, fallback_to_python)
        result_df = pd.concat([result_df, diff_df], axis=1)
    
    # Create percentage change features
    if pct_change_periods:
        pct_change_df = create_pct_change_features(series, pct_change_periods, fallback_to_python)
        result_df = pd.concat([result_df, pct_change_df], axis=1)
    
    # Create rolling window features
    if rolling_windows and rolling_feature_types:
        for feature_type in rolling_feature_types:
            rolling_df = create_rolling_window_features(
                series, rolling_windows, feature_type, fallback_to_python
            )
            result_df = pd.concat([result_df, rolling_df], axis=1)
    
    # Create EMA features
    if ema_spans:
        ema_df = create_ema_features(series, ema_spans, fallback_to_python=fallback_to_python)
        result_df = pd.concat([result_df, ema_df], axis=1)
    
    return result_df
