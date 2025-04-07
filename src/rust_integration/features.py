"""
Rust-accelerated feature engineering module.

This module provides Python wrappers for feature engineering functions implemented in Rust.
"""
import numpy as np
from typing import Union, List, Optional, Tuple
import pandas as pd

try:
    from ai_trading_agent_rs import rust_extensions
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust extensions not available. Falling back to Python implementations.")


def create_lag_features(series: Union[List[float], np.ndarray, pd.Series], lags: List[int]) -> np.ndarray:
    """
    Create lag features from a time series.
    
    Args:
        series: Input time series as a list, numpy array, or pandas Series
        lags: List of lag periods
        
    Returns:
        numpy.ndarray: 2D array with each column representing a lag feature
        
    Raises:
        ValueError: If series is empty or lags is empty
        TypeError: If series is not a list, numpy array, or pandas Series
    """
    # Handle pandas Series
    if isinstance(series, pd.Series):
        series = series.values
    
    if not isinstance(series, (list, np.ndarray)):
        raise TypeError("series must be a list, numpy array, or pandas Series")
    
    if len(series) == 0:
        raise ValueError("series cannot be empty")
        
    if not lags:
        raise ValueError("lags list cannot be empty")
    
    # Convert to numpy array if needed
    if isinstance(series, list):
        series = np.array(series, dtype=float)
    
    if RUST_AVAILABLE:
        # Use Rust implementation
        return rust_extensions.create_lag_features_rs(series, lags)
    else:
        # Fallback to Python implementation
        n_samples = len(series)
        n_lags = len(lags)
        result = np.zeros((n_samples, n_lags))
        
        for i, lag in enumerate(lags):
            for j in range(n_samples):
                if j >= lag:
                    result[j, i] = series[j - lag]
                else:
                    result[j, i] = np.nan
                    
        return result


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
