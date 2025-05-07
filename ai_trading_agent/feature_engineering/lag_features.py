"""
Lag Features Module

This module provides high-performance implementations of lag features
using Rust extensions for maximum performance.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any
import logging

# Import Rust extension
try:
    from ai_trading_agent.rust_extensions import (
        create_lag_features_rs,
        create_return_features_rs,
        create_rolling_features_rs
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust extensions not available for lag features. Using slower Python implementations.")

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
        lags: List of lag periods to create
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
            df = pd.DataFrame(index=index)
            
            for i, lag in enumerate(lags):
                # Extract lag values
                lag_values = [row[i] for row in result]
                
                # Add to DataFrame
                df[f'lag_{lag}'] = lag_values
            
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


def create_return_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    periods: List[int],
    log_returns: bool = True,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create return features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        periods: List of periods for calculating returns
        log_returns: Whether to calculate log returns (True) or simple returns (False)
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with return features as columns
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
            result = create_return_features_rs(series_values, periods, log_returns)
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            for i, period in enumerate(periods):
                # Extract return values
                return_values = [row[i] for row in result]
                
                # Add to DataFrame
                df[f'return_{period}'] = return_values
            
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
        
        # Calculate return features
        for period in periods:
            if period <= 0:
                raise ValueError(f"Return periods must be positive integers, got {period}")
            
            if log_returns:
                # Log returns: ln(P_t / P_{t-period})
                df[f'return_{period}'] = np.log(series / series.shift(period))
            else:
                # Simple returns: (P_t / P_{t-period}) - 1
                df[f'return_{period}'] = (series / series.shift(period)) - 1
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create return features")


def create_rolling_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    windows: List[int],
    functions: List[str] = ['mean', 'std', 'min', 'max'],
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create rolling window features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        windows: List of window sizes for rolling calculations
        functions: List of functions to apply to rolling windows
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with rolling features as columns
    """
    # Input validation
    if not windows:
        raise ValueError("windows must be a non-empty list of integers")
    
    if not functions:
        raise ValueError("functions must be a non-empty list of strings")
    
    # Validate functions
    valid_functions = ['mean', 'std', 'min', 'max', 'median', 'sum', 'var', 'skew', 'kurt']
    for func in functions:
        if func not in valid_functions:
            raise ValueError(f"Invalid function: {func}. Valid functions are: {valid_functions}")
    
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
            result = create_rolling_features_rs(series_values, windows, functions)
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            for i, window in enumerate(windows):
                for j, func in enumerate(functions):
                    # Extract feature values
                    feature_values = [row[i][j] for row in result]
                    
                    # Add to DataFrame
                    df[f'rolling_{window}_{func}'] = feature_values
            
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
        
        # Calculate rolling features
        for window in windows:
            if window <= 0:
                raise ValueError(f"Window sizes must be positive integers, got {window}")
            
            rolling = series.rolling(window=window)
            
            for func in functions:
                if func == 'mean':
                    df[f'rolling_{window}_mean'] = rolling.mean()
                elif func == 'std':
                    df[f'rolling_{window}_std'] = rolling.std()
                elif func == 'min':
                    df[f'rolling_{window}_min'] = rolling.min()
                elif func == 'max':
                    df[f'rolling_{window}_max'] = rolling.max()
                elif func == 'median':
                    df[f'rolling_{window}_median'] = rolling.median()
                elif func == 'sum':
                    df[f'rolling_{window}_sum'] = rolling.sum()
                elif func == 'var':
                    df[f'rolling_{window}_var'] = rolling.var()
                elif func == 'skew':
                    df[f'rolling_{window}_skew'] = rolling.skew()
                elif func == 'kurt':
                    df[f'rolling_{window}_kurt'] = rolling.kurt()
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create rolling features")


def create_all_lag_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    lags: Optional[List[int]] = None,
    return_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    rolling_functions: List[str] = ['mean', 'std'],
    log_returns: bool = True,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create all types of lag features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        lags: List of lag periods to create
        return_periods: List of periods for calculating returns
        rolling_windows: List of window sizes for rolling calculations
        rolling_functions: List of functions to apply to rolling windows
        log_returns: Whether to calculate log returns (True) or simple returns (False)
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with all lag features as columns
    """
    # Set default values if not provided
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    
    if return_periods is None:
        return_periods = [1, 2, 5, 10]
    
    if rolling_windows is None:
        rolling_windows = [5, 10, 20, 50]
    
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
    
    # Create return features
    if return_periods:
        return_df = create_return_features(series, return_periods, log_returns, fallback_to_python)
        result_df = pd.concat([result_df, return_df], axis=1)
    
    # Create rolling features
    if rolling_windows and rolling_functions:
        rolling_df = create_rolling_features(series, rolling_windows, rolling_functions, fallback_to_python)
        result_df = pd.concat([result_df, rolling_df], axis=1)
    
    return result_df


# Example usage
if __name__ == "__main__":
    # Generate sample data
    import numpy as np
    
    # Create a sample price series
    np.random.seed(42)
    n_samples = 1000
    price = 100.0
    prices = [price]
    
    for _ in range(1, n_samples):
        change_pct = np.random.normal(0, 0.01)
        price *= (1 + change_pct)
        prices.append(price)
    
    # Convert to pandas Series
    price_series = pd.Series(prices)
    
    # Create lag features
    lag_df = create_lag_features(price_series, lags=[1, 5, 10])
    
    # Create return features
    return_df = create_return_features(price_series, periods=[1, 5, 10], log_returns=True)
    
    # Create rolling features
    rolling_df = create_rolling_features(
        price_series, 
        windows=[5, 10, 20], 
        functions=['mean', 'std', 'min', 'max']
    )
    
    # Create all features at once
    all_features = create_all_lag_features(
        price_series,
        lags=[1, 5, 10],
        return_periods=[1, 5, 10],
        rolling_windows=[5, 10, 20]
    )
    
    # Print results
    print("Lag features:")
    print(lag_df.head())
    
    print("\nReturn features:")
    print(return_df.head())
    
    print("\nRolling features:")
    print(rolling_df.head())
    
    print("\nAll features:")
    print(all_features.head())
