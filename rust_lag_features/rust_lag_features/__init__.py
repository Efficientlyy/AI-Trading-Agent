"""
Rust-accelerated lag features for time series analysis.

This module provides Python wrappers for feature engineering functions implemented in Rust.
"""

# Import the Rust extension module
try:
    from rust_lag_features import (
        create_lag_features,
        create_diff_features,
        create_pct_change_features
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    
    # Python fallback implementations if Rust extension is not available
    import numpy as np
    from typing import List, Union
    import pandas as pd
    
    def create_lag_features(series, lags):
        """
        Create lag features from a time series.
        
        Args:
            series: Input time series as a list, numpy array, or pandas Series
            lags: List of lag periods
            
        Returns:
            List of lists: Each inner list represents a lag feature
        """
        # Convert series to list if it's not already
        if isinstance(series, np.ndarray):
            series_list = series.tolist()
        elif isinstance(series, pd.Series):
            series_list = series.tolist()
        else:
            series_list = list(series)
            
        n_samples = len(series_list)
        n_features = len(lags)
        result = [[None for _ in range(n_features)] for _ in range(n_samples)]
        
        for i, lag in enumerate(lags):
            for j in range(n_samples):
                if j >= lag:
                    result[j][i] = series_list[j - lag]
                    
        return result
    
    def create_diff_features(series, periods):
        """
        Create difference features from a time series.
        
        Args:
            series: Input time series as a list, numpy array, or pandas Series
            periods: List of periods for calculating differences
            
        Returns:
            List of lists: Each inner list represents a difference feature
        """
        # Convert series to list if it's not already
        if isinstance(series, np.ndarray):
            series_list = series.tolist()
        elif isinstance(series, pd.Series):
            series_list = series.tolist()
        else:
            series_list = list(series)
            
        n_samples = len(series_list)
        n_features = len(periods)
        result = [[None for _ in range(n_features)] for _ in range(n_samples)]
        
        for i, period in enumerate(periods):
            for j in range(n_samples):
                if j >= period:
                    result[j][i] = series_list[j] - series_list[j - period]
                    
        return result
    
    def create_pct_change_features(series, periods):
        """
        Create percentage change features from a time series.
        
        Args:
            series: Input time series as a list, numpy array, or pandas Series
            periods: List of periods for calculating percentage changes
            
        Returns:
            List of lists: Each inner list represents a percentage change feature
        """
        # Convert series to list if it's not already
        if isinstance(series, np.ndarray):
            series_list = series.tolist()
        elif isinstance(series, pd.Series):
            series_list = series.tolist()
        else:
            series_list = list(series)
            
        n_samples = len(series_list)
        n_features = len(periods)
        result = [[None for _ in range(n_features)] for _ in range(n_samples)]
        
        for i, period in enumerate(periods):
            for j in range(n_samples):
                if j >= period:
                    previous_value = series_list[j - period]
                    if previous_value != 0:
                        result[j][i] = (series_list[j] - previous_value) / previous_value
                    
        return result
