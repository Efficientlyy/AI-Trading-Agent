"""
Time series analysis module for sentiment data.

This module provides functions for analyzing sentiment data over time,
including lag features, rolling statistics, and other time series transformations.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple

from src.rust_integration.features import (
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    create_rolling_window_features,
    create_feature_matrix
)

# Set up logging
logger = logging.getLogger(__name__)


class SentimentTimeSeriesAnalyzer:
    """
    Analyzer for sentiment time series data.
    
    This class provides methods for analyzing sentiment data over time,
    including creating lag features, calculating rolling statistics,
    and detecting trends and anomalies in sentiment data.
    """
    
    def __init__(self, default_lags: Optional[List[int]] = None, default_windows: Optional[List[int]] = None):
        """
        Initialize the SentimentTimeSeriesAnalyzer.
        
        Args:
            default_lags: Default lag periods to use for lag features
            default_windows: Default window sizes to use for rolling statistics
        """
        self.default_lags = default_lags or [1, 2, 3, 5, 10, 21]
        self.default_windows = default_windows or [5, 10, 21, 63]
        
    def create_sentiment_features(
        self, 
        sentiment_data: pd.DataFrame,
        sentiment_columns: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
        include_diff: bool = True,
        include_pct_change: bool = True,
        include_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Create time series features from sentiment data.
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            sentiment_columns: List of column names containing sentiment scores
                If None, will try to use ['compound', 'positive', 'negative', 'neutral']
            lags: List of lag periods to use
            windows: List of window sizes to use for rolling statistics
            include_diff: Whether to include difference features
            include_pct_change: Whether to include percentage change features
            include_rolling: Whether to include rolling window features
            
        Returns:
            DataFrame with original data and added time series features
        """
        if sentiment_data.empty:
            logger.warning("Empty sentiment data provided")
            return sentiment_data
        
        # Use default parameters if not provided
        lags = lags or self.default_lags
        windows = windows or self.default_windows
        
        # Determine sentiment columns to process
        if sentiment_columns is None:
            default_columns = ['compound', 'positive', 'negative', 'neutral']
            sentiment_columns = [col for col in default_columns if col in sentiment_data.columns]
            
        if not sentiment_columns:
            logger.warning("No sentiment columns found in data")
            return sentiment_data
        
        # Make a copy of the input data
        result_df = sentiment_data.copy()
        
        # Process each sentiment column
        for col in sentiment_columns:
            if col not in sentiment_data.columns:
                logger.warning(f"Column {col} not found in sentiment data")
                continue
                
            # Get the sentiment series
            series = sentiment_data[col].values
            
            # Create lag features
            for lag in lags:
                lag_features = create_lag_features(series, [lag])
                result_df[f'{col}_lag_{lag}'] = lag_features[:, 0]
            
            # Create difference features
            if include_diff:
                for period in lags:
                    diff_features = create_diff_features(series, [period])
                    result_df[f'{col}_diff_{period}'] = diff_features[:, 0]
            
            # Create percentage change features
            if include_pct_change:
                for period in lags:
                    pct_features = create_pct_change_features(series, [period])
                    result_df[f'{col}_pct_{period}'] = pct_features[:, 0]
            
            # Create rolling window features
            if include_rolling:
                # Rolling mean
                for window in windows:
                    mean_features = create_rolling_window_features(series, [window], 'mean')
                    result_df[f'{col}_mean_{window}'] = mean_features[:, 0]
                
                # Rolling standard deviation (volatility)
                for window in windows:
                    std_features = create_rolling_window_features(series, [window], 'std')
                    result_df[f'{col}_std_{window}'] = std_features[:, 0]
        
        return result_df
    
    def detect_sentiment_trends(
        self,
        sentiment_data: pd.DataFrame,
        sentiment_column: str = 'compound',
        window_size: int = 10,
        threshold: float = 0.1
    ) -> pd.Series:
        """
        Detect trends in sentiment data.
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            sentiment_column: Column name containing sentiment scores
            window_size: Window size for trend detection
            threshold: Threshold for trend detection
            
        Returns:
            Series with trend indicators (1 for uptrend, -1 for downtrend, 0 for no trend)
        """
        if sentiment_column not in sentiment_data.columns:
            raise ValueError(f"Column {sentiment_column} not found in sentiment data")
        
        # Calculate rolling mean
        rolling_mean = create_rolling_window_features(
            sentiment_data[sentiment_column].values,
            [window_size],
            'mean'
        )[:, 0]
        
        # Calculate trend
        trend = np.zeros(len(sentiment_data))
        
        for i in range(window_size, len(sentiment_data)):
            # Current value vs rolling mean
            if sentiment_data[sentiment_column].iloc[i] > rolling_mean[i] + threshold:
                trend[i] = 1  # Uptrend
            elif sentiment_data[sentiment_column].iloc[i] < rolling_mean[i] - threshold:
                trend[i] = -1  # Downtrend
        
        return pd.Series(trend, index=sentiment_data.index)
    
    def detect_sentiment_anomalies(
        self,
        sentiment_data: pd.DataFrame,
        sentiment_column: str = 'compound',
        window_size: int = 21,
        std_threshold: float = 2.0
    ) -> pd.Series:
        """
        Detect anomalies in sentiment data.
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            sentiment_column: Column name containing sentiment scores
            window_size: Window size for anomaly detection
            std_threshold: Number of standard deviations for anomaly detection
            
        Returns:
            Series with anomaly indicators (1 for positive anomaly, -1 for negative anomaly, 0 for no anomaly)
        """
        if sentiment_column not in sentiment_data.columns:
            raise ValueError(f"Column {sentiment_column} not found in sentiment data")
        
        # Calculate rolling mean and standard deviation
        rolling_features = np.column_stack([
            create_rolling_window_features(
                sentiment_data[sentiment_column].values,
                [window_size],
                'mean'
            ),
            create_rolling_window_features(
                sentiment_data[sentiment_column].values,
                [window_size],
                'std'
            )
        ])
        
        rolling_mean = rolling_features[:, 0]
        rolling_std = rolling_features[:, 1]
        
        # Calculate anomalies
        anomalies = np.zeros(len(sentiment_data))
        
        for i in range(window_size, len(sentiment_data)):
            # Check if value is outside the expected range
            upper_bound = rolling_mean[i] + std_threshold * rolling_std[i]
            lower_bound = rolling_mean[i] - std_threshold * rolling_std[i]
            
            if sentiment_data[sentiment_column].iloc[i] > upper_bound:
                anomalies[i] = 1  # Positive anomaly
            elif sentiment_data[sentiment_column].iloc[i] < lower_bound:
                anomalies[i] = -1  # Negative anomaly
        
        return pd.Series(anomalies, index=sentiment_data.index)
    
    def calculate_sentiment_momentum(
        self,
        sentiment_data: pd.DataFrame,
        sentiment_column: str = 'compound',
        short_window: int = 5,
        long_window: int = 21
    ) -> pd.Series:
        """
        Calculate sentiment momentum.
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            sentiment_column: Column name containing sentiment scores
            short_window: Short window size for momentum calculation
            long_window: Long window size for momentum calculation
            
        Returns:
            Series with momentum values
        """
        if sentiment_column not in sentiment_data.columns:
            raise ValueError(f"Column {sentiment_column} not found in sentiment data")
        
        # Calculate short and long term moving averages
        rolling_features = np.column_stack([
            create_rolling_window_features(
                sentiment_data[sentiment_column].values,
                [short_window],
                'mean'
            ),
            create_rolling_window_features(
                sentiment_data[sentiment_column].values,
                [long_window],
                'mean'
            )
        ])
        
        short_ma = rolling_features[:, 0]
        long_ma = rolling_features[:, 1]
        
        # Calculate momentum (difference between short and long term moving averages)
        momentum = short_ma - long_ma
        
        return pd.Series(momentum, index=sentiment_data.index)
    
    def calculate_sentiment_volatility(
        self,
        sentiment_data: pd.DataFrame,
        sentiment_column: str = 'compound',
        window_size: int = 21
    ) -> pd.Series:
        """
        Calculate sentiment volatility.
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            sentiment_column: Column name containing sentiment scores
            window_size: Window size for volatility calculation
            
        Returns:
            Series with volatility values
        """
        if sentiment_column not in sentiment_data.columns:
            raise ValueError(f"Column {sentiment_column} not found in sentiment data")
        
        # Calculate rolling standard deviation (volatility)
        volatility = create_rolling_window_features(
            sentiment_data[sentiment_column].values,
            [window_size],
            'std'
        )[:, 0]
        
        return pd.Series(volatility, index=sentiment_data.index)
    
    def calculate_sentiment_rate_of_change(
        self,
        sentiment_data: pd.DataFrame,
        sentiment_column: str = 'compound',
        period: int = 5
    ) -> pd.Series:
        """
        Calculate sentiment rate of change.
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            sentiment_column: Column name containing sentiment scores
            period: Period for rate of change calculation
            
        Returns:
            Series with rate of change values
        """
        if sentiment_column not in sentiment_data.columns:
            raise ValueError(f"Column {sentiment_column} not found in sentiment data")
        
        # Calculate percentage change
        pct_change = create_pct_change_features(
            sentiment_data[sentiment_column].values,
            [period]
        )[:, 0]
        
        return pd.Series(pct_change, index=sentiment_data.index)
    
    def calculate_sentiment_acceleration(
        self,
        sentiment_data: pd.DataFrame,
        sentiment_column: str = 'compound',
        period: int = 5
    ) -> pd.Series:
        """
        Calculate sentiment acceleration (change in rate of change).
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            sentiment_column: Column name containing sentiment scores
            period: Period for acceleration calculation
            
        Returns:
            Series with acceleration values
        """
        # Calculate rate of change
        roc = self.calculate_sentiment_rate_of_change(
            sentiment_data,
            sentiment_column,
            period
        ).values
        
        # Calculate change in rate of change (acceleration)
        acceleration = create_diff_features(roc, [period])[:, 0]
        
        return pd.Series(acceleration, index=sentiment_data.index)
