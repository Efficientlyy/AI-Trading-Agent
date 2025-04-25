"""
Sentiment analyzer for trading signals.

This module provides a sentiment analyzer that fetches sentiment data from Alpha Vantage,
processes it using NLP techniques, and generates trading signals based on sentiment analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
from datetime import datetime, timedelta

from ai_trading_agent.data_sources.alpha_vantage_client import AlphaVantageClient
from ai_trading_agent.nlp_processing.sentiment_processor import SentimentProcessor
from ai_trading_agent.rust_integration.features import (
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    create_rolling_window_features,
    create_feature_matrix
)

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analyzer for trading signals.
    
    This class fetches sentiment data from Alpha Vantage, processes it using NLP techniques,
    and generates trading signals based on sentiment analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary for the sentiment analyzer
        """
        self.config = config or {}
        
        # Initialize Alpha Vantage client with advanced configuration
        alpha_vantage_config = self.config.get("alpha_vantage_client", {})
        api_key = alpha_vantage_config.get("api_key") or self.config.get("alpha_vantage_api_key")
        tier = alpha_vantage_config.get("tier", "free")
        
        self.alpha_vantage_client = AlphaVantageClient(api_key=api_key, tier=tier)
        
        # Configure caching for the Alpha Vantage client
        cache_ttl = alpha_vantage_config.get("cache_ttl", 3600)  # Default 1 hour
        use_cache = alpha_vantage_config.get("use_cache", True)
        
        self.alpha_vantage_client.cache_ttl = cache_ttl
        self.alpha_vantage_client.use_cache = use_cache
        
        # Initialize sentiment processor
        self.sentiment_processor = SentimentProcessor()
        
        # Default configuration values
        self.default_lags = self.config.get("default_lags", [1, 2, 3, 5, 10, 21])
        self.default_windows = self.config.get("default_windows", [5, 10, 21, 63])
        self.sentiment_threshold = self.config.get("sentiment_threshold", Decimal("0.2"))
        self.sentiment_window = self.config.get("sentiment_window", 5)
        
        # Feature weights
        self.feature_weights = self.config.get("feature_weights", {
            "sentiment_score": Decimal("1.0"),
            "sentiment_trend": Decimal("0.8"),
            "sentiment_momentum": Decimal("0.7"),
            "sentiment_anomaly": Decimal("0.5"),
            "sentiment_volatility": Decimal("-0.3"),
            "sentiment_roc": Decimal("0.6"),
            "sentiment_acceleration": Decimal("0.4")
        })
        
        # Fallback topics to use when specific queries fail
        self.fallback_topics = self.config.get("fallback_topics", [
            "blockchain", "cryptocurrency", "finance", "technology"
        ])
        
        # Update the Alpha Vantage client's fallback topics
        self.alpha_vantage_client.DEFAULT_FALLBACK_TOPICS = self.fallback_topics
    
    def fetch_sentiment_data(self, 
                            topic: Optional[str] = None,
                            crypto_ticker: Optional[str] = None,
                            days_back: int = 7) -> pd.DataFrame:
        """
        Fetch sentiment data from Alpha Vantage.
        
        Args:
            topic: Topic to fetch sentiment for (e.g., "blockchain", "cryptocurrency")
            crypto_ticker: Cryptocurrency ticker (e.g., "BTC", "ETH")
            days_back: Number of days to look back for news
            
        Returns:
            DataFrame containing sentiment data
        """
        if topic:
            response = self.alpha_vantage_client.get_sentiment_by_topic(topic, days_back)
        elif crypto_ticker:
            response = self.alpha_vantage_client.get_sentiment_by_crypto(crypto_ticker, days_back)
        else:
            logger.error("Either topic or crypto_ticker must be provided")
            return pd.DataFrame()
        
        if response.get("error"):
            logger.error(f"Error fetching sentiment data: {response['error']}")
            return pd.DataFrame()
        
        # Extract sentiment scores
        sentiment_data = self.alpha_vantage_client.extract_sentiment_scores(response)
        
        if not sentiment_data:
            logger.warning("No sentiment data extracted")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(sentiment_data)
        
        # Convert time_published to datetime
        if 'time_published' in df.columns:
            df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
            df = df.sort_values('time_published')
        
        # Process text data if available
        if 'summary' in df.columns:
            # Prepare data for sentiment processor
            raw_data = [{'text': summary} for summary in df['summary'].tolist()]
            
            # Process with VADER
            processed_data = self.sentiment_processor.process_data(raw_data)
            
            # Add VADER sentiment scores to DataFrame
            df['vader_sentiment_score'] = [entry.get('sentiment_score', 0) for entry in processed_data]
        
        return df
    
    def generate_time_series_features(self, df: pd.DataFrame, 
                                     sentiment_column: str = 'overall_sentiment_score',
                                     lags: Optional[List[int]] = None,
                                     windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Generate time series features from sentiment data.
        
        Args:
            df: DataFrame containing sentiment data
            sentiment_column: Column containing sentiment scores
            lags: List of lag periods for creating lag features
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with added time series features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot generate features")
            return df
        
        if sentiment_column not in df.columns:
            logger.error(f"Sentiment column '{sentiment_column}' not found in DataFrame")
            return df
        
        # Use default values if not provided
        lags = lags or self.default_lags
        windows = windows or self.default_windows
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Extract sentiment series
        sentiment_series = result_df[sentiment_column].values
        
        try:
            # Generate lag features
            lag_features = create_lag_features(sentiment_series, lags)
            for i, lag in enumerate(lags):
                result_df[f'{sentiment_column}_lag_{lag}'] = lag_features[:, i]
            
            # Generate difference features
            diff_features = create_diff_features(sentiment_series, lags)
            for i, lag in enumerate(lags):
                result_df[f'{sentiment_column}_diff_{lag}'] = diff_features[:, i]
            
            # Generate percentage change features
            pct_change_features = create_pct_change_features(sentiment_series, lags)
            for i, lag in enumerate(lags):
                result_df[f'{sentiment_column}_pct_change_{lag}'] = pct_change_features[:, i]
            
            # Generate rolling mean features
            rolling_mean_features = create_rolling_window_features(sentiment_series, windows, 'mean')
            for i, window in enumerate(windows):
                result_df[f'{sentiment_column}_rolling_mean_{window}'] = rolling_mean_features[:, i]
            
            # Generate rolling standard deviation features
            rolling_std_features = create_rolling_window_features(sentiment_series, windows, 'std')
            for i, window in enumerate(windows):
                result_df[f'{sentiment_column}_rolling_std_{window}'] = rolling_std_features[:, i]
            
            # Calculate sentiment momentum (difference between short and long-term moving averages)
            short_window = min(windows)
            long_window = max(windows)
            result_df['sentiment_momentum'] = (
                result_df[f'{sentiment_column}_rolling_mean_{short_window}'] - 
                result_df[f'{sentiment_column}_rolling_mean_{long_window}']
            )
            
            # Calculate sentiment trend
            result_df['sentiment_trend'] = 0
            # Uptrend: short-term average > long-term average
            result_df.loc[result_df['sentiment_momentum'] > self.sentiment_threshold, 'sentiment_trend'] = 1
            # Downtrend: short-term average < long-term average
            result_df.loc[result_df['sentiment_momentum'] < -self.sentiment_threshold, 'sentiment_trend'] = -1
            
            # Calculate sentiment volatility (normalized)
            if f'{sentiment_column}_rolling_std_{self.sentiment_window}' in result_df.columns:
                mean_std = result_df[f'{sentiment_column}_rolling_std_{self.sentiment_window}'].mean()
                if mean_std > 0:
                    result_df['sentiment_volatility'] = (
                        result_df[f'{sentiment_column}_rolling_std_{self.sentiment_window}'] / mean_std
                    )
                else:
                    result_df['sentiment_volatility'] = 1.0
            
            # Calculate rate of change (ROC)
            if f'{sentiment_column}_pct_change_{self.sentiment_window}' in result_df.columns:
                result_df['sentiment_roc'] = result_df[f'{sentiment_column}_pct_change_{self.sentiment_window}']
            
            # Calculate sentiment acceleration (change in momentum)
            if 'sentiment_momentum' in result_df.columns:
                result_df['sentiment_momentum_lag_1'] = result_df['sentiment_momentum'].shift(1)
                result_df['sentiment_acceleration'] = (
                    result_df['sentiment_momentum'] - result_df['sentiment_momentum_lag_1']
                )
            
            # Detect sentiment anomalies
            if 'sentiment_volatility' in result_df.columns:
                result_df['sentiment_anomaly'] = 0
                # Positive anomaly: sentiment > 2 standard deviations above mean
                result_df.loc[
                    result_df[sentiment_column] > (
                        result_df[f'{sentiment_column}_rolling_mean_{self.sentiment_window}'] + 
                        2 * result_df[f'{sentiment_column}_rolling_std_{self.sentiment_window}']
                    ),
                    'sentiment_anomaly'
                ] = 1
                # Negative anomaly: sentiment < 2 standard deviations below mean
                result_df.loc[
                    result_df[sentiment_column] < (
                        result_df[f'{sentiment_column}_rolling_mean_{self.sentiment_window}'] - 
                        2 * result_df[f'{sentiment_column}_rolling_std_{self.sentiment_window}']
                    ),
                    'sentiment_anomaly'
                ] = -1
            
        except Exception as e:
            logger.error(f"Error generating time series features: {str(e)}")
        
        return result_df
    
    def calculate_weighted_sentiment_score(self, df: pd.DataFrame,
                                          sentiment_column: str = 'overall_sentiment_score') -> pd.DataFrame:
        """
        Calculate weighted sentiment score from various features.
        
        Args:
            df: DataFrame containing sentiment data and features
            sentiment_column: Column containing sentiment scores
            
        Returns:
            DataFrame with added weighted_sentiment_score column
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot calculate weighted score")
            return df
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Initialize weighted score column with zeros
        result_df['weighted_sentiment_score'] = 0.0
        
        try:
            # Add base sentiment score
            if sentiment_column in result_df.columns:
                weight = float(self.feature_weights.get('sentiment_score', Decimal('1.0')))
                result_df['weighted_sentiment_score'] += weight * result_df[sentiment_column].astype(float)
            
            # Add sentiment trend
            if 'sentiment_trend' in result_df.columns:
                weight = float(self.feature_weights.get('sentiment_trend', Decimal('0.8')))
                result_df['weighted_sentiment_score'] += weight * result_df['sentiment_trend'].astype(float)
            
            # Add sentiment momentum
            if 'sentiment_momentum' in result_df.columns:
                weight = float(self.feature_weights.get('sentiment_momentum', Decimal('0.7')))
                result_df['weighted_sentiment_score'] += weight * result_df['sentiment_momentum'].astype(float)
            
            # Add sentiment anomaly
            if 'sentiment_anomaly' in result_df.columns:
                weight = float(self.feature_weights.get('sentiment_anomaly', Decimal('0.5')))
                result_df['weighted_sentiment_score'] += weight * result_df['sentiment_anomaly'].astype(float)
            
            # Add sentiment volatility (typically negative weight)
            if 'sentiment_volatility' in result_df.columns:
                weight = float(self.feature_weights.get('sentiment_volatility', Decimal('-0.3')))
                result_df['weighted_sentiment_score'] += weight * result_df['sentiment_volatility'].astype(float)
            
            # Add sentiment rate of change
            if 'sentiment_roc' in result_df.columns:
                weight = float(self.feature_weights.get('sentiment_roc', Decimal('0.6')))
                result_df['weighted_sentiment_score'] += weight * result_df['sentiment_roc'].astype(float)
            
            # Add sentiment acceleration
            if 'sentiment_acceleration' in result_df.columns:
                weight = float(self.feature_weights.get('sentiment_acceleration', Decimal('0.4')))
                result_df['weighted_sentiment_score'] += weight * result_df['sentiment_acceleration'].astype(float)
                
            # Convert the weighted score to Decimal for consistency
            result_df['weighted_sentiment_score'] = result_df['weighted_sentiment_score'].apply(lambda x: Decimal(str(x)))
            
        except Exception as e:
            logger.error(f"Error calculating weighted sentiment score: {str(e)}")
        
        return result_df
    
    def generate_trading_signals(self, df: pd.DataFrame,
                                sentiment_column: str = 'weighted_sentiment_score') -> pd.DataFrame:
        """
        Generate trading signals from sentiment data.
        
        Args:
            df: DataFrame containing sentiment data and features
            sentiment_column: Column containing sentiment scores to use for signal generation
            
        Returns:
            DataFrame with added signal column
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot generate signals")
            return df
        
        if sentiment_column not in df.columns:
            logger.error(f"Sentiment column '{sentiment_column}' not found in DataFrame")
            return df
        
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Initialize signal column
        result_df['signal'] = 0
        
        try:
            # Generate signals based on sentiment threshold
            # Long signal: sentiment > threshold
            result_df.loc[result_df[sentiment_column] > self.sentiment_threshold, 'signal'] = 1
            # Short signal: sentiment < -threshold
            result_df.loc[result_df[sentiment_column] < -self.sentiment_threshold, 'signal'] = -1
            
            # Adjust signals based on trend confirmation
            if 'sentiment_trend' in result_df.columns:
                # Reduce signal strength if trend contradicts
                trend_contradicts_long = (result_df['sentiment_trend'] < 0) & (result_df['signal'] > 0)
                trend_contradicts_short = (result_df['sentiment_trend'] > 0) & (result_df['signal'] < 0)
                
                # Reduce signal to 0.5 (partial position) if trend contradicts
                result_df.loc[trend_contradicts_long, 'signal'] = 0.5
                result_df.loc[trend_contradicts_short, 'signal'] = -0.5
            
            # Adjust signals based on volatility
            if 'sentiment_volatility' in result_df.columns:
                # Reduce signal strength in high volatility
                high_volatility = result_df['sentiment_volatility'] > 1.5
                
                # Reduce signal by 50% in high volatility
                result_df.loc[high_volatility, 'signal'] = result_df.loc[high_volatility, 'signal'] * 0.5
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
        
        return result_df
    
    def analyze_sentiment(self, 
                         topic: Optional[str] = None,
                         crypto_ticker: Optional[str] = None,
                         days_back: int = 7) -> pd.DataFrame:
        """
        Analyze sentiment and generate trading signals.
        
        This is the main method that combines all the steps:
        1. Fetch sentiment data from Alpha Vantage
        2. Generate time series features
        3. Calculate weighted sentiment score
        4. Generate trading signals
        
        Args:
            topic: Topic to fetch sentiment for (e.g., "blockchain", "cryptocurrency")
            crypto_ticker: Cryptocurrency ticker (e.g., "BTC", "ETH")
            days_back: Number of days to look back for news
            
        Returns:
            DataFrame containing sentiment data, features, and trading signals
        """
        # Fetch sentiment data
        df = self.fetch_sentiment_data(topic=topic, crypto_ticker=crypto_ticker, days_back=days_back)
        
        if df.empty:
            logger.warning("No sentiment data fetched, cannot analyze sentiment")
            return df
        
        # Generate time series features
        df = self.generate_time_series_features(df)
        
        # Calculate weighted sentiment score
        df = self.calculate_weighted_sentiment_score(df)
        
        # Generate trading signals
        df = self.generate_trading_signals(df)
        
        return df
