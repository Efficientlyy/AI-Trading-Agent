"""
Sentiment Analysis Service.

This module provides a unified service for the sentiment analysis system, coordinating
data collection, NLP processing, time series feature generation, and trading signal generation.
"""
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json

from ai_trading_agent.sentiment_analysis.data_collection import SentimentCollectionService
from ai_trading_agent.sentiment_analysis.nlp_processing import NLPPipeline
from ai_trading_agent.sentiment_analysis.strategy import DummySentimentStrategy
from ai_trading_agent.sentiment_analysis.time_series import SentimentTimeSeriesAnalyzer
from ..trading_engine.models import Order

logger = logging.getLogger(__name__)


class SentimentAnalysisService:
    """
    Service for coordinating the sentiment analysis system.
    
    This service manages the data collection, NLP processing, time series feature generation,
    and trading signal generation components of the sentiment analysis system, providing a
    unified interface for the trading engine.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment analysis service.
        
        Args:
            config: Configuration dictionary with settings for all components
        """
        self.config = config
        
        # Initialize components
        self.collection_service = SentimentCollectionService(config.get('data_collection', {}))
        self.nlp_processor = NLPPipeline(config.get('nlp_processing', {}))
        self.strategy = DummySentimentStrategy(config.get('strategy', {}))
        self.time_series_analyzer = SentimentTimeSeriesAnalyzer(config.get('time_series', {}))
        
        # Cache for sentiment data
        self.sentiment_cache = {}
        self.cache_expiry = config.get('cache_expiry_hours', 24)
        
        # Output directory for saving sentiment data
        self.output_dir = config.get('output_dir', 'data/sentiment')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Sentiment analysis service initialized")
        
    def collect_sentiment_data(self, symbols: List[str], start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        Collect sentiment data for the specified symbols and time range.
        
        Args:
            symbols: List of asset symbols to collect sentiment data for
            start_date: Start date for data collection
            end_date: End date for data collection
            force_refresh: Whether to force refresh the cache
            
        Returns:
            DataFrame with sentiment data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            start_date = end_date - timedelta(days=self.config.get('default_lookback_days', 30))
            
        # Check cache
        cache_key = f"{'-'.join(sorted(symbols))}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        if not force_refresh and cache_key in self.sentiment_cache:
            cache_entry = self.sentiment_cache[cache_key]
            cache_time = cache_entry['timestamp']
            cache_expiry = cache_time + timedelta(hours=self.cache_expiry)
            
            if datetime.now() < cache_expiry:
                logger.info(f"Using cached sentiment data for {symbols} from {start_date} to {end_date}")
                return cache_entry['data']
                
        # Collect sentiment data
        logger.info(f"Collecting sentiment data for {symbols} from {start_date} to {end_date}")
        sentiment_data = self.collection_service.collect_all(symbols, start_date, end_date)
        
        # Process sentiment data with NLP
        sentiment_data = self._process_sentiment_data(sentiment_data)
        
        # Save to cache
        self.sentiment_cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': sentiment_data
        }
        
        # Save to file
        self._save_sentiment_data(sentiment_data, symbols, start_date, end_date)
        
        return sentiment_data
    
    def _process_sentiment_data(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process sentiment data with NLP.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            
        Returns:
            DataFrame with processed sentiment data
        """
        if sentiment_data.empty:
            logger.warning("Empty sentiment data")
            return sentiment_data
        
        # Check if the data already has sentiment scores
        if 'compound' in sentiment_data.columns:
            logger.info("Sentiment data already processed")
            return sentiment_data
        
        # Process text with NLP
        if 'text' in sentiment_data.columns:
            logger.info("Processing sentiment data with NLP")
            
            # Preprocess text
            sentiment_data['processed_text'] = sentiment_data['text'].apply(
                lambda x: self.nlp_processor.preprocessor.preprocess(x) if isinstance(x, str) else ""
            )
            
            # Calculate sentiment scores
            sentiment_scores = sentiment_data['processed_text'].apply(
                lambda x: self.nlp_processor.sentiment_analyzer.analyze_sentiment(x) if x else {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
            )
            
            # Extract sentiment scores
            sentiment_data['compound'] = sentiment_scores.apply(lambda x: x['compound'])
            sentiment_data['positive'] = sentiment_scores.apply(lambda x: x['positive'])
            sentiment_data['negative'] = sentiment_scores.apply(lambda x: x['negative'])
            sentiment_data['neutral'] = sentiment_scores.apply(lambda x: x['neutral'])
            
            # Extract entities
            sentiment_data['entities'] = sentiment_data['processed_text'].apply(
                lambda x: self.nlp_processor.entity_extractor.extract_entities(x) if x else []
            )
        
        return sentiment_data
    
    def _save_sentiment_data(self, data: pd.DataFrame, symbols: List[str], 
                            start_date: datetime, end_date: datetime):
        """
        Save sentiment data to file.
        
        Args:
            data: DataFrame with sentiment data
            symbols: List of asset symbols
            start_date: Start date for data collection
            end_date: End date for data collection
        """
        if data.empty:
            logger.warning("No sentiment data to save")
            return
            
        # Create filename
        symbols_str = '-'.join(sorted(symbols))
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        filename = f"sentiment_{symbols_str}_{start_str}_{end_str}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to CSV
        try:
            data.to_csv(filepath, index=False)
            logger.info(f"Saved sentiment data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving sentiment data to {filepath}: {e}")
    
    def generate_trading_signals(self, market_data: pd.DataFrame, symbols: List[str],
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               force_refresh: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on sentiment analysis.
        
        Args:
            market_data: DataFrame with market data (OHLCV)
            symbols: List of asset symbols to generate signals for
            start_date: Start date for data collection
            end_date: End date for data collection
            force_refresh: Whether to force refresh the sentiment data cache
            
        Returns:
            DataFrame with trading signals
        """
        # Collect sentiment data
        sentiment_data = self.collect_sentiment_data(symbols, start_date, end_date, force_refresh)
        
        if sentiment_data.empty:
            logger.warning("No sentiment data available for signal generation")
            return pd.DataFrame()
            
        # Generate time series features
        features = self.generate_features(sentiment_data)
        
        # Generate signals
        signals = self.strategy.generate_signals(features)
        
        # Save signals to file
        self._save_signals(signals, symbols, start_date, end_date)
        
        return signals
    
    def generate_features(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from sentiment data.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            
        Returns:
            DataFrame with features
        """
        if sentiment_data.empty:
            logger.warning("Empty sentiment data for feature generation")
            return pd.DataFrame()
            
        logger.info("Generating time series features from sentiment data")
        
        # Create time series features
        features_df = self.time_series_analyzer.create_sentiment_features(
            sentiment_data,
            sentiment_columns=['compound', 'positive', 'negative', 'neutral'] if all(col in sentiment_data.columns for col in ['compound', 'positive', 'negative', 'neutral']) else ['compound'],
            lags=self.config.get('feature_lags', [1, 2, 3, 5, 10, 21]),
            windows=self.config.get('feature_windows', [5, 10, 21, 63]),
            include_diff=True,
            include_pct_change=True,
            include_rolling=True
        )
        
        # Add trend detection
        if 'compound' in sentiment_data.columns:
            try:
                features_df['sentiment_trend'] = self.time_series_analyzer.detect_sentiment_trends(
                    sentiment_data,
                    sentiment_column='compound',
                    window_size=self.config.get('trend_window', 10),
                    threshold=self.config.get('trend_threshold', 0.1)
                )
                
                # Add anomaly detection
                features_df['sentiment_anomaly'] = self.time_series_analyzer.detect_sentiment_anomalies(
                    sentiment_data,
                    sentiment_column='compound',
                    window_size=self.config.get('anomaly_window', 21),
                    std_threshold=self.config.get('anomaly_threshold', 2.0)
                )
                
                # Add momentum
                features_df['sentiment_momentum'] = self.time_series_analyzer.calculate_sentiment_momentum(
                    sentiment_data,
                    sentiment_column='compound',
                    short_window=self.config.get('momentum_short_window', 5),
                    long_window=self.config.get('momentum_long_window', 21)
                )
                
                # Add volatility
                features_df['sentiment_volatility'] = self.time_series_analyzer.calculate_sentiment_volatility(
                    sentiment_data,
                    sentiment_column='compound',
                    window_size=self.config.get('volatility_window', 21)
                )
                
                # Add rate of change
                features_df['sentiment_roc'] = self.time_series_analyzer.calculate_sentiment_rate_of_change(
                    sentiment_data,
                    sentiment_column='compound',
                    period=self.config.get('roc_period', 5)
                )
                
                # Add acceleration
                features_df['sentiment_acceleration'] = self.time_series_analyzer.calculate_sentiment_acceleration(
                    sentiment_data,
                    sentiment_column='compound',
                    period=self.config.get('acceleration_period', 5)
                )
            except Exception as e:
                logger.error(f"Error generating advanced sentiment features: {e}")
                logger.debug("Continuing with basic features only")
        
        logger.info(f"Generated {features_df.shape[1] - sentiment_data.shape[1]} new features")
        return features_df
    
    def _save_signals(self, signals: pd.DataFrame, symbols: List[str],
                     start_date: Optional[datetime], end_date: Optional[datetime]):
        """
        Save trading signals to file.
        
        Args:
            signals: DataFrame with trading signals
            symbols: List of asset symbols
            start_date: Start date for data collection
            end_date: End date for data collection
        """
        if signals.empty:
            logger.warning("No signals to save")
            return
            
        # Create filename
        symbols_str = '-'.join(sorted(symbols))
        start_str = start_date.strftime('%Y%m%d') if start_date else 'none'
        end_str = end_date.strftime('%Y%m%d') if end_date else 'none'
        filename = f"signals_{symbols_str}_{start_str}_{end_str}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to CSV
        try:
            signals.to_csv(filepath)
            logger.info(f"Saved trading signals to {filepath}")
        except Exception as e:
            logger.error(f"Error saving trading signals to {filepath}: {e}")
    
    def generate_orders(self, signals: pd.DataFrame, timestamp: pd.Timestamp,
                       current_positions: Dict[str, Any]) -> List[Order]:
        """
        Generate orders based on trading signals.
        
        Args:
            signals: DataFrame with trading signals
            timestamp: Current timestamp
            current_positions: Dictionary of current positions by symbol
            
        Returns:
            List of Order objects
        """
        return self.strategy.generate_orders(signals, timestamp, current_positions)
    
    def update_trade_history(self, trade_result: Dict[str, Any]):
        """
        Update trade history with a new trade result.
        
        Args:
            trade_result: Dictionary with trade result information
        """
        self.strategy.update_trade_history(trade_result)
    
    def get_sentiment_summary(self, symbols: List[str], days: int = 7) -> Dict[str, Any]:
        """
        Get a summary of sentiment data for the specified symbols.
        
        Args:
            symbols: List of asset symbols to get sentiment summary for
            days: Number of days to include in the summary
            
        Returns:
            Dictionary with sentiment summary information
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Collect sentiment data
        sentiment_data = self.collect_sentiment_data(symbols, start_date, end_date)
        
        if sentiment_data.empty:
            logger.warning("No sentiment data available for summary")
            return {}
            
        # Create summary
        summary = {}
        
        for symbol in symbols:
            symbol_data = sentiment_data[sentiment_data['symbol'] == symbol]
            
            if symbol_data.empty:
                continue
                
            # Calculate average sentiment score by source
            source_sentiment = {}
            for source in symbol_data['source'].unique():
                source_data = symbol_data[symbol_data['source'] == source]
                avg_score = source_data['sentiment_score'].mean() if 'sentiment_score' in source_data else 0.0
                source_sentiment[source] = avg_score
                
            # Calculate overall sentiment score
            overall_score = 0.0
            total_weight = 0.0
            
            for source, score in source_sentiment.items():
                weight = self.strategy.source_weights.get(source, 0.1)
                overall_score += score * weight
                total_weight += weight
                
            if total_weight > 0:
                overall_score /= total_weight
                
            # Get most recent sentiment data
            recent_data = symbol_data.sort_values('timestamp', ascending=False).head(1)
            recent_score = recent_data['sentiment_score'].iloc[0] if not recent_data.empty and 'sentiment_score' in recent_data else 0.0
            
            # Get sentiment trend
            if 'timestamp' in symbol_data.columns:
                symbol_data = symbol_data.sort_values('timestamp')
                if 'sentiment_score' in symbol_data.columns:
                    scores = symbol_data['sentiment_score'].tolist()
                    trend = 'up' if len(scores) > 1 and scores[-1] > scores[0] else 'down' if len(scores) > 1 and scores[-1] < scores[0] else 'flat'
                else:
                    trend = 'unknown'
            else:
                trend = 'unknown'
                
            # Create symbol summary
            summary[symbol] = {
                'overall_score': overall_score,
                'recent_score': recent_score,
                'trend': trend,
                'source_sentiment': source_sentiment,
                'data_points': len(symbol_data)
            }
            
        return summary
