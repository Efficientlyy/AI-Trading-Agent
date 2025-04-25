"""
Alpha Vantage Sentiment Data Connector

This module connects the Alpha Vantage client to the rest of the trading system,
transforming the raw sentiment data into formats compatible with the visualization
components and trading strategies.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from ..data_sources.alpha_vantage_client import AlphaVantageClient
from ..common.time_utils import to_utc_naive

logger = logging.getLogger(__name__)

class AlphaVantageSentimentConnector:
    """Connector for Alpha Vantage sentiment data to the trading system."""
    
    def __init__(self, client: Optional[AlphaVantageClient] = None, cache_ttl: int = 3600):
        """
        Initialize the connector.
        
        Args:
            client: AlphaVantageClient instance. If None, a new one will be created.
            cache_ttl: Time-to-live for cached data in seconds
        """
        self.client = client or AlphaVantageClient()
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = cache_ttl
    
    def get_sentiment_for_symbol(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """
        Get sentiment data for a specific symbol and convert to DataFrame format.
        
        Args:
            symbol: Symbol to get sentiment for (e.g., 'BTC', 'ETH')
            days_back: Number of days to look back
            
        Returns:
            DataFrame with sentiment data
        """
        cache_key = f"{symbol}_{days_back}"
        
        # Check cache first
        if cache_key in self.cache and datetime.now().timestamp() - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl:
            logger.info(f"Using cached sentiment data for {symbol}")
            return self.cache[cache_key]
        
        logger.info(f"Fetching sentiment data for {symbol} from Alpha Vantage")
        
        # For crypto symbols, use get_sentiment_by_crypto
        if symbol.upper() in ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 'SHIB', 'AVAX', 'LINK']:
            raw_data = self.client.get_sentiment_by_crypto(symbol, days_back=days_back)
        else:
            # For other symbols, try get_sentiment_by_topic with the symbol as topic
            # This is a fallback but may not work well for non-crypto assets
            raw_data = self.client.get_sentiment_by_topic(symbol.lower(), days_back=days_back)
        
        # Extract sentiment scores from raw data
        sentiment_scores = self.client.extract_sentiment_scores(raw_data)
        
        if not sentiment_scores:
            logger.warning(f"No sentiment data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(sentiment_scores)
        
        # Convert time_published to datetime
        if 'time_published' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time_published'])
        else:
            df['timestamp'] = pd.Timestamp.now()  # Fallback
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Store in cache
        self.cache[cache_key] = df
        self.cache_timestamps[cache_key] = datetime.now().timestamp()
        
        return df
    
    def get_sentiment_signals(self, symbol: str, days_back: int = 7, 
                              threshold: float = 0.2, window_size: int = 3) -> Dict[str, Any]:
        """
        Generate trading signals based on sentiment data.
        
        Args:
            symbol: Symbol to get signals for
            days_back: Number of days to look back
            threshold: Sentiment threshold for generating signals
            window_size: Window size for rolling calculations
            
        Returns:
            Dictionary with sentiment signals
        """
        # Get sentiment data
        df = self.get_sentiment_for_symbol(symbol, days_back)
        
        if df.empty:
            logger.warning(f"No sentiment data available for {symbol}")
            return {
                'symbol': symbol,
                'signal': 'hold',
                'strength': 0.0,
                'score': 0.0,
                'trend': 0.0,
                'volatility': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate sentiment metrics
        # We'll use overall_sentiment_score or ticker_sentiment_score if available
        if 'ticker_sentiment_score' in df.columns:
            sentiment_column = 'ticker_sentiment_score'
        else:
            sentiment_column = 'overall_sentiment_score'
        
        # Current sentiment score (latest)
        current_score = df[sentiment_column].iloc[-1] if not df.empty else 0.0
        
        # Calculate trend (change over time)
        if len(df) >= window_size:
            # Use rolling average to smooth out noise
            df['rolling_sentiment'] = df[sentiment_column].rolling(window=window_size).mean()
            # Calculate trend as slope over recent periods
            trend = df['rolling_sentiment'].diff().iloc[-window_size:].mean()
        else:
            trend = 0.0
            df['rolling_sentiment'] = df[sentiment_column]
        
        # Calculate volatility (standard deviation of sentiment)
        volatility = df[sentiment_column].std() if len(df) > 1 else 0.0
        
        # Generate signal based on current score and trend
        signal = 'hold'
        if current_score > threshold:
            signal = 'buy'
        elif current_score < -threshold:
            signal = 'sell'
        
        # Calculate signal strength (0.0 to 1.0)
        # This is based on both the absolute sentiment score and the trend
        raw_strength = abs(current_score) * (1 + abs(trend))
        strength = min(1.0, raw_strength)
        
        return {
            'symbol': symbol,
            'signal': signal,
            'strength': float(strength),
            'score': float(current_score),
            'trend': float(trend),
            'volatility': float(volatility),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_sentiment_summary(self, symbols: List[str], days_back: int = 7) -> Dict[str, Any]:
        """
        Get sentiment summary for multiple symbols.
        
        Args:
            symbols: List of symbols to get sentiment for
            days_back: Number of days to look back
            
        Returns:
            Dictionary with sentiment summary
        """
        sentiment_data = {}
        
        for symbol in symbols:
            try:
                signals = self.get_sentiment_signals(symbol, days_back)
                sentiment_data[symbol] = signals
            except Exception as e:
                logger.error(f"Error getting sentiment for {symbol}: {e}")
                # Provide a default neutral signal
                sentiment_data[symbol] = {
                    'symbol': symbol,
                    'signal': 'hold',
                    'strength': 0.0,
                    'score': 0.0,
                    'trend': 0.0,
                    'volatility': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'sentimentData': sentiment_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_historical_sentiment(self, symbol: str, timeframe: str = '1M') -> List[Dict[str, Any]]:
        """
        Get historical sentiment data for a specific symbol.
        
        Args:
            symbol: Symbol to get historical sentiment for
            timeframe: Timeframe ('1D', '1W', '1M', '3M', '1Y')
            
        Returns:
            List of dictionaries with historical sentiment data
        """
        # Convert timeframe to days_back
        days_map = {
            '1D': 1,
            '1W': 7,
            '1M': 30,
            '3M': 90,
            '1Y': 365
        }
        days_back = days_map.get(timeframe, 30)  # Default to 30 days
        
        # Get sentiment data
        df = self.get_sentiment_for_symbol(symbol, days_back)
        
        if df.empty:
            logger.warning(f"No historical sentiment data for {symbol}")
            return []
        
        # Determine which sentiment column to use
        if 'ticker_sentiment_score' in df.columns:
            sentiment_column = 'ticker_sentiment_score'
        else:
            sentiment_column = 'overall_sentiment_score'
        
        # Calculate rolling average to smooth out data
        window_size = min(3, len(df))
        if window_size > 0:
            df['rolling_sentiment'] = df[sentiment_column].rolling(window=window_size).mean()
            # Fill NaN values at the beginning
            df['rolling_sentiment'] = df['rolling_sentiment'].fillna(df[sentiment_column])
        else:
            df['rolling_sentiment'] = df[sentiment_column]
        
        # Convert to list of dictionaries for the frontend
        result = []
        for timestamp, row in df.iterrows():
            result.append({
                'timestamp': timestamp.isoformat(),
                'score': float(row['rolling_sentiment']),
                'raw_score': float(row[sentiment_column])
            })
        
        return result