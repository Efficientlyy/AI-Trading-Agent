"""
Real-time optimized trading strategies for paper trading.

This module provides strategy implementations that are optimized for real-time data
with enhanced error handling, data validation, and performance optimizations.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from ..common import logger
from ..agent.simple_strategy_manager import BaseStrategy


class RealtimeMACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover strategy optimized for real-time data.
    
    Features:
    - Data validation to handle missing or incomplete data
    - Caching of calculations to improve performance
    - Confidence scoring based on signal strength and data quality
    - Detailed metadata for monitoring and debugging
    """
    
    def __init__(self, name: str = "RealtimeMACrossover", 
                 short_window: int = 10, 
                 long_window: int = 50,
                 min_data_points: int = 0,
                 signal_smoothing: bool = True,
                 cache_calculations: bool = True):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            short_window: Short moving average window
            long_window: Long moving average window
            min_data_points: Minimum number of data points required (0 = use long_window)
            signal_smoothing: Whether to smooth signals to reduce noise
            cache_calculations: Whether to cache calculations for performance
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.min_data_points = min_data_points if min_data_points > 0 else long_window
        self.signal_smoothing = signal_smoothing
        self.cache_calculations = cache_calculations
        
        # Cache for calculations
        self._cache = {}
        self._cache_timestamp = {}
        self._last_signals = {}
        
        logger.info(f"Initialized {name} with short_window={short_window}, "
                   f"long_window={long_window}, min_data_points={self.min_data_points}, "
                   f"signal_smoothing={signal_smoothing}, cache_calculations={cache_calculations}")
    
    def _calculate_moving_averages(self, df: pd.DataFrame, symbol: str) -> Tuple[float, float, bool]:
        """
        Calculate moving averages with caching for performance.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol to calculate for
            
        Returns:
            Tuple of (short_ma, long_ma, is_valid)
        """
        current_time = time.time()
        cache_key = f"{symbol}_{id(df)}"
        
        # Check if we have a valid cached calculation
        if (self.cache_calculations and 
            cache_key in self._cache and 
            current_time - self._cache_timestamp.get(cache_key, 0) < 60):  # Cache for 60 seconds
            return self._cache[cache_key]
        
        # Validate data
        if len(df) < self.min_data_points:
            return 0, 0, False
        
        try:
            # Calculate moving averages
            short_ma = df['close'].rolling(window=self.short_window).mean().iloc[-1]
            long_ma = df['close'].rolling(window=self.long_window).mean().iloc[-1]
            
            # Check for NaN values
            if pd.isna(short_ma) or pd.isna(long_ma):
                return 0, 0, False
            
            # Cache the result
            if self.cache_calculations:
                self._cache[cache_key] = (short_ma, long_ma, True)
                self._cache_timestamp[cache_key] = current_time
            
            return short_ma, long_ma, True
        
        except Exception as e:
            logger.error(f"{self.name}: Error calculating moving averages for {symbol}: {e}")
            return 0, 0, False
    
    def _calculate_signal_strength(self, short_ma: float, long_ma: float, 
                                  prev_signal: float = 0) -> float:
        """
        Calculate signal strength with optional smoothing.
        
        Args:
            short_ma: Short moving average value
            long_ma: Long moving average value
            prev_signal: Previous signal value for smoothing
            
        Returns:
            Signal strength value (-1 to 1)
        """
        # Calculate raw signal
        if short_ma > long_ma:
            raw_signal = 1.0  # Buy
        elif short_ma < long_ma:
            raw_signal = -1.0  # Sell
        else:
            raw_signal = 0.0  # Hold
        
        # Calculate signal strength based on the difference
        if long_ma > 0:
            # Normalize the difference between MAs as a percentage of the long MA
            diff_pct = (short_ma - long_ma) / long_ma
            # Scale to a reasonable range (-1 to 1)
            strength = max(min(diff_pct * 10, 1.0), -1.0)
        else:
            strength = raw_signal
        
        # Apply smoothing if enabled
        if self.signal_smoothing and prev_signal != 0:
            # 70% new signal, 30% previous signal
            smoothed_signal = 0.7 * strength + 0.3 * prev_signal
            return smoothed_signal
        
        return strength
    
    def _calculate_confidence(self, df: pd.DataFrame, is_valid: bool) -> float:
        """
        Calculate confidence score based on data quality and signal strength.
        
        Args:
            df: DataFrame with market data
            is_valid: Whether the calculation is valid
            
        Returns:
            Confidence score (0 to 1)
        """
        if not is_valid:
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Adjust based on data quality
        if len(df) < self.long_window * 2:
            confidence *= 0.8  # Reduce confidence if limited data
        
        # Check for gaps in data
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            if len(timestamps) > 1:
                avg_interval = (timestamps.iloc[-1] - timestamps.iloc[0]) / (len(timestamps) - 1)
                max_interval = timestamps.diff().max()
                
                if max_interval > avg_interval * 3:
                    confidence *= 0.9  # Reduce confidence if there are gaps
        
        # Check for recent volatility
        if 'close' in df.columns and len(df) > 10:
            recent_volatility = df['close'].pct_change().abs().mean() * 100
            if recent_volatility > 2.0:  # More than 2% average change
                confidence *= 0.9  # Reduce confidence in volatile markets
        
        return min(confidence, 1.0)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], 
                        current_portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for each symbol.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with market data
            current_portfolio: Current portfolio state (optional)
        
        Returns:
            Dictionary mapping symbols to signal dictionaries
        """
        signals = {}
        
        for symbol, df in data.items():
            try:
                # Get previous signal for smoothing if available
                prev_signal = self._last_signals.get(symbol, {}).get('signal_strength', 0)
                
                # Calculate moving averages
                short_ma, long_ma, is_valid = self._calculate_moving_averages(df, symbol)
                
                if is_valid:
                    # Calculate signal strength
                    signal_strength = self._calculate_signal_strength(short_ma, long_ma, prev_signal)
                    
                    # Calculate confidence
                    confidence_score = self._calculate_confidence(df, is_valid)
                    
                    # Create signal dictionary
                    signals[symbol] = {
                        'signal_strength': signal_strength,
                        'confidence_score': confidence_score,
                        'metadata': {
                            'strategy': self.name,
                            'short_ma': short_ma,
                            'long_ma': long_ma,
                            'ma_diff_pct': ((short_ma - long_ma) / long_ma) if long_ma > 0 else 0,
                            'data_points': len(df),
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                else:
                    signals[symbol] = {
                        'signal_strength': 0,
                        'confidence_score': 0.0,
                        'metadata': {
                            'strategy': self.name,
                            'reason': 'Insufficient or invalid data',
                            'data_points': len(df),
                            'required_points': self.min_data_points,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
            except Exception as e:
                logger.error(f"{self.name}: Error generating signal for {symbol}: {e}")
                signals[symbol] = {
                    'signal_strength': 0,
                    'confidence_score': 0.0,
                    'metadata': {
                        'strategy': self.name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                }
        
        # Store signals for next iteration
        self._last_signals = signals
        
        return signals


class RealtimeSentimentStrategy(BaseStrategy):
    """
    Sentiment analysis strategy optimized for real-time data.
    
    Features:
    - Time-weighted sentiment scoring
    - Trend detection in sentiment changes
    - Confidence scoring based on sentiment consistency
    - Detailed metadata for monitoring and debugging
    """
    
    def __init__(self, name: str = "RealtimeSentiment", 
                 sentiment_threshold: float = 0.2, 
                 confidence_threshold: float = 0.6,
                 time_window: int = 5,
                 trend_weight: float = 0.3):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            sentiment_threshold: Threshold for sentiment to generate signals
            confidence_threshold: Threshold for confidence to generate signals
            time_window: Number of recent data points to consider for trend
            trend_weight: Weight to give to trend vs. current sentiment (0-1)
        """
        super().__init__(name)
        self.sentiment_threshold = sentiment_threshold
        self.confidence_threshold = confidence_threshold
        self.time_window = time_window
        self.trend_weight = trend_weight
        
        # Cache for previous sentiments
        self._sentiment_history = {}
        
        logger.info(f"Initialized {name} with sentiment_threshold={sentiment_threshold}, "
                   f"confidence_threshold={confidence_threshold}, time_window={time_window}, "
                   f"trend_weight={trend_weight}")
    
    def _calculate_sentiment_trend(self, df: pd.DataFrame, symbol: str) -> Tuple[float, float, float]:
        """
        Calculate sentiment trend from recent data.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol to calculate for
            
        Returns:
            Tuple of (current_sentiment, trend_sentiment, combined_sentiment)
        """
        # Get sentiment data
        if 'sentiment_score' not in df.columns:
            return 0, 0, 0
        
        # Get current sentiment
        current_sentiment = df['sentiment_score'].iloc[-1]
        
        # Calculate trend if we have enough data
        window_size = min(self.time_window, len(df))
        if window_size > 1:
            recent_sentiments = df['sentiment_score'].iloc[-window_size:].values
            
            # Calculate trend (simple linear regression slope)
            x = np.arange(window_size)
            slope = np.polyfit(x, recent_sentiments, 1)[0]
            
            # Normalize slope to a reasonable range
            trend_sentiment = max(min(slope * window_size, 1.0), -1.0)
            
            # Store in history
            if symbol not in self._sentiment_history:
                self._sentiment_history[symbol] = []
            
            self._sentiment_history[symbol].append(current_sentiment)
            if len(self._sentiment_history[symbol]) > 20:  # Keep last 20 values
                self._sentiment_history[symbol].pop(0)
        else:
            trend_sentiment = 0
        
        # Combine current sentiment with trend
        combined_sentiment = (1 - self.trend_weight) * current_sentiment + self.trend_weight * trend_sentiment
        
        return current_sentiment, trend_sentiment, combined_sentiment
    
    def _calculate_confidence(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Calculate confidence based on sentiment consistency and data quality.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol to calculate for
            
        Returns:
            Confidence score (0 to 1)
        """
        # Get confidence from data if available
        if 'confidence_score' in df.columns:
            base_confidence = df['confidence_score'].iloc[-1]
        else:
            base_confidence = 0.7  # Default confidence
        
        # Adjust based on sentiment consistency if we have history
        if symbol in self._sentiment_history and len(self._sentiment_history[symbol]) > 3:
            history = self._sentiment_history[symbol]
            # Calculate standard deviation of recent sentiments
            std_dev = np.std(history)
            
            # Lower confidence if sentiment is volatile
            if std_dev > 0.3:
                base_confidence *= 0.8
            elif std_dev > 0.1:
                base_confidence *= 0.9
        
        # Adjust based on data recency
        if 'timestamp' in df.columns:
            latest_time = pd.to_datetime(df['timestamp'].iloc[-1])
            now = pd.Timestamp.now()
            age_hours = (now - latest_time).total_seconds() / 3600
            
            # Reduce confidence for older data
            if age_hours > 24:
                base_confidence *= 0.5
            elif age_hours > 12:
                base_confidence *= 0.7
            elif age_hours > 6:
                base_confidence *= 0.9
        
        return min(base_confidence, 1.0)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], 
                        current_portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for each symbol.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with market data
            current_portfolio: Current portfolio state (optional)
        
        Returns:
            Dictionary mapping symbols to signal dictionaries
        """
        signals = {}
        
        for symbol, df in data.items():
            try:
                if 'sentiment_score' in df.columns:
                    # Calculate sentiment and trend
                    current_sentiment, trend_sentiment, combined_sentiment = self._calculate_sentiment_trend(df, symbol)
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(df, symbol)
                    
                    # Generate signal
                    if combined_sentiment > self.sentiment_threshold and confidence > self.confidence_threshold:
                        signal_strength = min(combined_sentiment * 2, 1.0)  # Scale to 0-1 range
                    elif combined_sentiment < -self.sentiment_threshold and confidence > self.confidence_threshold:
                        signal_strength = max(combined_sentiment * 2, -1.0)  # Scale to -1-0 range
                    else:
                        signal_strength = 0
                    
                    signals[symbol] = {
                        'signal_strength': signal_strength,
                        'confidence_score': confidence,
                        'metadata': {
                            'strategy': self.name,
                            'current_sentiment': current_sentiment,
                            'trend_sentiment': trend_sentiment,
                            'combined_sentiment': combined_sentiment,
                            'sentiment_history_length': len(self._sentiment_history.get(symbol, [])),
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                else:
                    signals[symbol] = {
                        'signal_strength': 0,
                        'confidence_score': 0.0,
                        'metadata': {
                            'strategy': self.name,
                            'reason': 'No sentiment data',
                            'timestamp': datetime.now().isoformat()
                        }
                    }
            except Exception as e:
                logger.error(f"{self.name}: Error generating signal for {symbol}: {e}")
                signals[symbol] = {
                    'signal_strength': 0,
                    'confidence_score': 0.0,
                    'metadata': {
                        'strategy': self.name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                }
        
        return signals


class RealtimeVolumeStrategy(BaseStrategy):
    """
    Volume analysis strategy optimized for real-time data.
    
    Features:
    - Volume spike detection
    - Relative volume analysis
    - Price-volume correlation
    - Detailed metadata for monitoring and debugging
    """
    
    def __init__(self, name: str = "RealtimeVolume", 
                 volume_threshold: float = 2.0,
                 price_correlation_weight: float = 0.5,
                 lookback_period: int = 20):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            volume_threshold: Threshold for relative volume to generate signals
            price_correlation_weight: Weight to give to price-volume correlation
            lookback_period: Number of periods to look back for volume baseline
        """
        super().__init__(name)
        self.volume_threshold = volume_threshold
        self.price_correlation_weight = price_correlation_weight
        self.lookback_period = lookback_period
        
        logger.info(f"Initialized {name} with volume_threshold={volume_threshold}, "
                   f"price_correlation_weight={price_correlation_weight}, "
                   f"lookback_period={lookback_period}")
    
    def _calculate_relative_volume(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate relative volume compared to baseline.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Tuple of (relative_volume, current_volume, avg_volume)
        """
        if 'volume' not in df.columns or len(df) < 2:
            return 1.0, 0, 0
        
        # Get current volume
        current_volume = df['volume'].iloc[-1]
        
        # Calculate average volume over lookback period
        lookback = min(self.lookback_period, len(df) - 1)
        avg_volume = df['volume'].iloc[-lookback-1:-1].mean()
        
        # Calculate relative volume
        if avg_volume > 0:
            relative_volume = current_volume / avg_volume
        else:
            relative_volume = 1.0
        
        return relative_volume, current_volume, avg_volume
    
    def _calculate_price_volume_correlation(self, df: pd.DataFrame) -> float:
        """
        Calculate correlation between price changes and volume.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if 'close' not in df.columns or 'volume' not in df.columns or len(df) < 5:
            return 0
        
        try:
            # Calculate price changes
            price_changes = df['close'].pct_change()
            
            # Calculate correlation between absolute price changes and volume
            correlation = price_changes.abs().corr(df['volume'])
            
            if pd.isna(correlation):
                return 0
            
            return correlation
        except Exception:
            return 0
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], 
                        current_portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for each symbol.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with market data
            current_portfolio: Current portfolio state (optional)
        
        Returns:
            Dictionary mapping symbols to signal dictionaries
        """
        signals = {}
        
        for symbol, df in data.items():
            try:
                if 'volume' in df.columns and 'close' in df.columns and len(df) > self.lookback_period:
                    # Calculate relative volume
                    relative_volume, current_volume, avg_volume = self._calculate_relative_volume(df)
                    
                    # Calculate price-volume correlation
                    correlation = self._calculate_price_volume_correlation(df)
                    
                    # Calculate price direction
                    if len(df) >= 2:
                        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                    else:
                        price_change = 0
                    
                    # Generate signal
                    signal_strength = 0
                    confidence = 0.6  # Base confidence
                    
                    # High volume with positive correlation is a stronger signal
                    if relative_volume > self.volume_threshold:
                        # Direction based on price change
                        direction = 1 if price_change > 0 else -1
                        
                        # Adjust strength based on relative volume and correlation
                        volume_factor = min((relative_volume - 1) / 2, 1.0)
                        corr_factor = abs(correlation) * self.price_correlation_weight
                        
                        # Calculate signal strength
                        signal_strength = direction * volume_factor * (1 + corr_factor)
                        
                        # Adjust confidence based on correlation
                        confidence = 0.6 + (abs(correlation) * 0.3)
                    
                    signals[symbol] = {
                        'signal_strength': max(min(signal_strength, 1.0), -1.0),  # Clamp to [-1, 1]
                        'confidence_score': min(confidence, 1.0),
                        'metadata': {
                            'strategy': self.name,
                            'relative_volume': relative_volume,
                            'current_volume': current_volume,
                            'avg_volume': avg_volume,
                            'price_volume_correlation': correlation,
                            'price_change_pct': price_change * 100,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                else:
                    signals[symbol] = {
                        'signal_strength': 0,
                        'confidence_score': 0.0,
                        'metadata': {
                            'strategy': self.name,
                            'reason': 'Insufficient volume data',
                            'timestamp': datetime.now().isoformat()
                        }
                    }
            except Exception as e:
                logger.error(f"{self.name}: Error generating signal for {symbol}: {e}")
                signals[symbol] = {
                    'signal_strength': 0,
                    'confidence_score': 0.0,
                    'metadata': {
                        'strategy': self.name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                }
        
        return signals
