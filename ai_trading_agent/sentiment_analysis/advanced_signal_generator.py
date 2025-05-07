"""
Advanced Trading Signal Generator

This module provides an advanced trading signal generator that combines sentiment analysis
with technical indicators to generate more robust trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

from .signal_generator import SentimentSignalGenerator
from ..feature_engineering.advanced_features import (
    create_bollinger_bands,
    create_rsi_features,
    create_macd_features,
    calculate_fibonacci_retracement,
    calculate_pivot_points,
    calculate_volume_profile
)

class AdvancedSignalGenerator:
    """
    Generates trading signals by combining sentiment analysis with technical indicators.
    
    This generator uses a multi-factor approach to generate trading signals, considering:
    1. Sentiment scores from various sources (news, social media, fear & greed index)
    2. Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    3. Market conditions and volatility
    
    The signals are weighted and combined to produce a final signal strength and direction.
    """
    
    def __init__(
        self,
        sentiment_threshold: float = 0.1,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        bollinger_band_width_threshold: float = 0.05,
        macd_signal_threshold: float = 0.0,
        volume_weight: float = 0.2,
        sentiment_weight: float = 0.4,
        technical_weight: float = 0.4,
        use_adaptive_thresholds: bool = True,
        sentiment_window: int = 20,
        sentiment_quantile: float = 0.8,
        use_crypto_specific_settings: bool = True
    ):
        """
        Initialize the advanced signal generator.
        
        Args:
            sentiment_threshold: Threshold for sentiment signals
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            bollinger_band_width_threshold: Threshold for Bollinger Band width to identify volatility
            macd_signal_threshold: Threshold for MACD signal line crossovers
            volume_weight: Weight for volume-based signals
            sentiment_weight: Weight for sentiment-based signals
            technical_weight: Weight for technical indicator-based signals
            use_adaptive_thresholds: Whether to use adaptive thresholds for sentiment
            sentiment_window: Window size for adaptive sentiment thresholds
            sentiment_quantile: Quantile for adaptive sentiment thresholds
            use_crypto_specific_settings: Whether to use crypto-specific settings
        """
        self.sentiment_threshold = sentiment_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bollinger_band_width_threshold = bollinger_band_width_threshold
        self.macd_signal_threshold = macd_signal_threshold
        
        # Ensure weights sum to 1.0
        total_weight = volume_weight + sentiment_weight + technical_weight
        self.volume_weight = volume_weight / total_weight
        self.sentiment_weight = sentiment_weight / total_weight
        self.technical_weight = technical_weight / total_weight
        
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.sentiment_window = sentiment_window
        self.sentiment_quantile = sentiment_quantile
        
        # Initialize sentiment signal generator
        self.sentiment_generator = SentimentSignalGenerator(
            buy_threshold=sentiment_threshold,
            sell_threshold=-sentiment_threshold,
            adaptive=use_adaptive_thresholds,
            window=sentiment_window,
            quantile=sentiment_quantile
        )
        
        # Crypto-specific settings
        self.use_crypto_specific_settings = use_crypto_specific_settings
        if use_crypto_specific_settings:
            # Crypto markets are more volatile, adjust thresholds
            self.rsi_overbought = 75.0
            self.rsi_oversold = 25.0
            self.bollinger_band_width_threshold = 0.08
            # Crypto markets are more sentiment-driven
            total_weight = volume_weight + (sentiment_weight * 1.5) + technical_weight
            self.volume_weight = volume_weight / total_weight
            self.sentiment_weight = (sentiment_weight * 1.5) / total_weight
            self.technical_weight = technical_weight / total_weight
    
    def generate_signals(
        self,
        price_data: pd.DataFrame,
        sentiment_data: pd.DataFrame,
        volume_data: Optional[pd.Series] = None,
        additional_features: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals by combining sentiment and technical indicators.
        
        Args:
            price_data: DataFrame with price data (must have 'open', 'high', 'low', 'close' columns)
            sentiment_data: DataFrame with sentiment scores
            volume_data: Optional Series with volume data
            additional_features: Optional dictionary of additional feature DataFrames
            
        Returns:
            DataFrame with trading signals and signal strengths
        """
        if price_data.empty or sentiment_data.empty:
            return pd.DataFrame()
        
        # Ensure required columns exist in price_data
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in price_data.columns for col in required_columns):
            raise ValueError(f"Price data must contain columns: {required_columns}")
        
        # Calculate technical indicators
        technical_signals = self._calculate_technical_signals(price_data, volume_data)
        
        # Process sentiment data
        sentiment_signals = self._process_sentiment_data(sentiment_data, price_data.index)
        
        # Combine signals
        combined_signals = self._combine_signals(
            technical_signals,
            sentiment_signals,
            price_data,
            additional_features
        )
        
        return combined_signals
    
    def _calculate_technical_signals(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate technical indicators and generate signals.
        
        Args:
            price_data: DataFrame with price data
            volume_data: Optional Series with volume data
            
        Returns:
            DataFrame with technical signals
        """
        signals = pd.DataFrame(index=price_data.index)
        
        # Calculate RSI
        rsi_features = create_rsi_features(price_data['close'], [14])
        signals['rsi_14'] = rsi_features['rsi_14']
        signals['rsi_signal'] = 0
        signals.loc[signals['rsi_14'] < self.rsi_oversold, 'rsi_signal'] = 1  # Oversold -> Buy
        signals.loc[signals['rsi_14'] > self.rsi_overbought, 'rsi_signal'] = -1  # Overbought -> Sell
        
        # Calculate MACD
        macd_features = create_macd_features(price_data['close'])
        signals['macd'] = macd_features['macd']
        signals['macd_signal'] = macd_features['macd_signal']
        signals['macd_histogram'] = macd_features['macd_histogram']
        
        # MACD signal: 1 when MACD crosses above signal line, -1 when crosses below
        signals['macd_crossover'] = 0
        signals.loc[signals['macd_histogram'] > self.macd_signal_threshold, 'macd_crossover'] = 1
        signals.loc[signals['macd_histogram'] < -self.macd_signal_threshold, 'macd_crossover'] = -1
        
        # Calculate Bollinger Bands
        bb_features = create_bollinger_bands(price_data['close'], [20])
        signals['bb_upper'] = bb_features['bb_upper_20']
        signals['bb_middle'] = bb_features['bb_middle_20']
        signals['bb_lower'] = bb_features['bb_lower_20']
        
        # Calculate BB width (volatility indicator)
        signals['bb_width'] = (signals['bb_upper'] - signals['bb_lower']) / signals['bb_middle']
        
        # BB signals: 1 when price crosses below lower band, -1 when crosses above upper band
        signals['bb_signal'] = 0
        signals.loc[price_data['close'] < signals['bb_lower'], 'bb_signal'] = 1
        signals.loc[price_data['close'] > signals['bb_upper'], 'bb_signal'] = -1
        
        # Calculate Fibonacci retracement levels
        try:
            # Find recent high and low for Fibonacci calculation
            window = min(100, len(price_data))
            recent_high = price_data['high'].rolling(window=window).max().iloc[-1]
            recent_low = price_data['low'].rolling(window=window).min().iloc[-1]
            
            fib_levels = calculate_fibonacci_retracement(recent_high, recent_low)
            for level_name, level_value in fib_levels.items():
                signals[f'fib_{level_name}'] = level_value
            
            # Fibonacci signals
            close_price = price_data['close'].iloc[-1]
            signals['fib_signal'] = 0
            
            # Buy signal if price is near 0.618 or 0.786 retracement (support)
            if close_price <= fib_levels['0.618'] * 1.01 and close_price >= fib_levels['0.618'] * 0.99:
                signals['fib_signal'].iloc[-1] = 1
            elif close_price <= fib_levels['0.786'] * 1.01 and close_price >= fib_levels['0.786'] * 0.99:
                signals['fib_signal'].iloc[-1] = 1
            
            # Sell signal if price is near 0.236 or 0.382 retracement (resistance)
            elif close_price >= fib_levels['0.236'] * 0.99 and close_price <= fib_levels['0.236'] * 1.01:
                signals['fib_signal'].iloc[-1] = -1
            elif close_price >= fib_levels['0.382'] * 0.99 and close_price <= fib_levels['0.382'] * 1.01:
                signals['fib_signal'].iloc[-1] = -1
        except Exception as e:
            # If Fibonacci calculation fails, continue without it
            signals['fib_signal'] = 0
        
        # Calculate pivot points if we have enough data
        try:
            if len(price_data) >= 2:
                # Get previous day's data for pivot point calculation
                prev_day = price_data.iloc[-2]
                pivot_points = calculate_pivot_points(
                    prev_day['high'],
                    prev_day['low'],
                    prev_day['close']
                )
                
                for pp_name, pp_value in pivot_points.items():
                    signals[f'pivot_{pp_name}'] = pp_value
                
                # Pivot point signals
                close_price = price_data['close'].iloc[-1]
                signals['pivot_signal'] = 0
                
                # Buy signal if price is near S1 or S2 (support)
                if close_price <= pivot_points['S1'] * 1.01 and close_price >= pivot_points['S1'] * 0.99:
                    signals['pivot_signal'].iloc[-1] = 1
                elif close_price <= pivot_points['S2'] * 1.01 and close_price >= pivot_points['S2'] * 0.99:
                    signals['pivot_signal'].iloc[-1] = 1
                
                # Sell signal if price is near R1 or R2 (resistance)
                elif close_price >= pivot_points['R1'] * 0.99 and close_price <= pivot_points['R1'] * 1.01:
                    signals['pivot_signal'].iloc[-1] = -1
                elif close_price >= pivot_points['R2'] * 0.99 and close_price <= pivot_points['R2'] * 1.01:
                    signals['pivot_signal'].iloc[-1] = -1
        except Exception as e:
            # If pivot point calculation fails, continue without it
            signals['pivot_signal'] = 0
        
        # Volume profile if volume data is available
        if volume_data is not None and not volume_data.empty:
            try:
                volume_profile = calculate_volume_profile(price_data, volume_data)
                signals['volume_profile_poc'] = volume_profile['poc']
                signals['volume_profile_vah'] = volume_profile['vah']
                signals['volume_profile_val'] = volume_profile['val']
                
                # Volume profile signals
                close_price = price_data['close'].iloc[-1]
                signals['volume_profile_signal'] = 0
                
                # Buy signal if price is near Value Area Low (support)
                if close_price <= volume_profile['val'] * 1.01 and close_price >= volume_profile['val'] * 0.99:
                    signals['volume_profile_signal'].iloc[-1] = 1
                
                # Sell signal if price is near Value Area High (resistance)
                elif close_price >= volume_profile['vah'] * 0.99 and close_price <= volume_profile['vah'] * 1.01:
                    signals['volume_profile_signal'].iloc[-1] = -1
            except Exception as e:
                # If volume profile calculation fails, continue without it
                signals['volume_profile_signal'] = 0
        
        # Combine technical signals
        signals['technical_signal'] = (
            signals['rsi_signal'] + 
            signals['macd_crossover'] + 
            signals['bb_signal']
        )
        
        # Add Fibonacci and pivot signals if available
        if 'fib_signal' in signals.columns:
            signals['technical_signal'] += signals['fib_signal']
        
        if 'pivot_signal' in signals.columns:
            signals['technical_signal'] += signals['pivot_signal']
        
        if 'volume_profile_signal' in signals.columns:
            signals['technical_signal'] += signals['volume_profile_signal']
        
        # Normalize technical signal to range [-1, 1]
        max_possible_signals = 3  # RSI, MACD, BB are always calculated
        if 'fib_signal' in signals.columns:
            max_possible_signals += 1
        if 'pivot_signal' in signals.columns:
            max_possible_signals += 1
        if 'volume_profile_signal' in signals.columns:
            max_possible_signals += 1
        
        signals['technical_signal'] = signals['technical_signal'] / max_possible_signals
        
        return signals
    
    def _process_sentiment_data(
        self,
        sentiment_data: pd.DataFrame,
        price_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Process sentiment data and generate sentiment signals.
        
        Args:
            sentiment_data: DataFrame with sentiment scores
            price_index: DatetimeIndex from price data for alignment
            
        Returns:
            DataFrame with sentiment signals
        """
        # Check if sentiment_data has a 'sentiment_score' column
        if 'sentiment_score' not in sentiment_data.columns:
            raise ValueError("Sentiment data must contain a 'sentiment_score' column")
        
        # Ensure sentiment_data has a datetime index
        if not isinstance(sentiment_data.index, pd.DatetimeIndex):
            if 'timestamp' in sentiment_data.columns:
                sentiment_data = sentiment_data.set_index('timestamp')
            else:
                raise ValueError("Sentiment data must have a datetime index or a 'timestamp' column")
        
        # Resample sentiment data to match price data frequency
        # First, determine the frequency of price data
        price_freq = pd.infer_freq(price_index)
        if price_freq is None:
            # If frequency can't be inferred, default to daily
            price_freq = 'D'
        
        # Resample sentiment data
        resampled_sentiment = sentiment_data.resample(price_freq).mean()
        
        # Forward fill missing values
        resampled_sentiment = resampled_sentiment.fillna(method='ffill')
        
        # Align with price index
        aligned_sentiment = resampled_sentiment.reindex(price_index, method='ffill')
        
        # Generate sentiment signals
        sentiment_signals = pd.DataFrame(index=price_index)
        sentiment_signals['sentiment_score'] = aligned_sentiment['sentiment_score']
        
        # Use sentiment generator to create signals
        sentiment_signals['sentiment_signal'] = self.sentiment_generator.generate_signals_from_scores(
            sentiment_signals['sentiment_score']
        )
        
        return sentiment_signals
    
    def _combine_signals(
        self,
        technical_signals: pd.DataFrame,
        sentiment_signals: pd.DataFrame,
        price_data: pd.DataFrame,
        additional_features: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Combine technical and sentiment signals to generate final trading signals.
        
        Args:
            technical_signals: DataFrame with technical signals
            sentiment_signals: DataFrame with sentiment signals
            price_data: DataFrame with price data
            additional_features: Optional dictionary of additional feature DataFrames
            
        Returns:
            DataFrame with combined signals
        """
        combined = pd.DataFrame(index=price_data.index)
        
        # Copy key signals
        combined['technical_signal'] = technical_signals['technical_signal']
        combined['sentiment_signal'] = sentiment_signals['sentiment_signal']
        
        # Calculate volatility factor from Bollinger Band width
        combined['volatility'] = technical_signals['bb_width']
        
        # Adjust weights based on volatility
        # In high volatility, reduce sentiment weight and increase technical weight
        volatility_factor = combined['volatility'] / self.bollinger_band_width_threshold
        volatility_factor = volatility_factor.clip(0.5, 2.0)  # Limit adjustment factor
        
        adjusted_sentiment_weight = self.sentiment_weight / volatility_factor
        adjusted_technical_weight = self.technical_weight * volatility_factor
        
        # Normalize weights to sum to 1.0 (excluding volume weight)
        total_weight = adjusted_sentiment_weight + adjusted_technical_weight
        adjusted_sentiment_weight = adjusted_sentiment_weight / total_weight * (1.0 - self.volume_weight)
        adjusted_technical_weight = adjusted_technical_weight / total_weight * (1.0 - self.volume_weight)
        
        # Calculate weighted signal
        combined['signal_strength'] = (
            combined['technical_signal'] * adjusted_technical_weight +
            combined['sentiment_signal'] * adjusted_sentiment_weight
        )
        
        # Add volume component if available
        if 'volume_profile_signal' in technical_signals.columns:
            combined['signal_strength'] += technical_signals['volume_profile_signal'] * self.volume_weight
        
        # Determine final signal direction
        combined['signal'] = 0
        combined.loc[combined['signal_strength'] > 0.2, 'signal'] = 1
        combined.loc[combined['signal_strength'] < -0.2, 'signal'] = -1
        
        # Add confidence score (absolute value of signal strength)
        combined['confidence'] = combined['signal_strength'].abs()
        
        # Add additional features if provided
        if additional_features is not None:
            for feature_name, feature_df in additional_features.items():
                if not feature_df.empty:
                    # Align feature with price index
                    aligned_feature = feature_df.reindex(price_data.index, method='ffill')
                    
                    # Add to combined signals
                    for col in aligned_feature.columns:
                        combined[f'{feature_name}_{col}'] = aligned_feature[col]
        
        return combined
    
    def get_trading_decision(
        self,
        signals: pd.DataFrame,
        current_position: float = 0.0,
        risk_tolerance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Get a trading decision based on the generated signals.
        
        Args:
            signals: DataFrame with trading signals
            current_position: Current position size (negative for short)
            risk_tolerance: Risk tolerance (0.0 to 1.0)
            
        Returns:
            Dictionary with trading decision
        """
        if signals.empty:
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': 'No signals available'
            }
        
        # Get the latest signal
        latest = signals.iloc[-1]
        
        # Determine action based on signal and current position
        action = 'hold'
        reason = ''
        
        if latest['signal'] == 1:  # Buy signal
            if current_position <= 0:
                action = 'buy'
                reason = 'Buy signal with no long position'
            elif latest['confidence'] > 0.8:
                action = 'increase'
                reason = 'Strong buy signal with existing long position'
        elif latest['signal'] == -1:  # Sell signal
            if current_position >= 0:
                action = 'sell'
                reason = 'Sell signal with no short position'
            elif latest['confidence'] > 0.8:
                action = 'increase_short'
                reason = 'Strong sell signal with existing short position'
        
        # Adjust position size based on confidence and risk tolerance
        position_size = latest['confidence'] * risk_tolerance
        
        # Compile decision
        decision = {
            'action': action,
            'position_size': position_size,
            'confidence': latest['confidence'],
            'reason': reason,
            'technical_signal': latest['technical_signal'],
            'sentiment_signal': latest['sentiment_signal'],
            'signal_strength': latest['signal_strength'],
            'volatility': latest['volatility']
        }
        
        return decision
