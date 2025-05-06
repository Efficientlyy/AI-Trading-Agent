"""
Sentiment Signal Processor

This module processes raw sentiment data into actionable trading signals,
with timeframe awareness and market regime detection capabilities.
It serves as a bridge between the sentiment analysis system and the trading engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from ai_trading_agent.sentiment_analysis.signal_generator import SentimentSignalGenerator
from ai_trading_agent.signal_processing.regime import MarketRegimeDetector


class TradingMode(Enum):
    """Trading modes based on timeframe."""
    SCALPING = "scalping"       # 1-5 minute charts - No sentiment
    INTRADAY = "intraday"       # 15-60 minute charts - Limited sentiment
    SWING = "swing"             # 4h-1d charts - Full sentiment
    POSITION = "position"       # Weekly/Monthly - Heavy sentiment


class SentimentSignal:
    """Represents a trading signal derived from sentiment analysis."""
    
    def __init__(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        signal_type: str,
        strength: float,
        confidence: float,
        timeframe: str,
        source: str = "sentiment_analysis",
        metadata: Optional[Dict] = None
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.signal_type = signal_type  # 'buy', 'sell', 'hold'
        self.strength = strength        # 0.0 to 1.0
        self.confidence = confidence    # 0.0 to 1.0
        self.timeframe = timeframe
        self.source = source
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert the signal to a dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "signal_type": self.signal_type,
            "strength": self.strength,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "source": self.source,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SentimentSignal':
        """Create a signal from a dictionary."""
        return cls(
            symbol=data["symbol"],
            timestamp=data["timestamp"],
            signal_type=data["signal_type"],
            strength=data["strength"],
            confidence=data["confidence"],
            timeframe=data["timeframe"],
            source=data.get("source", "sentiment_analysis"),
            metadata=data.get("metadata", {})
        )


class SentimentSignalProcessor:
    """
    Processes raw sentiment data into actionable trading signals.
    
    This class serves as a bridge between the sentiment analysis system and the trading engine,
    with timeframe awareness and market regime detection capabilities.
    """
    
    def __init__(
        self,
        threshold: float = 0.2,
        window_size: int = 3,
        sentiment_weight: float = 0.4,
        min_confidence: float = 0.6,
        enable_regime_detection: bool = True
    ):
        """
        Initialize the sentiment signal processor.
        
        Args:
            threshold: The sentiment threshold for generating signals (0.0 to 1.0)
            window_size: The number of periods to consider for trend analysis
            sentiment_weight: The weight of sentiment in combined signals (0.0 to 1.0)
            min_confidence: Minimum confidence level for signals to be considered valid
            enable_regime_detection: Whether to adjust weights based on market regime
        """
        self.threshold = threshold
        self.window_size = window_size
        self.sentiment_weight = sentiment_weight
        self.min_confidence = min_confidence
        self.enable_regime_detection = enable_regime_detection
        
        # Initialize the signal generator with our threshold
        self.signal_generator = SentimentSignalGenerator(
            buy_threshold=threshold,
            sell_threshold=-threshold,
            adaptive=True,
            window=window_size
        )
        
        # Initialize the market regime detector if enabled
        self.regime_detector = MarketRegimeDetector() if enable_regime_detection else None
    
    def select_trading_mode(self, timeframe: str) -> TradingMode:
        """
        Select the appropriate trading mode based on the timeframe.
        
        Args:
            timeframe: The timeframe string (e.g., '1m', '5m', '15m', '1h', '4h', '1d', '1w')
            
        Returns:
            The appropriate trading mode for the timeframe
        """
        timeframe = timeframe.lower()
        
        if timeframe in ['1m', '5m']:
            return TradingMode.SCALPING
        elif timeframe in ['15m', '30m', '1h']:
            return TradingMode.INTRADAY
        elif timeframe in ['4h', '1d']:
            return TradingMode.SWING
        elif timeframe in ['1w', '1m']:
            return TradingMode.POSITION
        else:
            # Default to swing trading for unknown timeframes
            return TradingMode.SWING
    
    def get_sentiment_weight(self, trading_mode: TradingMode, market_regime: Optional[Union[str, 'MarketRegime']] = None) -> float:
        """
        Get the appropriate sentiment weight based on trading mode and market regime.
        
        Args:
            trading_mode: The trading mode
            market_regime: The market regime (if available), can be a string or MarketRegime enum
            
        Returns:
            The sentiment weight (0.0 to 1.0)
        """
        # Base weights by trading mode
        base_weights = {
            TradingMode.SCALPING: 0.0,    # No sentiment for scalping
            TradingMode.INTRADAY: 0.3,    # Limited sentiment for intraday
            TradingMode.SWING: 0.6,       # Moderate sentiment for swing trading
            TradingMode.POSITION: 0.8     # High sentiment for position trading
        }
        
        base_weight = base_weights.get(trading_mode, 0.5)  # Default to 0.5
        
        if self.enable_regime_detection and market_regime and self.regime_detector:
            regime_multipliers = {
                "trending": 1.0,    # Full weight in trending markets
                "ranging": 0.7,     # Reduced weight in ranging markets
                "volatile": 0.5     # Minimal weight in volatile markets
            }
            
            # Handle both string and enum values for market_regime
            regime_key = market_regime
            if hasattr(market_regime, 'value'):  # It's an enum
                regime_key = market_regime.value  # Use the enum's value (already a string)
            elif isinstance(market_regime, str):
                regime_key = market_regime.lower()  # Convert string to lowercase
            
            multiplier = regime_multipliers.get(regime_key, 1.0)
            return base_weight * multiplier
        
        return base_weight
    
    def process_sentiment_data(
        self,
        symbol: str,
        historical_sentiment: pd.Series,
        timeframe: str,
        price_data: Optional[pd.DataFrame] = None
    ) -> List[SentimentSignal]:
        """
        Process historical sentiment data into trading signals.
        
        Args:
            symbol: The trading symbol
            historical_sentiment: A pandas Series of sentiment scores with datetime index
            timeframe: The timeframe string (e.g., '1m', '5m', '15m', '1h', '4h', '1d', '1w')
            price_data: Optional price data for correlation analysis and regime detection
            
        Returns:
            A list of SentimentSignal objects
        """
        # Select trading mode based on timeframe
        trading_mode = self.select_trading_mode(timeframe)
        
        # If scalping mode, return empty list (no sentiment signals)
        if trading_mode == TradingMode.SCALPING:
            return []
        
        # Detect market regime if enabled and price data is available
        market_regime = None
        if self.enable_regime_detection and price_data is not None and self.regime_detector:
            market_regime = self.regime_detector.detect_regime(price_data)
        
        # Get appropriate sentiment weight
        sentiment_weight = self.get_sentiment_weight(trading_mode, market_regime)
        
        # If sentiment weight is 0, return empty list
        if sentiment_weight == 0:
            return []
        
        # Generate raw signals using the signal generator
        raw_signals = self.signal_generator.generate_signals_from_scores(historical_sentiment)
        
        # Calculate signal strength and confidence
        signal_strength = self._calculate_signal_strength(historical_sentiment, raw_signals)
        signal_confidence = self._calculate_signal_confidence(historical_sentiment, raw_signals)
        
        # Create SentimentSignal objects
        signals = []
        for timestamp, signal_value in raw_signals.items():
            if signal_value == 0:  # Skip hold signals
                continue
                
            signal_type = "buy" if signal_value > 0 else "sell"
            strength = signal_strength.get(timestamp, 0.5)
            confidence = signal_confidence.get(timestamp, 0.5)
            
            # Skip signals with low confidence
            if confidence < self.min_confidence:
                continue
            
            # Create metadata with additional information
            metadata = {
                "raw_score": historical_sentiment.get(timestamp, 0),
                "market_regime": market_regime,
                "sentiment_weight": sentiment_weight
            }
            
            signals.append(
                SentimentSignal(
                    symbol=symbol,
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    timeframe=timeframe,
                    metadata=metadata
                )
            )
        
        return signals
    
    def _calculate_signal_strength(
        self, 
        sentiment_scores: pd.Series, 
        signals: pd.Series
    ) -> Dict[pd.Timestamp, float]:
        """
        Calculate the strength of each signal based on sentiment magnitude.
        
        Args:
            sentiment_scores: Historical sentiment scores
            signals: Generated trading signals (1, -1, 0)
            
        Returns:
            Dictionary mapping timestamps to signal strength (0.0 to 1.0)
        """
        strengths = {}
        
        # Normalize sentiment scores to 0-1 range for strength calculation
        abs_scores = sentiment_scores.abs()
        max_score = max(abs_scores.max(), 1.0)  # Avoid division by zero
        
        for timestamp, signal in signals.items():
            if signal == 0:  # Skip hold signals
                continue
                
            # Get the absolute sentiment score
            score = abs(sentiment_scores.get(timestamp, 0))
            
            # Calculate strength as normalized score
            strength = min(score / max_score, 1.0)
            
            # Enhance strength based on trend if possible
            if self.window_size > 1 and len(sentiment_scores) > self.window_size:
                # Get window of scores ending at this timestamp
                idx = sentiment_scores.index.get_loc(timestamp)
                if idx >= self.window_size - 1:
                    window = sentiment_scores.iloc[idx - (self.window_size - 1):idx + 1]
                    
                    # Check if trend is consistent with signal
                    if signal > 0 and window.is_monotonic_increasing:
                        strength = min(strength * 1.2, 1.0)  # Boost strength for consistent uptrend
                    elif signal < 0 and window.is_monotonic_decreasing:
                        strength = min(strength * 1.2, 1.0)  # Boost strength for consistent downtrend
            
            strengths[timestamp] = strength
        
        return strengths
    
    def _calculate_signal_confidence(
        self, 
        sentiment_scores: pd.Series, 
        signals: pd.Series
    ) -> Dict[pd.Timestamp, float]:
        """
        Calculate the confidence level of each signal.
        
        Args:
            sentiment_scores: Historical sentiment scores
            signals: Generated trading signals (1, -1, 0)
            
        Returns:
            Dictionary mapping timestamps to confidence levels (0.0 to 1.0)
        """
        confidences = {}
        
        for timestamp, signal in signals.items():
            if signal == 0:  # Skip hold signals
                continue
                
            # Base confidence starts at 0.7
            confidence = 0.7
            
            # Adjust confidence based on sentiment volatility if possible
            if self.window_size > 1 and len(sentiment_scores) > self.window_size:
                # Get window of scores ending at this timestamp
                idx = sentiment_scores.index.get_loc(timestamp)
                if idx >= self.window_size - 1:
                    window = sentiment_scores.iloc[idx - (self.window_size - 1):idx + 1]
                    
                    # Calculate volatility as standard deviation
                    volatility = window.std()
                    
                    # Higher volatility reduces confidence
                    confidence -= min(volatility * 0.5, 0.3)
                    
                    # Check consistency of signals in window
                    if self.signal_generator:
                        window_signals = self.signal_generator.generate_signals_from_scores(window)
                        consistency = (window_signals == signal).mean()
                        
                        # Higher consistency increases confidence
                        confidence += consistency * 0.3
            
            confidences[timestamp] = max(min(confidence, 1.0), 0.0)  # Ensure 0-1 range
        
        return confidences
    
    def combine_with_technical(
        self,
        sentiment_signals: List[SentimentSignal],
        technical_signals: List,  # Type depends on technical signal implementation
        sentiment_weight: Optional[float] = None
    ) -> List:
        """
        Combine sentiment signals with technical signals.
        
        Args:
            sentiment_signals: List of sentiment signals
            technical_signals: List of technical signals
            sentiment_weight: Optional override for sentiment weight
            
        Returns:
            List of combined signals
        """
        # This is a placeholder for the implementation
        # The actual implementation would depend on the structure of technical signals
        # and how you want to combine them with sentiment signals
        
        # For now, we'll just return the sentiment signals
        return sentiment_signals
