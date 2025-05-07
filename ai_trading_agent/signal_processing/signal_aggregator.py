"""
Signal Aggregation System.

This module provides a framework for combining trading signals from multiple sources
with configurable weighting and conflict resolution strategies.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

from ai_trading_agent.signal_processing.sentiment_processor import SentimentSignal
from ai_trading_agent.signal_processing.regime import MarketRegime, MarketRegimeDetector

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Enum for different types of trading signals."""
    PRICE = "price"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    CUSTOM = "custom"


class SignalDirection(Enum):
    """Enum for signal directions."""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalTimeframe(Enum):
    """Enum for signal timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class ConflictResolutionStrategy(Enum):
    """Enum for conflict resolution strategies."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    PRIORITY_BASED = "priority_based"
    CONSENSUS_REQUIRED = "consensus_required"
    VETO_POWER = "veto_power"


class TradingSignal:
    """
    Generic trading signal class that can represent signals from any source.
    
    This is a unified representation that can hold technical, sentiment, or any other
    type of signal in a consistent format.
    """
    
    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        direction: SignalDirection,
        strength: float,
        confidence: float,
        timeframe: Union[str, SignalTimeframe],
        source: str,
        timestamp: Optional[datetime] = None,
        expiry: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a trading signal.
        
        Args:
            symbol: The trading symbol (e.g., 'BTC', 'ETH')
            signal_type: Type of signal (technical, sentiment, etc.)
            direction: Direction of the signal (buy, sell, neutral)
            strength: Strength of the signal (0.0 to 1.0)
            confidence: Confidence in the signal (0.0 to 1.0)
            timeframe: Timeframe the signal applies to
            source: Source of the signal (e.g., 'RSI', 'MACD', 'SentimentAnalysis')
            timestamp: When the signal was generated
            expiry: When the signal expires
            metadata: Additional information about the signal
        """
        self.symbol = symbol
        self.signal_type = signal_type if isinstance(signal_type, SignalType) else SignalType(signal_type)
        self.direction = direction if isinstance(direction, SignalDirection) else SignalDirection(direction)
        self.strength = max(0.0, min(1.0, strength))  # Clamp between 0 and 1
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
        
        if isinstance(timeframe, str):
            try:
                self.timeframe = SignalTimeframe(timeframe)
            except ValueError:
                # Default to 1d if not a valid enum value
                logger.warning(f"Invalid timeframe: {timeframe}, defaulting to 1d")
                self.timeframe = SignalTimeframe.D1
        else:
            self.timeframe = timeframe
            
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.expiry = expiry
        self.metadata = metadata or {}
        
        # Calculate a weighted score that combines strength and confidence
        self.score = self.strength * self.confidence
        
        # Track if this signal has been used
        self.used = False
        
    def is_valid(self) -> bool:
        """Check if the signal is still valid (not expired)."""
        if not self.expiry:
            return True
        return datetime.now() < self.expiry
    
    def mark_as_used(self):
        """Mark this signal as used in a trading decision."""
        self.used = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the signal to a dictionary."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'direction': self.direction.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'timeframe': self.timeframe.value,
            'source': self.source,
            'timestamp': self.timestamp,
            'expiry': self.expiry,
            'score': self.score,
            'used': self.used,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_sentiment_signal(cls, sentiment_signal: SentimentSignal) -> 'TradingSignal':
        """
        Convert a SentimentSignal to a TradingSignal.
        
        Args:
            sentiment_signal: SentimentSignal object
            
        Returns:
            TradingSignal object
        """
        # Map sentiment signal type to direction
        direction_map = {
            'buy': SignalDirection.BUY,
            'sell': SignalDirection.SELL,
            'strong_buy': SignalDirection.STRONG_BUY,
            'strong_sell': SignalDirection.STRONG_SELL,
            'neutral': SignalDirection.NEUTRAL,
            None: SignalDirection.NEUTRAL
        }
        
        direction = direction_map.get(sentiment_signal.signal_type, SignalDirection.NEUTRAL)
        
        return cls(
            symbol=sentiment_signal.symbol,
            signal_type=SignalType.SENTIMENT,
            direction=direction,
            strength=sentiment_signal.strength,
            confidence=sentiment_signal.confidence,
            timeframe=sentiment_signal.timeframe,
            source=f"SentimentAnalysis:{sentiment_signal.source if hasattr(sentiment_signal, 'source') else 'Unknown'}",
            timestamp=sentiment_signal.timestamp,
            metadata=sentiment_signal.metadata
        )
    
    def __str__(self) -> str:
        """String representation of the signal."""
        return (f"{self.symbol} {self.direction.value.upper()} signal from {self.source} "
                f"(strength: {self.strength:.2f}, confidence: {self.confidence:.2f}) "
                f"on {self.timeframe.value} timeframe")


class SignalAggregator:
    """
    System for combining signals from multiple sources with configurable weighting and conflict resolution.
    
    This class aggregates different types of trading signals (technical, sentiment, etc.)
    and produces a unified signal based on configurable rules and weights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signal aggregator.
        
        Args:
            config: Configuration dictionary for the aggregator
        """
        self.config = config or {}
        
        # Default weights for different signal types
        self.signal_weights = self.config.get('signal_weights', {
            SignalType.PRICE.value: 0.7,
            SignalType.TECHNICAL.value: 0.8,
            SignalType.SENTIMENT.value: 0.5,
            SignalType.FUNDAMENTAL.value: 0.6,
            SignalType.VOLUME.value: 0.4,
            SignalType.VOLATILITY.value: 0.5,
            SignalType.CUSTOM.value: 0.5
        })
        
        # Default weights for different timeframes
        self.timeframe_weights = self.config.get('timeframe_weights', {
            SignalTimeframe.M1.value: 0.3,
            SignalTimeframe.M5.value: 0.4,
            SignalTimeframe.M15.value: 0.5,
            SignalTimeframe.M30.value: 0.6,
            SignalTimeframe.H1.value: 0.7,
            SignalTimeframe.H4.value: 0.8,
            SignalTimeframe.D1.value: 0.9,
            SignalTimeframe.W1.value: 1.0,
            SignalTimeframe.MN1.value: 1.0
        })
        
        # Default weights for different sources
        self.source_weights = self.config.get('source_weights', {})
        
        # Default conflict resolution strategy
        self.conflict_strategy = ConflictResolutionStrategy(
            self.config.get('conflict_strategy', ConflictResolutionStrategy.WEIGHTED_AVERAGE.value)
        )
        
        # Minimum confidence threshold for considering a signal
        self.min_confidence = float(self.config.get('min_confidence', 0.5))
        
        # Minimum strength threshold for considering a signal
        self.min_strength = float(self.config.get('min_strength', 0.3))
        
        # Minimum number of signals required for aggregation
        self.min_signals = int(self.config.get('min_signals', 1))
        
        # Maximum age of signals to consider (in hours)
        self.max_signal_age_hours = float(self.config.get('max_signal_age_hours', 24))
        
        # Signal history for performance tracking
        self.signal_history = []
        
        # Initialize the market regime detector if enabled
        self.regime_detector = None
        if self.config.get('enable_regime_detection', True):
            self.regime_detector = MarketRegimeDetector()
        
        # Custom filters for signals
        self.custom_filters = []
        
        # Signal quality tracking
        self.signal_quality = {}  # source -> quality score
        
        logger.info(f"Initialized SignalAggregator with {self.conflict_strategy.value} conflict resolution")
    
    def add_custom_filter(self, filter_func: Callable[[TradingSignal], bool]):
        """
        Add a custom filter function for signals.
        
        Args:
            filter_func: Function that takes a TradingSignal and returns True if it should be included
        """
        self.custom_filters.append(filter_func)
        logger.info(f"Added custom signal filter, total filters: {len(self.custom_filters)}")
    
    def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Filter signals based on confidence, strength, age, and custom filters.
        
        Args:
            signals: List of signals to filter
            
        Returns:
            Filtered list of signals
        """
        filtered_signals = []
        now = datetime.now()
        
        for signal in signals:
            # Skip expired signals
            if signal.expiry and now > signal.expiry:
                continue
                
            # Skip signals that are too old
            age_hours = (now - signal.timestamp).total_seconds() / 3600
            if age_hours > self.max_signal_age_hours:
                continue
                
            # Skip signals with low confidence
            if signal.confidence < self.min_confidence:
                continue
                
            # Skip signals with low strength
            if signal.strength < self.min_strength:
                continue
                
            # Apply custom filters
            if all(filter_func(signal) for filter_func in self.custom_filters):
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def get_signal_weight(self, signal: TradingSignal) -> float:
        """
        Calculate the weight for a signal based on its type, timeframe, source, and quality.
        
        Args:
            signal: Signal to calculate weight for
            
        Returns:
            Weight for the signal
        """
        # Base weight from signal type
        type_weight = self.signal_weights.get(
            signal.signal_type.value, 
            0.5  # Default weight if not specified
        )
        
        # Timeframe weight
        timeframe_weight = self.timeframe_weights.get(
            signal.timeframe.value, 
            0.5  # Default weight if not specified
        )
        
        # Source weight
        source_weight = self.source_weights.get(
            signal.source, 
            0.5  # Default weight if not specified
        )
        
        # Quality weight based on historical performance
        quality_weight = self.signal_quality.get(
            signal.source, 
            0.5  # Default weight if no history
        )
        
        # Combine weights
        combined_weight = (type_weight * 0.3 + 
                          timeframe_weight * 0.3 + 
                          source_weight * 0.2 + 
                          quality_weight * 0.2)
        
        return combined_weight
    
    def adjust_weights_for_regime(
        self, 
        signals: List[TradingSignal], 
        weights: List[float], 
        regime: MarketRegime
    ) -> List[float]:
        """
        Adjust signal weights based on the current market regime.
        
        Args:
            signals: List of signals
            weights: Current weights for signals
            regime: Current market regime
            
        Returns:
            Adjusted weights
        """
        if not self.regime_detector:
            return weights
            
        adjusted_weights = weights.copy()
        
        # Get regime-specific parameters
        regime_params = self.regime_detector.get_regime_parameters(regime)
        
        for i, signal in enumerate(signals):
            # Adjust weight based on signal type and regime
            if signal.signal_type == SignalType.TECHNICAL:
                adjusted_weights[i] *= regime_params.get('technical_weight', 1.0)
            elif signal.signal_type == SignalType.SENTIMENT:
                adjusted_weights[i] *= regime_params.get('sentiment_weight', 1.0)
            elif signal.signal_type == SignalType.VOLUME:
                adjusted_weights[i] *= regime_params.get('volume_weight', 1.0)
            elif signal.signal_type == SignalType.VOLATILITY:
                adjusted_weights[i] *= regime_params.get('volatility_weight', 1.0)
        
        return adjusted_weights
    
    def resolve_conflicts_majority_vote(
        self, 
        signals: List[TradingSignal], 
        weights: List[float]
    ) -> Tuple[SignalDirection, float, float]:
        """
        Resolve conflicts using majority vote strategy.
        
        Args:
            signals: List of signals
            weights: Weights for signals
            
        Returns:
            Tuple of (direction, strength, confidence)
        """
        direction_counts = {direction: 0 for direction in SignalDirection}
        
        for signal, weight in zip(signals, weights):
            direction_counts[signal.direction] += weight
        
        # Find direction with highest weighted count
        max_direction = max(direction_counts.items(), key=lambda x: x[1])
        
        # Calculate total votes
        total_votes = sum(direction_counts.values())
        
        # Calculate strength as proportion of votes for winning direction
        strength = max_direction[1] / total_votes if total_votes > 0 else 0.5
        
        # Calculate confidence based on how decisive the vote was
        if total_votes > 0:
            # Sort directions by vote count
            sorted_counts = sorted(direction_counts.values(), reverse=True)
            
            if len(sorted_counts) > 1:
                # Difference between top two directions
                margin = (sorted_counts[0] - sorted_counts[1]) / total_votes
                confidence = 0.5 + (margin / 2)  # Scale to 0.5-1.0 range
            else:
                confidence = 1.0  # Only one direction received votes
        else:
            confidence = 0.5  # No votes
        
        return max_direction[0], strength, confidence
    
    def resolve_conflicts_weighted_average(
        self, 
        signals: List[TradingSignal], 
        weights: List[float]
    ) -> Tuple[SignalDirection, float, float]:
        """
        Resolve conflicts using weighted average strategy.
        
        Args:
            signals: List of signals
            weights: Weights for signals
            
        Returns:
            Tuple of (direction, strength, confidence)
        """
        # Convert directions to numeric values for averaging
        direction_values = {
            SignalDirection.STRONG_SELL: -2.0,
            SignalDirection.SELL: -1.0,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.BUY: 1.0,
            SignalDirection.STRONG_BUY: 2.0
        }
        
        total_weight = sum(weights)
        if total_weight == 0:
            return SignalDirection.NEUTRAL, 0.5, 0.5
        
        # Calculate weighted average
        weighted_sum = sum(direction_values[signal.direction] * weight * signal.strength 
                          for signal, weight in zip(signals, weights))
        
        weighted_avg = weighted_sum / total_weight
        
        # Determine direction from weighted average
        if weighted_avg <= -1.5:
            direction = SignalDirection.STRONG_SELL
        elif weighted_avg < -0.5:
            direction = SignalDirection.SELL
        elif weighted_avg <= 0.5:
            direction = SignalDirection.NEUTRAL
        elif weighted_avg < 1.5:
            direction = SignalDirection.BUY
        else:
            direction = SignalDirection.STRONG_BUY
        
        # Calculate strength based on distance from neutral
        strength = min(1.0, abs(weighted_avg) / 2.0)
        
        # Calculate confidence based on agreement among signals
        signal_directions = [signal.direction for signal in signals]
        if len(set(signal_directions)) == 1:
            # All signals agree
            confidence = 1.0
        else:
            # Calculate standard deviation of direction values
            direction_nums = [direction_values[signal.direction] for signal in signals]
            std_dev = np.std(direction_nums)
            
            # Lower standard deviation means higher confidence
            confidence = max(0.5, 1.0 - (std_dev / 4.0))  # Scale to 0.5-1.0 range
        
        return direction, strength, confidence
    
    def resolve_conflicts_priority_based(
        self, 
        signals: List[TradingSignal], 
        weights: List[float]
    ) -> Tuple[SignalDirection, float, float]:
        """
        Resolve conflicts using priority-based strategy.
        
        Args:
            signals: List of signals
            weights: Weights for signals
            
        Returns:
            Tuple of (direction, strength, confidence)
        """
        # Sort signals by weight (highest priority first)
        sorted_signals = [s for _, s in sorted(zip(weights, signals), key=lambda pair: pair[0], reverse=True)]
        
        # Take the highest priority signal
        top_signal = sorted_signals[0]
        
        # Use its direction and strength
        direction = top_signal.direction
        strength = top_signal.strength
        
        # Calculate confidence based on weight difference
        if len(sorted_signals) > 1:
            # Difference between top two signals
            weight_diff = weights[0] - weights[1]
            confidence = 0.5 + (weight_diff / 2)  # Scale to 0.5-1.0 range
        else:
            confidence = top_signal.confidence
        
        return direction, strength, confidence
    
    def resolve_conflicts_consensus_required(
        self, 
        signals: List[TradingSignal], 
        weights: List[float]
    ) -> Tuple[SignalDirection, float, float]:
        """
        Resolve conflicts using consensus required strategy.
        
        Args:
            signals: List of signals
            weights: Weights for signals
            
        Returns:
            Tuple of (direction, strength, confidence)
        """
        # Count signals for each direction
        direction_counts = {direction: 0 for direction in SignalDirection}
        
        for signal in signals:
            direction_counts[signal.direction] += 1
        
        # Calculate percentage of signals for each direction
        total_signals = len(signals)
        direction_percentages = {d: count / total_signals for d, count in direction_counts.items()}
        
        # Find direction with highest percentage
        max_direction = max(direction_percentages.items(), key=lambda x: x[1])
        
        # Check if we have consensus (more than 2/3 of signals agree)
        if max_direction[1] >= 0.67:
            direction = max_direction[0]
            
            # Calculate strength as average of agreeing signals
            agreeing_signals = [s for s in signals if s.direction == direction]
            strength = sum(s.strength for s in agreeing_signals) / len(agreeing_signals)
            
            # Confidence is high since we have consensus
            confidence = 0.7 + (max_direction[1] - 0.67) * (0.3 / 0.33)  # Scale 0.67-1.0 to 0.7-1.0
        else:
            # No consensus, return neutral with low confidence
            direction = SignalDirection.NEUTRAL
            strength = 0.5
            confidence = 0.5
        
        return direction, strength, confidence
    
    def resolve_conflicts_veto_power(
        self, 
        signals: List[TradingSignal], 
        weights: List[float]
    ) -> Tuple[SignalDirection, float, float]:
        """
        Resolve conflicts using veto power strategy.
        
        Args:
            signals: List of signals
            weights: Weights for signals
            
        Returns:
            Tuple of (direction, strength, confidence)
        """
        # Check for veto signals
        veto_threshold = 0.8  # Signals with weight >= this can veto
        
        # Find signals with veto power
        veto_signals = [s for s, w in zip(signals, weights) if w >= veto_threshold]
        
        if veto_signals:
            # If we have veto signals, use majority vote among them
            veto_weights = [w for s, w in zip(signals, weights) if w >= veto_threshold]
            return self.resolve_conflicts_majority_vote(veto_signals, veto_weights)
        else:
            # Otherwise use weighted average for all signals
            return self.resolve_conflicts_weighted_average(signals, weights)
    
    def aggregate_signals(
        self, 
        signals: List[TradingSignal], 
        price_data: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None
    ) -> Optional[TradingSignal]:
        """
        Aggregate multiple signals into a single trading signal.
        
        Args:
            signals: List of signals to aggregate
            price_data: Optional price data for regime detection
            symbol: Symbol for the aggregated signal
            
        Returns:
            Aggregated trading signal or None if no valid signals
        """
        if not signals:
            logger.warning("No signals to aggregate")
            return None
        
        # Filter signals
        filtered_signals = self.filter_signals(signals)
        
        if len(filtered_signals) < self.min_signals:
            logger.warning(f"Not enough valid signals: {len(filtered_signals)} < {self.min_signals}")
            return None
        
        # Get symbol from signals if not provided
        if not symbol:
            # Check if all signals have the same symbol
            symbols = set(signal.symbol for signal in filtered_signals)
            if len(symbols) == 1:
                symbol = next(iter(symbols))
            else:
                logger.warning(f"Multiple symbols in signals: {symbols}")
                # Use the most common symbol
                symbol_counts = {}
                for s in filtered_signals:
                    symbol_counts[s.symbol] = symbol_counts.get(s.symbol, 0) + 1
                symbol = max(symbol_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate weights for each signal
        weights = [self.get_signal_weight(signal) for signal in filtered_signals]
        
        # Detect market regime if price data is provided
        regime = None
        if price_data is not None and self.regime_detector:
            regime = self.regime_detector.detect_regime(price_data)
            logger.info(f"Detected {regime.value} regime for {symbol}")
            
            # Adjust weights based on regime
            weights = self.adjust_weights_for_regime(filtered_signals, weights, regime)
        
        # Resolve conflicts based on strategy
        if self.conflict_strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            direction, strength, confidence = self.resolve_conflicts_majority_vote(filtered_signals, weights)
        elif self.conflict_strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            direction, strength, confidence = self.resolve_conflicts_weighted_average(filtered_signals, weights)
        elif self.conflict_strategy == ConflictResolutionStrategy.PRIORITY_BASED:
            direction, strength, confidence = self.resolve_conflicts_priority_based(filtered_signals, weights)
        elif self.conflict_strategy == ConflictResolutionStrategy.CONSENSUS_REQUIRED:
            direction, strength, confidence = self.resolve_conflicts_consensus_required(filtered_signals, weights)
        elif self.conflict_strategy == ConflictResolutionStrategy.VETO_POWER:
            direction, strength, confidence = self.resolve_conflicts_veto_power(filtered_signals, weights)
        else:
            # Default to weighted average
            direction, strength, confidence = self.resolve_conflicts_weighted_average(filtered_signals, weights)
        
        # Create aggregated signal
        sources = ", ".join(set(signal.source for signal in filtered_signals))
        timeframe = filtered_signals[0].timeframe  # Use timeframe from first signal
        
        aggregated_signal = TradingSignal(
            symbol=symbol,
            signal_type=SignalType.CUSTOM,
            direction=direction,
            strength=strength,
            confidence=confidence,
            timeframe=timeframe,
            source=f"Aggregated({sources})",
            timestamp=datetime.now(),
            metadata={
                'component_signals': len(filtered_signals),
                'regime': regime.value if regime else None,
                'conflict_strategy': self.conflict_strategy.value
            }
        )
        
        # Add to signal history
        self.signal_history.append({
            'timestamp': datetime.now(),
            'aggregated_signal': aggregated_signal.to_dict(),
            'component_signals': [signal.to_dict() for signal in filtered_signals],
            'weights': weights
        })
        
        # Mark component signals as used
        for signal in filtered_signals:
            signal.mark_as_used()
        
        logger.info(f"Aggregated {len(filtered_signals)} signals into: {aggregated_signal}")
        
        return aggregated_signal
    
    def update_signal_quality(self, source: str, success: bool, profit_loss: float = 0.0):
        """
        Update the quality score for a signal source based on its performance.
        
        Args:
            source: Source of the signal
            success: Whether the signal led to a successful trade
            profit_loss: Profit or loss from the trade (positive for profit)
        """
        # Initialize if not exists
        if source not in self.signal_quality:
            self.signal_quality[source] = 0.5  # Start with neutral quality
        
        # Update quality score
        current_quality = self.signal_quality[source]
        
        if success:
            # Successful trade increases quality
            # Weight by profit_loss if available
            if profit_loss > 0:
                # Scale profit_loss to a reasonable range (0.01 - 0.1)
                pl_factor = min(0.1, max(0.01, abs(profit_loss) / 10))
                self.signal_quality[source] = min(1.0, current_quality + pl_factor)
            else:
                # Small increase for success without profit data
                self.signal_quality[source] = min(1.0, current_quality + 0.02)
        else:
            # Failed trade decreases quality
            if profit_loss < 0:
                # Scale profit_loss to a reasonable range (0.01 - 0.1)
                pl_factor = min(0.1, max(0.01, abs(profit_loss) / 10))
                self.signal_quality[source] = max(0.1, current_quality - pl_factor)
            else:
                # Small decrease for failure without loss data
                self.signal_quality[source] = max(0.1, current_quality - 0.02)
        
        logger.info(f"Updated quality for {source}: {current_quality:.2f} -> {self.signal_quality[source]:.2f}")
    
    def get_signal_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for signals.
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.signal_history:
            return {'error': 'No signal history available'}
        
        # Count signals by source
        source_counts = {}
        for entry in self.signal_history:
            for signal in entry['component_signals']:
                source = signal['source']
                source_counts[source] = source_counts.get(source, 0) + 1
        
        # Count signals by type
        type_counts = {}
        for entry in self.signal_history:
            for signal in entry['component_signals']:
                signal_type = signal['signal_type']
                type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
        
        # Count signals by direction
        direction_counts = {}
        for entry in self.signal_history:
            direction = entry['aggregated_signal']['direction']
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Calculate average confidence and strength
        avg_confidence = sum(entry['aggregated_signal']['confidence'] for entry in self.signal_history) / len(self.signal_history)
        avg_strength = sum(entry['aggregated_signal']['strength'] for entry in self.signal_history) / len(self.signal_history)
        
        # Return statistics
        return {
            'total_aggregations': len(self.signal_history),
            'source_distribution': source_counts,
            'type_distribution': type_counts,
            'direction_distribution': direction_counts,
            'avg_confidence': avg_confidence,
            'avg_strength': avg_strength,
            'signal_quality': self.signal_quality
        }
