"""
Meta-strategy framework for combining signals from multiple strategies.

This module provides a mechanism to combine signals from multiple trading strategies,
apply filtering rules, and generate consensus-based trading decisions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable

from src.common.logging import get_logger
from src.models.signals import Signal, SignalType
from src.models.market_data import TimeFrame
# Use relative imports to avoid circular imports
from .strategy import Strategy


class StrategyWeighting:
    """Configuration for strategy weighting within a meta-strategy."""
    
    def __init__(
        self,
        strategy_id: str,
        weight: float = 1.0,
        min_confidence: float = 0.0,
        enabled: bool = True
    ):
        """Initialize strategy weighting configuration.
        
        Args:
            strategy_id: Identifier of the strategy
            weight: Relative weight of the strategy's signals (default: 1.0)
            min_confidence: Minimum confidence threshold to consider signals (default: 0.0)
            enabled: Whether the strategy is enabled (default: True)
        """
        self.strategy_id = strategy_id
        self.weight = weight
        self.min_confidence = min_confidence
        self.enabled = enabled


class SignalCombinationMethod:
    """Enumeration of methods for combining signals from multiple strategies."""
    
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    UNANIMOUS = "unanimous"
    ANY = "any"
    CUSTOM = "custom"


class MetaStrategy(Strategy):
    """A strategy that combines signals from multiple sub-strategies.
    
    The meta-strategy aggregates signals from various strategies and applies
    combination rules to generate consensus-based trading decisions.
    """
    
    def __init__(
        self,
        strategy_id: str,
        sub_strategies: Optional[List[Strategy]] = None,
        strategy_weights: Optional[Dict[str, StrategyWeighting]] = None,
        combination_method: str = SignalCombinationMethod.WEIGHTED_AVERAGE,
        min_consensus_pct: float = 0.5,
        min_overall_confidence: float = 0.5,
        signal_window: int = 300,  # 5 minutes in seconds
        custom_combination_func: Optional[Callable] = None
    ):
        """Initialize the meta-strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy
            sub_strategies: List of strategies to combine
            strategy_weights: Dictionary mapping strategy_ids to their weights
            combination_method: Method to combine signals
            min_consensus_pct: Minimum percentage of strategies needed for consensus
            min_overall_confidence: Minimum combined confidence to generate a signal
            signal_window: Time window in seconds to consider signals as concurrent
            custom_combination_func: Custom function for combining signals (if combination_method is CUSTOM)
        """
        super().__init__(strategy_id=strategy_id)
        
        self.sub_strategies = sub_strategies or []
        self.strategy_weights = strategy_weights or {}
        self.combination_method = combination_method
        self.min_consensus_pct = min_consensus_pct
        self.min_overall_confidence = min_overall_confidence
        self.signal_window = signal_window
        self.custom_combination_func = custom_combination_func
        
        # Active signals from sub-strategies
        self.active_signals: Dict[str, List[Signal]] = {}
        
        # Recent signals for tracking purposes
        self.recent_signals: Dict[str, List[Signal]] = {}
        
        # Logger
        self.logger = get_logger("strategy", f"{strategy_id}")
    
    async def _strategy_initialize(self) -> None:
        """Initialize the meta-strategy and all sub-strategies."""
        self.logger.info(f"Initializing meta-strategy '{self.strategy_id}'")
        
        # Initialize all sub-strategies
        for strategy in self.sub_strategies:
            if not strategy.is_initialized:
                strategy.initialize()
        
        # Initialize signal storage
        for strategy in self.sub_strategies:
            self.active_signals[strategy.strategy_id] = []
            self.recent_signals[strategy.strategy_id] = []
        
        # Set up signal handlers for sub-strategies
        for strategy in self.sub_strategies:
            await self._register_signal_handler(strategy)
    
    async def _strategy_start(self) -> None:
        """Start the meta-strategy and all sub-strategies."""
        self.logger.info(f"Starting meta-strategy '{self.strategy_id}'")
        
        # Start all sub-strategies
        for strategy in self.sub_strategies:
            if not strategy.is_running:
                strategy.start()
    
    async def _strategy_stop(self) -> None:
        """Stop the meta-strategy and all sub-strategies."""
        self.logger.info(f"Stopping meta-strategy '{self.strategy_id}'")
        
        # Stop all sub-strategies
        for strategy in self.sub_strategies:
            if strategy.is_running:
                strategy.stop()
    
    # Implement required abstract methods from Strategy
    async def process_candle(self, symbol: str, timeframe: TimeFrame, candle: Dict[str, Any]) -> None:
        """Process candle data (implementation required by Strategy).
        
        MetaStrategy doesn't directly process candles, as it relies on sub-strategies.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe of the candle
            candle: Candle data dictionary
        """
        # MetaStrategy doesn't directly process candles - sub-strategies do
        pass
    
    async def process_trade(self, symbol: str, trade: Dict[str, Any]) -> None:
        """Process trade data (implementation required by Strategy).
        
        MetaStrategy doesn't directly process trades, as it relies on sub-strategies.
        
        Args:
            symbol: Trading pair symbol
            trade: Trade data dictionary
        """
        # MetaStrategy doesn't directly process trades - sub-strategies do
        pass
    
    async def _register_signal_handler(self, strategy: Strategy) -> None:
        """Register a signal handler for a sub-strategy.
        
        Args:
            strategy: The sub-strategy to register a handler for
        """
        # Store the original publish_signal method
        original_publish_signal = strategy.publish_signal
        
        # Create a new signal handler
        async def signal_handler(signal: Signal) -> None:
            # Process the signal within the meta-strategy
            await self._process_sub_strategy_signal(strategy.strategy_id, signal)
            
            # Still call the original method to ensure signals propagate
            await original_publish_signal(signal)
        
        # Replace the strategy's publish_signal method
        strategy.publish_signal = signal_handler
    
    async def _process_sub_strategy_signal(self, strategy_id: str, signal: Signal) -> None:
        """Process a signal from a sub-strategy.
        
        Args:
            strategy_id: ID of the strategy that generated the signal
            signal: The signal to process
        """
        self.logger.debug(f"Received signal from {strategy_id}: {signal.direction} {signal.signal_type.name}")
        
        # Check if strategy is enabled and signal meets confidence threshold
        weighting = self.strategy_weights.get(strategy_id)
        if weighting and not weighting.enabled:
            self.logger.debug(f"Ignoring signal from disabled strategy {strategy_id}")
            return
        
        if weighting and signal.confidence < weighting.min_confidence:
            self.logger.debug(f"Signal confidence {signal.confidence} below threshold {weighting.min_confidence}")
            return
        
        # Store the signal
        self.active_signals[strategy_id].append(signal)
        self.recent_signals[strategy_id].append(signal)
        
        # Clean up old signals
        self._cleanup_old_signals()
        
        # Process signals to check for consensus
        await self._check_for_consensus(signal.symbol)
    
    def _cleanup_old_signals(self) -> None:
        """Remove signals that are outside the signal window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.signal_window)
        
        for strategy_id in self.active_signals:
            # Remove old signals
            self.active_signals[strategy_id] = [
                s for s in self.active_signals[strategy_id]
                if s.timestamp > cutoff
            ]
            
            # Keep only the 100 most recent signals for history
            self.recent_signals[strategy_id] = self.recent_signals[strategy_id][-100:]
    
    async def _check_for_consensus(self, symbol: str) -> None:
        """Check if there's a consensus among sub-strategies for a symbol.
        
        Args:
            symbol: The trading pair symbol to check consensus for
        """
        # Get active entry signals for the symbol
        long_signals: List[Tuple[str, Signal]] = []
        short_signals: List[Tuple[str, Signal]] = []
        
        for strategy_id, signals in self.active_signals.items():
            # Filter for entry signals for this symbol
            for signal in signals:
                if signal.symbol == symbol and signal.signal_type == SignalType.ENTRY:
                    if signal.direction == "long":
                        long_signals.append((strategy_id, signal))
                    elif signal.direction == "short":
                        short_signals.append((strategy_id, signal))
        
        # Check for long consensus
        if long_signals:
            await self._evaluate_consensus(symbol, "long", long_signals)
        
        # Check for short consensus
        if short_signals:
            await self._evaluate_consensus(symbol, "short", short_signals)
    
    async def _evaluate_consensus(self, symbol: str, direction: str, signals: List[Tuple[str, Signal]]) -> None:
        """Evaluate if there's a consensus for a direction and generate a meta-signal if appropriate.
        
        Args:
            symbol: Trading pair symbol
            direction: Signal direction ("long" or "short")
            signals: List of (strategy_id, signal) tuples with active signals
        """
        # Count unique strategies with signals
        unique_strategies = set(strategy_id for strategy_id, _ in signals)
        
        # Calculate consensus percentage
        total_active_strategies = len([s for s in self.sub_strategies if s.is_running])
        consensus_pct = len(unique_strategies) / total_active_strategies if total_active_strategies > 0 else 0
        
        # Check if we have minimum consensus
        if consensus_pct < self.min_consensus_pct:
            self.logger.debug(
                f"Consensus {consensus_pct:.2f} below threshold {self.min_consensus_pct} for {symbol} {direction}"
            )
            return
        
        # Combine signals based on the selected method
        combined_confidence, price, metadata = self._combine_signals(unique_strategies, signals)
        
        # Check if combined confidence meets threshold
        if combined_confidence < self.min_overall_confidence:
            self.logger.debug(
                f"Combined confidence {combined_confidence:.2f} below threshold {self.min_overall_confidence}"
            )
            return
        
        # Generate a meta-strategy signal
        sub_strat_ids = ", ".join(unique_strategies)
        reason = f"Consensus from {len(unique_strategies)} strategies ({sub_strat_ids})"
        
        await self.publish_signal(
            Signal(
                symbol=symbol,
                strategy_id=self.strategy_id,
                signal_type=SignalType.ENTRY,
                direction=direction,
                price=price,
                confidence=combined_confidence,
                timestamp=datetime.now(),
                reason=reason,
                metadata={
                    "consensus_pct": consensus_pct,
                    "contributing_strategies": list(unique_strategies),
                    **metadata
                }
            )
        )
        
        self.logger.info(
            f"Generated {direction} signal for {symbol} "
            f"with confidence {combined_confidence:.2f} based on {len(unique_strategies)} strategies"
        )
    
    def _combine_signals(
        self, 
        strategy_ids: Set[str], 
        signals: List[Tuple[str, Signal]]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Combine signals using the specified method.
        
        Args:
            strategy_ids: Set of strategy IDs that generated signals
            signals: List of (strategy_id, signal) tuples
            
        Returns:
            Tuple of (combined_confidence, price, metadata)
        """
        # Default values
        combined_confidence = 0.0
        price = 0.0
        total_weight = 0.0
        metadata = {"signal_weights": {}}
        
        # Get the most recent signal from each strategy
        latest_signals: Dict[str, Signal] = {}
        for strategy_id, signal in signals:
            if strategy_id not in latest_signals or signal.timestamp > latest_signals[strategy_id].timestamp:
                latest_signals[strategy_id] = signal
        
        # Weighted average method
        if self.combination_method == SignalCombinationMethod.WEIGHTED_AVERAGE:
            weighted_sum = 0.0
            price_sum = 0.0
            
            for strategy_id, signal in latest_signals.items():
                # Get weight for this strategy (default to 1.0)
                weight = 1.0
                if strategy_id in self.strategy_weights:
                    weight_obj = self.strategy_weights.get(strategy_id)
                    if weight_obj:
                        weight = weight_obj.weight
                
                weighted_sum += signal.confidence * weight
                price_sum += signal.price * weight
                total_weight += weight
                
                # Store weight in metadata
                metadata["signal_weights"][strategy_id] = weight
            
            if total_weight > 0:
                combined_confidence = weighted_sum / total_weight
                price = price_sum / total_weight
        
        # Majority vote method
        elif self.combination_method == SignalCombinationMethod.MAJORITY_VOTE:
            # Count strategies with confidence above 0.5
            strong_signals = [
                (sid, sig) for sid, sig in latest_signals.items()
                if sig.confidence > 0.5
            ]
            
            if len(strong_signals) > len(latest_signals) / 2:
                # Majority has strong confidence
                combined_confidence = sum(sig.confidence for _, sig in strong_signals) / len(strong_signals)
                price = sum(sig.price for _, sig in strong_signals) / len(strong_signals)
            else:
                # No majority with strong confidence
                combined_confidence = 0.0
                price = 0.0
        
        # Unanimous method
        elif self.combination_method == SignalCombinationMethod.UNANIMOUS:
            # All strategies must have signals
            if len(strategy_ids) == len(self.sub_strategies):
                # Fix: Using list of Signal objects instead of trying to iterate dictionary values
                confidence_values = [signal.confidence for signal in latest_signals.values()]
                price_values = [signal.price for signal in latest_signals.values()]
                
                combined_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
                price = sum(price_values) / len(price_values) if price_values else 0.0
            else:
                combined_confidence = 0.0
                price = 0.0
        
        # Any signal method
        elif self.combination_method == SignalCombinationMethod.ANY:
            # Find the signal with highest confidence
            if latest_signals:
                best_signal = max(latest_signals.values(), key=lambda s: s.confidence)
                combined_confidence = best_signal.confidence
                price = best_signal.price
        
        # Custom method
        elif self.combination_method == SignalCombinationMethod.CUSTOM and self.custom_combination_func:
            combined_confidence, price, custom_metadata = self.custom_combination_func(latest_signals)
            if custom_metadata:
                metadata.update(custom_metadata)
        
        return combined_confidence, price, metadata
    
    def get_strategy_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """Get a sub-strategy by its ID.
        
        Args:
            strategy_id: ID of the strategy to find
            
        Returns:
            The strategy if found, None otherwise
        """
        for strategy in self.sub_strategies:
            if strategy.strategy_id == strategy_id:
                return strategy
        return None
    
    def add_strategy(self, strategy: Strategy, weight: float = 1.0, min_confidence: float = 0.0) -> None:
        """Add a strategy to the meta-strategy.
        
        Args:
            strategy: The strategy to add
            weight: Weight for the strategy's signals
            min_confidence: Minimum confidence threshold for signals
        """
        if strategy not in self.sub_strategies:
            self.sub_strategies.append(strategy)
            self.active_signals[strategy.strategy_id] = []
            self.recent_signals[strategy.strategy_id] = []
            
            # Add weighting
            self.strategy_weights[strategy.strategy_id] = StrategyWeighting(
                strategy_id=strategy.strategy_id,
                weight=weight,
                min_confidence=min_confidence
            )
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the meta-strategy.
        
        Args:
            strategy_id: ID of the strategy to remove
            
        Returns:
            True if the strategy was removed, False otherwise
        """
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy:
            self.sub_strategies.remove(strategy)
            if strategy_id in self.active_signals:
                del self.active_signals[strategy_id]
            if strategy_id in self.recent_signals:
                del self.recent_signals[strategy_id]
            if strategy_id in self.strategy_weights:
                del self.strategy_weights[strategy_id]
            return True
        return False
    
    def update_strategy_weight(self, strategy_id: str, weight: float) -> bool:
        """Update the weight for a strategy.
        
        Args:
            strategy_id: ID of the strategy to update
            weight: New weight value
            
        Returns:
            True if the weight was updated, False if strategy not found
        """
        if strategy_id in self.strategy_weights:
            self.strategy_weights[strategy_id].weight = weight
            return True
        return False
    
    def enable_strategy(self, strategy_id: str) -> bool:
        """Enable a strategy.
        
        Args:
            strategy_id: ID of the strategy to enable
            
        Returns:
            True if the strategy was enabled, False if not found
        """
        if strategy_id in self.strategy_weights:
            self.strategy_weights[strategy_id].enabled = True
            return True
        return False
    
    def disable_strategy(self, strategy_id: str) -> bool:
        """Disable a strategy.
        
        Args:
            strategy_id: ID of the strategy to disable
            
        Returns:
            True if the strategy was disabled, False if not found
        """
        if strategy_id in self.strategy_weights:
            self.strategy_weights[strategy_id].enabled = False
            return True
        return False
    
    def get_active_signals(self, strategy_id: Optional[str] = None) -> Dict[str, List[Signal]]:
        """Get active signals from sub-strategies.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            
        Returns:
            Dictionary of strategy_id to list of active signals
        """
        if strategy_id:
            return {strategy_id: self.active_signals.get(strategy_id, [])}
        return self.active_signals
    
    def get_recent_signals(self, strategy_id: Optional[str] = None) -> Dict[str, List[Signal]]:
        """Get recent signals from sub-strategies.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            
        Returns:
            Dictionary of strategy_id to list of recent signals
        """
        if strategy_id:
            return {strategy_id: self.recent_signals.get(strategy_id, [])}
        return self.recent_signals
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the meta-strategy.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "num_sub_strategies": len(self.sub_strategies),
            "combination_method": self.combination_method,
            "min_consensus_pct": self.min_consensus_pct,
            "min_overall_confidence": self.min_overall_confidence,
            "signal_window": self.signal_window,
            "strategy_weights": {
                sid: sw.weight for sid, sw in self.strategy_weights.items()
            },
            "strategy_enabled": {
                sid: sw.enabled for sid, sw in self.strategy_weights.items()
            },
            "active_signals_count": {
                sid: len(signals) for sid, signals in self.active_signals.items()
            },
            "recent_signals_count": {
                sid: len(signals) for sid, signals in self.recent_signals.items()
            }
        }
        
        return stats 