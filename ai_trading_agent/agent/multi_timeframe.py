"""
Multi-Timeframe Analysis Module

This module provides classes and utilities for combining signals from multiple
timeframes to produce more reliable trading signals with higher confidence.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .strategy_manager import Strategy, BaseStrategy, SignalDirection
from ..common.utils import get_logger


class MultiTimeframeStrategy(BaseStrategy):
    """
    Strategy that combines signals from multiple timeframes for confirmation.
    
    Rather than analyzing a single timeframe, this strategy runs the base strategy
    on multiple timeframes and only generates a signal when a minimum number of
    timeframes agree.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Multi-Timeframe strategy.
        
        Args:
            config: Dictionary with configuration parameters
                - base_strategy: The strategy to run on each timeframe
                - timeframes: List of timeframes to analyze
                - min_confirmations: Minimum number of timeframes that must agree
                - weighting: How to weight signals (equal, longer_higher, shorter_higher)
                - base_config: Configuration for the base strategy
        """
        name = config.get("name", "Multi_TF")
        description = config.get("description", "Multi-Timeframe Confirmation Strategy")
        super().__init__(name, description, config)
        
        # Extract strategy parameters with defaults
        self.base_strategy_name = config.get("base_strategy", "MA_Cross")
        self.timeframes = config.get("timeframes", ["1h", "4h", "1d"])
        self.min_confirmations = config.get("min_confirmations", 2)
        self.weighting = config.get("weighting", "longer_higher")  # equal, longer_higher, shorter_higher
        self.base_config = config.get("base_config", {})
        
        # Initialize base strategies
        self.base_strategies = self._init_base_strategies()
        
        # Validate parameters
        if self.min_confirmations > len(self.timeframes):
            self.logger.warning(
                f"min_confirmations ({self.min_confirmations}) is greater than "
                f"the number of timeframes ({len(self.timeframes)}). "
                f"Setting min_confirmations to {len(self.timeframes)}."
            )
            self.min_confirmations = len(self.timeframes)
            
        # Precompute weights for timeframes
        self.tf_weights = self._compute_weights()
        
        self.logger.info(
            f"Initialized Multi-Timeframe Strategy with base={self.base_strategy_name}, "
            f"timeframes={self.timeframes}, min_confirmations={self.min_confirmations}, "
            f"weighting={self.weighting}"
        )
    
    def _init_base_strategies(self) -> Dict[str, Strategy]:
        """Initialize base strategies for each timeframe."""
        strategies = {}
        
        # Import strategy classes
        from .strategy_manager import (
            MovingAverageCrossStrategy,
            RSIOverboughtOversoldStrategy
        )
        
        # Map strategy names to classes
        strategy_classes = {
            "MA_Cross": MovingAverageCrossStrategy,
            "RSI_OB_OS": RSIOverboughtOversoldStrategy
        }
        
        # Check if base strategy is supported
        if self.base_strategy_name not in strategy_classes:
            self.logger.error(f"Unsupported base strategy: {self.base_strategy_name}")
            return strategies
        
        # Initialize a strategy for each timeframe
        for tf in self.timeframes:
            # Create a unique name for the timeframe-specific strategy
            tf_name = f"{self.base_strategy_name}_{tf}"
            
            # Copy the base config and add timeframe-specific adjustments
            tf_config = self.base_config.copy()
            
            # Create the strategy instance
            strategies[tf] = strategy_classes[self.base_strategy_name](tf_config)
            
        return strategies
    
    def _compute_weights(self) -> Dict[str, float]:
        """Compute weights for each timeframe based on weighting strategy."""
        weights = {}
        
        if self.weighting == "equal":
            # Equal weighting
            for tf in self.timeframes:
                weights[tf] = 1.0 / len(self.timeframes)
                
        elif self.weighting == "longer_higher":
            # Higher weight for longer timeframes
            # Convert timeframes to relative weights
            total_weight = 0
            for i, tf in enumerate(self.timeframes):
                weight = i + 1  # Weight increases with index
                weights[tf] = weight
                total_weight += weight
            
            # Normalize weights
            for tf in weights:
                weights[tf] /= total_weight
                
        elif self.weighting == "shorter_higher":
            # Higher weight for shorter timeframes
            # Convert timeframes to relative weights
            total_weight = 0
            for i, tf in enumerate(self.timeframes):
                weight = len(self.timeframes) - i  # Weight decreases with index
                weights[tf] = weight
                total_weight += weight
            
            # Normalize weights
            for tf in weights:
                weights[tf] /= total_weight
                
        else:
            # Default to equal weighting
            for tf in self.timeframes:
                weights[tf] = 1.0 / len(self.timeframes)
        
        return weights
    
    def generate_signals(
        self, 
        market_data: Dict[str, Dict[str, pd.DataFrame]], 
        indicators: Dict[str, Dict[str, Dict]], 
        symbols: List[str]
    ) -> List[Dict]:
        """
        Generate trading signals based on multi-timeframe confirmation.
        
        Args:
            market_data: Dictionary mapping symbols to timeframe-specific market data DataFrames
                Format: {symbol: {timeframe: DataFrame}}
            indicators: Dictionary with calculated indicator values per timeframe
                Format: {symbol: {timeframe: {indicator_name: values}}}
            symbols: List of symbols to generate signals for
            
        Returns:
            List of signal dictionaries
        """
        start_time = datetime.now()
        signals = []
        
        # Validate inputs
        if not isinstance(market_data, dict) or not isinstance(indicators, dict):
            self.logger.error("Invalid market_data or indicators format")
            return signals
        
        for symbol in symbols:
            # Check if we have data for this symbol
            if symbol not in market_data or symbol not in indicators:
                continue
            
            symbol_data = market_data[symbol]
            symbol_indicators = indicators[symbol]
            
            # Collect signals from each timeframe
            tf_signals = {}
            for tf in self.timeframes:
                # Check if we have data for this timeframe
                if tf not in symbol_data or tf not in symbol_indicators:
                    continue
                
                # Get strategy for this timeframe
                tf_strategy = self.base_strategies.get(tf)
                if tf_strategy is None:
                    continue
                
                # Generate signals for this timeframe
                tf_signals_list = tf_strategy.generate_signals(
                    {symbol: symbol_data[tf]},
                    {symbol: symbol_indicators[tf]},
                    [symbol]
                )
                
                # Store the signal (if any)
                if tf_signals_list:
                    tf_signals[tf] = tf_signals_list[0]  # Take the first signal
            
            # Check if we have enough signals
            if len(tf_signals) < self.min_confirmations:
                continue
            
            # Analyze signal agreement
            buy_signals = {}
            sell_signals = {}
            
            for tf, signal in tf_signals.items():
                signal_value = signal["payload"]["signal"]
                if signal_value > 0:
                    buy_signals[tf] = signal
                elif signal_value < 0:
                    sell_signals[tf] = signal
            
            # Generate a combined signal if we have enough confirmations
            final_signal = None
            
            if len(buy_signals) >= self.min_confirmations:
                # Combine buy signals
                final_signal = self._combine_signals(buy_signals, symbol, SignalDirection.BUY)
                
            elif len(sell_signals) >= self.min_confirmations:
                # Combine sell signals
                final_signal = self._combine_signals(sell_signals, symbol, SignalDirection.SELL)
            
            # Add the final signal if we have one
            if final_signal:
                signals.append(final_signal)
        
        # Update metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["signals_generated"] += len(signals)
        self.metrics["avg_execution_time_ms"] = (
            (self.metrics["avg_execution_time_ms"] * (self.metrics["signals_generated"] - len(signals)) + 
             execution_time * len(signals)) / 
            max(1, self.metrics["signals_generated"])
        )
        
        return signals
    
    def _combine_signals(
        self, 
        tf_signals: Dict[str, Dict], 
        symbol: str, 
        direction: SignalDirection
    ) -> Dict:
        """
        Combine signals from multiple timeframes into a single signal.
        
        Args:
            tf_signals: Dictionary mapping timeframes to signals
            symbol: Trading symbol
            direction: Signal direction (BUY or SELL)
            
        Returns:
            Combined signal dictionary
        """
        # Calculate weighted average signal strength
        total_strength = 0.0
        total_weight = 0.0
        
        for tf, signal in tf_signals.items():
            weight = self.tf_weights.get(tf, 1.0 / len(self.timeframes))
            signal_strength = abs(signal["payload"]["signal"])
            total_strength += signal_strength * weight
            total_weight += weight
        
        # Normalize strength
        avg_strength = total_strength / max(total_weight, 0.001)
        
        # Get latest price from any timeframe
        latest_close = next(iter(tf_signals.values()))["payload"]["price_at_signal"]
        
        # Create the combined signal
        combined_signal = {
            "type": "technical_signal",
            "payload": {
                "symbol": symbol,
                "signal": direction.value * avg_strength,
                "confidence": avg_strength,
                "strategy": self.name,
                "price_at_signal": float(latest_close),
                "timestamp": datetime.now().isoformat(),
                "indicators_used": {},  # Will be populated below
                "metadata": {
                    "timeframes_analyzed": list(self.timeframes),
                    "timeframes_confirmed": list(tf_signals.keys()),
                    "confirmation_count": len(tf_signals),
                    "min_confirmations": self.min_confirmations,
                    "weighting_method": self.weighting,
                    "base_strategy": self.base_strategy_name
                }
            }
        }
        
        # Add indicators used from each timeframe
        for tf, signal in tf_signals.items():
            combined_signal["payload"]["indicators_used"][tf] = signal["payload"]["indicators_used"]
        
        # Log the combined signal
        self.logger.info(
            f"Generated {direction.name} signal for {symbol} with "
            f"strength {avg_strength:.4f} from {len(tf_signals)} timeframes"
        )
        
        return combined_signal


class TimeframeManager:
    """
    Manages market data and indicators across multiple timeframes.
    
    This class is responsible for organizing data in the format required
    by the MultiTimeframeStrategy.
    """
    
    def __init__(self, timeframes: List[str]):
        """
        Initialize the TimeframeManager.
        
        Args:
            timeframes: List of timeframes to manage
        """
        self.timeframes = timeframes
        self.logger = get_logger("TimeframeManager")
        
    def organize_market_data(
        self, 
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Organize market data by symbol and timeframe.
        
        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
                Format: {symbol: DataFrame}
                
        Returns:
            Organized data in format: {symbol: {timeframe: DataFrame}}
        """
        organized_data = {}
        
        for symbol, df in market_data.items():
            organized_data[symbol] = {}
            
            # For demonstration, assume the main dataframe is the lowest timeframe
            # In a real implementation, you would reframe data to different timeframes
            for tf in self.timeframes:
                # In a real implementation, you would resample the data to the appropriate timeframe
                # For now, we'll just use the same data for all timeframes
                organized_data[symbol][tf] = df
        
        return organized_data
    
    def organize_indicators(
        self, 
        indicators: Dict[str, Dict]
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Organize indicators by symbol and timeframe.
        
        Args:
            indicators: Dictionary with calculated indicator values
                Format: {symbol: {indicator_name: values}}
                
        Returns:
            Organized data in format: {symbol: {timeframe: {indicator_name: values}}}
        """
        organized_indicators = {}
        
        for symbol, indicator_dict in indicators.items():
            organized_indicators[symbol] = {}
            
            # For demonstration, assume the indicators are the same for all timeframes
            # In a real implementation, you would calculate indicators for each timeframe
            for tf in self.timeframes:
                # In a real implementation, you would calculate the indicators for each timeframe
                # For now, we'll just use the same indicators for all timeframes
                organized_indicators[symbol][tf] = indicator_dict
        
        return organized_indicators
