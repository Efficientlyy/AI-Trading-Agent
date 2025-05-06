"""
Simple strategy manager module for the AI Trading Agent.

This module provides a simplified strategy manager implementation
with support for real-time data processing.
"""

from typing import Dict, List, Any, Optional, Set
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from ..common import logger


class BaseStrategy:
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
        """
        self.name = name
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for each symbol.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with market data
            current_portfolio: Current portfolio state (optional)
        
        Returns:
            Dictionary mapping symbols to signal dictionaries
        """
        raise NotImplementedError("Subclasses must implement generate_signals")


class SimpleStrategyManager:
    """
    Simple strategy manager that combines signals from multiple strategies.
    Enhanced with real-time data handling capabilities.
    """
    
    def __init__(self, name: str = "SimpleStrategyManager", 
                 aggregation_method: str = "weighted_average",
                 strategy_weights: Optional[Dict[str, float]] = None,
                 min_signal_interval: int = 60,
                 stale_data_threshold: int = 300):
        """
        Initialize the strategy manager.
        
        Args:
            name: Name of the strategy manager
            aggregation_method: Method to combine signals ('weighted_average', 'majority_vote')
            strategy_weights: Dictionary mapping strategy names to weights
            min_signal_interval: Minimum time in seconds between signal changes (prevents excessive trading)
            stale_data_threshold: Maximum age of data in seconds before considering it stale
        """
        self.name = name
        self.aggregation_method = aggregation_method
        self.strategy_weights = strategy_weights or {}
        self.strategies = []
        self.min_signal_interval = min_signal_interval
        self.stale_data_threshold = stale_data_threshold
        
        # Real-time data handling
        self.last_signal_time = {}  # Tracks last signal time per symbol
        self.last_signal_value = {}  # Tracks last signal value per symbol
        self.data_timestamps = {}  # Tracks last data timestamp per symbol
        
        logger.info(f"Initialized {self.name} with aggregation_method={aggregation_method}, " 
                   f"min_signal_interval={min_signal_interval}s, stale_data_threshold={stale_data_threshold}s")
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Add a strategy to the manager.
        
        Args:
            strategy: Strategy instance to add
        """
        self.strategies.append(strategy)
        
        # Set default weight if not provided
        if strategy.name not in self.strategy_weights:
            self.strategy_weights[strategy.name] = 1.0
        
        logger.info(f"Added strategy {strategy.name} to {self.name}")
    
    def process_data_and_generate_signals(self, current_data: Dict[str, pd.Series], 
                                         historical_data: Dict[str, pd.DataFrame],
                                         current_portfolio: Optional[Dict[str, Any]] = None,
                                         **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        """
        Process data and generate combined signals.
        Enhanced with real-time data validation and signal throttling.
        
        Args:
            current_data: Dictionary mapping symbols to Series with current market data
            historical_data: Dictionary mapping symbols to DataFrames with historical market data
            current_portfolio: Current portfolio state (optional)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary mapping symbols to combined signal dictionaries
        """
        if not self.strategies:
            logger.warning(f"{self.name}: No strategies available to generate signals")
            return {symbol: {'signal_strength': 0.0, 'confidence_score': 0.0, 'metadata': {'reason': 'No strategies'}} 
                   for symbol in historical_data.keys()}
        
        # Validate data timestamps
        current_time = time.time()
        stale_symbols = set()
        for symbol, data in current_data.items():
            # Get timestamp from data if available
            if isinstance(data, pd.Series) and 'timestamp' in data:
                timestamp = data['timestamp']
                if isinstance(timestamp, (pd.Timestamp, datetime)):
                    data_time = timestamp.timestamp()
                else:
                    data_time = float(timestamp)
            elif isinstance(data, dict) and 'timestamp' in data:
                timestamp = data['timestamp']
                if isinstance(timestamp, (pd.Timestamp, datetime)):
                    data_time = timestamp.timestamp()
                else:
                    data_time = float(timestamp)
            else:
                # Use the index of the last row in historical data
                try:
                    last_idx = historical_data[symbol].index[-1]
                    if isinstance(last_idx, pd.Timestamp):
                        data_time = last_idx.timestamp()
                    else:
                        # If we can't determine the timestamp, assume it's current
                        data_time = current_time
                except (IndexError, KeyError):
                    data_time = current_time
            
            # Update data timestamp tracking
            self.data_timestamps[symbol] = data_time
            
            # Check if data is stale
            if current_time - data_time > self.stale_data_threshold:
                stale_symbols.add(symbol)
                logger.warning(f"{self.name}: Stale data detected for {symbol}, age: {current_time - data_time:.1f}s")
        
        # Generate signals from all strategies
        all_strategy_signals = {}
        for strategy in self.strategies:
            try:
                # Filter out stale symbols for this strategy
                filtered_historical_data = {
                    symbol: data for symbol, data in historical_data.items() 
                    if symbol not in stale_symbols
                }
                
                if not filtered_historical_data:
                    logger.warning(f"{self.name}: No valid data for {strategy.name} after filtering stale symbols")
                    continue
                
                signals = strategy.generate_signals(filtered_historical_data, current_portfolio)
                all_strategy_signals[strategy.name] = signals
                logger.info(f"{self.name}: Generated signals from {strategy.name}")
            except Exception as e:
                logger.error(f"{self.name}: Error generating signals from {strategy.name}: {e}")
                all_strategy_signals[strategy.name] = {symbol: {'signal_strength': 0.0, 'confidence_score': 0.0, 
                                                              'metadata': {'error': str(e)}}
                                                     for symbol in historical_data.keys()}
        
        # Combine signals
        raw_combined_signals = self._combine_signals(all_strategy_signals)
        
        # Apply signal throttling
        combined_signals = {}
        for symbol, signal in raw_combined_signals.items():
            # Skip stale symbols
            if symbol in stale_symbols:
                logger.info(f"{self.name}: Skipping signal generation for stale symbol {symbol}")
                combined_signals[symbol] = {
                    'signal_strength': 0.0, 
                    'confidence_score': 0.0, 
                    'metadata': {'reason': 'Stale data', 'stale_age': current_time - self.data_timestamps.get(symbol, 0)}
                }
                continue
                
            # Get the current signal strength
            current_signal_strength = signal['signal_strength']
            
            # Check if we have a previous signal
            if symbol in self.last_signal_value:
                last_signal_time = self.last_signal_time.get(symbol, 0)
                last_signal_value = self.last_signal_value.get(symbol, 0)
                
                # Determine if signal direction has changed
                signal_direction_changed = (
                    (last_signal_value > 0.2 and current_signal_strength <= 0.2) or
                    (last_signal_value < -0.2 and current_signal_strength >= -0.2) or
                    (abs(last_signal_value) <= 0.2 and abs(current_signal_strength) > 0.2)
                )
                
                # Apply throttling if signal direction changed and not enough time has passed
                if signal_direction_changed and (current_time - last_signal_time) < self.min_signal_interval:
                    logger.info(f"{self.name}: Throttling signal change for {symbol}, " 
                               f"last change: {current_time - last_signal_time:.1f}s ago")
                    
                    # Keep the previous signal
                    signal['signal_strength'] = last_signal_value
                    
                    # Add throttling info to metadata
                    if 'metadata' not in signal:
                        signal['metadata'] = {}
                    signal['metadata']['throttled'] = True
                    signal['metadata']['original_signal'] = current_signal_strength
                    signal['metadata']['time_since_last_change'] = current_time - last_signal_time
                else:
                    # Update the last signal time and value
                    self.last_signal_time[symbol] = current_time
                    self.last_signal_value[symbol] = current_signal_strength
            else:
                # First signal for this symbol
                self.last_signal_time[symbol] = current_time
                self.last_signal_value[symbol] = current_signal_strength
            
            combined_signals[symbol] = signal
        
        logger.info(f"{self.name}: Generated combined signals for {len(combined_signals)} symbols")
        
        return combined_signals
    
    def _combine_signals(self, all_strategy_signals: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Combine signals from multiple strategies.
        
        Args:
            all_strategy_signals: Dictionary mapping strategy names to their signal dictionaries
        
        Returns:
            Dictionary mapping symbols to combined signal dictionaries
        """
        if not all_strategy_signals:
            return {}
        
        # Get all unique symbols
        all_symbols = set()
        for strategy_signals in all_strategy_signals.values():
            all_symbols.update(strategy_signals.keys())
        
        # Combine signals for each symbol
        combined_signals = {}
        for symbol in all_symbols:
            if self.aggregation_method == "weighted_average":
                combined_signals[symbol] = self._apply_weighted_average(all_strategy_signals, symbol)
            elif self.aggregation_method == "majority_vote":
                combined_signals[symbol] = self._apply_majority_vote(all_strategy_signals, symbol)
            else:
                # Default to simple average
                combined_signals[symbol] = self._apply_simple_average(all_strategy_signals, symbol)
        
        return combined_signals
    
    def _apply_weighted_average(self, all_strategy_signals: Dict[str, Dict[str, Dict[str, Any]]], 
                              symbol: str) -> Dict[str, Any]:
        """
        Apply weighted average to combine signals.
        
        Args:
            all_strategy_signals: Dictionary mapping strategy names to their signal dictionaries
            symbol: Symbol to combine signals for
        
        Returns:
            Combined signal dictionary
        """
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        
        # Collect metadata from all strategies
        metadata = {'strategies': {}}
        
        for strategy_name, strategy_signals in all_strategy_signals.items():
            if symbol in strategy_signals:
                signal = strategy_signals[symbol]
                weight = self.strategy_weights.get(strategy_name, 1.0)
                
                weighted_sum += signal['signal_strength'] * weight
                confidence_sum += signal['confidence_score'] * weight
                total_weight += weight
                
                # Add strategy signal to metadata
                metadata['strategies'][strategy_name] = {
                    'signal_strength': signal['signal_strength'],
                    'confidence_score': signal['confidence_score']
                }
        
        # Calculate weighted average
        if total_weight > 0:
            signal_strength = weighted_sum / total_weight
            confidence_score = confidence_sum / total_weight
        else:
            signal_strength = 0.0
            confidence_score = 0.0
        
        # Add aggregation method to metadata
        metadata['aggregation_method'] = 'weighted_average'
        
        return {
            'signal_strength': signal_strength,
            'confidence_score': confidence_score,
            'metadata': metadata
        }
    
    def _apply_simple_average(self, all_strategy_signals: Dict[str, Dict[str, Dict[str, Any]]], 
                            symbol: str) -> Dict[str, Any]:
        """
        Apply simple average to combine signals.
        
        Args:
            all_strategy_signals: Dictionary mapping strategy names to their signal dictionaries
            symbol: Symbol to combine signals for
        
        Returns:
            Combined signal dictionary
        """
        signal_sum = 0.0
        confidence_sum = 0.0
        count = 0
        
        # Collect metadata from all strategies
        metadata = {'strategies': {}}
        
        for strategy_name, strategy_signals in all_strategy_signals.items():
            if symbol in strategy_signals:
                signal = strategy_signals[symbol]
                
                signal_sum += signal['signal_strength']
                confidence_sum += signal['confidence_score']
                count += 1
                
                # Add strategy signal to metadata
                metadata['strategies'][strategy_name] = {
                    'signal_strength': signal['signal_strength'],
                    'confidence_score': signal['confidence_score']
                }
        
        # Calculate simple average
        if count > 0:
            signal_strength = signal_sum / count
            confidence_score = confidence_sum / count
        else:
            signal_strength = 0.0
            confidence_score = 0.0
        
        # Add aggregation method to metadata
        metadata['aggregation_method'] = 'simple_average'
        
        return {
            'signal_strength': signal_strength,
            'confidence_score': confidence_score,
            'metadata': metadata
        }
    
    def _apply_majority_vote(self, all_strategy_signals: Dict[str, Dict[str, Dict[str, Any]]], 
                           symbol: str) -> Dict[str, Any]:
        """
        Apply majority vote to combine signals.
        
        Args:
            all_strategy_signals: Dictionary mapping strategy names to their signal dictionaries
            symbol: Symbol to combine signals for
        
        Returns:
            Combined signal dictionary
        """
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        confidence_sum = 0.0
        count = 0
        
        # Collect metadata from all strategies
        metadata = {'strategies': {}, 'votes': {}}
        
        for strategy_name, strategy_signals in all_strategy_signals.items():
            if symbol in strategy_signals:
                signal = strategy_signals[symbol]
                signal_strength = signal['signal_strength']
                
                if signal_strength > 0.2:
                    buy_votes += 1
                elif signal_strength < -0.2:
                    sell_votes += 1
                else:
                    hold_votes += 1
                
                confidence_sum += signal['confidence_score']
                count += 1
                
                # Add strategy signal to metadata
                metadata['strategies'][strategy_name] = {
                    'signal_strength': signal_strength,
                    'confidence_score': signal['confidence_score']
                }
        
        # Determine majority
        metadata['votes'] = {'buy': buy_votes, 'sell': sell_votes, 'hold': hold_votes}
        
        if buy_votes > sell_votes and buy_votes > hold_votes:
            signal_strength = 1.0
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            signal_strength = -1.0
        else:
            signal_strength = 0.0
        
        # Calculate average confidence
        confidence_score = confidence_sum / count if count > 0 else 0.0
        
        # Add aggregation method to metadata
        metadata['aggregation_method'] = 'majority_vote'
        
        return {
            'signal_strength': signal_strength,
            'confidence_score': confidence_score,
            'metadata': metadata
        }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], 
                        current_portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals for each symbol.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with market data
            current_portfolio: Current portfolio state (optional)
        
        Returns:
            Dictionary mapping symbols to signal dictionaries
        """
        # Convert data to the format expected by process_data_and_generate_signals
        current_data = {symbol: df.iloc[-1] for symbol, df in data.items()}
        
        return self.process_data_and_generate_signals(
            current_data=current_data,
            historical_data=data,
            current_portfolio=current_portfolio
        )