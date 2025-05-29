"""
Strategy Coordination Module

This module handles coordination between multiple trading strategies to:
1. Analyze correlations between strategy performance
2. Allocate resources efficiently across strategies
3. Generate combined signals considering strategy overlaps
4. Prevent conflicting positions and optimize capital usage

The coordination system helps the AI Trading Agent achieve superior performance
by leveraging the strengths of multiple strategies while mitigating their weaknesses.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from ai_trading_agent.agent.strategy import BaseStrategy, RichSignal, RichSignalsDict
from ai_trading_agent.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class StrategyCoordinator:
    """
    Coordinates multiple trading strategies to improve overall performance
    by analyzing correlations, allocating resources, and resolving conflicts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Strategy Coordinator.
        
        Args:
            config: Configuration dictionary with parameters
                - strategies: List of strategy names to coordinate
                - lookback_periods: Number of periods to consider for correlation analysis
                - min_correlation_threshold: Minimum correlation to consider significant
                - max_position_overlap: Maximum allowed position overlap between strategies
                - conflict_resolution_method: Method to resolve conflicting signals
                - capital_allocation_method: Method to allocate capital across strategies
                - coordination_frequency: How often to perform full coordination analysis
                - enable_adaptive_allocation: Whether to adapt allocations based on performance
        """
        self.strategies = config.get("strategies", [])
        self.lookback_periods = config.get("lookback_periods", 50)
        self.min_correlation_threshold = config.get("min_correlation_threshold", 0.3)
        self.max_position_overlap = config.get("max_position_overlap", 0.7)
        self.conflict_resolution_method = config.get("conflict_resolution_method", "performance_weighted")
        self.capital_allocation_method = config.get("capital_allocation_method", "dynamic")
        self.coordination_frequency = config.get("coordination_frequency", 5)
        self.enable_adaptive_allocation = config.get("enable_adaptive_allocation", True)
        
        # Strategy performance tracking
        self.performance_history = {}  # strategy -> symbol -> list of performance metrics
        self.correlation_matrix = {}  # (strategy1, strategy2) -> correlation value
        self.strategy_allocations = {}  # strategy -> allocation percentage
        self.coordination_counter = 0
        self.active_symbols = set()
        
        # Initialize strategy allocations
        self._initialize_allocations()
        
        logger.info(f"Strategy Coordinator initialized with {len(self.strategies)} strategies")
    
    def _initialize_allocations(self) -> None:
        """Initialize capital allocations for strategies."""
        if not self.strategies:
            return
            
        # Start with equal allocation
        equal_allocation = 1.0 / len(self.strategies)
        for strategy in self.strategies:
            self.strategy_allocations[strategy] = equal_allocation
            
        logger.info(f"Initialized equal strategy allocations: {equal_allocation:.2f} each")
    
    def update_performance(self, strategy_name: str, symbol: str, 
                          timestamp: str, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            timestamp: Timestamp of the performance record
            metrics: Performance metrics (returns, sharpe, drawdown, etc.)
        """
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = {}
            
        if symbol not in self.performance_history[strategy_name]:
            self.performance_history[strategy_name][symbol] = []
            
        # Add timestamp to metrics
        metrics["timestamp"] = timestamp
        
        # Add to performance history
        self.performance_history[strategy_name][symbol].append(metrics)
        
        # Trim history if needed
        max_history = self.lookback_periods * 2
        if len(self.performance_history[strategy_name][symbol]) > max_history:
            self.performance_history[strategy_name][symbol] = \
                self.performance_history[strategy_name][symbol][-max_history:]
                
        # Add to tracked symbols
        self.active_symbols.add(symbol)
        
        # Increment coordination counter
        self.coordination_counter += 1
        
        # Check if it's time to perform coordination analysis
        if self.coordination_counter >= self.coordination_frequency:
            self._analyze_strategy_correlations()
            self._update_capital_allocations()
            self.coordination_counter = 0
    
    def _analyze_strategy_correlations(self) -> None:
        """
        Analyze correlations between strategies based on performance.
        Updates the correlation matrix.
        """
        if len(self.strategies) < 2:
            return
            
        # Get list of symbols with performance data for multiple strategies
        common_symbols = set()
        for symbol in self.active_symbols:
            strategies_with_data = sum(1 for s in self.strategies 
                                     if s in self.performance_history 
                                     and symbol in self.performance_history[s]
                                     and len(self.performance_history[s][symbol]) >= 10)
            if strategies_with_data >= 2:
                common_symbols.add(symbol)
                
        if not common_symbols:
            logger.warning("Not enough common data to analyze strategy correlations")
            return
            
        # Calculate correlations between strategies
        for i, strategy1 in enumerate(self.strategies):
            for strategy2 in self.strategies[i+1:]:
                correlations = []
                
                for symbol in common_symbols:
                    if (strategy1 in self.performance_history 
                        and strategy2 in self.performance_history
                        and symbol in self.performance_history[strategy1]
                        and symbol in self.performance_history[strategy2]):
                        
                        # Get returns for both strategies
                        returns1 = [p.get("returns", 0) for p in self.performance_history[strategy1][symbol][-self.lookback_periods:]]
                        returns2 = [p.get("returns", 0) for p in self.performance_history[strategy2][symbol][-self.lookback_periods:]]
                        
                        # Trim to equal length if needed
                        min_len = min(len(returns1), len(returns2))
                        if min_len >= 10:  # Need at least 10 data points for meaningful correlation
                            returns1 = returns1[-min_len:]
                            returns2 = returns2[-min_len:]
                            
                            # Calculate correlation
                            try:
                                corr = np.corrcoef(returns1, returns2)[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                            except Exception as e:
                                logger.error(f"Error calculating correlation: {e}")
                
                # Average correlation across symbols
                if correlations:
                    avg_corr = np.mean(correlations)
                    self.correlation_matrix[(strategy1, strategy2)] = avg_corr
                    self.correlation_matrix[(strategy2, strategy1)] = avg_corr
                    
                    logger.info(f"Correlation between {strategy1} and {strategy2}: {avg_corr:.3f}")
    
    def _update_capital_allocations(self) -> None:
        """
        Update capital allocations for strategies based on performance and correlations.
        """
        if not self.enable_adaptive_allocation or len(self.strategies) < 2:
            return
            
        # Calculate performance metrics for each strategy
        strategy_metrics = {}
        for strategy in self.strategies:
            if strategy not in self.performance_history:
                continue
                
            # Calculate average Sharpe ratio across symbols
            sharpe_values = []
            for symbol in self.performance_history[strategy]:
                symbol_data = self.performance_history[strategy][symbol]
                if len(symbol_data) >= 10:
                    recent_data = symbol_data[-10:]  # Use last 10 periods
                    sharpe_values.extend([p.get("sharpe_ratio", 0) for p in recent_data if "sharpe_ratio" in p])
            
            if sharpe_values:
                avg_sharpe = np.mean(sharpe_values)
                strategy_metrics[strategy] = {
                    "sharpe_ratio": avg_sharpe,
                    "correlation_penalty": 0.0,  # Will be calculated next
                    "allocation_score": 0.0      # Will be calculated next
                }
                
        # Calculate correlation penalties
        for strategy in strategy_metrics:
            penalty = 0.0
            correlation_count = 0
            
            for other_strategy in strategy_metrics:
                if strategy != other_strategy:
                    key = (strategy, other_strategy)
                    if key in self.correlation_matrix:
                        # Higher positive correlation = higher penalty
                        corr = self.correlation_matrix[key]
                        if abs(corr) > self.min_correlation_threshold:
                            # Only penalize positive correlations
                            if corr > 0:
                                penalty += corr
                            correlation_count += 1
            
            # Average the penalty
            if correlation_count > 0:
                strategy_metrics[strategy]["correlation_penalty"] = penalty / correlation_count
            
            # Calculate final score (Sharpe - Correlation penalty)
            raw_score = strategy_metrics[strategy]["sharpe_ratio"] - 0.5 * strategy_metrics[strategy]["correlation_penalty"]
            strategy_metrics[strategy]["allocation_score"] = max(0.1, raw_score)  # Ensure minimum allocation
        
        # Normalize scores to get allocations
        total_score = sum(m["allocation_score"] for m in strategy_metrics.values())
        
        if total_score > 0:
            for strategy in strategy_metrics:
                allocation = strategy_metrics[strategy]["allocation_score"] / total_score
                self.strategy_allocations[strategy] = allocation
                
            logger.info(f"Updated strategy allocations: {self.strategy_allocations}")
        else:
            logger.warning("Could not update allocations: total score is zero")
    
    def coordinate_signals(self, strategy_signals: Dict[str, RichSignalsDict]) -> Dict[str, Dict[str, RichSignal]]:
        """
        Coordinate signals from multiple strategies to produce optimal combined signals.
        
        Args:
            strategy_signals: Dictionary mapping strategy names to their signal dictionaries
            
        Returns:
            Coordinated signals dictionary
        """
        if not strategy_signals:
            return {}
            
        # Track all symbols across strategies
        all_symbols = set()
        for strategy, signals in strategy_signals.items():
            all_symbols.update(signals.keys())
            
        # Result dictionary
        coordinated_signals = {}
        
        for symbol in all_symbols:
            # Collect all signals for this symbol
            symbol_signals = {}
            for strategy, signals in strategy_signals.items():
                if symbol in signals:
                    symbol_signals[strategy] = signals[symbol]
            
            # Coordinate signals for this symbol
            if len(symbol_signals) == 1:
                # Only one strategy has a signal, use it with its allocation
                strategy = next(iter(symbol_signals.keys()))
                allocation = self.strategy_allocations.get(strategy, 1.0)
                
                # Apply allocation to signal quantities
                coordinated_signal = self._apply_allocation_to_signal(
                    symbol_signals[strategy],
                    allocation
                )
                coordinated_signals[symbol] = coordinated_signal
                
            elif len(symbol_signals) > 1:
                # Multiple strategies have signals, need conflict resolution
                coordinated_signal = self._resolve_signal_conflicts(
                    symbol,
                    symbol_signals
                )
                coordinated_signals[symbol] = coordinated_signal
                
        return coordinated_signals
    
    def _apply_allocation_to_signal(self, 
                                   signal_dict: Dict[str, RichSignal],
                                   allocation: float) -> Dict[str, RichSignal]:
        """
        Apply capital allocation to a signal.
        
        Args:
            signal_dict: Dictionary of signals
            allocation: Capital allocation factor
            
        Returns:
            Adjusted signal dictionary
        """
        result = {}
        
        for timestamp, signal in signal_dict.items():
            # Create a copy of the signal
            adjusted_signal = RichSignal(
                action=signal.action,
                quantity=signal.quantity * allocation,
                price=signal.price,
                metadata=signal.metadata.copy() if signal.metadata else {}
            )
            
            # Add allocation info to metadata
            if adjusted_signal.metadata is None:
                adjusted_signal.metadata = {}
            adjusted_signal.metadata["capital_allocation"] = allocation
            
            result[timestamp] = adjusted_signal
            
        return result
    
    def _resolve_signal_conflicts(self,
                                 symbol: str,
                                 strategy_signals: Dict[str, Dict[str, RichSignal]]) -> Dict[str, RichSignal]:
        """
        Resolve conflicts between signals from multiple strategies.
        
        Args:
            symbol: Trading symbol
            strategy_signals: Dictionary mapping strategy names to their signal dictionaries
            
        Returns:
            Resolved signal dictionary
        """
        # Use specified conflict resolution method
        if self.conflict_resolution_method == "performance_weighted":
            return self._performance_weighted_resolution(symbol, strategy_signals)
        elif self.conflict_resolution_method == "majority_vote":
            return self._majority_vote_resolution(symbol, strategy_signals)
        elif self.conflict_resolution_method == "conservative":
            return self._conservative_resolution(symbol, strategy_signals)
        else:
            # Default to performance weighted
            return self._performance_weighted_resolution(symbol, strategy_signals)
    
    def _performance_weighted_resolution(self, 
                                        symbol: str,
                                        strategy_signals: Dict[str, Dict[str, RichSignal]]) -> Dict[str, RichSignal]:
        """
        Resolve conflicts using performance-weighted averaging.
        
        Args:
            symbol: Trading symbol
            strategy_signals: Dictionary mapping strategy names to their signal dictionaries
            
        Returns:
            Resolved signal dictionary
        """
        result = {}
        timestamps = set()
        
        # Collect all timestamps
        for signals in strategy_signals.values():
            timestamps.update(signals.keys())
        
        for timestamp in timestamps:
            # Collect signals for this timestamp
            timestamp_signals = {}
            for strategy, signals in strategy_signals.items():
                if timestamp in signals:
                    timestamp_signals[strategy] = signals[timestamp]
            
            if not timestamp_signals:
                continue
                
            # Calculate performance weights
            weights = {}
            total_weight = 0.0
            
            for strategy in timestamp_signals:
                # Use strategy allocation as base weight
                weight = self.strategy_allocations.get(strategy, 1.0 / len(strategy_signals))
                
                # Boost weight based on recent performance if available
                if (strategy in self.performance_history and 
                    symbol in self.performance_history[strategy] and
                    len(self.performance_history[strategy][symbol]) > 0):
                    
                    # Get recent Sharpe ratio
                    recent_data = self.performance_history[strategy][symbol][-5:]
                    sharpe_values = [p.get("sharpe_ratio", 0) for p in recent_data if "sharpe_ratio" in p]
                    
                    if sharpe_values:
                        avg_sharpe = np.mean(sharpe_values)
                        # Scale sharpe to a reasonable weight factor (0.5 to 1.5)
                        sharpe_factor = 0.5 + min(1.0, max(0.0, avg_sharpe / 2))
                        weight *= sharpe_factor
                
                weights[strategy] = weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                for strategy in weights:
                    weights[strategy] /= total_weight
            
            # Calculate weighted signal
            net_action = 0
            net_quantity = 0
            price_sum = 0
            price_count = 0
            all_metadata = {}
            
            for strategy, signal in timestamp_signals.items():
                weight = weights.get(strategy, 0)
                
                net_action += np.sign(signal.action) * weight
                net_quantity += signal.quantity * weight
                
                if signal.price is not None:
                    price_sum += signal.price * weight
                    price_count += weight
                
                # Collect metadata
                if signal.metadata:
                    for key, value in signal.metadata.items():
                        if key not in all_metadata:
                            all_metadata[key] = []
                        all_metadata[key].append((value, weight))
            
            # Determine final action
            final_action = 1 if net_action > 0.2 else (-1 if net_action < -0.2 else 0)
            
            # Only set quantity if action is non-zero
            final_quantity = abs(net_quantity) if final_action != 0 else 0
            
            # Calculate weighted average price
            final_price = price_sum / price_count if price_count > 0 else None
            
            # Process metadata
            final_metadata = dict()
            for key, values in all_metadata.items():
                try:
                    # Try to compute weighted average for numeric values
                    if all(isinstance(v[0], (int, float)) for v in values):
                        final_metadata[key] = sum(v[0] * v[1] for v in values) / sum(v[1] for v in values)
                    else:
                        # For non-numeric, take the value with highest weight
                        final_metadata[key] = max(values, key=lambda x: x[1])[0]
                except Exception as e:
                    logger.error(f"Error processing metadata {key}: {e}")
            
            # Add coordination information
            final_metadata["coordinated"] = True
            final_metadata["strategies_used"] = list(timestamp_signals.keys())
            final_metadata["strategy_weights"] = weights
            
            # Create final signal
            result[timestamp] = RichSignal(
                action=final_action,
                quantity=final_quantity,
                price=final_price,
                metadata=final_metadata
            )
            
        return result
