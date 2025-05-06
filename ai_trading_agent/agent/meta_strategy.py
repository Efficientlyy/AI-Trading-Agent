"""
Meta-strategy for dynamically selecting the best signal aggregation method based on market conditions.

This module implements a meta-strategy that:
1. Analyzes current market conditions (volatility, trend strength, etc.)
2. Selects the most appropriate signal aggregation method based on historical performance
3. Dynamically adjusts strategy weights based on recent performance
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..common import logger
from .integrated_manager import IntegratedStrategyManager
from .data_manager import DataManagerABC
from .market_analyzer import MarketAnalyzer

class DynamicAggregationMetaStrategy:
    """
    Meta-strategy that dynamically selects the best signal aggregation method based on market conditions.
    
    This class wraps around an IntegratedStrategyManager and dynamically selects the best
    aggregation method based on current market conditions and historical performance.
    """
    
    def __init__(self, config: Dict[str, Any], data_manager: DataManagerABC):
        """
        Initialize the meta-strategy.
        
        Args:
            config: Configuration dictionary with the following keys:
                - 'config_path': Path to the meta-strategy configuration file
                - 'lookback_window': Number of days to look back for performance evaluation
                - 'update_frequency': How often to update the best method (in days)
                - 'default_method': Default aggregation method if no best method can be determined
            data_manager: Data manager instance
        """
        self.config = config
        self.data_manager = data_manager
        self.market_analyzer = MarketAnalyzer()
        
        # Load meta-strategy configuration
        self.config_path = config.get('config_path', None)
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.meta_config = json.load(f)
        else:
            # Default configuration if file not found
            self.meta_config = {
                'name': 'DynamicAggregationMetaStrategy',
                'description': 'Dynamically selects the best aggregation method based on market conditions',
                'market_condition_mapping': {
                    'normal': 'weighted_average',
                    'trending': 'dynamic_contextual',
                    'volatile': 'rule_based',
                    'crisis': 'majority_vote'
                }
            }
        
        # Initialize parameters
        self.lookback_window = config.get('lookback_window', 30)  # 30 days lookback by default
        self.update_frequency = config.get('update_frequency', 7)  # Update every 7 days by default
        self.default_method = config.get('default_method', 'weighted_average')
        
        # Initialize the strategy managers for each aggregation method
        self.strategy_managers = {}
        self.initialize_strategy_managers()
        
        # Track performance of each method
        self.performance_history = {}
        self.last_update = None
        self.current_best_method = self.default_method
        
        logger.info(f"DynamicAggregationMetaStrategy initialized with default method: {self.default_method}")
    
    def initialize_strategy_managers(self):
        """Initialize strategy managers for each aggregation method."""
        aggregation_methods = ['weighted_average', 'dynamic_contextual', 'rule_based', 'majority_vote']
        
        for method in aggregation_methods:
            # Create a config for this method
            method_config = dict(self.config)
            method_config['aggregation_method'] = method
            method_config['name'] = f"IntegratedManager_{method}"
            
            # Create a strategy manager for this method
            self.strategy_managers[method] = IntegratedStrategyManager(
                config=method_config,
                data_manager=self.data_manager
            )
            
            # Copy strategies from the config if available
            if 'strategies' in self.config:
                for strategy in self.config['strategies']:
                    self.strategy_managers[method].add_strategy(strategy)
    
    def analyze_market_conditions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze current market conditions.
        
        Args:
            data: Dictionary mapping symbols to their historical data
        
        Returns:
            Dictionary with market condition analysis
        """
        market_conditions = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Determine volatility level
            if volatility < 0.15:
                volatility_level = 'low'
            elif volatility < 0.30:
                volatility_level = 'medium'
            else:
                volatility_level = 'high'
            
            # Calculate trend strength (absolute value of slope of linear regression)
            x = np.arange(len(df))
            y = df['close'].values
            slope, _ = np.polyfit(x, y, 1)
            trend_strength = abs(slope) / df['close'].mean()
            
            # Determine trend strength level
            if trend_strength < 0.001:
                trend_level = 'weak'
            elif trend_strength < 0.003:
                trend_level = 'medium'
            else:
                trend_level = 'strong'
            
            # Determine market regime
            if volatility_level == 'high' and trend_level == 'weak':
                regime = 'volatile'
            elif trend_level == 'strong':
                regime = 'trending'
            elif volatility_level == 'high' and trend_level == 'strong':
                regime = 'crisis'
            else:
                regime = 'normal'
            
            market_conditions[symbol] = {
                'volatility': volatility,
                'volatility_level': volatility_level,
                'trend_strength': trend_strength,
                'trend_level': trend_level,
                'regime': regime
            }
        
        # Determine overall market regime (majority vote)
        regimes = [cond['regime'] for cond in market_conditions.values()]
        if regimes:
            from collections import Counter
            overall_regime = Counter(regimes).most_common(1)[0][0]
        else:
            overall_regime = 'normal'  # Default if no data
        
        return {
            'symbol_conditions': market_conditions,
            'overall_regime': overall_regime
        }
    
    def select_best_method(self, market_conditions: Dict[str, Any]) -> str:
        """
        Select the best aggregation method based on market conditions.
        
        Args:
            market_conditions: Dictionary with market condition analysis
        
        Returns:
            Name of the best aggregation method
        """
        overall_regime = market_conditions['overall_regime']
        
        # Use the mapping from the meta-strategy configuration
        if 'market_condition_mapping' in self.meta_config:
            mapping = self.meta_config['market_condition_mapping']
            if overall_regime in mapping:
                base_method = mapping[overall_regime]
            else:
                base_method = self.default_method
        else:
            base_method = self.default_method
        
        # Consider recent performance if available
        if self.performance_history:
            # Get recent performance for each method
            performance = {m: self.get_recent_performance(m) for m in self.strategy_managers.keys()}
            
            # Find the method with the best performance
            best_performing_method = max(performance.items(), key=lambda x: x[1])[0]
            
            # If the best performing method is significantly better than the base method,
            # use it instead of the one selected based on market conditions
            if performance[best_performing_method] > performance.get(base_method, 0.0) * 1.2:  # 20% better
                logger.info(f"Overriding method selection based on performance: {best_performing_method} (performance: {performance[best_performing_method]:.2f}) vs {base_method} (performance: {performance.get(base_method, 0.0):.2f})")
                return best_performing_method
        
        # Consider symbol-specific conditions
        symbol_conditions = market_conditions.get('symbol_conditions', {})
        if symbol_conditions:
            # Count regimes across symbols
            regime_counts = {}
            for symbol, condition in symbol_conditions.items():
                regime = condition.get('regime', 'normal')
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # If there's a significant disagreement between symbols, use a more robust method
            if len(regime_counts) > 1:
                # Calculate entropy to measure diversity of regimes
                total_symbols = sum(regime_counts.values())
                entropy = -sum((count / total_symbols) * np.log2(count / total_symbols) for count in regime_counts.values())
                
                # If entropy is high (diverse regimes), use rule-based method which is more robust
                if entropy > 1.0:  # Threshold for high diversity
                    logger.info(f"Mixed market regimes detected (entropy: {entropy:.2f}), using rule-based method for robustness")
                    return 'rule_based'
        
        # Consider volatility level
        avg_volatility = 0.0
        volatility_count = 0
        
        for symbol, condition in symbol_conditions.items():
            if 'volatility' in condition:
                avg_volatility += condition['volatility']
                volatility_count += 1
        
        if volatility_count > 0:
            avg_volatility /= volatility_count
            
            # In extremely high volatility, majority vote can be more stable
            if avg_volatility > 0.5:  # Very high volatility threshold
                logger.info(f"Extremely high volatility detected ({avg_volatility:.2f}), using majority_vote method for stability")
                return 'majority_vote'
        
        # Use the base method selected from the mapping
        return base_method
    
    def update_performance_history(self, method: str, signals: Dict[str, Any], 
                                  actual_returns: Dict[str, float]) -> None:
        """
        Update the performance history for a given method.
        
        Args:
            method: Name of the aggregation method
            signals: Signals generated by the method
            actual_returns: Actual returns observed after the signals
        """
        if method not in self.performance_history:
            self.performance_history[method] = []
        
        # Calculate performance metrics
        correct_direction = 0
        total_signals = 0
        
        for symbol, signal in signals.items():
            if symbol not in actual_returns:
                continue
            
            signal_strength = signal.get('signal_strength', 0.0)
            actual_return = actual_returns[symbol]
            
            # Check if signal direction matches return direction
            if (signal_strength > 0 and actual_return > 0) or (signal_strength < 0 and actual_return < 0):
                correct_direction += 1
            
            total_signals += 1
        
        # Calculate accuracy
        accuracy = correct_direction / total_signals if total_signals > 0 else 0.0
        
        # Add to performance history
        self.performance_history[method].append({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'correct_direction': correct_direction,
            'total_signals': total_signals
        })
    
    def get_recent_performance(self, method: str) -> float:
        """
        Get the recent performance of a given method.
        
        Args:
            method: Name of the aggregation method
        
        Returns:
            Average accuracy over the lookback window
        """
        if method not in self.performance_history or not self.performance_history[method]:
            return 0.0
        
        # Get performance records within the lookback window
        cutoff_time = datetime.now() - timedelta(days=self.lookback_window)
        recent_records = [r for r in self.performance_history[method] 
                         if r['timestamp'] > cutoff_time]
        
        if not recent_records:
            return 0.0
        
        # Calculate average accuracy
        avg_accuracy = sum(r['accuracy'] for r in recent_records) / len(recent_records)
        return avg_accuracy
    
    def should_update_method(self) -> bool:
        """
        Determine if it's time to update the best method.
        
        Returns:
            True if it's time to update, False otherwise
        """
        if self.last_update is None:
            return True
        
        time_since_update = datetime.now() - self.last_update
        return time_since_update.days >= self.update_frequency
    
    def process_data_and_generate_signals(self, 
                                         current_data: Dict[str, pd.Series], 
                                         historical_data: Dict[str, pd.DataFrame],
                                         current_portfolio: Optional[Dict[str, Any]] = None,
                                         **kwargs: Any) -> Dict[str, Any]:
        """
        Process data and generate signals using the best aggregation method.
        
        Args:
            current_data: Dictionary mapping symbols to their current data
            historical_data: Dictionary mapping symbols to their historical data
            current_portfolio: Optional dictionary with current portfolio state
            **kwargs: Additional keyword arguments
        
        Returns:
            Dictionary with combined signals
        """
        # Analyze market conditions
        market_conditions = self.analyze_market_conditions(historical_data)
        logger.info(f"Current market regime: {market_conditions['overall_regime']}")
        
        # Check if we should update the best method
        if self.should_update_method():
            # Select the best method based on market conditions
            best_method = self.select_best_method(market_conditions)
            
            # If we have performance history, consider that too
            if self.performance_history:
                # Get recent performance for each method
                performance = {m: self.get_recent_performance(m) for m in self.strategy_managers.keys()}
                
                # Find the method with the best performance
                best_performing_method = max(performance.items(), key=lambda x: x[1])[0]
                
                # If the best performing method has significantly better performance,
                # use it instead of the one selected based on market conditions
                if performance[best_performing_method] > performance.get(best_method, 0.0) * 1.2:  # 20% better
                    best_method = best_performing_method
                    logger.info(f"Overriding method selection based on performance: {best_method}")
            
            self.current_best_method = best_method
            self.last_update = datetime.now()
            logger.info(f"Updated best method to {self.current_best_method}")
        
        # Use the current best method to generate signals
        logger.info(f"Generating signals using {self.current_best_method} method")
        strategy_manager = self.strategy_managers[self.current_best_method]
        
        # Generate signals
        signals = strategy_manager.process_data_and_generate_signals(
            current_data=current_data,
            historical_data=historical_data,
            current_portfolio=current_portfolio,
            **kwargs
        )
        
        # Add meta-strategy information to the signals
        for symbol, signal in signals.items():
            if 'metadata' not in signal:
                signal['metadata'] = {}
            signal['metadata']['meta_strategy'] = {
                'selected_method': self.current_best_method,
                'market_regime': market_conditions['overall_regime']
            }
        
        return signals
    
    def add_strategy(self, strategy):
        """
        Add a strategy to all strategy managers.
        
        Args:
            strategy: Strategy instance to add
        """
        for manager in self.strategy_managers.values():
            manager.add_strategy(strategy)
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove a strategy from all strategy managers.
        
        Args:
            strategy_name: Name of the strategy to remove
        
        Returns:
            True if the strategy was removed, False otherwise
        """
        success = True
        for manager in self.strategy_managers.values():
            if not manager.remove_strategy(strategy_name):
                success = False
        return success
    
    def get_strategies(self) -> List[str]:
        """
        Get the names of all strategies managed by this meta-strategy.
        
        Returns:
            List of strategy names
        """
        # All managers should have the same strategies, so just use the first one
        if not self.strategy_managers:
            return []
        
        first_manager = next(iter(self.strategy_managers.values()))
        return first_manager.get_strategies()
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], 
                        current_portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """
        Generate simple integer signals (-1, 0, 1) for each symbol.
        
        This method is required to maintain compatibility with the StrategyManagerABC interface.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            current_portfolio: Optional dictionary with current portfolio state
        
        Returns:
            Dictionary mapping symbols to integer signals (-1, 0, 1)
        """
        # Extract current data from the last row of historical data
        current_data = {symbol: df.iloc[-1] for symbol, df in data.items() if not df.empty}
        
        # Generate rich signals
        rich_signals = self.process_data_and_generate_signals(
            current_data=current_data,
            historical_data=data,
            current_portfolio=current_portfolio
        )
        
        # Convert rich signals to integer signals
        integer_signals = {}
        for symbol, signal in rich_signals.items():
            strength = signal.get('signal_strength', 0.0)
            if strength > 0.1:
                integer_signals[symbol] = 1  # Buy
            elif strength < -0.1:
                integer_signals[symbol] = -1  # Sell
            else:
                integer_signals[symbol] = 0  # Hold
        
        return integer_signals
