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
from ..market_regime import (
    MarketRegimeType,
    TemporalPatternRecognition,
    TemporalPatternOptimizer
)
from ..optimization.reinforcement_learning import StrategyRL

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
                - 'enable_predictive_switching': Whether to use predictive strategy switching
                - 'enable_reinforcement_learning': Whether to use RL for strategy selection
                - 'rl_model_path': Path to load/save reinforcement learning model
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
        
        # Advanced strategy switching features
        self.enable_predictive_switching = config.get('enable_predictive_switching', True)
        self.enable_reinforcement_learning = config.get('enable_reinforcement_learning', False)
        self.rl_model_path = config.get('rl_model_path', None)
        
        # Initialize temporal pattern recognition if predictive switching is enabled
        if self.enable_predictive_switching:
            self.temporal_pattern_recognizer = TemporalPatternRecognition()
            self.strategy_optimizer = TemporalPatternOptimizer(self.temporal_pattern_recognizer)
            logger.info("Predictive strategy switching enabled with temporal pattern recognition")
            
        # Initialize reinforcement learning if enabled
        if self.enable_reinforcement_learning:
            # Initialize with default market regimes
            market_regimes = [regime.value for regime in MarketRegimeType]
            aggregation_methods = ['weighted_average', 'dynamic_contextual', 'rule_based', 'majority_vote']
            
            self.strategy_rl = StrategyRL(
                available_strategies=aggregation_methods,
                market_regimes=market_regimes,
                observation_window=10,
                model_path=self.rl_model_path
            )
            logger.info("Reinforcement learning enabled for strategy selection")
        
        # Initialize the strategy managers for each aggregation method
        self.strategy_managers = {}
        self.initialize_strategy_managers()
        
        # Track performance of each method
        self.performance_history = {}
        self.last_update = None
        self.current_best_method = self.default_method
        
        # Method effectiveness by regime
        self.regime_effectiveness = {
            regime.value: {method: [] for method in self.meta_config['market_condition_mapping'].values()}
            for regime in MarketRegimeType
        }
        
        # Ensemble weights based on effectiveness (initialize with equal weights)
        aggregation_methods = list(set(self.meta_config['market_condition_mapping'].values()))
        self.ensemble_weights = {method: 1.0 / len(aggregation_methods) for method in aggregation_methods}
        
        # Tracking for reinforcement learning
        self.last_method = self.current_best_method
        self.last_regime = None
        self.last_signals = {}
        self.last_metrics = {}
        
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
        
        Features:
        - Basic mapping based on detected market regime
        - Performance-based override using historical performance
        - Predictive strategy switching using temporal pattern recognition
        - Reinforcement learning for optimized strategy selection
        - Ensemble weighting based on regime effectiveness
        
        Args:
            market_conditions: Dictionary with market condition analysis
            
        Returns:
            Name of the best aggregation method
        """
        overall_regime = market_conditions['overall_regime']
        
        # Track the last regime for reinforcement learning
        self.last_regime = overall_regime
        
        # 1. BASIC MAPPING-BASED SELECTION
        # Use the mapping from the meta-strategy configuration
        if 'market_condition_mapping' in self.meta_config:
            mapping = self.meta_config['market_condition_mapping']
            if overall_regime in mapping:
                base_method = mapping[overall_regime]
            else:
                base_method = self.default_method
        else:
            base_method = self.default_method
            
        logger.info(f"Base method: {base_method} based on {overall_regime} market regime")
        
        # 2. PERFORMANCE-BASED OVERRIDE
        performance_method = base_method
        
        if self.performance_history:
            # Get recent performance for each method
            performance = {m: self.get_recent_performance(m) for m in self.strategy_managers.keys()}
            
            # Find the method with the best performance
            best_performing_method = max(performance.items(), key=lambda x: x[1])[0]
            
            # Only override if the best method is significantly better
            if performance[best_performing_method] > performance.get(base_method, 0.0) * 1.2:  # 20% better
                performance_method = best_performing_method
                logger.info(f"Performance override: {performance_method} based on superior performance")
                
            # Update regime effectiveness tracking
            for method, perf_value in performance.items():
                if overall_regime in self.regime_effectiveness:
                    if method in self.regime_effectiveness[overall_regime]:
                        self.regime_effectiveness[overall_regime][method].append(perf_value)
                        # Keep only the last 20 performance measurements
                        if len(self.regime_effectiveness[overall_regime][method]) > 20:
                            self.regime_effectiveness[overall_regime][method] = \
                                self.regime_effectiveness[overall_regime][method][-20:]
        
        # 3. PREDICTIVE STRATEGY SWITCHING
        predictive_method = performance_method
        
        if self.enable_predictive_switching and hasattr(self, 'temporal_pattern_recognizer'):
            # Extract price data for temporal analysis
            symbol_data = {}
            for symbol, conditions in market_conditions.get('symbol_conditions', {}).items():
                if 'data' in market_conditions and symbol in market_conditions['data']:
                    symbol_data[symbol] = market_conditions['data'][symbol]
            
            if symbol_data:
                # Use the first symbol for temporal analysis (could be enhanced to use all symbols)
                symbol = next(iter(symbol_data))
                data = symbol_data[symbol]
                
                if len(data) > 60:  # Need enough data for meaningful analysis
                    # Perform temporal pattern analysis
                    try:
                        temporal_result = self.temporal_pattern_recognizer.analyze_temporal_patterns(
                            prices=data['close'] if 'close' in data else data.iloc[:, 0],
                            volumes=data.get('volume', None),
                            asset_id=f"{symbol}_meta"
                        )
                        
                        # Look for regime transition opportunities
                        transition_opp = self.temporal_pattern_recognizer.detect_regime_transition_opportunity(f"{symbol}_meta")
                        
                        if transition_opp.get('transition_opportunity', False):
                            next_regime = transition_opp.get('potential_next_regime')
                            
                            if next_regime and next_regime in self.meta_config.get('market_condition_mapping', {}):
                                # Look ahead to the method that would be best in the predicted regime
                                predictive_method = self.meta_config['market_condition_mapping'][next_regime]
                                logger.info(f"Predictive override: {predictive_method} based on anticipated transition to {next_regime}")
                                
                                # Store the transition opportunity for later use
                                market_conditions['temporal_analysis'] = {
                                    'transition_opportunity': True,
                                    'potential_next_regime': next_regime,
                                    'confidence': transition_opp.get('confidence', 0.0)
                                }
                        
                        # Check for timeframe agreement
                        alignment = self.temporal_pattern_recognizer.get_timeframe_alignment_signal(f"{symbol}_meta")
                        
                        if alignment.get('has_alignment', False):
                            # With high alignment, be more confident in our regime detection
                            alignment_score = alignment.get('agreement_score', 0.0)
                            
                            if 'temporal_analysis' not in market_conditions:
                                market_conditions['temporal_analysis'] = {}
                            market_conditions['temporal_analysis']['timeframe_agreement'] = alignment_score
                            
                            logger.info(f"Timeframe alignment: {alignment_score:.2f} for confirmed regime {alignment.get('aligned_regime')}")
                    
                    except Exception as e:
                        logger.warning(f"Error in temporal pattern analysis: {str(e)}")
        
        # 4. REINFORCEMENT LEARNING SELECTION
        final_method = predictive_method
        
        if self.enable_reinforcement_learning and hasattr(self, 'strategy_rl'):
            # Prepare performance metrics for RL
            metrics = {}
            for method, history in self.performance_history.items():
                if history:
                    latest = history[-1]
                    metrics[method] = {
                        'accuracy': latest.get('accuracy', 0.5),
                        'return': latest.get('return', 0.0),
                        'sharpe_ratio': latest.get('sharpe', 0.0),
                        'profit_factor': latest.get('profit_factor', 1.0),
                        'max_drawdown': latest.get('drawdown', 0.0),
                        'win_rate': latest.get('win_rate', 0.5)
                    }
            
            # If we have metrics for the last method, update the RL
            if self.last_method and self.last_method in metrics and self.last_regime:
                import random
                
                # Select strategy using reinforcement learning
                rl_method = self.strategy_rl.select_strategy(
                    current_strategy=self.last_method,
                    market_regime=self.last_regime,
                    performance_metrics=metrics[self.last_method],
                    training=True
                )
                
                # Only use RL if we're confident in its exploration
                if self.strategy_rl.get_exploration_rate() < 0.3:
                    final_method = rl_method
                    logger.info(f"RL override: {final_method} based on learned policy")
                    
                # Save the RL model periodically
                if self.rl_model_path and random.random() < 0.1:  # 10% chance to save
                    self.strategy_rl.save_model(self.rl_model_path)
        
        # 5. SYMBOL-SPECIFIC CONSIDERATIONS
        # This part is preserved from the original implementation
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
        
        # 6. VOLATILITY CONSIDERATIONS
        # This part is preserved from the original implementation
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
        
        # 7. ENSEMBLE MODEL WEIGHTING
        # Update weights based on effectiveness in similar regimes
        self._update_ensemble_weights(overall_regime)
        
        # Remember the selected method for next iteration's RL update
        self.last_method = final_method
        
        return final_method
        
    def _update_ensemble_weights(self, current_regime: str) -> None:
        """
        Update ensemble weights based on effectiveness in the current regime.
        
        Args:
            current_regime: Current market regime
        """
        if current_regime not in self.regime_effectiveness:
            return
            
        # Calculate average performance for each method in this regime
        regime_performances = {}
        for method, performances in self.regime_effectiveness[current_regime].items():
            if performances:
                regime_performances[method] = sum(performances) / len(performances)
            else:
                regime_performances[method] = 0.0
        
        # Normalize performances to get weights
        total_performance = sum(p for p in regime_performances.values() if p > 0)
        
        if total_performance > 0:
            # Update weights based on relative performance
            for method, perf in regime_performances.items():
                if perf > 0:
                    self.ensemble_weights[method] = perf / total_performance
                else:
                    self.ensemble_weights[method] = 0.01  # Small non-zero weight for poor performers
                    
            # Renormalize weights
            total_weight = sum(self.ensemble_weights.values())
            for method in self.ensemble_weights:
                self.ensemble_weights[method] /= total_weight
    
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
