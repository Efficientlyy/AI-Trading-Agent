import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .strategy import BaseStrategy, RichSignal, RichSignalsDict, BaseStrategyManager
from .data_manager import DataManagerABC
from ..common import logger

class IntegratedStrategyManager(BaseStrategyManager):
    """Manages multiple strategies and combines their rich signals into a unified output.

    Uses a configurable aggregation method (e.g., confidence-weighted average)
    to produce a final signal strength and confidence score for each symbol.
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManagerABC):
        """
        Initializes the IntegratedStrategyManager.

        Args:
            config: Configuration dictionary. Expected keys:
                - 'aggregation_method' (str): Method to combine signals ('weighted_average', 'simple_average', 'majority_vote'). Default 'weighted_average'.
                - 'strategy_configs' (Dict[str, Dict]): Configurations for individual strategies to be loaded.
                - 'strategy_weights' (Dict[str, float], optional): Weights assigned to each strategy for aggregation.
                - 'name' (str, optional): Name for this strategy manager instance.
            data_manager: An instance of a DataManager conforming to DataManagerABC.
        """
        super().__init__(config, data_manager)
        self.name = config.get('name', 'IntegratedStrategyManager')  # Add name attribute
        self.aggregation_method = config.get('aggregation_method', 'weighted_average')
        self.strategy_weights = config.get('strategy_weights', {})
        
        # Initialize strategies dictionary
        self._strategies = {}
        
        # Initialize additional attributes
        self.data_manager = data_manager
        
        logger.info(f"IntegratedStrategyManager initialized with aggregation method: {self.aggregation_method}")
        # Strategy loading logic would go here based on 'strategy_configs'
        # For now, strategies will be added manually via add_strategy

    def _combine_signals(self, all_strategy_signals: Dict[str, RichSignalsDict]) -> RichSignalsDict:
        """
        Combines signals from multiple strategies for each symbol.

        Args:
            all_strategy_signals: A dictionary where keys are strategy names
                                   and values are the RichSignalsDict output from each strategy.
                                   e.g., {'MA_Crossover': {'AAPL': {...}, 'MSFT': {...}},
                                          'Sentiment':    {'AAPL': {...}, 'MSFT': {...}}}

        Returns:
            A single RichSignalsDict containing the combined signal for each symbol.
        """
        combined_signals: RichSignalsDict = {}
        if not all_strategy_signals:
            return combined_signals

        # Get a list of all unique symbols across all strategies
        all_symbols = set()
        for strategy_output in all_strategy_signals.values():
            all_symbols.update(strategy_output.keys())

        # Check if we should use meta-learner to dynamically select aggregation method
        use_meta_learner = self.config.get('use_meta_learner', False)
        if use_meta_learner:
            # Meta-learner will determine the best aggregation method based on recent performance
            selected_method = self._apply_meta_learner(all_strategy_signals)
            logger.info(f"Meta-learner selected '{selected_method}' as the best aggregation method")
            # Temporarily override the configured aggregation method
            original_method = self.aggregation_method
            self.aggregation_method = selected_method

        for symbol in all_symbols:
            signals_for_symbol: List[Tuple[str, RichSignal]] = [] # Store (strategy_name, signal)
            for strategy_name, strategy_output in all_strategy_signals.items():
                if symbol in strategy_output:
                    signals_for_symbol.append((strategy_name, strategy_output[symbol]))

            if not signals_for_symbol:
                # Should not happen if symbol came from the outputs, but safety check
                combined_signals[symbol] = {
                    'signal_strength': 0.0,
                    'confidence_score': 0.0,
                    'signal_type': 'combined',
                    'metadata': {'reason': 'No signals found for symbol'}
                }
                continue

            # --- Apply Aggregation Logic --- 
            if self.aggregation_method == 'weighted_average':
                combined_signals[symbol] = self._apply_weighted_average(signals_for_symbol, symbol)
            elif self.aggregation_method == 'dynamic_contextual':
                combined_signals[symbol] = self._apply_dynamic_contextual(signals_for_symbol, symbol)
            elif self.aggregation_method == 'rule_based':
                combined_signals[symbol] = self._apply_rule_based(signals_for_symbol, symbol)
            elif self.aggregation_method == 'majority_vote':
                combined_signals[symbol] = self._apply_majority_vote(signals_for_symbol, symbol)
            else:
                logger.warning(f"Unsupported aggregation method: {self.aggregation_method}. Defaulting to weighted_average for {symbol}.")
                combined_signals[symbol] = self._apply_weighted_average(signals_for_symbol, symbol)

            # Add the aggregation method used to the metadata
            if 'metadata' not in combined_signals[symbol]:
                combined_signals[symbol]['metadata'] = {}
            combined_signals[symbol]['metadata']['aggregation_method'] = self.aggregation_method

        # Restore original aggregation method if we used meta-learner
        if use_meta_learner:
            self.aggregation_method = original_method

        return combined_signals

    def _apply_meta_learner(self, all_strategy_signals: Dict[str, RichSignalsDict]) -> str:
        """
        Use a meta-learning approach to dynamically select the best aggregation method.
        
        This method analyzes recent performance of different aggregation methods and
        selects the one that has performed best under current market conditions.
        
        Args:
            all_strategy_signals: Dictionary of signals from all strategies
            
        Returns:
            String name of the selected aggregation method
        """
        # Get available aggregation methods
        available_methods = ['weighted_average', 'dynamic_contextual', 'rule_based', 'majority_vote']
        
        # Get performance history if available
        performance_history = self.config.get('performance_history', {})
        market_regime = self.config.get('current_market_regime', 'normal')
        
        # If we don't have performance history, use the configured method
        if not performance_history:
            return self.aggregation_method
            
        # Calculate scores for each method based on recent performance
        method_scores = {}
        
        for method in available_methods:
            # Skip methods that don't have performance data
            if method not in performance_history:
                method_scores[method] = 0.0
                continue
                
            # Get performance metrics for this method
            method_perf = performance_history[method]
            
            # Calculate a score based on multiple metrics
            # Higher is better
            score = 0.0
            
            # Add Sharpe ratio (risk-adjusted return)
            if 'sharpe_ratio' in method_perf:
                score += method_perf['sharpe_ratio'] * 2.0  # Weight Sharpe more heavily
                
            # Add win rate
            if 'win_rate' in method_perf:
                score += method_perf['win_rate']
                
            # Add profit factor
            if 'profit_factor' in method_perf:
                score += method_perf['profit_factor']
                
            # Subtract max drawdown (lower is better)
            if 'max_drawdown' in method_perf:
                score -= method_perf['max_drawdown'] * 2.0  # Penalize drawdowns more heavily
                
            # Add market regime specific adjustments
            if market_regime == 'volatile' and method == 'dynamic_contextual':
                score *= 1.2  # Boost dynamic contextual in volatile markets
            elif market_regime == 'trending' and method == 'weighted_average':
                score *= 1.1  # Boost weighted average in trending markets
            elif market_regime == 'ranging' and method == 'majority_vote':
                score *= 1.1  # Boost majority vote in ranging markets
                
            method_scores[method] = score
            
        # Find the method with the highest score
        if not method_scores:
            return self.aggregation_method
            
        best_method = max(method_scores.items(), key=lambda x: x[1])[0]
        
        # Log the scores for debugging
        logger.debug(f"Meta-learner scores: {method_scores}")
        logger.info(f"Meta-learner selected '{best_method}' with score {method_scores[best_method]}")
        
        return best_method

    def _apply_weighted_average(self, signals_for_symbol, symbol):
        """
        Apply weighted average aggregation method to combine signals.
        
        Args:
            signals_for_symbol: List of (strategy_name, signal) tuples
            symbol: The symbol being processed
            
        Returns:
            Combined RichSignal
        """
        signal_sum = 0.0
        confidence_sum = 0.0
        weight_sum = 0.0
        signal_types = set()
        
        # Collect metadata from all strategies
        metadata = {'strategies': {}, 'weights': {}, 'signal_distribution': {}}
        
        # Track signal distribution for analysis
        signal_distribution = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for strategy_name, signal in signals_for_symbol:
            # Get weight for this strategy (default to 1.0 if not specified)
            weight = self.strategy_weights.get(strategy_name, 1.0)
            
            # Apply confidence as a multiplier to the weight
            confidence = signal.get('confidence_score', 0.5)
            effective_weight = weight * confidence
            
            # Accumulate weighted signal
            signal_strength = signal.get('signal_strength', 0.0)
            signal_sum += signal_strength * effective_weight
            confidence_sum += confidence * weight
            weight_sum += weight
            
            # Track signal type
            signal_type = signal.get('signal_type', 'unknown')
            signal_types.add(signal_type)
            
            # Track signal distribution
            if signal_strength > 0.1:
                signal_distribution['buy'] += 1
            elif signal_strength < -0.1:
                signal_distribution['sell'] += 1
            else:
                signal_distribution['hold'] += 1
            
            # Add strategy signal and weight to metadata
            metadata['strategies'][strategy_name] = {
                'signal_strength': signal_strength,
                'confidence_score': confidence,
                'weight': weight,
                'effective_weight': effective_weight,
                'signal_type': signal_type,
                'additional_info': signal.get('metadata', {})
            }
        
        # Calculate weighted average
        if weight_sum > 0:
            final_signal_strength = signal_sum / weight_sum
            final_confidence = confidence_sum / weight_sum
        else:
            final_signal_strength = 0.0
            final_confidence = 0.0
        
        # Apply signal strength thresholding if configured
        threshold = self.config.get('signal_strength_threshold', 0.1)
        if abs(final_signal_strength) < threshold:
            final_signal_strength = 0.0
        
        # Add aggregation method to metadata
        metadata['aggregation_method'] = 'weighted_average'
        metadata['final_weights'] = {'weight_sum': weight_sum}
        metadata['signal_distribution'] = signal_distribution
        
        # Determine the most appropriate signal type
        combined_signal_type = 'combined'
        if len(signal_types) == 1:
            combined_signal_type = next(iter(signal_types))
        
        return {
            'signal_strength': final_signal_strength,
            'confidence_score': final_confidence,
            'signal_type': combined_signal_type,
            'metadata': metadata
        }

    def _apply_dynamic_contextual(self, signals_for_symbol, symbol):
        """
        Apply dynamic contextual weighting based on market regime.
        
        This method adjusts strategy weights based on current market conditions
        and recent performance before applying weighted averaging.
        
        Args:
            signals_for_symbol: List of (strategy_name, signal) tuples
            symbol: The symbol being processed
            
        Returns:
            Combined RichSignal with dynamically adjusted weights
        """
        # Start with base weights from configuration
        dynamic_weights = {}
        for strategy_name, _ in signals_for_symbol:
            dynamic_weights[strategy_name] = self.strategy_weights.get(strategy_name, 1.0)
        
        # Get market regime indicators if available
        # In a real implementation, these would come from market analysis
        # For now, we'll use simple defaults or config values
        market_regime = self.config.get('market_regime', 'normal')
        volatility = self.config.get('volatility', 'medium')
        trend_strength = self.config.get('trend_strength', 'medium')
        
        # Adjust weights based on market regime
        for strategy_name in dynamic_weights:
            # Example: Boost technical strategies in trending markets
            if 'crossover' in strategy_name.lower() or 'technical' in strategy_name.lower():
                if trend_strength == 'strong':
                    dynamic_weights[strategy_name] *= 1.5
                elif trend_strength == 'weak':
                    dynamic_weights[strategy_name] *= 0.7
            
            # Example: Boost sentiment strategies in volatile markets
            if 'sentiment' in strategy_name.lower():
                if volatility == 'high':
                    dynamic_weights[strategy_name] *= 1.3
                elif volatility == 'low':
                    dynamic_weights[strategy_name] *= 0.8
        
        # Normalize adjusted weights
        total_weight = sum(dynamic_weights.values())
        if total_weight > 0:
            for strategy_name in dynamic_weights:
                dynamic_weights[strategy_name] /= total_weight
        
        # Now use these dynamic weights for weighted averaging
        total_weighted_strength = 0.0
        total_confidence_weight = 0.0
        confidences = []
        metadata_list = []
        
        for strategy_name, signal in signals_for_symbol:
            strength = signal.get('signal_strength', 0.0)
            confidence = signal.get('confidence_score', 0.0)
            
            # Apply dynamic weight and confidence
            weight = dynamic_weights.get(strategy_name, 0.0) * confidence
            total_weighted_strength += strength * weight
            total_confidence_weight += weight
            confidences.append(confidence)
            
            metadata_list.append({
                'strategy': strategy_name,
                'strength': strength,
                'confidence': confidence,
                'base_weight': self.strategy_weights.get(strategy_name, 1.0),
                'dynamic_weight': dynamic_weights.get(strategy_name, 0.0),
                'effective_weight': weight
            })
        
        if total_confidence_weight > 1e-6:
            combined_strength = total_weighted_strength / total_confidence_weight
        else:
            combined_strength = 0.0
        
        combined_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'signal_strength': np.clip(combined_strength, -1.0, 1.0),
            'confidence_score': np.clip(combined_confidence, 0.0, 1.0),
            'signal_type': 'combined',
            'metadata': {
                'aggregation_method': 'dynamic_contextual',
                'market_regime': market_regime,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'contributing_signals': metadata_list
            }
        }
    
    def _apply_rule_based(self, signals_for_symbol, symbol):
        """
        Apply rule-based priority system to combine signals.
        
        This method applies configurable rules to prioritize or override signals
        based on specific conditions.
        
        Args:
            signals_for_symbol: List of (strategy_name, signal) tuples
            symbol: The symbol being processed
            
        Returns:
            Combined RichSignal based on priority rules
        """
        # Load rules from configuration
        rules = self.config.get('priority_rules', [])
        if not rules:
            # Default rules if none configured
            rules = [
                # Example rule: If any strategy has very high confidence (>0.9), use its signal
                {'condition': 'confidence_score > 0.9', 'action': 'use_highest_confidence'},
                # Example rule: If strategies disagree significantly, reduce confidence
                {'condition': 'signal_disagreement > 0.8', 'action': 'reduce_confidence'},
                # Default fallback to weighted average
                {'condition': 'default', 'action': 'weighted_average'}
            ]
        
        # Extract signal properties for rule evaluation
        confidences = [s[1].get('confidence_score', 0.5) for s in signals_for_symbol]
        strengths = [s[1].get('signal_strength', 0.0) for s in signals_for_symbol]
        
        # Calculate metrics for rule conditions
        max_confidence = max(confidences) if confidences else 0.5
        max_confidence_idx = confidences.index(max_confidence) if confidences else -1
        max_confidence_strategy = signals_for_symbol[max_confidence_idx][0] if max_confidence_idx >= 0 else None
        
        # Measure signal disagreement (standard deviation of strengths)
        signal_disagreement = np.std(strengths) if len(strengths) > 1 else 0.0
        
        # Apply rules in order
        for rule in rules:
            condition = rule.get('condition')
            action = rule.get('action')
            
            # Evaluate rule condition
            condition_met = False
            if condition == 'default':
                condition_met = True
            elif condition == 'confidence_score > 0.9' and max_confidence > 0.9:
                condition_met = True
            elif condition == 'signal_disagreement > 0.8' and signal_disagreement > 0.8:
                condition_met = True
            
            # Apply rule action if condition is met
            if condition_met:
                if action == 'use_highest_confidence' and max_confidence_strategy:
                    # Use the signal with highest confidence
                    signal = dict(signals_for_symbol[max_confidence_idx][1])  # Make a copy
                    signal['metadata'] = signal.get('metadata', {})
                    signal['metadata']['rule_applied'] = f"{condition} -> {action}"
                    signal['metadata']['selected_strategy'] = max_confidence_strategy
                    signal['metadata']['aggregation_method'] = 'rule_based'
                    return signal
                    
                elif action == 'reduce_confidence':
                    # Use weighted average but reduce confidence
                    avg_signal = self._apply_weighted_average(signals_for_symbol, symbol)
                    avg_signal['confidence_score'] *= 0.7  # Reduce confidence by 30%
                    avg_signal['metadata']['rule_applied'] = f"{condition} -> {action}"
                    avg_signal['metadata']['aggregation_method'] = 'rule_based'
                    return avg_signal
                    
                elif action == 'weighted_average':
                    # Fall back to weighted average
                    avg_signal = self._apply_weighted_average(signals_for_symbol, symbol)
                    avg_signal['metadata']['rule_applied'] = 'default -> weighted_average'
                    avg_signal['metadata']['aggregation_method'] = 'rule_based'
                    return avg_signal
        
        # If no rules matched (shouldn't happen with default rule), fall back to weighted average
        return self._apply_weighted_average(signals_for_symbol, symbol)
    
    def _apply_majority_vote(self, signals_for_symbol, symbol):
        """
        Apply majority vote approach to combine signals.
        
        This method counts the number of strategies giving positive, negative, or neutral signals
        and uses the majority direction, weighted by confidence.
        
        Args:
            signals_for_symbol: List of (strategy_name, signal) tuples
            symbol: The symbol being processed
            
        Returns:
            Combined RichSignal based on majority vote
        """
        # Count votes in each direction, weighted by confidence
        positive_votes = 0.0
        negative_votes = 0.0
        neutral_votes = 0.0
        metadata_list = []
        
        for strategy_name, signal in signals_for_symbol:
            strength = signal.get('signal_strength', 0.0)
            confidence = signal.get('confidence_score', 0.5)
            
            # Categorize signal direction and add weighted vote
            direction = 'neutral'
            if strength > 0.1:  # Positive signal
                positive_votes += confidence
                direction = 'positive'
            elif strength < -0.1:  # Negative signal
                negative_votes += confidence
                direction = 'negative'
            else:  # Neutral signal
                neutral_votes += confidence
                direction = 'neutral'
                
            metadata_list.append({
                'strategy': strategy_name,
                'strength': strength,
                'confidence': confidence,
                'direction': direction
            })
        
        # Determine majority direction
        total_votes = positive_votes + negative_votes + neutral_votes
        if total_votes == 0:
            total_votes = 1  # Avoid division by zero
            
        # Calculate final signal strength based on vote distribution
        if positive_votes >= negative_votes and positive_votes >= neutral_votes:
            # Positive majority
            signal_strength = positive_votes / total_votes
        elif negative_votes >= positive_votes and negative_votes >= neutral_votes:
            # Negative majority
            signal_strength = -negative_votes / total_votes
        else:
            # Neutral majority
            signal_strength = 0.0
        
        # Calculate confidence based on vote consensus
        max_votes = max(positive_votes, negative_votes, neutral_votes)
        vote_consensus = max_votes / total_votes if total_votes > 0 else 0.5
        
        return {
            'signal_strength': np.clip(signal_strength, -1.0, 1.0),
            'confidence_score': np.clip(vote_consensus, 0.0, 1.0),
            'signal_type': 'combined',
            'metadata': {
                'aggregation_method': 'majority_vote',
                'positive_votes': positive_votes,
                'negative_votes': negative_votes,
                'neutral_votes': neutral_votes,
                'vote_consensus': vote_consensus,
                'contributing_signals': metadata_list
            }
        }

    def process_data_and_generate_signals(
        self,
        current_data: Dict[str, pd.Series], 
        historical_data: Dict[str, pd.DataFrame],
        current_portfolio: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> RichSignalsDict:
        """
        Processes data through managed strategies and returns the *combined* final signal.

        Args:
            current_data: Dictionary mapping symbol to Series for the current timestamp.
            historical_data: Dictionary mapping symbol to DataFrame for historical data.
            current_portfolio: Optional dictionary with current portfolio state.
            **kwargs: Additional keyword arguments to pass to strategies.
                - timestamp: Current timestamp for signal generation
                - market_regime: Optional market regime information
                - risk_parameters: Optional risk management parameters

        Returns:
            A RichSignalsDict containing the combined signal for each symbol.
        """
        all_strategy_signals: Dict[str, RichSignalsDict] = {}  # strategy_name -> signals_dict
        timestamp = kwargs.get('timestamp', pd.Timestamp.now())
        market_regime = kwargs.get('market_regime', None)
        risk_parameters = kwargs.get('risk_parameters', {})

        # Check if we have any strategies
        if not self._strategies:
            logger.warning(f"{self.name}: No strategies loaded. Returning neutral signals.")
            symbols = list(current_data.keys()) if current_data else []
            return {symbol: {'signal_strength': 0.0, 'confidence_score': 0.0, 'signal_type': 'combined', 
                             'metadata': {'reason': 'No strategies loaded', 'timestamp': timestamp}} 
                    for symbol in symbols}

        # 1. Generate signals from all managed strategies
        for strategy_name, strategy_instance in self._strategies.items():
            try:
                # Prepare strategy-specific parameters
                strategy_params = {
                    'timestamp': timestamp,
                    'market_regime': market_regime,
                }
                
                # Add strategy-specific risk parameters if available
                if strategy_name in risk_parameters:
                    strategy_params['risk_parameters'] = risk_parameters[strategy_name]
                
                # Pass the relevant data slices to each strategy
                strategy_signals = strategy_instance.generate_signals(
                    data=historical_data,  # Pass the full historical data
                    current_portfolio=current_portfolio,
                    current_data=current_data,  # Pass current tick too if needed
                    **strategy_params
                )
                
                # Validate and sanitize strategy signals
                sanitized_signals = {}
                for symbol, signal in strategy_signals.items():
                    # Ensure required fields are present
                    if 'signal_strength' not in signal:
                        signal['signal_strength'] = 0.0
                    if 'confidence_score' not in signal:
                        signal['confidence_score'] = 0.5
                    if 'signal_type' not in signal:
                        signal['signal_type'] = strategy_name
                    if 'metadata' not in signal:
                        signal['metadata'] = {}
                    
                    # Ensure values are within bounds
                    signal['signal_strength'] = np.clip(signal['signal_strength'], -1.0, 1.0)
                    signal['confidence_score'] = np.clip(signal['confidence_score'], 0.0, 1.0)
                    
                    # Add timestamp to metadata
                    signal['metadata']['timestamp'] = timestamp
                    
                    sanitized_signals[symbol] = signal
                
                all_strategy_signals[strategy_name] = sanitized_signals
                logger.debug(f"Signals generated by '{strategy_name}': {sanitized_signals}")
            except Exception as e:
                logger.error(f"Error generating signals from strategy '{strategy_name}': {e}", exc_info=True)
                # Handle error for this strategy - generate default HOLD signals
                symbols = list(current_data.keys()) if current_data else []
                all_strategy_signals[strategy_name] = {
                    symbol: {
                        'signal_strength': 0.0, 
                        'confidence_score': 0.0, 
                        'signal_type': strategy_name, 
                        'metadata': {
                            'error': str(e),
                            'timestamp': timestamp,
                            'error_type': type(e).__name__
                        }
                    } for symbol in symbols
                }

        # 2. Combine the collected signals
        combined_signals = self._combine_signals(all_strategy_signals)
        
        # 3. Apply risk management if configured
        if self.config.get('apply_risk_management', False) and current_portfolio:
            combined_signals = self._apply_risk_management(combined_signals, current_portfolio, risk_parameters)
        
        # 4. Apply signal filtering if configured
        if self.config.get('apply_signal_filtering', False):
            combined_signals = self._apply_signal_filtering(combined_signals, timestamp)
        
        logger.info(f"{self.name}: Generated combined signals for {len(combined_signals)} symbols using {self.aggregation_method} method.")
        
        # Add timestamp to all signals
        for symbol, signal in combined_signals.items():
            if 'metadata' not in signal:
                signal['metadata'] = {}
            signal['metadata']['timestamp'] = timestamp
            signal['metadata']['aggregation_method'] = self.aggregation_method
        
        return combined_signals

    def generate_signals(self, current_data: Dict[str, pd.Series], historical_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """
        Generates final integer trading signals (1, -1, 0) for each symbol.

        This method fulfills the StrategyManagerABC interface requirement.
        It calls the internal signal processing and combination logic and then
        simplifies the rich signal output into a basic integer signal.

        Args:
            current_data: Dictionary mapping symbols to their current market data.
            historical_data: Dictionary mapping symbols to their historical market data.

        Returns:
            A dictionary mapping symbols to their final trading signal (1, -1, or 0).
        """
        # Call the existing detailed signal generation method
        # Pass None for current_portfolio as it's not available here
        rich_signals = self.process_data_and_generate_signals(
            current_data=current_data,
            historical_data=historical_data,
            current_portfolio=None  # Portfolio state not available in this interface method
        )

        # Convert rich signals (strength/confidence) to simple integer signals
        final_signals: Dict[str, int] = {}
        for symbol, signal_data in rich_signals.items():
            strength = signal_data.get('signal_strength', 0.0)
            # Simple conversion: positive strength -> buy (1), negative -> sell (-1), zero -> hold (0)
            if strength > 1e-6:  # Use a small threshold to handle floating point noise
                final_signals[symbol] = 1
            elif strength < -1e-6:
                final_signals[symbol] = -1
            else:
                final_signals[symbol] = 0

        return final_signals

    def _apply_risk_management(self, signals: RichSignalsDict, current_portfolio: Dict[str, Any], risk_parameters: Dict[str, Any]) -> RichSignalsDict:
        """
        Apply risk management rules to modify signals based on current portfolio state.
        
        This method can adjust signal strength or completely filter out signals based on:
        - Current portfolio exposure
        - Concentration limits
        - Drawdown protection
        - Volatility-based position sizing
        
        Args:
            signals: The combined signals dictionary to apply risk management to
            current_portfolio: Current portfolio state including positions, cash, etc.
            risk_parameters: Dictionary of risk parameters to apply
            
        Returns:
            Modified signals dictionary with risk management applied
        """
        if not signals or not current_portfolio:
            return signals
            
        # Make a copy to avoid modifying the original
        adjusted_signals = signals.copy()
        
        # Get risk parameters with defaults
        max_portfolio_exposure = risk_parameters.get('max_portfolio_exposure', 1.0)  # 100% by default
        max_position_size = risk_parameters.get('max_position_size', 0.25)  # 25% by default
        max_concentration = risk_parameters.get('max_concentration', 0.5)  # 50% by default
        drawdown_threshold = risk_parameters.get('drawdown_threshold', 0.1)  # 10% by default
        
        # Calculate current portfolio metrics
        total_portfolio_value = current_portfolio.get('total_value', 0.0)
        current_exposure = current_portfolio.get('current_exposure', 0.0) / total_portfolio_value if total_portfolio_value > 0 else 0.0
        current_drawdown = current_portfolio.get('current_drawdown', 0.0)
        
        # Check if we're in a drawdown protection mode
        in_drawdown_protection = current_drawdown > drawdown_threshold
        
        # Get current positions
        positions = current_portfolio.get('positions', {})
        
        # Apply risk rules to each signal
        for symbol, signal in adjusted_signals.items():
            # Skip signals that are already neutral
            if abs(signal['signal_strength']) < 1e-6:
                continue
                
            # Get current position for this symbol if any
            position_value = positions.get(symbol, {}).get('value', 0.0)
            position_ratio = position_value / total_portfolio_value if total_portfolio_value > 0 else 0.0
            
            # Check position concentration limit
            if position_ratio >= max_position_size and signal['signal_strength'] > 0:
                # Reduce buy signal if position already at maximum size
                signal['signal_strength'] *= (1.0 - position_ratio / max_position_size)
                signal['metadata']['risk_adjustment'] = 'position_size_limit'
                
            # Check overall portfolio exposure
            if current_exposure >= max_portfolio_exposure and signal['signal_strength'] > 0:
                # Reduce buy signal if portfolio exposure at maximum
                signal['signal_strength'] *= (1.0 - current_exposure / max_portfolio_exposure)
                signal['metadata']['risk_adjustment'] = 'portfolio_exposure_limit'
                
            # Apply drawdown protection if needed
            if in_drawdown_protection:
                # Reduce all signal strengths during drawdown
                protection_factor = 1.0 - (current_drawdown / drawdown_threshold)
                signal['signal_strength'] *= max(0.0, protection_factor)
                signal['metadata']['risk_adjustment'] = 'drawdown_protection'
                
            # Ensure signal strength is within bounds after adjustments
            signal['signal_strength'] = np.clip(signal['signal_strength'], -1.0, 1.0)
            
        return adjusted_signals
        
    def _apply_signal_filtering(self, signals: RichSignalsDict, timestamp: pd.Timestamp) -> RichSignalsDict:
        """
        Apply filtering rules to signals to reduce noise and false positives.
        
        This method can filter signals based on:
        - Minimum confidence threshold
        - Minimum signal strength threshold
        - Signal persistence (requiring signals to persist for multiple periods)
        - Time-of-day rules
        
        Args:
            signals: The combined signals dictionary to filter
            timestamp: Current timestamp for time-based filtering
            
        Returns:
            Filtered signals dictionary
        """
        if not signals:
            return signals
            
        # Make a copy to avoid modifying the original
        filtered_signals = signals.copy()
        
        # Get filtering parameters from config
        min_confidence = self.config.get('min_confidence_threshold', 0.3)
        min_signal_strength = self.config.get('min_signal_strength', 0.1)
        apply_time_filtering = self.config.get('apply_time_filtering', False)
        
        # Get trading hours if time filtering is enabled
        trading_start_hour = self.config.get('trading_start_hour', 9)  # Default market open
        trading_end_hour = self.config.get('trading_end_hour', 16)  # Default market close
        avoid_market_open = self.config.get('avoid_market_open', False)  # Avoid trading at market open
        avoid_market_close = self.config.get('avoid_market_close', False)  # Avoid trading at market close
        
        # Current hour for time-based filtering
        current_hour = timestamp.hour
        current_minute = timestamp.minute
        
        # Apply filtering to each signal
        for symbol, signal in list(filtered_signals.items()):
            # Check confidence threshold
            if signal['confidence_score'] < min_confidence:
                # Either remove the signal or set to neutral
                if self.config.get('remove_low_confidence', False):
                    del filtered_signals[symbol]
                    continue
                else:
                    signal['signal_strength'] = 0.0
                    signal['metadata']['filter_reason'] = 'low_confidence'
            
            # Check signal strength threshold
            if abs(signal['signal_strength']) < min_signal_strength:
                # Set to exactly zero to indicate a deliberate hold
                signal['signal_strength'] = 0.0
                signal['metadata']['filter_reason'] = 'insufficient_strength'
            
            # Apply time-of-day filtering if enabled
            if apply_time_filtering:
                # Check if outside trading hours
                if current_hour < trading_start_hour or current_hour >= trading_end_hour:
                    signal['signal_strength'] = 0.0
                    signal['metadata']['filter_reason'] = 'outside_trading_hours'
                
                # Check market open/close avoidance
                if avoid_market_open and current_hour == trading_start_hour and current_minute < 30:
                    signal['signal_strength'] = 0.0
                    signal['metadata']['filter_reason'] = 'market_open_avoidance'
                
                if avoid_market_close and current_hour == trading_end_hour - 1 and current_minute > 30:
                    signal['signal_strength'] = 0.0
                    signal['metadata']['filter_reason'] = 'market_close_avoidance'
        
        return filtered_signals
    
    # Override the add_strategy method to accept both a strategy name and a strategy instance
    def add_strategy(self, strategy_name: str, strategy: BaseStrategy) -> None:
        """
        Adds a strategy to the manager with the specified name.
        
        Args:
            strategy_name: The name to use for this strategy in the manager
            strategy: The strategy instance to add
        """
        # Store the strategy in our strategies dictionary
        self._strategies[strategy_name] = strategy
        
        # Add default weight if not already present
        if strategy_name not in self.strategy_weights:
            self.strategy_weights[strategy_name] = 1.0
            
        logger.info(f"Added strategy '{strategy_name}' to {self.name}")
    
    def remove_strategy(self, strategy_name: str) -> None:
        """
        Removes a strategy from the manager by name.
        
        Args:
            strategy_name: The name of the strategy to remove
        """
        if strategy_name in self._strategies:
            del self._strategies[strategy_name]
            logger.info(f"Removed strategy '{strategy_name}' from {self.name}")
        else:
            logger.warning(f"Strategy '{strategy_name}' not found in {self.name}")
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """
        Retrieves a strategy by name.
        
        Args:
            name: The name of the strategy to retrieve
            
        Returns:
            The strategy instance or None if not found
        """
        return self._strategies.get(name)
