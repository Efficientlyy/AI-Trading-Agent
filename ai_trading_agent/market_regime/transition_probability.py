"""
Regime Transition Probability Modeling

This module provides tools for modeling and predicting transitions between
different market regimes using probability models and machine learning approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from .core_definitions import MarketRegimeType, MarketRegimeInfo

# Set up logger
logger = logging.getLogger(__name__)


class TransitionProbabilityModel:
    """
    Class for modeling transitions between different market regimes.
    
    Uses historical data to build transition probability matrices and
    predict likely future regime transitions.
    """
    
    def __init__(self, transition_window: int = 5):
        """
        Initialize the transition probability model.
        
        Args:
            transition_window: Default time window (in periods) to consider
                               for a regime transition
        """
        self.transition_window = transition_window
        self.regime_history = {}
        self.transition_matrices = {}
        self.conditional_matrices = {}
    
    def add_regime_observation(self,
                              regime_info: MarketRegimeInfo,
                              asset_id: str = "default"):
        """
        Add a new regime observation to the history.
        
        Args:
            regime_info: Current market regime information
            asset_id: Identifier for the asset
        """
        if asset_id not in self.regime_history:
            self.regime_history[asset_id] = []
            
        self.regime_history[asset_id].append(regime_info)
        
        # Limit history to prevent memory issues
        if len(self.regime_history[asset_id]) > 1000:
            self.regime_history[asset_id] = self.regime_history[asset_id][-1000:]
    
    def build_transition_matrix(self,
                               asset_id: str = "default",
                               window: Optional[int] = None) -> Dict[str, any]:
        """
        Build a transition probability matrix based on historical data.
        
        Args:
            asset_id: Identifier for the asset
            window: Time window to consider for transitions (overrides default)
            
        Returns:
            Dictionary with transition matrix and statistics
        """
        if asset_id not in self.regime_history or len(self.regime_history[asset_id]) < 10:
            logger.warning(f"Insufficient regime history for {asset_id} to build transition matrix")
            return {
                "transition_matrix": None,
                "regime_counts": {},
                "start_probabilities": {}
            }
            
        # Use provided window or default
        window = window or self.transition_window
            
        # Count transitions
        transitions = defaultdict(Counter)
        regime_counts = Counter()
        
        history = self.regime_history[asset_id]
        for i in range(len(history) - window):
            start_regime = history[i].regime_type
            end_regime = history[i + window].regime_type
            
            transitions[start_regime.value][end_regime.value] += 1
            regime_counts[start_regime.value] += 1
        
        # Calculate probabilities
        transition_matrix = {}
        total_observations = sum(regime_counts.values())
        
        # Initial state probabilities
        start_probabilities = {regime: count / total_observations 
                              for regime, count in regime_counts.items()}
        
        # Transition probabilities
        for start_regime, end_regimes in transitions.items():
            transition_matrix[start_regime] = {}
            
            total = sum(end_regimes.values())
            for end_regime, count in end_regimes.items():
                transition_matrix[start_regime][end_regime] = count / total
        
        # Store the result
        self.transition_matrices[asset_id] = {
            "matrix": transition_matrix,
            "window": window,
            "regime_counts": regime_counts,
            "start_probabilities": start_probabilities,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "transition_matrix": transition_matrix,
            "regime_counts": dict(regime_counts),
            "start_probabilities": start_probabilities,
            "data_points": len(history)
        }
    
    def build_conditional_matrix(self,
                                asset_id: str = "default",
                                condition_variable: str = "volatility_regime",
                                window: Optional[int] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Build a conditional transition matrix based on an additional variable.
        
        Args:
            asset_id: Identifier for the asset
            condition_variable: Variable to condition on (e.g., 'volatility_regime')
            window: Time window to consider for transitions
            
        Returns:
            Dictionary mapping conditions to transition matrices
        """
        if asset_id not in self.regime_history or len(self.regime_history[asset_id]) < 20:
            logger.warning(f"Insufficient regime history for {asset_id} to build conditional matrix")
            return {}
            
        # Use provided window or default
        window = window or self.transition_window
            
        # Group transitions by condition
        conditional_transitions = defaultdict(lambda: defaultdict(Counter))
        conditional_counts = defaultdict(Counter)
        
        history = self.regime_history[asset_id]
        for i in range(len(history) - window):
            start_regime = history[i].regime_type
            end_regime = history[i + window].regime_type
            
            # Get condition value
            if hasattr(history[i], condition_variable):
                condition = getattr(history[i], condition_variable)
                condition_value = condition.value if hasattr(condition, "value") else str(condition)
                
                # Count transition with this condition
                conditional_transitions[condition_value][start_regime.value][end_regime.value] += 1
                conditional_counts[condition_value][start_regime.value] += 1
        
        # Calculate conditional probabilities
        conditional_matrices = {}
        
        for condition, transitions in conditional_transitions.items():
            conditional_matrices[condition] = {}
            
            for start_regime, end_regimes in transitions.items():
                conditional_matrices[condition][start_regime] = {}
                
                total = sum(end_regimes.values())
                for end_regime, count in end_regimes.items():
                    conditional_matrices[condition][start_regime][end_regime] = count / total
        
        # Store the result
        self.conditional_matrices[asset_id] = {
            "matrices": conditional_matrices,
            "condition_variable": condition_variable,
            "window": window,
            "timestamp": datetime.now().isoformat()
        }
        
        return conditional_matrices
    
    def predict_next_regime(self,
                           current_regime: MarketRegimeType,
                           asset_id: str = "default",
                           horizon: int = 1,
                           condition: Optional[str] = None) -> Dict[str, float]:
        """
        Predict the probability distribution of the next regime.
        
        Args:
            current_regime: Current market regime
            asset_id: Identifier for the asset
            horizon: Number of periods to forecast ahead
            condition: Optional conditioning variable value
            
        Returns:
            Dictionary mapping possible regimes to their probabilities
        """
        # Check if we need to build matrices first
        if asset_id not in self.transition_matrices:
            self.build_transition_matrix(asset_id)
            
        # If still no matrix, return uniform distribution
        if asset_id not in self.transition_matrices or not self.transition_matrices[asset_id]["matrix"]:
            # Return uniform distribution over all possible regimes
            regimes = [r.value for r in MarketRegimeType]
            return {regime: 1.0 / len(regimes) for regime in regimes}
            
        # Get the appropriate matrix
        if condition and asset_id in self.conditional_matrices:
            matrices = self.conditional_matrices[asset_id]["matrices"]
            if condition in matrices and current_regime.value in matrices[condition]:
                matrix = matrices[condition]
            else:
                matrix = self.transition_matrices[asset_id]["matrix"]
        else:
            matrix = self.transition_matrices[asset_id]["matrix"]
            
        # Handle case where current regime not in matrix
        if current_regime.value not in matrix:
            # Return uniform distribution over all regimes in the matrix
            regimes = list(set([r for transitions in matrix.values() for r in transitions.keys()]))
            return {regime: 1.0 / len(regimes) for regime in regimes}
            
        # For horizon=1, simply return the row for current regime
        if horizon == 1:
            return matrix[current_regime.value]
            
        # For longer horizons, perform matrix multiplication (not included for simplicity)
        # This would involve raising the transition matrix to the power of horizon
        # For now, we'll just return the one-step transition
        logger.warning(f"Multi-step forecasting not fully implemented; returning one-step forecast")
        return matrix[current_regime.value]
    
    def get_most_likely_transition(self,
                                  current_regime: MarketRegimeType,
                                  asset_id: str = "default",
                                  condition: Optional[str] = None,
                                  threshold: float = 0.15) -> Optional[str]:
        """
        Get the most likely regime transition if probability exceeds threshold.
        
        Args:
            current_regime: Current market regime
            asset_id: Identifier for the asset
            condition: Optional conditioning variable value
            threshold: Probability threshold for signaling a likely transition
            
        Returns:
            Most likely next regime or None if no probability exceeds threshold
        """
        probabilities = self.predict_next_regime(current_regime, asset_id, condition=condition)
        
        # Find most likely transition
        most_likely_regime = None
        highest_prob = 0.0
        
        for regime, prob in probabilities.items():
            if regime != current_regime.value and prob > highest_prob:
                highest_prob = prob
                most_likely_regime = regime
                
        # Return only if probability exceeds threshold
        if highest_prob > threshold:
            return most_likely_regime, highest_prob
        else:
            return None, 0.0
    
    def get_regime_stability(self,
                            asset_id: str = "default",
                            window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate the stability (self-transition probability) of each regime.
        
        Args:
            asset_id: Identifier for the asset
            window: Time window for transition analysis
            
        Returns:
            Dictionary mapping regime types to stability scores
        """
        # Check if we need to build matrices first
        if asset_id not in self.transition_matrices:
            self.build_transition_matrix(asset_id, window=window)
            
        # If still no matrix, return empty result
        if asset_id not in self.transition_matrices or not self.transition_matrices[asset_id]["matrix"]:
            return {}
            
        matrix = self.transition_matrices[asset_id]["matrix"]
        
        # Calculate stability (probability of staying in the same regime)
        stability = {}
        for regime, transitions in matrix.items():
            stability[regime] = transitions.get(regime, 0.0)
            
        return stability
    
    def get_average_regime_duration(self,
                                   asset_id: str = "default") -> Dict[str, float]:
        """
        Calculate the average duration of each regime from the history.
        
        Args:
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary mapping regime types to average durations
        """
        if asset_id not in self.regime_history or len(self.regime_history[asset_id]) < 10:
            return {}
            
        history = self.regime_history[asset_id]
        
        # Find regime runs
        runs = []
        current_regime = None
        run_start = 0
        
        for i, info in enumerate(history):
            if info.regime_type != current_regime:
                if current_regime is not None:
                    runs.append((current_regime.value, i - run_start))
                current_regime = info.regime_type
                run_start = i
                
        # Add the last run
        if current_regime is not None and run_start < len(history):
            runs.append((current_regime.value, len(history) - run_start))
            
        # Calculate average durations
        durations = defaultdict(list)
        for regime, length in runs:
            durations[regime].append(length)
            
        return {regime: sum(lengths) / len(lengths) 
               for regime, lengths in durations.items() if lengths}
