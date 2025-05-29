"""
Reinforcement Learning-based Strategy Selector for the Adaptive Trading Agent.

This module implements reinforcement learning techniques to autonomously select
the optimal trading strategy based on current market conditions and historical
performance.
"""

import logging
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import deque
import random

# ML imports
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RLStrategySelector:
    """
    Reinforcement Learning-based Strategy Selector.
    
    Uses Q-learning and contextual bandits to learn which trading strategies
    perform best in different market conditions, and autonomously selects
    the optimal strategy as market conditions evolve.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RL Strategy Selector.
        
        Args:
            config: Configuration dictionary with parameters:
                - strategy_list: List of available strategies
                - learning_rate: Learning rate for Q-learning
                - discount_factor: Discount factor for future rewards
                - exploration_rate: Initial exploration rate
                - min_exploration_rate: Minimum exploration rate
                - exploration_decay: Rate at which exploration decays
                - memory_size: Size of replay memory
                - batch_size: Batch size for training
                - model_path: Path to save/load model
                - stateful: Whether to maintain state across instances
                - state_features: List of state feature names to use
                - reward_metrics: Dictionary mapping metrics to reward weights
        """
        self.name = "RLStrategySelector"
        
        # RL parameters
        self.strategy_list = config.get('strategy_list', [])
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.95)
        self.exploration_rate = config.get('exploration_rate', 1.0)
        self.min_exploration_rate = config.get('min_exploration_rate', 0.05)
        self.exploration_decay = config.get('exploration_decay', 0.995)
        self.memory_size = config.get('memory_size', 1000)
        self.batch_size = config.get('batch_size', 32)
        self.model_path = config.get('model_path', 'rl_strategy_model.pkl')
        self.stateful = config.get('stateful', True)
        
        # State and reward configuration
        self.state_features = config.get('state_features', [
            'trend_strength', 'volatility', 'rsi', 'volume_change',
            'market_regime', 'hour_of_day', 'day_of_week'
        ])
        
        self.reward_metrics = config.get('reward_metrics', {
            'return': 1.0,          # Return has highest weight
            'sharpe_ratio': 0.7,    # Sharpe ratio (risk-adjusted return)
            'max_drawdown': -0.5,   # Drawdown (negative weight because lower is better)
            'win_rate': 0.3,        # Win rate
            'profit_factor': 0.3    # Profit factor
        })
        
        # Initialize Q-table with zeros
        # Each row is a state, each column is a strategy
        self.n_strategies = len(self.strategy_list)
        
        # Use dictionary-based Q-table since we'll have sparse high-dimensional states
        self.q_table = {}
        
        # Initialize replay memory
        self.replay_memory = deque(maxlen=self.memory_size)
        
        # State encoding/normalization
        self.scaler = StandardScaler()
        self.fitted_scaler = False
        
        # Track current state, action and performance
        self.current_state = None
        self.current_strategy_idx = None
        self.current_strategy = None
        self.latest_performance = {}
        self.episode_count = 0
        
        # Load existing model if available and stateful is True
        if self.stateful and os.path.exists(self.model_path):
            self._load_model()
        
        logger.info(f"Initialized {self.name} with {len(self.strategy_list)} strategies")
    
    def select_strategy(self, market_features: Dict[str, Any], 
                       performance_history: List[Dict[str, Any]] = None) -> str:
        """
        Select a strategy using reinforcement learning.
        
        Args:
            market_features: Dictionary of current market features/indicators
            performance_history: Optional list of recent performance metrics
            
        Returns:
            Selected strategy name
        """
        # Extract state from market features
        state = self._extract_state(market_features)
        self.current_state = state
        
        # Select action using epsilon-greedy policy
        strategy_idx = self._select_action(state)
        self.current_strategy_idx = strategy_idx
        self.current_strategy = self.strategy_list[strategy_idx]
        
        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay
        
        logger.info(f"RL Strategy Selector chose strategy: {self.current_strategy} " 
                    f"(exploration rate: {self.exploration_rate:.3f})")
        
        return self.current_strategy
    
    def update_performance(self, performance_metrics: Dict[str, float]) -> float:
        """
        Update the RL model with the performance of the selected strategy.
        
        Args:
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            Calculated reward value
        """
        if self.current_state is None or self.current_strategy_idx is None:
            logger.warning("Cannot update performance without prior strategy selection")
            return 0.0
        
        # Calculate reward from performance metrics
        reward = self._calculate_reward(performance_metrics)
        
        # Update Q-table
        self._update_q_value(self.current_state, self.current_strategy_idx, reward)
        
        # Store transition in replay memory
        self.latest_performance = performance_metrics
        self._add_to_replay_memory(self.current_state, self.current_strategy_idx, 
                                 reward, self.current_state)  # Next state is same for now
        
        # Perform batch learning if memory has enough samples
        if len(self.replay_memory) >= self.batch_size:
            self._batch_learn()
        
        self.episode_count += 1
        
        # Save model periodically
        if self.stateful and self.episode_count % 10 == 0:
            self._save_model()
        
        logger.info(f"Updated RL model with performance metrics, reward: {reward:.4f}")
        
        return reward
    
    def _extract_state(self, market_features: Dict[str, Any]) -> Tuple:
        """
        Extract and normalize state features from market data.
        
        Args:
            market_features: Dictionary of market features/indicators
            
        Returns:
            Tuple representing the current state
        """
        # Extract relevant features for state representation
        state_values = []
        
        for feature in self.state_features:
            if feature in market_features:
                # Handle categorical features
                if feature == 'market_regime':
                    # One-hot encode market regime
                    regime = market_features[feature]
                    
                    # Convert regime values to integers for state representation
                    regime_mapping = {
                        'bull': 0, 'bear': 1, 'volatile': 2, 'consolidation': 3, 
                        'sideways': 4, 'unknown': 5
                    }
                    regime_int = regime_mapping.get(regime.lower(), 5)
                    state_values.append(regime_int)
                    
                elif feature == 'hour_of_day':
                    # Bin hours into trading sessions
                    hour = int(market_features.get(feature, 0))
                    # Asian, European, American, and Overnight sessions
                    if 0 <= hour < 6:
                        session = 0  # Asian
                    elif 6 <= hour < 12:
                        session = 1  # European
                    elif 12 <= hour < 20:
                        session = 2  # American
                    else:
                        session = 3  # Overnight
                    state_values.append(session)
                    
                elif feature == 'day_of_week':
                    # Use as is (0-6)
                    day = int(market_features.get(feature, 0))
                    state_values.append(day)
                    
                else:
                    # Numerical features
                    value = float(market_features[feature])
                    state_values.append(value)
            else:
                # Use a default value if feature is missing
                state_values.append(0.0)
        
        # Use discrete state buckets for categorical features,
        # but normalize continuous features
        continuous_indices = [i for i, feature in enumerate(self.state_features) 
                            if feature not in ['market_regime', 'hour_of_day', 'day_of_week']]
        
        if continuous_indices:
            continuous_values = [state_values[i] for i in continuous_indices]
            
            # Reshape for scaler
            continuous_array = np.array(continuous_values).reshape(1, -1)
            
            # Fit scaler if not already fitted
            if not self.fitted_scaler:
                self.scaler.fit(continuous_array)
                self.fitted_scaler = True
            
            # Normalize continuous values
            normalized_continuous = self.scaler.transform(continuous_array)[0]
            
            # Replace continuous values with normalized ones
            for i, idx in enumerate(continuous_indices):
                state_values[idx] = normalized_continuous[i]
        
        # Convert floating-point values to fixed precision to create discrete states
        state_values = [round(val, 2) if isinstance(val, float) else val for val in state_values]
        
        # Convert list to tuple for hashing (as dictionary key)
        return tuple(state_values)
    
    def _select_action(self, state: Tuple) -> int:
        """
        Select a strategy index using epsilon-greedy policy.
        
        Args:
            state: Current state tuple
            
        Returns:
            Index of selected strategy
        """
        # Exploration: random strategy
        if random.random() < self.exploration_rate:
            return random.randint(0, self.n_strategies - 1)
        
        # Exploitation: best strategy for this state
        else:
            # If state not in Q-table, initialize it
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.n_strategies)
            
            # Return strategy with highest Q-value
            return np.argmax(self.q_table[state])
    
    def _update_q_value(self, state: Tuple, action: int, reward: float, 
                      next_state: Optional[Tuple] = None) -> None:
        """
        Update Q-value for a state-action pair using Q-learning.
        
        Args:
            state: Current state
            action: Action taken (strategy index)
            reward: Reward received
            next_state: Next state (if None, assume same as current)
        """
        # Initialize state in Q-table if not present
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_strategies)
        
        # If next_state is None, use current state
        if next_state is None:
            next_state = state
        
        # Initialize next_state in Q-table if not present
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_strategies)
        
        # Q-learning update rule
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        
        # Updated Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        
        # Update Q-table
        self.q_table[state][action] = new_q
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calculate reward from performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Calculated reward value
        """
        reward = 0.0
        
        # Sum weighted metrics to get reward
        for metric, weight in self.reward_metrics.items():
            if metric in metrics:
                reward += metrics[metric] * weight
        
        return reward
    
    def _add_to_replay_memory(self, state: Tuple, action: int, 
                            reward: float, next_state: Tuple) -> None:
        """
        Add transition to replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        self.replay_memory.append((state, action, reward, next_state))
    
    def _batch_learn(self) -> None:
        """
        Perform batch learning from replay memory.
        """
        # Sample a batch of transitions
        if len(self.replay_memory) < self.batch_size:
            return
            
        batch = random.sample(self.replay_memory, self.batch_size)
        
        # Update Q-values for each transition in the batch
        for state, action, reward, next_state in batch:
            self._update_q_value(state, action, reward, next_state)
    
    def _save_model(self) -> None:
        """
        Save the RL model to disk.
        """
        model_data = {
            'q_table': self.q_table,
            'scaler': self.scaler,
            'fitted_scaler': self.fitted_scaler,
            'exploration_rate': self.exploration_rate,
            'episode_count': self.episode_count
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Saved RL model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save RL model: {e}")
    
    def _load_model(self) -> None:
        """
        Load the RL model from disk.
        """
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data['q_table']
            self.scaler = model_data['scaler']
            self.fitted_scaler = model_data['fitted_scaler']
            self.exploration_rate = model_data['exploration_rate']
            self.episode_count = model_data['episode_count']
            
            logger.info(f"Loaded RL model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
    
    def get_strategy_rankings(self, market_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Get Q-values/rankings for all strategies in the current state.
        
        Args:
            market_features: Dictionary of current market features/indicators
            
        Returns:
            Dictionary mapping strategy names to their Q-values
        """
        state = self._extract_state(market_features)
        
        # Initialize state in Q-table if not present
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_strategies)
        
        # Get Q-values for this state
        q_values = self.q_table[state]
        
        # Create dictionary mapping strategy names to Q-values
        return {strategy: q_values[i] for i, strategy in enumerate(self.strategy_list)}
    
    def get_performance_history(self) -> Dict[str, List[float]]:
        """
        Get learning performance history.
        
        Returns:
            Dictionary with learning metrics
        """
        return {
            'episode_count': self.episode_count,
            'exploration_rate': self.exploration_rate,
            'latest_reward': self._calculate_reward(self.latest_performance) if self.latest_performance else 0.0
        }
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        Get exploration statistics.
        
        Returns:
            Dictionary with exploration metrics
        """
        # Calculate how many states we've explored
        states_explored = len(self.q_table)
        
        # Calculate percentage of non-zero Q-values
        total_q_values = states_explored * self.n_strategies
        non_zero_count = sum(np.count_nonzero(q_values) for q_values in self.q_table.values())
        
        # Exploration percentage (percentage of Q-values that are non-zero)
        if total_q_values > 0:
            exploration_percentage = non_zero_count / total_q_values * 100
        else:
            exploration_percentage = 0.0
        
        return {
            'states_explored': states_explored,
            'q_values_count': total_q_values,
            'non_zero_q_values': non_zero_count,
            'exploration_percentage': exploration_percentage,
            'current_exploration_rate': self.exploration_rate
        }
