"""
Reinforcement Learning for Strategy Selection Optimization

This module provides reinforcement learning capabilities for adaptively selecting
trading strategies based on market conditions and performance history.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import random
from collections import deque

logger = logging.getLogger(__name__)


class StrategyEnvironment:
    """
    Environment for strategy selection reinforcement learning.
    
    Provides a reinforcement learning environment that tracks the performance
    of different strategies under various market conditions.
    """
    
    def __init__(self, available_strategies: List[str], 
                market_regimes: List[str],
                reward_metrics: List[str] = None,
                observation_window: int = 10):
        """
        Initialize the strategy environment.
        
        Args:
            available_strategies: List of available strategy names
            market_regimes: List of possible market regimes
            reward_metrics: List of metrics to use for reward calculation
            observation_window: Number of past periods to include in state
        """
        self.available_strategies = available_strategies
        self.market_regimes = market_regimes
        self.reward_metrics = reward_metrics or ['sharpe_ratio', 'profit_factor', 'win_rate', 'max_drawdown']
        self.observation_window = observation_window
        
        # Strategy performance history
        self.strategy_history = {
            strategy: [] for strategy in available_strategies
        }
        
        # Current state information
        self.current_strategy = None
        self.current_regime = None
        self.current_metrics = {}
        
        # Observation history (for state representation)
        self.observation_history = deque(maxlen=observation_window)
        
        # Initialize the history with zeros
        zero_metrics = {metric: 0.0 for metric in self.reward_metrics}
        for _ in range(observation_window):
            self.observation_history.append({
                'regime': 'unknown',
                'strategy': '',
                'metrics': zero_metrics
            })
    
    def update_environment(self, 
                         strategy: str, 
                         market_regime: str,
                         performance_metrics: Dict[str, float]) -> None:
        """
        Update the environment with new performance data.
        
        Args:
            strategy: Currently active strategy
            market_regime: Current market regime
            performance_metrics: Performance metrics for the strategy
        """
        self.current_strategy = strategy
        self.current_regime = market_regime
        self.current_metrics = performance_metrics
        
        # Store in history
        if strategy in self.strategy_history:
            self.strategy_history[strategy].append({
                'timestamp': datetime.now(),
                'regime': market_regime,
                'metrics': performance_metrics
            })
        
        # Update observation history
        filtered_metrics = {
            metric: performance_metrics.get(metric, 0.0) 
            for metric in self.reward_metrics
        }
        
        self.observation_history.append({
            'regime': market_regime,
            'strategy': strategy,
            'metrics': filtered_metrics
        })
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state representation for RL.
        
        Returns:
            Numpy array representing the current state
        """
        # Convert regime to one-hot encoding
        regime_onehot = np.zeros(len(self.market_regimes))
        if self.current_regime in self.market_regimes:
            regime_onehot[self.market_regimes.index(self.current_regime)] = 1
        
        # Get recent metrics
        metrics_array = []
        for obs in self.observation_history:
            # Extract metrics in consistent order
            metrics_vector = [
                obs['metrics'].get(metric, 0.0) for metric in self.reward_metrics
            ]
            metrics_array.extend(metrics_vector)
        
        # Combine regime and metrics into state vector
        state = np.concatenate([regime_onehot, np.array(metrics_array)])
        return state
    
    def calculate_reward(self) -> float:
        """
        Calculate the reward based on recent performance.
        
        Returns:
            Scalar reward value
        """
        # Default reward is 0
        if not self.current_metrics:
            return 0.0
        
        # Calculate reward based on key metrics
        reward = 0.0
        
        # Sharpe ratio (higher is better)
        sharpe = self.current_metrics.get('sharpe_ratio', 0)
        reward += sharpe * 2.0  # Higher weight for Sharpe ratio
        
        # Profit factor (higher is better, capped at 3)
        profit_factor = min(3.0, self.current_metrics.get('profit_factor', 1.0))
        reward += (profit_factor - 1.0)  # 0 reward for PF=1 (breakeven)
        
        # Win rate (higher is better)
        win_rate = self.current_metrics.get('win_rate', 0.5)
        reward += (win_rate - 0.5) * 2.0  # 0 reward for 50% win rate
        
        # Drawdown (lower is better, negative contribution)
        drawdown = self.current_metrics.get('max_drawdown', 0)
        reward -= drawdown * 5.0  # Higher penalty for drawdowns
        
        return reward
    
    def get_strategy_index(self, strategy: str) -> int:
        """
        Get the index of a strategy in the available strategies list.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Index of the strategy or -1 if not found
        """
        try:
            return self.available_strategies.index(strategy)
        except ValueError:
            return -1


class QAgent:
    """
    Q-learning agent for strategy selection.
    
    Uses reinforcement learning to learn optimal strategy selection policy
    based on market conditions and strategy performance.
    """
    
    def __init__(self, environment: StrategyEnvironment,
                learning_rate: float = 0.1,
                discount_factor: float = 0.95,
                exploration_rate: float = 1.0,
                exploration_decay: float = 0.995,
                min_exploration_rate: float = 0.1):
        """
        Initialize the Q-learning agent.
        
        Args:
            environment: The strategy environment
            learning_rate: Alpha - learning rate for Q-value updates
            discount_factor: Gamma - discount factor for future rewards
            exploration_rate: Epsilon - initial exploration rate
            exploration_decay: Rate at which exploration decays
            min_exploration_rate: Minimum exploration rate
        """
        self.environment = environment
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Get state dimensions
        state_size = len(environment.market_regimes) + (
            len(environment.reward_metrics) * environment.observation_window)
        
        # Get action dimensions (one per strategy)
        action_size = len(environment.available_strategies)
        
        # Initialize Q-table with zeros
        # Using a simple table representation for now
        # For more complex state spaces, would use function approximation
        self.q_table = {}
    
    def _get_q_value(self, state_key: str, action: int) -> float:
        """
        Get Q-value for a state-action pair, or 0 if not yet visited.
        
        Args:
            state_key: String representation of state
            action: Action (strategy index)
            
        Returns:
            Q-value
        """
        if state_key in self.q_table:
            return self.q_table[state_key].get(action, 0.0)
        return 0.0
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """
        Convert state vector to a string key for the Q-table.
        
        Args:
            state: State vector
            
        Returns:
            String representation
        """
        # For simplicity, use a string of rounded values
        # In production, would use more sophisticated state discretization
        rounded = np.round(state, 2)
        return str(rounded.tolist())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action (strategy) based on current state.
        
        Args:
            state: Current state vector
            training: Whether we're in training mode (exploration enabled)
            
        Returns:
            Selected action (strategy index)
        """
        state_key = self._state_to_key(state)
        
        # Exploration-exploitation tradeoff
        if training and random.random() < self.exploration_rate:
            # Explore - random strategy
            return random.randint(0, len(self.environment.available_strategies) - 1)
        else:
            # Exploit - best known strategy
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
                
            # Get action with highest Q-value
            q_values = self.q_table[state_key]
            
            if not q_values:  # If no values yet, initialize with zeros
                for a in range(len(self.environment.available_strategies)):
                    q_values[a] = 0.0
                    
            # Return action with highest Q-value
            return max(q_values, key=q_values.get)
    
    def update_q_table(self, state: np.ndarray, action: int, 
                      reward: float, next_state: np.ndarray) -> None:
        """
        Update Q-values based on observed transition and reward.
        
        Args:
            state: Current state vector
            action: Action taken (strategy index)
            reward: Reward received
            next_state: Next state vector
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Initialize if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
            
        # Maximum Q-value for next state
        max_next_q = 0.0
        if next_state_key in self.q_table:
            next_q_values = self.q_table[next_state_key]
            if next_q_values:
                max_next_q = max(next_q_values.values())
        
        # Q-learning update formula
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q)
            
        # Update Q-value
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
    def get_best_strategy(self, state: np.ndarray) -> str:
        """
        Get the best strategy for the current state based on learned Q-values.
        
        Args:
            state: Current state vector
            
        Returns:
            Name of the best strategy
        """
        action = self.select_action(state, training=False)
        return self.environment.available_strategies[action]
    
    def save_model(self, file_path: str) -> None:
        """
        Save the Q-table to a file.
        
        Args:
            file_path: Path to save the model
        """
        import json
        
        # Convert keys from string to actual keys
        serializable_q_table = {}
        for state_key, actions in self.q_table.items():
            serializable_q_table[state_key] = {
                str(action): value for action, value in actions.items()
            }
            
        with open(file_path, 'w') as f:
            json.dump({
                'q_table': serializable_q_table,
                'exploration_rate': self.exploration_rate,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor
            }, f)
        
        logger.info(f"Saved Q-learning model to {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """
        Load the Q-table from a file.
        
        Args:
            file_path: Path to load the model from
        """
        import json
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Convert action keys back to integers
            q_table = {}
            for state_key, actions in data['q_table'].items():
                q_table[state_key] = {
                    int(action): value for action, value in actions.items()
                }
                
            self.q_table = q_table
            self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
            self.learning_rate = data.get('learning_rate', self.learning_rate)
            self.discount_factor = data.get('discount_factor', self.discount_factor)
            
            logger.info(f"Loaded Q-learning model from {file_path}")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading Q-learning model: {str(e)}")


class StrategyRL:
    """
    Main class for reinforcement learning strategy selection.
    
    Integrates the environment and agent for reinforcement learning of
    optimal strategy selection based on market conditions.
    """
    
    def __init__(self, 
                available_strategies: List[str], 
                market_regimes: List[str],
                observation_window: int = 10,
                learning_rate: float = 0.1,
                discount_factor: float = 0.95,
                model_path: Optional[str] = None):
        """
        Initialize the RL strategy selector.
        
        Args:
            available_strategies: List of available strategy names
            market_regimes: List of possible market regimes
            observation_window: Number of past observations to include in state
            learning_rate: Learning rate for Q-learning
            discount_factor: Discount factor for future rewards
            model_path: Optional path to load existing model
        """
        # Create environment
        self.environment = StrategyEnvironment(
            available_strategies=available_strategies,
            market_regimes=market_regimes,
            observation_window=observation_window
        )
        
        # Create agent
        self.agent = QAgent(
            environment=self.environment,
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )
        
        # Load model if provided
        if model_path:
            self.agent.load_model(model_path)
            
        self.last_state = None
        self.last_action = None
    
    def select_strategy(self, 
                       current_strategy: str,
                       market_regime: str,
                       performance_metrics: Dict[str, float],
                       training: bool = True) -> str:
        """
        Select the best strategy based on the current state.
        
        Args:
            current_strategy: Currently active strategy
            market_regime: Current market regime
            performance_metrics: Performance metrics for the strategy
            training: Whether to train the model during selection
            
        Returns:
            Selected strategy name
        """
        # Update environment with current information
        self.environment.update_environment(
            strategy=current_strategy,
            market_regime=market_regime,
            performance_metrics=performance_metrics
        )
        
        # Get current state
        current_state = self.environment.get_state()
        
        # Calculate reward if we have previous state and action
        if training and self.last_state is not None and self.last_action is not None:
            reward = self.environment.calculate_reward()
            
            # Update Q-values
            self.agent.update_q_table(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=current_state
            )
        
        # Select action (strategy)
        strategy_idx = self.agent.select_action(current_state, training=training)
        selected_strategy = self.environment.available_strategies[strategy_idx]
        
        # Store state and action for next update
        self.last_state = current_state
        self.last_action = strategy_idx
        
        return selected_strategy
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            file_path: Path to save the model
        """
        self.agent.save_model(file_path)
    
    def get_exploration_rate(self) -> float:
        """
        Get current exploration rate.
        
        Returns:
            Current exploration rate
        """
        return self.agent.exploration_rate
