"""
Reinforcement Learning Module for the AI Trading Agent

This module implements reinforcement learning capabilities for meta-parameter optimization
and trading strategy enhancement. It uses a Deep Q-Network (DQN) approach with experience
replay for stability and improved learning.

Key features:
- State representation based on market conditions and system performance
- Action space for parameter adjustments and strategy selection
- Reward function based on risk-adjusted returns
- Exploration vs exploitation management with epsilon-greedy approach
- Integration with the trading orchestrator
"""

import numpy as np
import pandas as pd
import random
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, cast

# Attempt to import TensorFlow, but make it optional
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import TensorBoard
    HAS_TENSORFLOW = True
except ImportError:
    logger_setup = False
    # Create placeholder classes for type checking
    class Sequential:
        def __init__(self, *args, **kwargs):
            pass
    
    class Model:
        def __init__(self, *args, **kwargs):
            pass
    
    # Inform about missing TensorFlow
    print("WARNING: TensorFlow not found. Running with limited functionality.")

from ai_trading_agent.utils.logging import get_logger
from ai_trading_agent.risk.risk_manager import RiskManager

# Configure logger
logger = get_logger(__name__)

class ReinforcementLearningAgent:
    """
    Reinforcement Learning Agent for optimizing trading strategy parameters
    using Deep Q-Learning with experience replay.
    
    This agent can operate in two modes:
    1. Full mode: Using TensorFlow for deep learning (when available)
    2. Simple mode: Using a rule-based approach (when TensorFlow is not available)
    """
    
    def __init__(
        self,
        state_size: int = 30,
        action_size: int = 10,
        memory_size: int = 2000,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        model_path: Optional[str] = None,
        tensorboard_log_dir: Optional[str] = None
    ):
        """
        Initialize the Reinforcement Learning Agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            memory_size: Maximum size of experience replay memory
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
            model_path: Path to load a pre-trained model from
            tensorboard_log_dir: Directory for TensorBoard logs
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_path = model_path
        
        # Create TensorBoard callback if log directory is provided
        self.tensorboard = None
        if tensorboard_log_dir:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f"{tensorboard_log_dir}/{current_time}"
            self.tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # Build or load the model
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = load_model(model_path)
        else:
            logger.info("Building new RL model")
            self.model = self._build_model()
        
        # Track training metrics
        self.train_loss_history = []
        self.rewards_history = []
        self.epsilon_history = []
    
    def _build_model(self) -> Model:
        """
        Build the Deep Q-Network model.
        
        Returns:
            Keras Model for the DQN
        """
        # Deep Q-Network
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        logger.info(f"Model built with state size {self.state_size} and action size {self.action_size}")
        return model
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.rand() <= self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)
        
        # Exploitation: choose best action based on model prediction
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: Optional[int] = None) -> float:
        """
        Train the model using experience replay.
        
        Args:
            batch_size: Size of the batch to sample from memory
            
        Returns:
            Training loss
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Check if we have enough samples in memory
        if len(self.memory) < batch_size:
            logger.warning(f"Not enough samples in memory ({len(self.memory)}) for replay")
            return 0.0
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract data
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # Current Q value for the action taken
            target = reward
            
            if not done:
                # Add discounted future reward if not done
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            
            # Get current Q values from the model
            targets[i] = self.model.predict(state, verbose=0)
            
            # Update the Q value for the action
            targets[i, action] = target
            
            # Store state
            states[i] = state
        
        # Train the model on this batch
        callbacks = [self.tensorboard] if self.tensorboard else None
        history = self.model.fit(
            states, targets, epochs=1, verbose=0, batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Extract and store the loss
        loss = history.history['loss'][0]
        self.train_loss_history.append(loss)
        
        # Decay epsilon for less exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon_history.append(self.epsilon)
        
        return loss
    
    def load(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
        """
        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")
    
    def save(self, path: str) -> None:
        """
        Save the current model.
        
        Args:
            path: Path to save the model to
        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def update_reward_history(self, reward: float) -> None:
        """
        Update the history of rewards.
        
        Args:
            reward: The reward to add to history
        """
        self.rewards_history.append(reward)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "epsilon": self.epsilon,
            "memory_usage": len(self.memory) / self.memory.maxlen if self.memory.maxlen else 0,
            "avg_loss": np.mean(self.train_loss_history[-100:]) if self.train_loss_history else None,
            "avg_reward": np.mean(self.rewards_history[-100:]) if self.rewards_history else None
        }
        return metrics


class TradingRLAgent(ReinforcementLearningAgent):
    """
    Specialized RL Agent for trading that extends the base RL Agent
    with trading-specific state representation, actions, and rewards.
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        market_features: List[str],
        strategy_params: List[str],
        performance_metrics: List[str],
        action_map: Dict[int, Dict[str, float]],
        **kwargs
    ):
        """
        Initialize the Trading RL Agent.
        
        Args:
            risk_manager: Risk Manager instance for risk-adjusted rewards
            market_features: List of market features to include in state
            strategy_params: List of strategy parameters to optimize
            performance_metrics: List of performance metrics to include in state
            action_map: Mapping from action indices to parameter adjustments
            **kwargs: Additional arguments for the base RL Agent
        """
        # Calculate state size based on features
        state_size = len(market_features) + len(strategy_params) + len(performance_metrics)
        
        # Initialize the base RL Agent
        super().__init__(state_size=state_size, action_size=len(action_map), **kwargs)
        
        self.risk_manager = risk_manager
        self.market_features = market_features
        self.strategy_params = strategy_params
        self.performance_metrics = performance_metrics
        self.action_map = action_map
        
        # Feature scaling parameters for normalization
        self.feature_scaling = {}
        
        logger.info(f"Trading RL Agent initialized with {len(action_map)} possible actions")
    
    def preprocess_state(self, market_data: pd.DataFrame, 
                         strategy_state: Dict[str, float],
                         performance_data: Dict[str, float]) -> np.ndarray:
        """
        Create a state representation from market data, strategy parameters, 
        and performance metrics.
        
        Args:
            market_data: DataFrame with market features
            strategy_state: Current strategy parameters
            performance_data: Current performance metrics
            
        Returns:
            Processed state as numpy array
        """
        state_values = []
        
        # Extract and normalize market features
        for feature in self.market_features:
            if feature in market_data.columns:
                # Get the most recent value
                value = market_data[feature].iloc[-1]
                
                # Update scaling info if needed
                if feature not in self.feature_scaling:
                    # Use a simple min-max over recent history for initial scaling
                    feature_min = market_data[feature].min()
                    feature_max = market_data[feature].max()
                    # Prevent division by zero with a small epsilon
                    feature_range = max(feature_max - feature_min, 1e-8)
                    
                    self.feature_scaling[feature] = {
                        'min': feature_min,
                        'max': feature_max,
                        'range': feature_range
                    }
                
                # Normalize the value
                norm_value = (value - self.feature_scaling[feature]['min']) / self.feature_scaling[feature]['range']
                state_values.append(norm_value)
            else:
                # If feature is missing, use a default value
                state_values.append(0.0)
                logger.warning(f"Market feature {feature} not found in data")
        
        # Extract strategy parameters
        for param in self.strategy_params:
            if param in strategy_state:
                # Assume strategy parameters are already normalized or use simple clipping
                value = strategy_state[param]
                # Simple clipping to [0, 1] if needed
                norm_value = max(0.0, min(value, 1.0))
                state_values.append(norm_value)
            else:
                state_values.append(0.5)  # Default mid-range value
                logger.warning(f"Strategy parameter {param} not found in state")
        
        # Extract performance metrics
        for metric in self.performance_metrics:
            if metric in performance_data:
                value = performance_data[metric]
                
                # Update scaling info if needed
                if metric not in self.feature_scaling:
                    # Initial scaling based on typical ranges for common metrics
                    if 'return' in metric.lower():
                        # Returns typically in [-0.1, 0.1] range
                        self.feature_scaling[metric] = {'min': -0.1, 'max': 0.1, 'range': 0.2}
                    elif 'sharpe' in metric.lower():
                        # Sharpe ratio typically in [-3, 3] range
                        self.feature_scaling[metric] = {'min': -3.0, 'max': 3.0, 'range': 6.0}
                    elif 'drawdown' in metric.lower():
                        # Drawdown typically in [0, 0.3] range
                        self.feature_scaling[metric] = {'min': 0.0, 'max': 0.3, 'range': 0.3}
                    else:
                        # Default scaling
                        self.feature_scaling[metric] = {'min': -1.0, 'max': 1.0, 'range': 2.0}
                
                # Normalize the value
                norm_value = (value - self.feature_scaling[metric]['min']) / self.feature_scaling[metric]['range']
                # Clip to [0, 1] range
                norm_value = max(0.0, min(norm_value, 1.0))
                state_values.append(norm_value)
            else:
                state_values.append(0.0)
                logger.warning(f"Performance metric {metric} not found in data")
        
        # Convert to numpy array and reshape for the model
        state = np.array(state_values, dtype=np.float32).reshape(1, self.state_size)
        return state
    
    def get_action_params(self, action: int) -> Dict[str, float]:
        """
        Convert action index to parameter adjustments.
        
        Args:
            action: Action index
            
        Returns:
            Dictionary of parameter adjustments
        """
        if action in self.action_map:
            return self.action_map[action]
        
        logger.error(f"Invalid action index: {action}")
        # Return no change as fallback
        return {param: 0.0 for param in self.strategy_params}
    
    def calculate_reward(self, 
                         returns: float, 
                         volatility: float, 
                         drawdown: float,
                         trade_count: int,
                         prev_sharpe: float = None) -> float:
        """
        Calculate the reward based on risk-adjusted returns.
        
        Args:
            returns: Period returns
            volatility: Period volatility
            drawdown: Maximum drawdown
            trade_count: Number of trades executed
            prev_sharpe: Previous Sharpe ratio for improvement comparison
            
        Returns:
            Calculated reward
        """
        # Calculate Sharpe ratio (risk-adjusted return)
        # Use a minimum volatility to prevent division by zero
        min_vol = 1e-8
        sharpe = returns / max(volatility, min_vol)
        
        # Base reward on Sharpe ratio
        reward = sharpe
        
        # Penalize for excessive drawdown
        if drawdown > 0.05:  # 5% drawdown threshold
            # Progressive penalty for larger drawdowns
            drawdown_penalty = (drawdown - 0.05) * 10.0
            reward -= drawdown_penalty
        
        # Reward improvement over previous performance
        if prev_sharpe is not None:
            improvement = sharpe - prev_sharpe
            # Add bonus for improvement, scaled by its magnitude
            reward += improvement * 2.0
        
        # Small penalty for no trading activity
        if trade_count == 0:
            reward -= 0.2
        
        # Clip reward to reasonable range to prevent exploding gradients
        reward = max(-10.0, min(reward, 10.0))
        
        return reward
    
    def adapt_strategy(self, 
                      current_state: Dict[str, Any],
                      market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Adapt the trading strategy parameters based on current state.
        
        Args:
            current_state: Current state including strategy parameters and performance
            market_data: Recent market data
            
        Returns:
            Dictionary of updated strategy parameters
        """
        # Extract relevant components from current state
        strategy_params = {p: current_state['strategy_params'].get(p, 0.5) 
                          for p in self.strategy_params}
        
        performance_metrics = {
            'returns': current_state['performance'].get('returns', 0.0),
            'volatility': current_state['performance'].get('volatility', 0.0),
            'drawdown': current_state['performance'].get('drawdown', 0.0),
            'sharpe_ratio': current_state['performance'].get('sharpe_ratio', 0.0),
            'trade_count': current_state['performance'].get('trade_count', 0)
        }
        
        # Preprocess state for the model
        state = self.preprocess_state(market_data, strategy_params, performance_metrics)
        
        # Choose an action based on current state
        action = self.act(state)
        
        # Get parameter adjustments for the chosen action
        param_adjustments = self.get_action_params(action)
        
        # Update strategy parameters based on adjustments
        updated_params = {}
        for param, current_value in strategy_params.items():
            if param in param_adjustments:
                # Apply the adjustment
                new_value = current_value + param_adjustments[param]
                # Clip to valid range [0, 1] for normalized parameters
                new_value = max(0.0, min(new_value, 1.0))
                updated_params[param] = new_value
            else:
                # Keep current value if no adjustment specified
                updated_params[param] = current_value
        
        # Calculate reward for the current state
        reward = self.calculate_reward(
            performance_metrics['returns'],
            performance_metrics['volatility'],
            performance_metrics['drawdown'],
            performance_metrics['trade_count'],
            prev_sharpe=current_state.get('prev_performance', {}).get('sharpe_ratio')
        )
        
        # Store experience for future learning
        if 'prev_state' in current_state and 'prev_action' in current_state:
            prev_state = current_state['prev_state']
            prev_action = current_state['prev_action']
            done = False  # In continuous trading, episodes don't really end
            
            self.remember(prev_state, prev_action, reward, state, done)
            self.update_reward_history(reward)
            
            # Perform experience replay learning if enough samples
            if len(self.memory) >= self.batch_size:
                self.replay()
        
        # Store current state and action for next iteration
        current_state['prev_state'] = state
        current_state['prev_action'] = action
        current_state['prev_performance'] = performance_metrics
        
        return updated_params
    
    def reset_history(self) -> None:
        """Reset the agent's historical tracking."""
        self.train_loss_history = []
        self.rewards_history = []
        self.epsilon_history = []


# Factory function to create an appropriate RL agent
def create_trading_rl_agent(
    config: dict,
    risk_manager: RiskManager
) -> 'TradingRLAgent':
    """
    Create a Trading RL Agent with the specified configuration.
    
    Args:
        config: Configuration dictionary
        risk_manager: Risk Manager instance
        
    Returns:
        Configured TradingRLAgent
    """
    # Check if TensorFlow is available
    if not HAS_TENSORFLOW:
        # Create simplified mock agent for testing when TensorFlow is not available
        logger.warning("TensorFlow not available. Using simplified RL agent for testing.")
        return SimplifiedTradingRLAgent(risk_manager=risk_manager, config=config)
    
    # Extract configuration parameters with defaults
    market_features = config.get('market_features', [
        'close', 'volume', 'volatility', 'rsi', 'macd', 'bb_width'
    ])
    
    strategy_params = config.get('strategy_params', [
        'window_size', 'threshold', 'risk_factor', 'position_size_factor'
    ])
    
    performance_metrics = config.get('performance_metrics', [
        'returns', 'volatility', 'drawdown', 'sharpe_ratio', 'trade_count'
    ])
    
    # Define action map: mapping from action indices to parameter adjustments
    # Each action represents a specific adjustment to one or more parameters
    action_map = config.get('action_map', {
        0: {'window_size': 0.0, 'threshold': 0.0, 'risk_factor': 0.0, 'position_size_factor': 0.0},  # No change
        1: {'window_size': 0.1, 'threshold': 0.0, 'risk_factor': 0.0, 'position_size_factor': 0.0},  # Increase window
        2: {'window_size': -0.1, 'threshold': 0.0, 'risk_factor': 0.0, 'position_size_factor': 0.0},  # Decrease window
        3: {'window_size': 0.0, 'threshold': 0.1, 'risk_factor': 0.0, 'position_size_factor': 0.0},  # Increase threshold
        4: {'window_size': 0.0, 'threshold': -0.1, 'risk_factor': 0.0, 'position_size_factor': 0.0},  # Decrease threshold
        5: {'window_size': 0.0, 'threshold': 0.0, 'risk_factor': 0.1, 'position_size_factor': 0.0},  # Increase risk
        6: {'window_size': 0.0, 'threshold': 0.0, 'risk_factor': -0.1, 'position_size_factor': 0.0},  # Decrease risk
        7: {'window_size': 0.0, 'threshold': 0.0, 'risk_factor': 0.0, 'position_size_factor': 0.1},  # Increase position size
        8: {'window_size': 0.0, 'threshold': 0.0, 'risk_factor': 0.0, 'position_size_factor': -0.1},  # Decrease position size
        9: {'window_size': 0.05, 'threshold': 0.05, 'risk_factor': 0.05, 'position_size_factor': 0.05},  # Increase all slightly
        10: {'window_size': -0.05, 'threshold': -0.05, 'risk_factor': -0.05, 'position_size_factor': -0.05},  # Decrease all slightly
    })
    
    # Extract RL hyperparameters
    rl_params = config.get('rl_params', {})
    
    # Create and return the agent
    agent = TradingRLAgent(
        risk_manager=risk_manager,
        market_features=market_features,
        strategy_params=strategy_params,
        performance_metrics=performance_metrics,
        action_map=action_map,
        **rl_params
    )
    
    return agent


# Simplified version for when TensorFlow is not available
class SimplifiedTradingRLAgent:
    """
    Simplified Trading RL Agent that works without TensorFlow.
    Provides basic functionality for testing purposes.
    """
    
    def __init__(self, risk_manager: RiskManager, config: dict):
        """
        Initialize a simplified agent.
        
        Args:
            risk_manager: Risk manager for calculating rewards
            config: Configuration dictionary
        """
        self.risk_manager = risk_manager
        self.config = config
        self.strategy_params = config.get('strategy_params', [
            'confidence_threshold', 'position_size_factor'
        ])
        self.epsilon = config.get('rl_params', dict()).get('epsilon', 0.5)
        self.epsilon_decay = config.get('rl_params', dict()).get('epsilon_decay', 0.99)
        self.gamma = config.get('rl_params', dict()).get('gamma', 0.95)
        self.rewards_history = []
        logger.info("Initialized SimplifiedTradingRLAgent")
        
    def calculate_reward(self, returns: float, volatility: float, drawdown: float, 
                        trade_count: int, prev_sharpe: float = None) -> float:
        """
        Calculate reward based on performance metrics.
        
        Args:
            returns: Period returns
            volatility: Period volatility
            drawdown: Maximum drawdown
            trade_count: Number of trades executed
            prev_sharpe: Previous Sharpe ratio for comparison
            
        Returns:
            Calculated reward value
        """
        # Simple reward function based on returns and drawdown
        reward = returns * 10 - drawdown * 5
        
        # Use risk manager if available
        if hasattr(self.risk_manager, 'calculate_adjusted_reward'):
            reward = self.risk_manager.calculate_adjusted_reward(
                returns, volatility, drawdown, trade_count
            )
        
        # Add comparative component if previous Sharpe is available
        if prev_sharpe is not None:
            current_sharpe = returns / volatility if volatility > 0 else 0
            sharpe_improvement = current_sharpe - prev_sharpe
            reward += sharpe_improvement
            
        # Store reward
        self.rewards_history.append(reward)
        
        return reward
        
    def adapt_strategy(self, current_state: dict, market_data: pd.DataFrame) -> dict:
        """
        Adapt strategy parameters based on current state and market data.
        
        Args:
            current_state: Dictionary with current strategy parameters and performance
            market_data: DataFrame with market data
            
        Returns:
            Updated strategy parameters
        """
        # Extract current parameters and performance metrics
        strategy_params = current_state.get('strategy_params', dict()).copy()
        performance_metrics = current_state.get('performance', dict())
        
        # Calculate reward if performance metrics are available
        reward = 0
        if all(k in performance_metrics for k in ['returns', 'volatility', 'drawdown']):
            reward = self.calculate_reward(
                performance_metrics['returns'],
                performance_metrics['volatility'],
                performance_metrics['drawdown'],
                performance_metrics.get('trade_count', 0),
                current_state.get('prev_performance', dict()).get('sharpe_ratio')
            )
        
        # Simple exploration-exploitation approach
        if np.random.random() < self.epsilon:
            # Exploration: make random adjustments
            for param in strategy_params:
                adjustment = (np.random.random() - 0.5) * 0.1  # Small random adjustment (-0.05 to 0.05)
                strategy_params[param] = strategy_params[param] + adjustment
                
                # Ensure parameters stay within reasonable bounds
                strategy_params[param] = max(0.1, min(0.9, strategy_params[param]))
        else:
            # Exploitation: adjust based on reward
            if reward > 0:
                # If reward is positive, make small adjustments in the same direction as before
                prev_params = current_state.get('prev_params', strategy_params.copy() if strategy_params else dict())
                for param in strategy_params:
                    # Calculate direction of previous change
                    if param in prev_params:
                        direction = strategy_params[param] - prev_params[param]
                        
                        # Make a small adjustment in the same direction
                        if abs(direction) > 0.001:  # Only if there was a significant previous change
                            adjustment = np.sign(direction) * 0.01
                            strategy_params[param] = strategy_params[param] + adjustment
                            
                            # Ensure parameters stay within reasonable bounds
                            strategy_params[param] = max(0.1, min(0.9, strategy_params[param]))
        
        # Decay epsilon for less exploration over time
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)
        
        # Store current parameters for next iteration
        current_state['prev_params'] = strategy_params.copy()
        
        return strategy_params
    
    def reset_history(self) -> None:
        """Reset the agent's history."""
        self.rewards_history = []
