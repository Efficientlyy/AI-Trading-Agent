"""
Multi-Asset Portfolio Strategy Module

This module implements portfolio-level strategies that optimize asset allocation
across multiple trading instruments, considering correlation, risk, and expected returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy.optimize import minimize
from scipy.stats import norm

from .strategy import BaseStrategy, RichSignal, RichSignalsDict
from ..common import logger


class PortfolioStrategy(BaseStrategy):
    """
    Strategy that optimizes portfolio allocation across multiple assets.
    
    This strategy:
    1. Analyzes correlations between assets
    2. Calculates optimal weights using modern portfolio theory
    3. Generates signals based on the difference between optimal and current weights
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PortfolioStrategy.
        
        Args:
            config: Configuration dictionary with parameters for the strategy.
                - name: Name of the strategy
                - optimization_method: Method for portfolio optimization ('mean_variance', 'risk_parity', 'min_variance')
                - risk_aversion: Risk aversion parameter for mean-variance optimization
                - rebalance_threshold: Minimum weight difference to generate a signal
                - max_position_size: Maximum position size for any single asset
                - lookback_window: Window for calculating returns and covariance
                - target_volatility: Target portfolio volatility (annualized)
                - use_shrinkage: Whether to use covariance shrinkage for more robust estimates
                - min_history: Minimum history required for optimization
        """
        super().__init__(config)
        self.name = config.get("name", "PortfolioStrategy")
        self.optimization_method = config.get("optimization_method", "mean_variance")
        self.risk_aversion = config.get("risk_aversion", 2.0)
        self.rebalance_threshold = config.get("rebalance_threshold", 0.05)
        self.max_position_size = config.get("max_position_size", 0.3)
        self.lookback_window = config.get("lookback_window", 60)
        self.target_volatility = config.get("target_volatility", 0.15)  # 15% annualized
        self.use_shrinkage = config.get("use_shrinkage", True)
        self.min_history = config.get("min_history", 30)
        
        # Store last optimized weights
        self.last_weights = {}
        
        logger.info(f"{self.name} initialized with {self.optimization_method} optimization method")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> RichSignalsDict:
        """
        Generate trading signals based on portfolio optimization.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            **kwargs: Additional keyword arguments
                - current_portfolio: Current portfolio state
                - timestamp: Current timestamp
                
        Returns:
            Dictionary mapping symbols to their signal dictionaries
        """
        if not data:
            logger.warning(f"{self.name}: No data provided for signal generation")
            return {}
        
        signals = {}
        timestamp = kwargs.get("timestamp", pd.Timestamp.now())
        current_portfolio = kwargs.get("current_portfolio", {})
        
        # Extract current positions and weights
        current_weights = self._extract_current_weights(current_portfolio)
        
        # Calculate optimal portfolio weights
        optimal_weights = self._optimize_portfolio(data)
        if not optimal_weights:
            logger.warning(f"{self.name}: Could not calculate optimal weights")
            return {}
        
        # Store optimal weights for future reference
        self.last_weights = optimal_weights
        
        # Generate signals based on weight differences
        for symbol, optimal_weight in optimal_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            weight_difference = optimal_weight - current_weight
            
            # Only generate signal if difference exceeds threshold
            if abs(weight_difference) >= self.rebalance_threshold:
                # Convert weight difference to signal strength (-1 to 1)
                # Normalize by max_position_size to keep within bounds
                signal_strength = np.clip(weight_difference / self.max_position_size, -1.0, 1.0)
                
                # Higher confidence for larger weight differences
                confidence_score = min(abs(weight_difference) / (2 * self.rebalance_threshold), 1.0)
                
                signals[symbol] = {
                    "signal_strength": signal_strength,
                    "confidence_score": confidence_score,
                    "signal_type": self.name,
                    "metadata": {
                        "optimal_weight": optimal_weight,
                        "current_weight": current_weight,
                        "weight_difference": weight_difference,
                        "optimization_method": self.optimization_method,
                        "timestamp": timestamp
                    }
                }
        
        return signals
    
    def _extract_current_weights(self, portfolio: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract current portfolio weights from portfolio state.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            Dictionary mapping symbols to their current weights
        """
        weights = {}
        
        # Check if portfolio data is available
        if not portfolio or 'total_value' not in portfolio or portfolio['total_value'] == 0:
            return weights
        
        total_value = portfolio.get('total_value', 0.0)
        positions = portfolio.get('positions', {})
        
        # Calculate weights
        for symbol, position in positions.items():
            position_value = position.get('value', 0.0)
            weights[symbol] = position_value / total_value if total_value > 0 else 0.0
        
        return weights
    
    def _optimize_portfolio(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights using the specified optimization method.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            
        Returns:
            Dictionary mapping symbols to their optimal weights
        """
        try:
            # Extract returns for all symbols
            returns_dict = {}
            for symbol, df in data.items():
                if len(df) < self.min_history:
                    logger.warning(f"{self.name}: Not enough history for {symbol}, skipping")
                    continue
                
                # Identify price column
                if 'close' in df.columns:
                    price_col = 'close'
                else:
                    price_cols = [col for col in df.columns if any(x in col.lower() for x in ['close', 'price', 'adj'])]
                    if not price_cols:
                        logger.warning(f"{self.name}: No suitable price column found for {symbol}")
                        continue
                    price_col = price_cols[0]
                
                # Calculate returns
                returns = df[price_col].pct_change().dropna()
                returns_dict[symbol] = returns.iloc[-self.lookback_window:] if len(returns) > self.lookback_window else returns
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna(how='all')
            
            # Check if we have enough data
            if len(returns_df) < self.min_history or len(returns_df.columns) < 2:
                logger.warning(f"{self.name}: Not enough data for portfolio optimization")
                return {}
            
            # Fill any remaining NaNs with zeros
            returns_df = returns_df.fillna(0)
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Apply shrinkage to covariance matrix if enabled
            if self.use_shrinkage:
                cov_matrix = self._shrink_covariance(cov_matrix)
            
            # Perform optimization based on selected method
            if self.optimization_method == "mean_variance":
                weights = self._mean_variance_optimization(expected_returns, cov_matrix)
            elif self.optimization_method == "min_variance":
                weights = self._minimum_variance_optimization(cov_matrix)
            elif self.optimization_method == "risk_parity":
                weights = self._risk_parity_optimization(cov_matrix)
            elif self.optimization_method == "max_sharpe":
                weights = self._maximum_sharpe_optimization(expected_returns, cov_matrix)
            else:
                logger.warning(f"{self.name}: Unknown optimization method {self.optimization_method}")
                return {}
            
            # Convert weights array to dictionary
            weight_dict = {symbol: weight for symbol, weight in zip(returns_df.columns, weights)}
            
            return weight_dict
            
        except Exception as e:
            logger.error(f"{self.name}: Error in portfolio optimization: {e}")
            return {}
    
    def _mean_variance_optimization(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Perform mean-variance optimization to maximize utility.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Array of optimal weights
        """
        n = len(expected_returns)
        
        # Define objective function (negative utility)
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
            return -utility
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((0.0, self.max_position_size) for _ in range(n))
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n) / n
        
        # Perform optimization
        result = minimize(objective, initial_weights, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning(f"{self.name}: Optimization failed: {result.message}")
            return initial_weights
    
    def _minimum_variance_optimization(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Perform minimum variance optimization.
        
        Args:
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Array of optimal weights
        """
        n = len(cov_matrix)
        
        # Define objective function (portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((0.0, self.max_position_size) for _ in range(n))
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n) / n
        
        # Perform optimization
        result = minimize(objective, initial_weights, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning(f"{self.name}: Optimization failed: {result.message}")
            return initial_weights
    
    def _risk_parity_optimization(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Perform risk parity optimization.
        
        Args:
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Array of optimal weights
        """
        n = len(cov_matrix)
        
        # Define objective function (sum of squared risk contribution differences)
        def objective(weights):
            weights = np.clip(weights, 1e-6, 1.0)  # Avoid division by zero
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_risk = np.dot(cov_matrix, weights)
            risk_contribution = weights * marginal_risk / portfolio_risk
            
            # Target equal risk contribution
            target_risk = portfolio_risk / n
            
            # Sum of squared differences from target
            return np.sum((risk_contribution - target_risk)**2)
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((0.0, self.max_position_size) for _ in range(n))
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n) / n
        
        # Perform optimization
        result = minimize(objective, initial_weights, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            logger.warning(f"{self.name}: Optimization failed: {result.message}")
            return initial_weights
    
    def _maximum_sharpe_optimization(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Perform maximum Sharpe ratio optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Array of optimal weights
        """
        n = len(expected_returns)
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Avoid division by zero
            if portfolio_volatility < 1e-8:
                return -portfolio_return / 1e-8
            return -portfolio_return / portfolio_volatility
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Define bounds
        bounds = tuple((0.0, self.max_position_size) for _ in range(n))
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n) / n
        
        # Perform optimization
        result = minimize(objective, initial_weights, method='SLSQP', 
                          bounds=bounds, constraints=constraints)
        
        if result.success:
            # Scale weights to target volatility
            weights = result.x
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_volatility > 0:
                scaling_factor = self.target_volatility / portfolio_volatility
                weights = weights * scaling_factor
                
                # Ensure weights sum to 1 after scaling
                weights = weights / np.sum(weights)
                
                # Ensure no weight exceeds max_position_size
                if np.max(weights) > self.max_position_size:
                    excess = weights - self.max_position_size
                    excess[excess < 0] = 0
                    weights = np.minimum(weights, self.max_position_size)
                    
                    # Redistribute excess weight
                    total_excess = np.sum(excess)
                    if total_excess > 0:
                        room = self.max_position_size - weights
                        room[room < 0] = 0
                        total_room = np.sum(room)
                        
                        if total_room > 0:
                            redistribution = room / total_room * total_excess
                            weights = weights + redistribution
            
            return weights
        else:
            logger.warning(f"{self.name}: Optimization failed: {result.message}")
            return initial_weights
    
    def _shrink_covariance(self, cov_matrix: pd.DataFrame, shrinkage_factor: Optional[float] = None) -> pd.DataFrame:
        """
        Apply shrinkage to covariance matrix for more robust estimates.
        
        Args:
            cov_matrix: Original covariance matrix
            shrinkage_factor: Shrinkage factor (0 to 1), if None, will be estimated
            
        Returns:
            Shrunk covariance matrix
        """
        # If shrinkage factor not provided, use a simple estimate
        if shrinkage_factor is None:
            # More samples -> less shrinkage needed
            n_samples = len(cov_matrix)
            shrinkage_factor = min(0.5, 1.0 / np.sqrt(n_samples))
        
        # Create target matrix (diagonal matrix with same variances)
        target = np.diag(np.diag(cov_matrix))
        
        # Apply shrinkage
        shrunk_cov = (1 - shrinkage_factor) * cov_matrix + shrinkage_factor * target
        
        return pd.DataFrame(shrunk_cov, index=cov_matrix.index, columns=cov_matrix.columns)
    
    def get_optimal_weights(self) -> Dict[str, float]:
        """
        Get the most recently calculated optimal weights.
        
        Returns:
            Dictionary mapping symbols to their optimal weights
        """
        return self.last_weights.copy()
    
    def calculate_portfolio_metrics(self, data: Dict[str, pd.DataFrame], 
                                   weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate key portfolio metrics based on historical data and weights.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            weights: Portfolio weights to use (if None, use last optimal weights)
            
        Returns:
            Dictionary of portfolio metrics
        """
        if weights is None:
            weights = self.last_weights
            
        if not weights:
            logger.warning(f"{self.name}: No weights available for portfolio metrics calculation")
            return {}
        
        try:
            # Extract returns for symbols in the portfolio
            returns_dict = {}
            for symbol, weight in weights.items():
                if symbol not in data:
                    continue
                    
                df = data[symbol]
                
                # Identify price column
                if 'close' in df.columns:
                    price_col = 'close'
                else:
                    price_cols = [col for col in df.columns if any(x in col.lower() for x in ['close', 'price', 'adj'])]
                    if not price_cols:
                        continue
                    price_col = price_cols[0]
                
                # Calculate returns
                returns = df[price_col].pct_change().dropna()
                returns_dict[symbol] = returns.iloc[-self.lookback_window:] if len(returns) > self.lookback_window else returns
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna(how='all')
            
            if len(returns_df) < 10:  # Need minimum data for meaningful metrics
                return {}
                
            # Convert weights dict to series aligned with returns columns
            weight_series = pd.Series([weights.get(col, 0.0) for col in returns_df.columns], index=returns_df.columns)
            
            # Calculate portfolio returns
            portfolio_returns = returns_df.dot(weight_series)
            
            # Calculate metrics
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Calculate max drawdown
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Calculate VaR and CVaR (95%)
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Calculate correlation with market (if available)
            market_correlation = None
            for market_symbol in ['SPY', 'QQQ', '^GSPC', '^IXIC']:
                if market_symbol in returns_dict:
                    market_correlation = portfolio_returns.corr(returns_dict[market_symbol])
                    break
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'market_correlation': market_correlation
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating portfolio metrics: {e}")
            return {}
            
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the strategy's configuration parameters dynamically.
        
        Args:
            config_updates: A dictionary containing parameters to update.
        """
        # Update main config
        self.config.update(config_updates)
        
        # Update specific parameters
        if "optimization_method" in config_updates:
            self.optimization_method = config_updates["optimization_method"]
            
        if "risk_aversion" in config_updates:
            self.risk_aversion = config_updates["risk_aversion"]
            
        if "rebalance_threshold" in config_updates:
            self.rebalance_threshold = config_updates["rebalance_threshold"]
            
        if "max_position_size" in config_updates:
            self.max_position_size = config_updates["max_position_size"]
            
        if "lookback_window" in config_updates:
            self.lookback_window = config_updates["lookback_window"]
            
        if "target_volatility" in config_updates:
            self.target_volatility = config_updates["target_volatility"]
            
        if "use_shrinkage" in config_updates:
            self.use_shrinkage = config_updates["use_shrinkage"]
            
        if "min_history" in config_updates:
            self.min_history = config_updates["min_history"]
            
        # Reset last weights if optimization method changes
        if "optimization_method" in config_updates:
            self.last_weights = {}
            
        logger.info(f"{self.name} configuration updated")
