#!/usr/bin/env python
"""
Position Risk Analyzer for the AI Trading System.

This module provides tools for analyzing risk metrics for trading positions
and portfolios, including Value at Risk (VaR), stress testing, and correlation
analysis.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)

class VaRMethod(Enum):
    """Methods for calculating Value at Risk."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"

class ConfidenceLevel(Enum):
    """Standard confidence levels for risk calculations."""
    CL_90 = 0.90
    CL_95 = 0.95
    CL_99 = 0.99
    CL_99_5 = 0.995
    CL_99_9 = 0.999

@dataclass
class Position:
    """Represents a trading position with risk parameters."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    exchange: str
    timestamp: datetime.datetime
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def position_value(self) -> float:
        """Calculate the current value of the position."""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate the unrealized profit/loss of the position."""
        return self.quantity * (self.current_price - self.entry_price)
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate the unrealized profit/loss percentage of the position."""
        return (self.current_price / self.entry_price - 1) * 100.0

@dataclass
class RiskMetrics:
    """Contains various risk metrics for a position or portfolio."""
    var_1d_95: float  # 1-day VaR at 95% confidence
    var_1d_99: float  # 1-day VaR at 99% confidence
    var_10d_99: float  # 10-day VaR at 99% confidence
    expected_shortfall: float  # Expected shortfall (average loss beyond VaR)
    max_drawdown: float  # Maximum historical drawdown
    volatility: float  # Historical volatility (standard deviation of returns)
    beta: Optional[float] = None  # Systematic risk relative to market
    sharpe_ratio: Optional[float] = None  # Risk-adjusted return measure
    sortino_ratio: Optional[float] = None  # Downside risk-adjusted return
    correlation_matrix: Optional[pd.DataFrame] = None  # Asset correlations
    stress_test_results: Optional[Dict[str, float]] = None  # Results of stress tests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for serialization."""
        result = {
            "var_1d_95": self.var_1d_95,
            "var_1d_99": self.var_1d_99,
            "var_10d_99": self.var_10d_99,
            "expected_shortfall": self.expected_shortfall,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility
        }
        
        if self.beta is not None:
            result["beta"] = self.beta
        
        if self.sharpe_ratio is not None:
            result["sharpe_ratio"] = self.sharpe_ratio
            
        if self.sortino_ratio is not None:
            result["sortino_ratio"] = self.sortino_ratio
        
        if self.stress_test_results is not None:
            result["stress_test_results"] = self.stress_test_results
        
        # Convert correlation matrix to a nested dictionary if it exists
        if self.correlation_matrix is not None:
            result["correlation_matrix"] = self.correlation_matrix.to_dict()
            
        return result

class PositionRiskAnalyzer:
    """
    Analyzes risk metrics for trading positions and portfolios.
    
    This class provides methods for calculating Value at Risk (VaR),
    expected shortfall, stress testing, and other risk metrics for
    both individual positions and portfolios of positions.
    """
    
    def __init__(self, market_data_provider=None, risk_free_rate: float = 0.0, time_horizon: int = 1):
        """
        Initialize the position risk analyzer.
        
        Args:
            market_data_provider: Provider for historical market data
            risk_free_rate: Annual risk-free rate used for Sharpe ratio calculation
            time_horizon: Default time horizon in days for VaR calculations
        """
        self.market_data_provider = market_data_provider
        self.risk_free_rate = risk_free_rate / 252  # Convert annual rate to daily
        self.time_horizon = time_horizon
        self.logger = logger
        
        self.logger.info(f"Initialized PositionRiskAnalyzer with time_horizon={time_horizon} days")
    
    def calculate_var_historical(
        self,
        returns: np.ndarray,
        position_value: float,
        confidence_level: float = 0.95,
        time_horizon: int = None
    ) -> float:
        """
        Calculate Value at Risk using the historical method.
        
        Args:
            returns: Historical returns as a numpy array
            position_value: Current value of the position
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            
        Returns:
            Value at Risk estimate
        """
        if time_horizon is None:
            time_horizon = self.time_horizon
            
        # Sort returns from lowest to highest
        sorted_returns = np.sort(returns)
        
        # Find the return at the specified percentile
        index = int(np.ceil(len(sorted_returns) * (1 - confidence_level))) - 1
        index = max(0, index)  # Ensure index is not negative
        
        # Get the critical return
        var_return = sorted_returns[index]
        
        # Scale to the position value and time horizon
        var = position_value * abs(var_return) * np.sqrt(time_horizon)
        
        return var
    
    def calculate_var_parametric(
        self,
        returns: np.ndarray,
        position_value: float,
        confidence_level: float = 0.95,
        time_horizon: int = None
    ) -> float:
        """
        Calculate Value at Risk using the parametric (variance-covariance) method.
        
        Args:
            returns: Historical returns as a numpy array
            position_value: Current value of the position
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            
        Returns:
            Value at Risk estimate
        """
        if time_horizon is None:
            time_horizon = self.time_horizon
            
        # Calculate mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Calculate z-score for the given confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var = position_value * (mean_return + z_score * std_return) * np.sqrt(time_horizon)
        
        # Make sure VaR is positive (representing a loss)
        var = abs(var)
        
        return var
    
    def calculate_var_monte_carlo(
        self,
        returns: np.ndarray,
        position_value: float,
        confidence_level: float = 0.95,
        time_horizon: int = None,
        num_simulations: int = 10000
    ) -> float:
        """
        Calculate Value at Risk using Monte Carlo simulation.
        
        Args:
            returns: Historical returns as a numpy array
            position_value: Current value of the position
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            num_simulations: Number of simulations to run
            
        Returns:
            Value at Risk estimate
        """
        if time_horizon is None:
            time_horizon = self.time_horizon
            
        # Calculate mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Generate random returns
        random_returns = np.random.normal(
            mean_return, 
            std_return, 
            size=num_simulations
        )
        
        # Adjust for time horizon
        random_returns = random_returns * np.sqrt(time_horizon)
        
        # Calculate portfolio values after the random returns
        portfolio_values = position_value * (1 + random_returns)
        
        # Calculate losses
        losses = position_value - portfolio_values
        
        # Sort losses
        sorted_losses = np.sort(losses)
        
        # Find the VaR at the specified confidence level
        var_index = int(np.ceil(num_simulations * confidence_level)) - 1
        var = sorted_losses[var_index]
        
        # Make sure VaR is positive (representing a loss)
        var = max(0, var)
        
        return var
    
    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        position_value: float,
        confidence_level: float = 0.95,
        time_horizon: int = None
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        The expected shortfall is the expected loss given that the loss
        exceeds the Value at Risk.
        
        Args:
            returns: Historical returns as a numpy array
            position_value: Current value of the position
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            
        Returns:
            Expected Shortfall estimate
        """
        if time_horizon is None:
            time_horizon = self.time_horizon
            
        # Sort returns from lowest to highest
        sorted_returns = np.sort(returns)
        
        # Find the index for the VaR threshold
        var_index = int(len(sorted_returns) * (1 - confidence_level))
        var_index = max(0, var_index)
        
        # Get tail returns beyond VaR
        tail_returns = sorted_returns[:var_index+1]
        
        # Calculate expected shortfall as the average of tail returns
        es_return = np.mean(tail_returns)
        
        # Scale to position value and time horizon
        es = position_value * abs(es_return) * np.sqrt(time_horizon)
        
        return es
    
    def calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate the maximum drawdown for a series of prices.
        
        Args:
            prices: Historical price data as a numpy array
            
        Returns:
            Maximum drawdown as a fraction of the peak value
        """
        # Calculate the running maximum
        running_max = np.maximum.accumulate(prices)
        
        # Calculate drawdowns
        drawdowns = (running_max - prices) / running_max
        
        # Find the maximum drawdown
        max_drawdown = np.max(drawdowns)
        
        return max_drawdown
    
    def calculate_portfolio_var(
        self,
        positions: List[Position],
        returns_data: Dict[str, np.ndarray],
        confidence_level: float = 0.95,
        time_horizon: int = None,
        method: VaRMethod = VaRMethod.PARAMETRIC
    ) -> float:
        """
        Calculate Value at Risk for a portfolio of positions.
        
        Args:
            positions: List of Position objects in the portfolio
            returns_data: Dictionary mapping symbols to return arrays
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: VaR calculation method
            
        Returns:
            Portfolio Value at Risk estimate
        """
        if time_horizon is None:
            time_horizon = self.time_horizon
            
        # For parametric VaR, we need to account for correlations
        if method == VaRMethod.PARAMETRIC:
            # Extract position values and returns for each position
            symbols = [pos.symbol for pos in positions]
            position_values = [pos.position_value for pos in positions]
            
            # Build a returns DataFrame for correlation calculation
            returns_df = pd.DataFrame({sym: returns_data[sym] for sym in symbols})
            
            # Calculate the covariance matrix
            cov_matrix = returns_df.cov()
            
            # Calculate portfolio variance using matrix multiplication
            portfolio_variance = 0.0
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions):
                    portfolio_variance += (
                        position_values[i] * 
                        position_values[j] * 
                        cov_matrix.iloc[i, j]
                    )
            
            # Calculate portfolio standard deviation
            portfolio_std = np.sqrt(portfolio_variance)
            
            # Calculate total portfolio value
            portfolio_value = sum(position_values)
            
            # Calculate the z-score for the given confidence level
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # Calculate VaR
            var = portfolio_value * z_score * portfolio_std * np.sqrt(time_horizon)
            var = abs(var)
            
            return var
        
        # For historical and Monte Carlo methods, simulate portfolio returns
        elif method in [VaRMethod.HISTORICAL, VaRMethod.MONTE_CARLO]:
            # If using Monte Carlo, generate simulated returns
            if method == VaRMethod.MONTE_CARLO:
                num_simulations = 10000
                simulated_portfolio_returns = []
                
                # Get mean and std for each asset
                means = {sym: np.mean(returns_data[sym]) for sym in returns_data}
                stds = {sym: np.std(returns_data[sym], ddof=1) for sym in returns_data}
                
                # Build correlation matrix
                symbols = list(returns_data.keys())
                returns_df = pd.DataFrame({sym: returns_data[sym] for sym in symbols})
                correlation_matrix = returns_df.corr()
                
                # Generate correlated random returns
                for _ in range(num_simulations):
                    # Generate standard normal random numbers
                    random_nums = np.random.standard_normal(len(symbols))
                    
                    # Apply Cholesky decomposition to get correlated random numbers
                    cholesky = np.linalg.cholesky(correlation_matrix.values)
                    correlated_random = np.dot(cholesky, random_nums)
                    
                    # Calculate simulated returns for each asset
                    simulated_returns = {
                        sym: means[sym] + stds[sym] * correlated_random[i]
                        for i, sym in enumerate(symbols)
                    }
                    
                    # Calculate portfolio return for this simulation
                    portfolio_return = sum(
                        pos.position_value * simulated_returns.get(pos.symbol, 0)
                        for pos in positions
                    ) / sum(pos.position_value for pos in positions)
                    
                    simulated_portfolio_returns.append(portfolio_return)
                
                # Use the simulated returns for VaR calculation
                portfolio_returns = np.array(simulated_portfolio_returns)
            else:
                # For historical method, calculate historical portfolio returns
                # First, ensure all return arrays have the same length
                min_length = min(len(returns) for returns in returns_data.values())
                aligned_returns = {
                    sym: returns[-min_length:] for sym, returns in returns_data.items()
                }
                
                # Calculate historical portfolio returns
                portfolio_returns = np.zeros(min_length)
                portfolio_value = sum(pos.position_value for pos in positions)
                
                for pos in positions:
                    if pos.symbol in aligned_returns:
                        weight = pos.position_value / portfolio_value
                        portfolio_returns += weight * aligned_returns[pos.symbol]
            
            # Calculate VaR using the portfolio returns
            portfolio_value = sum(pos.position_value for pos in positions)
            var = self.calculate_var_historical(
                portfolio_returns, 
                portfolio_value, 
                confidence_level, 
                time_horizon
            )
            
            return var
        
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
    
    def analyze_position_risk(
        self,
        position: Position,
        historical_prices: np.ndarray,
        confidence_level: float = 0.95,
        time_horizon: int = None,
        var_method: VaRMethod = VaRMethod.HISTORICAL
    ) -> RiskMetrics:
        """
        Perform comprehensive risk analysis for a single position.
        
        Args:
            position: The Position object to analyze
            historical_prices: Historical price data as a numpy array
            confidence_level: Confidence level for VaR calculations
            time_horizon: Time horizon in days
            var_method: Method to use for VaR calculation
            
        Returns:
            RiskMetrics object containing the risk analysis results
        """
        if time_horizon is None:
            time_horizon = self.time_horizon
            
        # Calculate returns from historical prices
        returns = np.diff(historical_prices) / historical_prices[:-1]
        
        # Calculate volatility (annualized)
        volatility = np.std(returns, ddof=1) * np.sqrt(252)
        
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown(historical_prices)
        
        # Calculate VaR for different confidence levels and time horizons
        if var_method == VaRMethod.HISTORICAL:
            var_1d_95 = self.calculate_var_historical(returns, position.position_value, 0.95, 1)
            var_1d_99 = self.calculate_var_historical(returns, position.position_value, 0.99, 1)
            var_10d_99 = self.calculate_var_historical(returns, position.position_value, 0.99, 10)
        elif var_method == VaRMethod.PARAMETRIC:
            var_1d_95 = self.calculate_var_parametric(returns, position.position_value, 0.95, 1)
            var_1d_99 = self.calculate_var_parametric(returns, position.position_value, 0.99, 1)
            var_10d_99 = self.calculate_var_parametric(returns, position.position_value, 0.99, 10)
        elif var_method == VaRMethod.MONTE_CARLO:
            var_1d_95 = self.calculate_var_monte_carlo(returns, position.position_value, 0.95, 1)
            var_1d_99 = self.calculate_var_monte_carlo(returns, position.position_value, 0.99, 1)
            var_10d_99 = self.calculate_var_monte_carlo(returns, position.position_value, 0.99, 10)
        else:
            raise ValueError(f"Unsupported VaR method: {var_method}")
        
        # Calculate expected shortfall (conditional VaR)
        expected_shortfall = self.calculate_expected_shortfall(returns, position.position_value, confidence_level)
        
        # Create and return the risk metrics
        return RiskMetrics(
            var_1d_95=var_1d_95,
            var_1d_99=var_1d_99,
            var_10d_99=var_10d_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            volatility=volatility
        )
    
    def analyze_portfolio_risk(
        self,
        positions: List[Position],
        historical_data: Dict[str, np.ndarray],
        confidence_level: float = 0.95,
        time_horizon: int = None,
        var_method: VaRMethod = VaRMethod.PARAMETRIC
    ) -> RiskMetrics:
        """
        Perform comprehensive risk analysis for a portfolio of positions.
        
        Args:
            positions: List of Position objects in the portfolio
            historical_data: Dict mapping symbols to price arrays
            confidence_level: Confidence level for VaR calculations
            time_horizon: Time horizon in days
            var_method: Method to use for VaR calculation
            
        Returns:
            RiskMetrics object containing the risk analysis results
        """
        if time_horizon is None:
            time_horizon = self.time_horizon
            
        # Calculate returns from historical prices for each position
        returns_data = {}
        for symbol, prices in historical_data.items():
            returns_data[symbol] = np.diff(prices) / prices[:-1]
        
        # Create a returns DataFrame for correlation analysis
        symbols = [pos.symbol for pos in positions]
        returns_df = pd.DataFrame({sym: returns_data[sym] for sym in symbols if sym in returns_data})
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Calculate portfolio volatility
        # Get position weights
        portfolio_value = sum(pos.position_value for pos in positions)
        weights = [pos.position_value / portfolio_value for pos in positions]
        
        # Calculate weighted volatility
        symbol_volatilities = {sym: np.std(returns, ddof=1) * np.sqrt(252) 
                              for sym, returns in returns_data.items()}
        
        # Calculate portfolio VaR
        var_1d_95 = self.calculate_portfolio_var(
            positions, returns_data, 0.95, 1, var_method
        )
        var_1d_99 = self.calculate_portfolio_var(
            positions, returns_data, 0.99, 1, var_method
        )
        var_10d_99 = self.calculate_portfolio_var(
            positions, returns_data, 0.99, 10, var_method
        )
        
        # Calculate portfolio expected shortfall
        # For simplicity, we'll use historical simulation method
        min_length = min(len(returns) for returns in returns_data.values())
        aligned_returns = {
            sym: returns[-min_length:] for sym, returns in returns_data.items()
        }
        
        # Calculate historical portfolio returns
        portfolio_returns = np.zeros(min_length)
        for i, pos in enumerate(positions):
            if pos.symbol in aligned_returns:
                portfolio_returns += weights[i] * aligned_returns[pos.symbol]
        
        expected_shortfall = self.calculate_expected_shortfall(
            portfolio_returns, portfolio_value, confidence_level
        )
        
        # Calculate portfolio max drawdown
        # First, need to calculate historical portfolio values
        min_length = min(len(prices) for prices in historical_data.values())
        portfolio_values = np.zeros(min_length)
        
        for pos in positions:
            if pos.symbol in historical_data:
                # Get the latest min_length prices
                prices = historical_data[pos.symbol][-min_length:]
                # Scale to current position size
                scaled_prices = (prices / prices[-1]) * pos.position_value
                portfolio_values += scaled_prices
                
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        # Calculate portfolio volatility
        portfolio_volatility = np.std(portfolio_returns, ddof=1) * np.sqrt(252)
        
        # Create and return the risk metrics
        return RiskMetrics(
            var_1d_95=var_1d_95,
            var_1d_99=var_1d_99,
            var_10d_99=var_10d_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            volatility=portfolio_volatility,
            correlation_matrix=correlation_matrix
        )
    
    def perform_stress_test(
        self,
        positions: List[Position],
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Perform stress testing on a portfolio by applying various stress scenarios.
        
        Args:
            positions: List of Position objects in the portfolio
            scenarios: Dict mapping scenario names to dicts of price changes by symbol
            
        Returns:
            Dict mapping scenario names to portfolio losses
        """
        results = {}
        portfolio_value = sum(pos.position_value for pos in positions)
        
        for scenario_name, price_changes in scenarios.items():
            # Calculate portfolio value after applying the scenario
            new_portfolio_value = 0.0
            
            for pos in positions:
                # Get price change for this symbol (default to 0 if not specified)
                price_change_pct = price_changes.get(pos.symbol, 0.0)
                
                # Calculate new price
                new_price = pos.current_price * (1 + price_change_pct / 100.0)
                
                # Add to new portfolio value
                new_portfolio_value += pos.quantity * new_price
            
            # Calculate loss (or gain) in portfolio value
            loss = portfolio_value - new_portfolio_value
            loss_pct = (loss / portfolio_value) * 100.0
            
            # Store the result
            results[scenario_name] = loss_pct
        
        return results
    
    def visualize_var(
        self,
        position: Position,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL,
        save_path: Optional[str] = None
    ):
        """
        Visualize Value at Risk for a position.
        
        Args:
            position: The Position object to analyze
            returns: Historical returns as a numpy array
            confidence_level: Confidence level for VaR calculation
            method: Method to use for VaR calculation
            save_path: Path to save the plot, if specified
        """
        plt.figure(figsize=(12, 6))
        
        # Plot histogram of returns
        sns.histplot(returns, bins=50, kde=True, color='skyblue')
        
        # Calculate VaR
        if method == VaRMethod.HISTORICAL:
            var = self.calculate_var_historical(
                returns, position.position_value, confidence_level, 1
            )
            var_return = np.sort(returns)[int(len(returns) * (1 - confidence_level))]
        elif method == VaRMethod.PARAMETRIC:
            var = self.calculate_var_parametric(
                returns, position.position_value, confidence_level, 1
            )
            var_return = stats.norm.ppf(1 - confidence_level, np.mean(returns), np.std(returns, ddof=1))
        elif method == VaRMethod.MONTE_CARLO:
            var = self.calculate_var_monte_carlo(
                returns, position.position_value, confidence_level, 1
            )
            # For visualization, use historical method to approximate the return
            var_return = np.sort(returns)[int(len(returns) * (1 - confidence_level))]
        
        # Convert VaR to a return value for plotting
        var_pct = var / position.position_value
        
        # Calculate expected shortfall
        es = self.calculate_expected_shortfall(
            returns, position.position_value, confidence_level
        )
        es_pct = es / position.position_value
        
        # Plot VaR line
        plt.axvline(x=var_return, color='red', linestyle='--', 
                   label=f'VaR ({confidence_level*100:.0f}%): {var_pct:.2%}')
        
        # Add expected shortfall
        plt.axvline(x=np.mean(returns[returns <= var_return]), color='purple', 
                   linestyle=':', label=f'ES ({confidence_level*100:.0f}%): {es_pct:.2%}')
        
        plt.title(f'Value at Risk Analysis for {position.symbol} using {method.value} method')
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Visualize a correlation matrix as a heatmap.
        
        Args:
            correlation_matrix: Pandas DataFrame containing the correlation matrix
            save_path: Path to save the plot, if specified
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            fmt='.2f'
        )
        
        plt.title('Asset Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_example_stress_scenarios() -> Dict[str, Dict[str, float]]:
    """
    Create example stress test scenarios.
    
    Returns:
        Dictionary mapping scenario names to dictionaries of price changes by symbol
    """
    scenarios = {
        "Market Crash": {
            "BTC/USD": -30.0,
            "ETH/USD": -40.0,
            "SOL/USD": -45.0,
            "BNB/USD": -35.0,
            "ADA/USD": -50.0,
            "XRP/USD": -40.0,
        },
        "Tech Selloff": {
            "BTC/USD": -15.0,
            "ETH/USD": -25.0,
            "SOL/USD": -35.0,
            "BNB/USD": -20.0,
            "ADA/USD": -30.0,
            "XRP/USD": -15.0,
        },
        "Crypto Bull Run": {
            "BTC/USD": 50.0,
            "ETH/USD": 75.0,
            "SOL/USD": 100.0,
            "BNB/USD": 60.0,
            "ADA/USD": 80.0,
            "XRP/USD": 70.0,
        },
        "BTC Dominance Rise": {
            "BTC/USD": 20.0,
            "ETH/USD": -5.0,
            "SOL/USD": -10.0,
            "BNB/USD": -7.0,
            "ADA/USD": -12.0,
            "XRP/USD": -8.0,
        },
        "Altcoin Season": {
            "BTC/USD": 5.0,
            "ETH/USD": 30.0,
            "SOL/USD": 50.0,
            "BNB/USD": 35.0,
            "ADA/USD": 45.0,
            "XRP/USD": 40.0,
        },
        "Regulatory Crackdown": {
            "BTC/USD": -20.0,
            "ETH/USD": -30.0,
            "SOL/USD": -25.0,
            "BNB/USD": -40.0,
            "ADA/USD": -35.0,
            "XRP/USD": -45.0,
        }
    }
    
    return scenarios 