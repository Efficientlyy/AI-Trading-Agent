"""
Correlation-based Portfolio Allocator for AI Trading Agent.

This module provides advanced portfolio allocation methods that take into account
correlations between assets to optimize risk-adjusted returns.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize

from ..common import logger

class CorrelationAllocator:
    """
    Correlation-based Portfolio Allocator.
    
    This class provides methods to adjust portfolio allocations based on asset correlations
    to achieve better diversification and risk-adjusted returns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the correlation allocator.
        
        Args:
            config: Configuration dictionary with parameters:
                - min_allocation: Minimum allocation per asset
                - max_allocation: Maximum allocation per asset
                - risk_aversion: Risk aversion parameter (higher = more conservative)
                - target_return: Target portfolio return (if using mean-variance optimization)
                - optimization_method: 'min_variance', 'max_sharpe', 'risk_parity', 'hierarchical'
                - correlation_lookback: Number of periods to use for correlation calculation
        """
        self.name = "CorrelationAllocator"
        self.min_allocation = config.get('min_allocation', 0.01)  # 1% minimum allocation
        self.max_allocation = config.get('max_allocation', 0.3)   # 30% maximum allocation
        self.risk_aversion = config.get('risk_aversion', 1.0)     # Risk aversion parameter
        self.target_return = config.get('target_return', None)    # Target return (optional)
        self.optimization_method = config.get('optimization_method', 'min_variance')
        self.correlation_lookback = config.get('correlation_lookback', 90)  # 90-day lookback
        
        logger.info(f"Initialized {self.name} with {self.optimization_method} optimization method")
    
    def calculate_allocations(self, returns: pd.DataFrame, 
                             covariance_matrix: Optional[pd.DataFrame] = None,
                             expected_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate optimal allocations based on asset returns and correlations.
        
        Args:
            returns: DataFrame of asset returns with assets as columns
            covariance_matrix: Optional pre-calculated covariance matrix
            expected_returns: Optional expected returns for each asset
            
        Returns:
            Dictionary mapping asset symbols to target allocation percentages
        """
        logger.info(f"Calculating optimal allocations using {self.optimization_method} method")
        
        # Get number of assets
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Calculate covariance matrix if not provided
        if covariance_matrix is None:
            covariance_matrix = returns.cov()
        
        # Calculate expected returns if not provided
        if expected_returns is None:
            expected_returns = returns.mean()
        
        # Calculate optimal weights based on selected method
        if self.optimization_method == 'min_variance':
            weights = self._minimum_variance_portfolio(covariance_matrix)
        elif self.optimization_method == 'max_sharpe':
            weights = self._maximum_sharpe_ratio(expected_returns, covariance_matrix)
        elif self.optimization_method == 'risk_parity':
            weights = self._risk_parity_portfolio(covariance_matrix)
        elif self.optimization_method == 'hierarchical':
            weights = self._hierarchical_risk_parity(returns)
        else:
            # Default to equal weight
            weights = np.ones(n_assets) / n_assets
        
        # Convert weights to allocation dictionary
        allocations = {}
        for i, asset in enumerate(assets):
            # Ensure allocations are within bounds
            allocation = max(min(weights[i], self.max_allocation), self.min_allocation)
            allocations[asset] = allocation
        
        # Normalize allocations to sum to 1.0
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            allocations = {k: v / total_allocation for k, v in allocations.items()}
        
        return allocations
    
    def _minimum_variance_portfolio(self, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """
        Calculate minimum variance portfolio weights.
        
        Args:
            covariance_matrix: Covariance matrix of asset returns
            
        Returns:
            Array of optimal portfolio weights
        """
        n_assets = len(covariance_matrix.columns)
        
        # Define optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]  # weights sum to 1
        bounds = tuple((self.min_allocation, self.max_allocation) for _ in range(n_assets))
        
        # Initial guess (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Define objective function (portfolio variance)
        def objective(weights):
            weights = np.array(weights)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            return portfolio_variance
        
        # Solve the optimization problem
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result['success']:
            return result['x']
        else:
            logger.warning(f"Optimization failed: {result['message']}")
            return initial_weights
    
    def _maximum_sharpe_ratio(self, expected_returns: pd.Series, 
                             covariance_matrix: pd.DataFrame,
                             risk_free_rate: float = 0.0) -> np.ndarray:
        """
        Calculate maximum Sharpe ratio portfolio weights.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Array of optimal portfolio weights
        """
        n_assets = len(covariance_matrix.columns)
        
        # Define optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]  # weights sum to 1
        bounds = tuple((self.min_allocation, self.max_allocation) for _ in range(n_assets))
        
        # Initial guess (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            weights = np.array(weights)
            portfolio_return = np.dot(weights, expected_returns) - risk_free_rate
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Handle division by zero
            if portfolio_volatility == 0:
                return -portfolio_return  # Just maximize return if vol is zero
            
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio  # Negative because we want to maximize
        
        # Solve the optimization problem
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result['success']:
            return result['x']
        else:
            logger.warning(f"Optimization failed: {result['message']}")
            return initial_weights
    
    def _risk_parity_portfolio(self, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """
        Calculate risk parity portfolio weights where each asset contributes 
        equally to total portfolio risk.
        
        Args:
            covariance_matrix: Covariance matrix of asset returns
            
        Returns:
            Array of optimal portfolio weights
        """
        n_assets = len(covariance_matrix.columns)
        
        # Define optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]  # weights sum to 1
        bounds = tuple((self.min_allocation, self.max_allocation) for _ in range(n_assets))
        
        # Initial guess (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Define objective function for risk parity
        def risk_contribution_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Individual asset contribution to risk
            marginal_risk_contribution = np.dot(covariance_matrix, weights) / portfolio_vol
            risk_contribution = weights * marginal_risk_contribution
            
            # Target is equal risk contribution
            target_risk = portfolio_vol / n_assets
            risk_diffs = risk_contribution - target_risk
            
            # Sum of squared differences as objective
            return np.sum(risk_diffs**2)
        
        # Solve the optimization problem
        result = minimize(risk_contribution_objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result['success']:
            return result['x']
        else:
            logger.warning(f"Risk parity optimization failed: {result['message']}")
            return initial_weights
    
    def _hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate portfolio weights using Hierarchical Risk Parity algorithm.
        This approach clusters assets based on their correlation structure.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Array of optimal portfolio weights
        """
        # Implementation based on Lopez de Prado's paper
        # "Building Diversified Portfolios that Outperform Out of Sample"
        
        # Get correlation matrix
        corr = returns.corr()
        
        # Convert to distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        
        # Hierarchical clustering
        from scipy.cluster import hierarchy
        link = hierarchy.linkage(dist, 'single')
        
        # Get quasi-diagonalization order
        sortIx = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link, dist))
        
        # Sort assets based on clustering
        sorted_assets = [returns.columns[i] for i in sortIx]
        
        # Calculate inverse-variance weights within clusters
        # This is a simplified implementation
        n_assets = len(returns.columns)
        cov = returns.cov()
        
        # Calculate cluster allocations based on inverse variance
        weights = np.zeros(n_assets)
        for i, asset_idx in enumerate(sortIx):
            weights[asset_idx] = 1 / np.sqrt(cov.iloc[asset_idx, asset_idx])
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def visualize_correlation_network(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a visualization of the correlation network between assets.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary with visualization data
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create network data structure
        nodes = []
        edges = []
        
        # Add nodes (assets)
        for i, asset in enumerate(returns.columns):
            nodes.append({
                'id': asset,
                'name': asset,
                'value': 1.0  # Size could be based on allocation or volatility
            })
        
        # Add edges (correlations)
        for i, asset1 in enumerate(returns.columns):
            for j, asset2 in enumerate(returns.columns):
                if i < j:  # Only add each pair once
                    correlation = corr_matrix.loc[asset1, asset2]
                    
                    # Only show significant correlations
                    if abs(correlation) > 0.3:
                        edges.append({
                            'source': asset1,
                            'target': asset2,
                            'value': abs(correlation),
                            'color': 'green' if correlation > 0 else 'red'
                        })
        
        # Return network visualization data
        return {
            'nodes': nodes,
            'edges': edges,
            'correlation_matrix': corr_matrix.to_dict()
        }
