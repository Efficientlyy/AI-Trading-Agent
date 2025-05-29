"""
Risk-Adjusted Portfolio Allocator for AI Trading Agent.

This module provides advanced portfolio allocation methods that optimize
allocations based on various risk metrics and target risk levels.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize

from ..common import logger

class RiskAdjustedAllocator:
    """
    Risk-Adjusted Portfolio Allocator.
    
    This class provides methods to adjust portfolio allocations based on
    various risk metrics, volatility regimes, and drawdown protection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the risk-adjusted allocator.
        
        Args:
            config: Configuration dictionary with parameters:
                - target_risk: Target portfolio volatility (annualized)
                - max_drawdown_limit: Maximum acceptable drawdown
                - vol_lookback: Lookback period for volatility calculation
                - vol_scaling: Whether to use volatility scaling
                - risk_measure: Risk measure to use ('volatility', 'var', 'cvar', 'drawdown')
                - dynamic_allocation: Whether to use dynamic allocation based on market regimes
        """
        self.name = "RiskAdjustedAllocator"
        self.target_risk = config.get('target_risk', 0.15)  # 15% target volatility
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.20)  # 20% max drawdown
        self.vol_lookback = config.get('vol_lookback', 63)  # ~3 months of trading days
        self.vol_scaling = config.get('vol_scaling', True)
        self.risk_measure = config.get('risk_measure', 'volatility')
        self.dynamic_allocation = config.get('dynamic_allocation', True)
        
        # Confidence level for VaR and CVaR
        self.confidence_level = config.get('confidence_level', 0.95)
        
        # Risk budget allocation
        self.risk_budget = config.get('risk_budget', {})
        
        logger.info(f"Initialized {self.name} with target_risk={self.target_risk*100:.1f}%, "
                   f"risk_measure={self.risk_measure}")
    
    def calculate_allocations(self, returns: pd.DataFrame, 
                             market_regime: Optional[str] = None,
                             risk_factors: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate risk-adjusted allocations based on historical returns and market regime.
        
        Args:
            returns: DataFrame of asset returns with assets as columns
            market_regime: Optional market regime identifier ('bullish', 'bearish', 'volatile', etc.)
            risk_factors: Optional asset-specific risk factors
            
        Returns:
            Dictionary mapping asset symbols to target allocation percentages
        """
        logger.info(f"Calculating risk-adjusted allocations using {self.risk_measure}")
        
        # Get assets
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(returns)
        
        # Adjust for market regime if available
        if self.dynamic_allocation and market_regime:
            allocations = self._regime_based_allocation(returns, market_regime, risk_metrics)
        else:
            # Calculate based on selected risk measure
            if self.risk_measure == 'volatility':
                allocations = self._volatility_adjusted_allocation(returns, risk_metrics)
            elif self.risk_measure == 'var':
                allocations = self._var_adjusted_allocation(returns, risk_metrics)
            elif self.risk_measure == 'cvar':
                allocations = self._cvar_adjusted_allocation(returns, risk_metrics)
            elif self.risk_measure == 'drawdown':
                allocations = self._drawdown_adjusted_allocation(returns, risk_metrics)
            else:
                # Default to equal weight
                allocations = {asset: 1.0 / n_assets for asset in assets}
        
        # Apply risk factors if available
        if risk_factors:
            allocations = self._apply_risk_factors(allocations, risk_factors)
        
        # Normalize allocations to sum to 1.0
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            allocations = {k: v / total_allocation for k, v in allocations.items()}
        
        return allocations
    
    def _calculate_risk_metrics(self, returns: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate various risk metrics for each asset.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary of risk metrics for each asset
        """
        risk_metrics = {}
        
        # Calculate metrics for each asset
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            
            # Skip assets with insufficient data
            if len(asset_returns) < self.vol_lookback:
                continue
            
            # Calculate volatility (annualized)
            volatility = asset_returns.std() * np.sqrt(252)
            
            # Calculate VaR
            var = asset_returns.quantile(1 - self.confidence_level)
            
            # Calculate CVaR (Expected Shortfall)
            cvar = asset_returns[asset_returns <= var].mean()
            
            # Calculate maximum drawdown
            cum_returns = (1 + asset_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns / rolling_max) - 1
            max_drawdown = abs(drawdowns.min())
            
            # Calculate additional metrics
            sharpe = (asset_returns.mean() * 252) / volatility if volatility > 0 else 0
            
            # Store metrics
            risk_metrics[asset] = {
                'volatility': volatility,
                'var': abs(var),
                'cvar': abs(cvar) if not pd.isna(cvar) else 0,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe
            }
        
        return risk_metrics
    
    def _volatility_adjusted_allocation(self, returns: pd.DataFrame, 
                                       risk_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate allocations inversely proportional to volatility.
        
        Args:
            returns: DataFrame of asset returns
            risk_metrics: Pre-calculated risk metrics
            
        Returns:
            Dictionary of allocations
        """
        allocations = {}
        
        # Inverse volatility weighting
        for asset, metrics in risk_metrics.items():
            vol = metrics['volatility']
            
            # Avoid division by zero
            if vol > 0:
                allocations[asset] = 1.0 / vol
            else:
                allocations[asset] = 0.0
        
        # Handle empty allocations
        if not allocations or sum(allocations.values()) == 0:
            equal_weight = 1.0 / len(returns.columns)
            allocations = {asset: equal_weight for asset in returns.columns}
        
        return allocations
    
    def _var_adjusted_allocation(self, returns: pd.DataFrame, 
                               risk_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate allocations inversely proportional to Value-at-Risk (VaR).
        
        Args:
            returns: DataFrame of asset returns
            risk_metrics: Pre-calculated risk metrics
            
        Returns:
            Dictionary of allocations
        """
        allocations = {}
        
        # Inverse VaR weighting
        for asset, metrics in risk_metrics.items():
            var = metrics['var']
            
            # Avoid division by zero
            if var > 0:
                allocations[asset] = 1.0 / var
            else:
                allocations[asset] = 0.0
        
        # Handle empty allocations
        if not allocations or sum(allocations.values()) == 0:
            equal_weight = 1.0 / len(returns.columns)
            allocations = {asset: equal_weight for asset in returns.columns}
        
        return allocations
    
    def _cvar_adjusted_allocation(self, returns: pd.DataFrame, 
                                risk_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate allocations inversely proportional to Conditional Value-at-Risk (CVaR).
        
        Args:
            returns: DataFrame of asset returns
            risk_metrics: Pre-calculated risk metrics
            
        Returns:
            Dictionary of allocations
        """
        allocations = {}
        
        # Inverse CVaR weighting
        for asset, metrics in risk_metrics.items():
            cvar = metrics['cvar']
            
            # Avoid division by zero
            if cvar > 0:
                allocations[asset] = 1.0 / cvar
            else:
                allocations[asset] = 0.0
        
        # Handle empty allocations
        if not allocations or sum(allocations.values()) == 0:
            equal_weight = 1.0 / len(returns.columns)
            allocations = {asset: equal_weight for asset in returns.columns}
        
        return allocations
    
    def _drawdown_adjusted_allocation(self, returns: pd.DataFrame, 
                                    risk_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate allocations inversely proportional to maximum drawdown.
        
        Args:
            returns: DataFrame of asset returns
            risk_metrics: Pre-calculated risk metrics
            
        Returns:
            Dictionary of allocations
        """
        allocations = {}
        
        # Inverse drawdown weighting
        for asset, metrics in risk_metrics.items():
            drawdown = metrics['max_drawdown']
            
            # Avoid division by zero and very small drawdowns
            if drawdown > 0.01:
                allocations[asset] = 1.0 / drawdown
            else:
                # For assets with very small drawdowns, use a default value
                allocations[asset] = 100.0  # High weight for low-drawdown assets
        
        # Handle empty allocations
        if not allocations or sum(allocations.values()) == 0:
            equal_weight = 1.0 / len(returns.columns)
            allocations = {asset: equal_weight for asset in returns.columns}
        
        return allocations
    
    def _regime_based_allocation(self, returns: pd.DataFrame, 
                               market_regime: str,
                               risk_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate regime-specific allocations based on market conditions.
        
        Args:
            returns: DataFrame of asset returns
            market_regime: Market regime identifier
            risk_metrics: Pre-calculated risk metrics
            
        Returns:
            Dictionary of allocations
        """
        # Define regime-specific allocation strategies
        if market_regime == 'bullish':
            # In bullish regimes, allocate based on Sharpe ratio
            allocations = {}
            for asset, metrics in risk_metrics.items():
                sharpe = metrics['sharpe_ratio']
                allocations[asset] = max(0, sharpe)  # Only positive Sharpe ratios
            
        elif market_regime == 'bearish':
            # In bearish regimes, focus on minimizing drawdown
            allocations = self._drawdown_adjusted_allocation(returns, risk_metrics)
            
        elif market_regime == 'volatile':
            # In volatile regimes, use minimum volatility approach
            allocations = self._volatility_adjusted_allocation(returns, risk_metrics)
            
        elif market_regime == 'stable':
            # In stable regimes, balanced approach using CVaR
            allocations = self._cvar_adjusted_allocation(returns, risk_metrics)
            
        else:
            # Default to volatility-adjusted allocation
            allocations = self._volatility_adjusted_allocation(returns, risk_metrics)
        
        # Handle empty allocations
        if not allocations or sum(allocations.values()) == 0:
            equal_weight = 1.0 / len(returns.columns)
            allocations = {asset: equal_weight for asset in returns.columns}
        
        return allocations
    
    def _apply_risk_factors(self, allocations: Dict[str, float], 
                          risk_factors: Dict[str, float]) -> Dict[str, float]:
        """
        Apply asset-specific risk factors to adjust allocations.
        
        Args:
            allocations: Current asset allocations
            risk_factors: Asset-specific risk factors (higher values = higher risk)
            
        Returns:
            Adjusted allocations
        """
        adjusted_allocations = {}
        
        # Apply risk factors as multipliers
        for asset, allocation in allocations.items():
            # Get risk factor (default to 1.0 if not provided)
            risk_factor = risk_factors.get(asset, 1.0)
            
            # Adjust allocation inversely proportional to risk factor
            if risk_factor > 0:
                adjusted_allocations[asset] = allocation / risk_factor
            else:
                adjusted_allocations[asset] = allocation
        
        return adjusted_allocations
    
    def calculate_portfolio_risk(self, allocations: Dict[str, float], 
                               covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics based on allocations.
        
        Args:
            allocations: Dictionary of asset allocations
            covariance_matrix: Covariance matrix of asset returns
            
        Returns:
            Dictionary of portfolio risk metrics
        """
        # Convert allocations to array
        assets = covariance_matrix.columns
        weights = np.array([allocations.get(asset, 0.0) for asset in assets])
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate marginal risk contribution
        marginal_risk_contribution = np.dot(covariance_matrix, weights) / portfolio_volatility
        risk_contribution = weights * marginal_risk_contribution
        
        # Calculate diversification ratio
        asset_volatilities = np.sqrt(np.diag(covariance_matrix))
        weighted_volatilities = weights * asset_volatilities
        weighted_vol_sum = np.sum(weighted_volatilities)
        diversification_ratio = weighted_vol_sum / portfolio_volatility
        
        # Return risk metrics
        return {
            'portfolio_volatility': portfolio_volatility,
            'diversification_ratio': diversification_ratio,
            'risk_contribution': {asset: rc for asset, rc in zip(assets, risk_contribution)},
            'risk_contribution_pct': {asset: rc / portfolio_volatility * 100 
                                   for asset, rc in zip(assets, risk_contribution)}
        }
    
    def scale_portfolio_to_risk_target(self, allocations: Dict[str, float], 
                                     returns: pd.DataFrame) -> Dict[str, float]:
        """
        Scale portfolio to match target risk level.
        
        Args:
            allocations: Dictionary of asset allocations
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary of scaled allocations and cash percentage
        """
        # Calculate covariance matrix
        covariance_matrix = returns.cov() * 252  # Annualized
        
        # Calculate current portfolio risk
        risk_metrics = self.calculate_portfolio_risk(allocations, covariance_matrix)
        current_risk = risk_metrics['portfolio_volatility']
        
        # Calculate scaling factor
        if current_risk > 0 and self.target_risk > 0:
            scaling_factor = self.target_risk / current_risk
        else:
            scaling_factor = 1.0
        
        # Limit scaling factor to avoid excessive leverage
        scaling_factor = min(scaling_factor, 2.0)
        
        # Scale allocations
        scaled_allocations = {}
        for asset, allocation in allocations.items():
            scaled_allocations[asset] = allocation * scaling_factor
        
        # Calculate cash allocation
        cash_allocation = max(0, 1.0 - sum(scaled_allocations.values()))
        scaled_allocations['CASH'] = cash_allocation
        
        logger.info(f"Scaled portfolio from {current_risk*100:.2f}% risk to {self.target_risk*100:.2f}% "
                  f"target risk, cash allocation: {cash_allocation*100:.2f}%")
        
        return scaled_allocations
