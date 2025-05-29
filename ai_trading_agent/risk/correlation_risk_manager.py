"""
Correlation-Based Portfolio Risk Manager

This module implements advanced correlation analysis techniques for portfolio risk management,
including dynamic correlation detection, regime shifts, and risk parity position sizing.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from scipy import stats, optimize

# Set up logger
logger = logging.getLogger(__name__)


class CorrelationRiskManager:
    """
    Correlation-based risk manager that detects correlation regime shifts
    and implements risk parity position sizing for optimal diversification.
    """
    
    def __init__(
        self,
        correlation_lookback: int = 60,
        short_lookback: int = 20,
        risk_parity_weight: float = 0.5,
        max_allocation: float = 0.25,
        min_allocation: float = 0.05,
        correlation_threshold: float = 0.7
    ):
        """
        Initialize the correlation-based risk manager.
        
        Args:
            correlation_lookback: Number of days for correlation calculation
            short_lookback: Number of days for short-term correlation
            risk_parity_weight: Weight of risk parity vs target allocations
            max_allocation: Maximum allocation to any single asset
            min_allocation: Minimum allocation if included in portfolio
            correlation_threshold: Threshold for high correlation warning
        """
        self.correlation_lookback = correlation_lookback
        self.short_lookback = short_lookback
        self.risk_parity_weight = risk_parity_weight
        self.max_allocation = max_allocation
        self.min_allocation = min_allocation
        self.correlation_threshold = correlation_threshold
        
        # State tracking
        self.long_correlation = None
        self.short_correlation = None
        self.correlation_change = None
        self.high_correlation_pairs = []
        self.correlation_clusters = {}
        self.in_correlation_regime_shift = False
        self.regime_shift_time = None
        
        # History
        self.correlation_history = []
        
        logger.info("Correlation Risk Manager initialized")
    
    def update_correlations(self, returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Update correlation matrices and detect regime shifts.
        
        Args:
            returns_data: Dictionary mapping symbols to return series
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        if len(returns_df) < self.short_lookback:
            logger.warning(f"Not enough data for correlation analysis (need {self.short_lookback}, got {len(returns_df)})")
            return {"success": False}
        
        try:
            # Calculate long-term correlation
            long_corr = returns_df.iloc[-self.correlation_lookback:].corr()
            
            # Calculate short-term correlation
            short_corr = returns_df.iloc[-self.short_lookback:].corr()
            
            # Calculate correlation change
            corr_change = short_corr - long_corr
            
            # Store updated correlations
            self.long_correlation = long_corr
            self.short_correlation = short_corr
            self.correlation_change = corr_change
            
            # Detect high correlation pairs
            self._detect_high_correlation_pairs()
            
            # Detect correlation clusters
            self._detect_correlation_clusters()
            
            # Check for correlation regime shift
            shift_detected = self._detect_correlation_regime_shift()
            
            # Store history
            self.correlation_history.append({
                "timestamp": datetime.now(),
                "avg_correlation": self._average_correlation(short_corr),
                "high_correlation_pairs": len(self.high_correlation_pairs),
                "correlation_clusters": self.correlation_clusters.copy(),
                "regime_shift_detected": shift_detected
            })
            
            # Trim history
            if len(self.correlation_history) > 500:
                self.correlation_history = self.correlation_history[-500:]
            
            return {
                "success": True,
                "avg_correlation": self._average_correlation(short_corr),
                "high_correlation_pairs": self.high_correlation_pairs,
                "correlation_clusters": self.correlation_clusters,
                "in_regime_shift": self.in_correlation_regime_shift,
                "regime_shift_detected": shift_detected
            }
            
        except Exception as e:
            logger.error(f"Error updating correlations: {str(e)}")
            return {"success": False}
    
    def _detect_high_correlation_pairs(self) -> None:
        """Detect highly correlated pairs of assets."""
        self.high_correlation_pairs = []
        
        if self.short_correlation is None:
            return
            
        for i, symbol1 in enumerate(self.short_correlation.index):
            for j, symbol2 in enumerate(self.short_correlation.columns):
                if i < j:  # Only check upper triangle
                    corr = self.short_correlation.loc[symbol1, symbol2]
                    if abs(corr) > self.correlation_threshold:
                        self.high_correlation_pairs.append({
                            "asset1": symbol1,
                            "asset2": symbol2,
                            "correlation": corr,
                            "change": self.correlation_change.loc[symbol1, symbol2]
                        })
        
        # Sort by correlation strength
        self.high_correlation_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    def _detect_correlation_clusters(self) -> None:
        """Detect clusters of highly correlated assets."""
        if self.short_correlation is None:
            return
            
        # Reset clusters
        self.correlation_clusters = {}
        
        # Use a simple algorithm to identify clusters
        symbols = list(self.short_correlation.index)
        visited = set()
        
        for i, symbol in enumerate(symbols):
            if symbol in visited:
                continue
                
            # Start a new cluster
            cluster = [symbol]
            visited.add(symbol)
            
            # Find correlated assets for this cluster
            for other in symbols:
                if other != symbol and other not in visited:
                    if abs(self.short_correlation.loc[symbol, other]) > self.correlation_threshold:
                        cluster.append(other)
                        visited.add(other)
            
            # Only store clusters with at least 2 members
            if len(cluster) > 1:
                self.correlation_clusters[f"cluster_{i+1}"] = cluster
    
    def _detect_correlation_regime_shift(self) -> bool:
        """
        Detect if there's a significant shift in correlation structure.
        
        Returns:
            True if a regime shift is detected, False otherwise
        """
        if self.long_correlation is None or self.short_correlation is None:
            return False
            
        # Calculate average absolute differences between correlations
        avg_change = np.mean(np.abs(self.correlation_change.values))
        
        # Check for significant change in correlation structure
        significant_change = avg_change > 0.3  # Threshold for significant change
        
        # Record regime shift if detected
        if significant_change and not self.in_correlation_regime_shift:
            self.in_correlation_regime_shift = True
            self.regime_shift_time = datetime.now()
            logger.warning(f"Correlation regime shift detected! Average change: {avg_change:.2f}")
            return True
        
        # Check if we need to exit regime shift state (after 5 days)
        if self.in_correlation_regime_shift and self.regime_shift_time:
            days_since_shift = (datetime.now() - self.regime_shift_time).days
            if days_since_shift > 5:
                self.in_correlation_regime_shift = False
                logger.info("Correlation regime shift period ended")
        
        return False
    
    def _average_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate average absolute correlation from correlation matrix."""
        n = len(correlation_matrix)
        if n <= 1:
            return 0.0
            
        # Sum absolute correlations in upper triangle (excluding diagonal)
        sum_corr = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                sum_corr += abs(correlation_matrix.iloc[i, j])
                count += 1
                
        if count == 0:
            return 0.0
            
        return sum_corr / count
    
    def calculate_risk_parity_weights(
        self,
        returns_data: Dict[str, pd.Series],
        target_allocations: Dict[str, float],
        volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate risk parity portfolio weights blended with target allocations.
        
        Args:
            returns_data: Dictionary mapping symbols to return series
            target_allocations: Target allocations for each asset
            volatilities: Volatility estimates for each asset
            
        Returns:
            Dictionary with optimized portfolio weights
        """
        # Get symbols that are in both returns_data and target_allocations
        symbols = [s for s in target_allocations.keys() if s in returns_data]
        
        if not symbols:
            logger.error("No valid symbols for risk parity calculation")
            return target_allocations
        
        try:
            # Create returns DataFrame
            returns_df = pd.DataFrame({sym: returns_data[sym] for sym in symbols})
            
            # Calculate correlation matrix
            correlation = returns_df.corr()
            
            # Create volatility vector
            vols = np.array([volatilities.get(sym, 0.2) for sym in symbols])
            
            # Create covariance matrix
            cov_matrix = np.outer(vols, vols) * correlation.values
            
            # Calculate risk parity weights (inverse of variance)
            # Equal risk contribution weights
            def risk_contributions(weights, cov):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
                contrib = weights * np.dot(cov, weights) / portfolio_vol
                return contrib
            
            def risk_parity_objective(weights, cov):
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize
                risk_target = np.ones(len(weights)) / len(weights)  # Equal risk
                risk_contrib = risk_contributions(weights, cov)
                return np.sum((risk_contrib - risk_target)**2)
            
            # Initial guess: inverse volatility weights
            inv_vol = 1.0 / vols
            initial_weights = inv_vol / np.sum(inv_vol)
            
            # Constraints: sum of weights = 1 and weights >= 0
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = tuple((self.min_allocation, self.max_allocation) for _ in range(len(symbols)))
            
            # Optimize to get risk parity weights
            result = optimize.minimize(
                risk_parity_objective,
                initial_weights,
                args=(cov_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                # Get the optimal weights
                risk_parity_weights = result.x
                # Normalize weights
                risk_parity_weights = risk_parity_weights / np.sum(risk_parity_weights)
                
                # Get the target allocation weights in the same order
                target_weights = np.array([target_allocations.get(sym, 0.0) for sym in symbols])
                
                # Normalize target weights
                if np.sum(target_weights) > 0:
                    target_weights = target_weights / np.sum(target_weights)
                
                # Blend risk parity with target allocations
                blended_weights = (self.risk_parity_weight * risk_parity_weights + 
                                 (1 - self.risk_parity_weight) * target_weights)
                
                # Normalize again
                blended_weights = blended_weights / np.sum(blended_weights)
                
                # Convert back to dictionary
                weights_dict = {sym: float(w) for sym, w in zip(symbols, blended_weights)}
                
                logger.info(f"Risk parity weights calculated for {len(symbols)} assets")
                return weights_dict
            else:
                logger.warning("Risk parity optimization failed, using inverse volatility")
                # Fall back to inverse volatility
                inv_vol = 1.0 / vols
                inv_vol_weights = inv_vol / np.sum(inv_vol)
                weights_dict = {sym: float(w) for sym, w in zip(symbols, inv_vol_weights)}
                return weights_dict
                
        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {str(e)}")
            # Return target allocations if calculation fails
            return target_allocations
    
    def optimize_portfolio_correlation(
        self,
        returns_data: Dict[str, pd.Series],
        target_allocations: Dict[str, float],
        max_portfolio_correlation: float = 0.5
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights to minimize average correlation.
        
        Args:
            returns_data: Dictionary mapping symbols to return series
            target_allocations: Target allocations for each asset
            max_portfolio_correlation: Maximum average correlation 
            
        Returns:
            Dictionary with correlation-optimized portfolio weights
        """
        # Get symbols that are in both returns_data and target_allocations
        symbols = [s for s in target_allocations.keys() if s in returns_data]
        
        if not symbols:
            logger.error("No valid symbols for correlation optimization")
            return target_allocations
        
        try:
            # Create returns DataFrame
            returns_df = pd.DataFrame({sym: returns_data[sym] for sym in symbols})
            
            # Calculate correlation matrix
            correlation = returns_df.corr()
            
            # Objective function: minimize portfolio correlation
            def portfolio_correlation(weights, corr_matrix):
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize
                # Calculate weighted average correlation
                avg_corr = 0.0
                total_weight = 0.0
                
                for i in range(len(weights)):
                    for j in range(i+1, len(weights)):
                        weight_product = weights[i] * weights[j]
                        avg_corr += weight_product * abs(corr_matrix.iloc[i, j])
                        total_weight += weight_product
                
                if total_weight > 0:
                    avg_corr = avg_corr / total_weight
                
                # Penalize deviation from target weights
                target_weights = np.array([target_allocations.get(sym, 0.0) for sym in symbols])
                if np.sum(target_weights) > 0:
                    target_weights = target_weights / np.sum(target_weights)
                    
                deviation_penalty = np.sum((weights - target_weights)**2)
                
                # Combined objective
                return avg_corr + 0.5 * deviation_penalty
            
            # Initial guess: target allocations
            initial_weights = np.array([target_allocations.get(sym, 0.0) for sym in symbols])
            if np.sum(initial_weights) > 0:
                initial_weights = initial_weights / np.sum(initial_weights)
            else:
                initial_weights = np.ones(len(symbols)) / len(symbols)
            
            # Constraints: sum of weights = 1 and weights >= min_allocation
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = tuple((self.min_allocation, self.max_allocation) for _ in range(len(symbols)))
            
            # Optimize to minimize correlation
            result = optimize.minimize(
                portfolio_correlation,
                initial_weights,
                args=(correlation,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                # Get the optimal weights
                optimal_weights = result.x
                # Normalize weights
                optimal_weights = optimal_weights / np.sum(optimal_weights)
                
                # Convert back to dictionary
                weights_dict = {sym: float(w) for sym, w in zip(symbols, optimal_weights)}
                
                # Calculate portfolio correlation with these weights
                portfolio_corr = portfolio_correlation(optimal_weights, correlation)
                logger.info(f"Portfolio correlation optimized to {portfolio_corr:.3f}")
                
                # Check if optimization achieved target
                if portfolio_corr > max_portfolio_correlation:
                    logger.warning(f"Could not achieve target correlation ({max_portfolio_correlation:.2f})")
                
                return weights_dict
            else:
                logger.warning("Correlation optimization failed, using target allocations")
                return target_allocations
                
        except Exception as e:
            logger.error(f"Error optimizing portfolio correlation: {str(e)}")
            # Return target allocations if calculation fails
            return target_allocations
    
    def get_correlation_status(self) -> Dict[str, Any]:
        """
        Get current correlation status for monitoring.
        
        Returns:
            Dictionary with correlation metrics and status
        """
        if self.short_correlation is None:
            return {"status": "not_initialized"}
        
        avg_correlation = self._average_correlation(self.short_correlation)
        
        status = {
            "avg_correlation": avg_correlation,
            "high_correlation_pairs": len(self.high_correlation_pairs),
            "clusters": {k: len(v) for k, v in self.correlation_clusters.items()},
            "in_regime_shift": self.in_correlation_regime_shift,
            "correlation_level": "low" if avg_correlation < 0.4 else 
                              "medium" if avg_correlation < 0.6 else
                              "high"
        }
        
        # Add warning flag
        status["warning"] = status["correlation_level"] == "high" or self.in_correlation_regime_shift
        
        return status
