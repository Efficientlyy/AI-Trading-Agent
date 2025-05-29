"""
Adaptive Risk Manager

This module implements advanced risk management techniques that adapt to market conditions,
providing volatility-based position sizing, correlation-based portfolio management,
and stress detection with automatic risk mitigation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta

from ai_trading_agent.market_regime import (
    MarketRegimeType,
    VolatilityRegimeType
)

# Set up logger
logger = logging.getLogger(__name__)


class AdaptiveRiskManager:
    """
    Adaptive Risk Manager that automatically adjusts risk parameters based on
    market conditions, volatility regimes, correlation shifts, and drawdowns.
    """
    
    def __init__(
        self,
        base_portfolio_risk: float = 0.02,  # 2% daily VaR by default
        max_position_size: float = 0.20,    # 20% max for any position
        drawdown_scale_factor: float = 0.5, # Scale by 50% at max drawdown
        correlation_threshold: float = 0.7,  # Threshold for high correlation
        stress_detection_window: int = 20,   # Days to look for stress
        use_monte_carlo: bool = False        # Whether to use MC simulations
    ):
        """
        Initialize the adaptive risk manager.
        
        Args:
            base_portfolio_risk: Base daily portfolio risk target (VaR)
            max_position_size: Maximum allowed position size as % of portfolio
            drawdown_scale_factor: How much to scale during max drawdowns
            correlation_threshold: Threshold to consider correlations high
            stress_detection_window: Window for stress detection
            use_monte_carlo: Whether to use Monte Carlo for risk estimation
        """
        self.base_portfolio_risk = base_portfolio_risk
        self.max_position_size = max_position_size
        self.drawdown_scale_factor = drawdown_scale_factor
        self.correlation_threshold = correlation_threshold
        self.stress_detection_window = stress_detection_window
        self.use_monte_carlo = use_monte_carlo
        
        # Current state
        self.current_portfolio_risk = base_portfolio_risk
        self.current_max_position = max_position_size
        self.current_risk_multiplier = 1.0
        self.in_stress_mode = False
        
        # Risk tracking
        self.risk_history = []
        self.correlation_matrix = None
        self.volatility_estimates = {}
        self.position_limits = {}
        self.stress_indicators = {
            "volatility_spike": False,
            "correlation_spike": False,
            "liquidity_drop": False,
            "price_gap": False,
            "extreme_volume": False
        }
        
        logger.info("Adaptive Risk Manager initialized with base portfolio risk: "
                   f"{self.base_portfolio_risk:.1%}")
    
    def update_market_data(self, market_data: Dict[str, Dict[str, pd.Series]]) -> None:
        """
        Update internal risk metrics based on new market data.
        
        Args:
            market_data: Dictionary with market data for multiple assets
        """
        self._update_volatility_estimates(market_data)
        self._update_correlation_matrix(market_data)
        self._detect_market_stress(market_data)
        
        # Update risk history
        self.risk_history.append({
            "timestamp": datetime.now(),
            "portfolio_risk": self.current_portfolio_risk,
            "max_position": self.current_max_position,
            "risk_multiplier": self.current_risk_multiplier,
            "in_stress_mode": self.in_stress_mode,
            "stress_indicators": self.stress_indicators.copy()
        })
        
        # Keep history limited to avoid memory issues
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
    
    def _update_volatility_estimates(self, market_data: Dict[str, Dict[str, pd.Series]]) -> None:
        """
        Update volatility estimates for all assets in the market data.
        
        Args:
            market_data: Dictionary with market data for multiple assets
        """
        for symbol, data in market_data.items():
            if "prices" in data and len(data["prices"]) > 20:
                # Calculate returns
                returns = data["prices"].pct_change().dropna()
                
                if len(returns) > 20:
                    # Standard volatility (20-day)
                    std_vol = returns[-20:].std() * np.sqrt(252)  # Annualized
                    
                    # EWMA volatility (more weight to recent observations)
                    ewma_vol = returns[-60:].ewm(span=20).std().iloc[-1] * np.sqrt(252)
                    
                    # Parkinson volatility estimator using high-low range
                    parkinson_vol = None
                    if "high" in data and "low" in data:
                        high_low_ratio = np.log(data["high"] / data["low"])
                        parkinson_vol = np.sqrt(
                            (1 / (4 * np.log(2))) * 
                            high_low_ratio[-20:].pow(2).mean() * 
                            252
                        )
                    
                    # Store volatility estimates
                    self.volatility_estimates[symbol] = {
                        "standard": std_vol,
                        "ewma": ewma_vol,
                        "parkinson": parkinson_vol,
                        "used": ewma_vol  # Default to EWMA volatility
                    }
    
    def _update_correlation_matrix(self, market_data: Dict[str, Dict[str, pd.Series]]) -> None:
        """
        Update correlation matrix for assets in the market data.
        
        Args:
            market_data: Dictionary with market data for multiple assets
        """
        price_data = {}
        
        # Extract price data
        for symbol, data in market_data.items():
            if "prices" in data and len(data["prices"]) > 60:
                price_data[symbol] = data["prices"]
        
        if len(price_data) > 1:
            # Create returns DataFrame
            returns_df = pd.DataFrame({
                symbol: prices.pct_change().dropna() 
                for symbol, prices in price_data.items()
            }).dropna()
            
            if len(returns_df) > 20:
                # Calculate correlation matrix (60-day correlation)
                self.correlation_matrix = returns_df[-60:].corr()
    
    def _detect_market_stress(self, market_data: Dict[str, Dict[str, pd.Series]]) -> None:
        """
        Detect market stress conditions using multiple indicators.
        
        Args:
            market_data: Dictionary with market data for multiple assets
        """
        # Reset stress indicators
        for indicator in self.stress_indicators:
            self.stress_indicators[indicator] = False
        
        # Look for volatility spikes in major indices
        for symbol in ["SPY", "QQQ", "IWM"]:
            if symbol in self.volatility_estimates:
                vol_now = self.volatility_estimates[symbol]["used"]
                vol_history = [h.get("volatility_estimates", {}).get(symbol, {}).get("used", 0) 
                             for h in self.risk_history[-30:] if "volatility_estimates" in h]
                
                if vol_history and vol_now > 1.5 * np.mean(vol_history):
                    self.stress_indicators["volatility_spike"] = True
                    logger.warning(f"Volatility spike detected in {symbol}: {vol_now:.1%}")
                    break
        
        # Check for correlation spikes (risk-on/risk-off behavior)
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 5:
            # Get average of upper triangle of correlation matrix
            corr_values = []
            for i in range(len(self.correlation_matrix.columns)):
                for j in range(i+1, len(self.correlation_matrix.columns)):
                    corr_values.append(self.correlation_matrix.iloc[i, j])
            
            if corr_values:
                avg_correlation = np.mean(corr_values)
                
                # Check if correlation is abnormally high
                if avg_correlation > self.correlation_threshold:
                    self.stress_indicators["correlation_spike"] = True
                    logger.warning(f"High correlation detected: {avg_correlation:.2f}")
        
        # Look for price gaps in major indices
        for symbol in ["SPY", "QQQ", "IWM"]:
            if symbol in market_data and "prices" in market_data[symbol]:
                prices = market_data[symbol]["prices"]
                if len(prices) > 2:
                    latest_return = (prices.iloc[-1] / prices.iloc[-2]) - 1
                    
                    # Check for large overnight gaps (> 2%)
                    if abs(latest_return) > 0.02:
                        self.stress_indicators["price_gap"] = True
                        logger.warning(f"Price gap detected in {symbol}: {latest_return:.1%}")
                        break
        
        # Update stress mode based on indicators
        indicator_count = sum(1 for v in self.stress_indicators.values() if v)
        self.in_stress_mode = indicator_count >= 2
        
        if self.in_stress_mode:
            logger.warning("Market stress detected - activating defensive risk settings")
            self._activate_stress_response()
    
    def _activate_stress_response(self) -> None:
        """Activate stress response - reduce risk and position sizing."""
        # Reduce overall portfolio risk
        self.current_risk_multiplier = 0.5
        self.current_portfolio_risk = self.base_portfolio_risk * self.current_risk_multiplier
        
        # Reduce maximum position size
        self.current_max_position = self.max_position_size * self.current_risk_multiplier
        
        logger.info(f"Stress response activated - portfolio risk reduced to {self.current_portfolio_risk:.1%}")
    
    def adjust_for_market_regime(
        self,
        market_regime: str,
        volatility_regime: Optional[str] = None,
        drawdown: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Adjust risk parameters based on market regime, volatility, and drawdown.
        
        Args:
            market_regime: Current market regime
            volatility_regime: Current volatility regime (optional)
            drawdown: Current drawdown as a decimal (e.g., 0.05 for 5%)
            
        Returns:
            Dictionary with updated risk parameters
        """
        # Base multiplier from market regime
        regime_multipliers = {
            MarketRegimeType.BULL.value: 1.0,
            MarketRegimeType.BEAR.value: 0.7,
            MarketRegimeType.VOLATILE.value: 0.6,
            MarketRegimeType.SIDEWAYS.value: 0.8,
            MarketRegimeType.TRENDING.value: 0.9,
            MarketRegimeType.CHOPPY.value: 0.7,
            MarketRegimeType.BREAKDOWN.value: 0.5,
            MarketRegimeType.RECOVERY.value: 0.8,
            MarketRegimeType.UNKNOWN.value: 0.7
        }
        
        # Get base multiplier for the current regime
        base_multiplier = regime_multipliers.get(market_regime, 0.7)
        
        # Adjust for volatility regime if provided
        vol_adjustment = 1.0
        if volatility_regime:
            vol_adjustments = {
                VolatilityRegimeType.LOW.value: 1.2,
                VolatilityRegimeType.MEDIUM.value: 1.0,
                VolatilityRegimeType.HIGH.value: 0.8,
                VolatilityRegimeType.EXTREME.value: 0.6
            }
            vol_adjustment = vol_adjustments.get(volatility_regime, 1.0)
        
        # Adjust for drawdown if provided
        dd_adjustment = 1.0
        if drawdown is not None:
            # Linear reduction as drawdown increases
            # At max_drawdown (e.g., 20%), multiply by drawdown_scale_factor
            max_drawdown = 0.20  # 20%
            if drawdown <= 0:
                dd_adjustment = 1.0
            elif drawdown >= max_drawdown:
                dd_adjustment = self.drawdown_scale_factor
            else:
                dd_adjustment = 1.0 - ((1.0 - self.drawdown_scale_factor) * drawdown / max_drawdown)
        
        # Combine adjustments
        self.current_risk_multiplier = base_multiplier * vol_adjustment * dd_adjustment
        
        # Apply stress mode override if active
        if self.in_stress_mode:
            self.current_risk_multiplier = min(self.current_risk_multiplier, 0.5)
        
        # Update current portfolio risk
        self.current_portfolio_risk = self.base_portfolio_risk * self.current_risk_multiplier
        
        # Update maximum position size
        self.current_max_position = self.max_position_size * self.current_risk_multiplier
        
        logger.info(f"Risk adjusted for {market_regime} regime - multiplier: {self.current_risk_multiplier:.2f}")
        
        return {
            "portfolio_risk": self.current_portfolio_risk,
            "max_position_size": self.current_max_position,
            "risk_multiplier": self.current_risk_multiplier
        }
    
    def calculate_position_sizes(
        self,
        portfolio_value: float,
        target_allocations: Dict[str, float],
        correlation_aware: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate optimal position sizes taking into account volatility and correlations.
        
        Args:
            portfolio_value: Current portfolio value
            target_allocations: Target allocation weights (should sum to 1.0)
            correlation_aware: Whether to adjust for correlations
            
        Returns:
            Dictionary with position sizes and limits for each asset
        """
        # Validate inputs
        if sum(target_allocations.values()) > 1.0:
            logger.warning("Target allocations sum to > 1.0, normalizing")
            total = sum(target_allocations.values())
            target_allocations = {k: v/total for k, v in target_allocations.items()}
        
        # Position sizing results
        position_sizing = {}
        
        # Simple volatility-adjusted position sizing
        if not correlation_aware or self.correlation_matrix is None:
            for symbol, target_weight in target_allocations.items():
                # Get volatility, default to 20% annualized if unknown
                vol = self.volatility_estimates.get(symbol, {}).get("used", 0.20)
                
                # Scale position size inversely with volatility
                # Higher volatility = smaller position size
                vol_adjustment = min(2.0, max(0.5, 0.20 / vol)) if vol > 0 else 1.0
                
                # Calculate position size with volatility adjustment but maintaining
                # the overall target allocation
                adjusted_weight = target_weight * vol_adjustment
                
                # Apply maximum position constraint
                final_weight = min(adjusted_weight, self.current_max_position)
                
                # Calculate dollar amount
                dollar_amount = portfolio_value * final_weight
                
                position_sizing[symbol] = {
                    "target_weight": target_weight,
                    "adjusted_weight": final_weight,
                    "dollar_amount": dollar_amount,
                    "volatility": vol
                }
        
        # Correlation-aware position sizing
        else:
            symbols = list(target_allocations.keys())
            weights = np.array([target_allocations[s] for s in symbols])
            
            # Create volatility vector
            vols = np.array([
                self.volatility_estimates.get(s, {}).get("used", 0.20)
                for s in symbols
            ])
            
            # Extract correlation submatrix for these symbols
            corr_matrix = self._extract_correlation_submatrix(symbols)
            
            # Create covariance matrix
            cov_matrix = np.outer(vols, vols) * corr_matrix
            
            # Risk parity weights (inverse volatility)
            inv_vol_weights = 1.0 / vols
            inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
            
            # Blend target weights with risk parity weights
            blend_ratio = 0.5  # 50% target, 50% risk parity
            blended_weights = blend_ratio * weights + (1 - blend_ratio) * inv_vol_weights
            
            # Normalize weights
            final_weights = blended_weights / np.sum(blended_weights)
            
            # Calculate portfolio volatility with these weights
            portfolio_vol = np.sqrt(final_weights.T @ cov_matrix @ final_weights)
            
            # If portfolio risk exceeds target, scale down weights
            if portfolio_vol > self.current_portfolio_risk:
                scaling_factor = self.current_portfolio_risk / portfolio_vol
                final_weights = final_weights * scaling_factor
            
            # Apply maximum position constraint
            for i, symbol in enumerate(symbols):
                if final_weights[i] > self.current_max_position:
                    final_weights[i] = self.current_max_position
            
            # Normalize again if needed
            if np.sum(final_weights) > 1.0:
                final_weights = final_weights / np.sum(final_weights)
            
            # Calculate dollar amounts and store results
            for i, symbol in enumerate(symbols):
                dollar_amount = portfolio_value * final_weights[i]
                
                position_sizing[symbol] = {
                    "target_weight": target_allocations[symbol],
                    "adjusted_weight": float(final_weights[i]),
                    "dollar_amount": float(dollar_amount),
                    "volatility": float(vols[i])
                }
            
            # Store additional portfolio metrics
            position_sizing["_portfolio"] = {
                "volatility": float(portfolio_vol),
                "target_risk": float(self.current_portfolio_risk),
                "correlation_adjusted": True
            }
        
        return position_sizing
    
    def _extract_correlation_submatrix(self, symbols: List[str]) -> np.ndarray:
        """
        Extract correlation submatrix for the given symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Correlation submatrix as numpy array
        """
        n = len(symbols)
        submatrix = np.ones((n, n))
        
        if self.correlation_matrix is not None:
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i == j:
                        submatrix[i, j] = 1.0
                    elif sym1 in self.correlation_matrix.index and sym2 in self.correlation_matrix.columns:
                        submatrix[i, j] = self.correlation_matrix.loc[sym1, sym2]
                    else:
                        # Default correlation if not available
                        submatrix[i, j] = 0.5
        
        return submatrix
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics for monitoring.
        
        Returns:
            Dictionary with current risk metrics
        """
        return {
            "portfolio_risk": self.current_portfolio_risk,
            "max_position_size": self.current_max_position,
            "risk_multiplier": self.current_risk_multiplier,
            "in_stress_mode": self.in_stress_mode,
            "stress_indicators": self.stress_indicators.copy(),
            "volatility_estimates": {
                k: v["used"] for k, v in self.volatility_estimates.items()
            }
        }
