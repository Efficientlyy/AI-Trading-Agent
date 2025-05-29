"""
Risk Orchestrator

This module integrates all components of the Advanced Risk Management Adaptivity system,
including volatility-based risk adjustments, correlation-based portfolio management,
and adaptive position sizing based on market conditions.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ai_trading_agent.risk.adaptive_risk_manager import AdaptiveRiskManager
from ai_trading_agent.risk.volatility_clustering import VolatilityClusteringModel
from ai_trading_agent.risk.correlation_risk_manager import CorrelationRiskManager
from ai_trading_agent.market_regime import MarketRegimeType, VolatilityRegimeType

# Set up logger
logger = logging.getLogger(__name__)


class RiskOrchestrator:
    """
    Orchestrates all risk management components and provides a unified interface
    for the trading system to access risk management capabilities.
    """
    
    def __init__(
        self,
        base_portfolio_risk: float = 0.02,
        max_position_size: float = 0.20,
        correlation_threshold: float = 0.7,
        enable_volatility_clustering: bool = True,
        enable_correlation_optimization: bool = True,
        enable_risk_parity: bool = True,
        enable_stress_detection: bool = True
    ):
        """
        Initialize the risk orchestrator.
        
        Args:
            base_portfolio_risk: Base daily portfolio VaR target
            max_position_size: Maximum allowed position size for any asset
            correlation_threshold: Threshold for high correlation warning
            enable_volatility_clustering: Whether to use GARCH models
            enable_correlation_optimization: Whether to optimize for correlation
            enable_risk_parity: Whether to use risk parity allocation
            enable_stress_detection: Whether to enable market stress detection
        """
        # Initialize component managers
        self.adaptive_risk_manager = AdaptiveRiskManager(
            base_portfolio_risk=base_portfolio_risk,
            max_position_size=max_position_size,
            correlation_threshold=correlation_threshold
        )
        
        self.volatility_model = VolatilityClusteringModel(
            lookback_window=500,
            forecast_horizon=10,
            use_cache=True
        ) if enable_volatility_clustering else None
        
        self.correlation_manager = CorrelationRiskManager(
            correlation_lookback=60,
            short_lookback=20,
            correlation_threshold=correlation_threshold
        ) if enable_correlation_optimization else None
        
        # Feature flags
        self.enable_volatility_clustering = enable_volatility_clustering
        self.enable_correlation_optimization = enable_correlation_optimization
        self.enable_risk_parity = enable_risk_parity
        self.enable_stress_detection = enable_stress_detection
        
        # State tracking
        self.last_update_time = None
        self.market_data = {}
        self.returns_data = {}
        self.risk_metrics = {}
        self.volatility_regimes = {}
        self.correlation_status = {}
        self.position_limits = {}
        self.alert_history = []
        
        logger.info("Risk Orchestrator initialized with Advanced Risk Management Adaptivity")
    
    def update_market_data(self, market_data: Dict[str, Dict[str, pd.Series]]) -> None:
        """
        Update all risk components with new market data.
        
        Args:
            market_data: Dictionary with market data for multiple assets
        """
        self.market_data = market_data
        self.last_update_time = datetime.now()
        
        # Extract returns data
        self.returns_data = self._extract_returns(market_data)
        
        # Update adaptive risk manager
        self.adaptive_risk_manager.update_market_data(market_data)
        
        # Update volatility clustering model
        if self.enable_volatility_clustering and self.volatility_model:
            self.volatility_regimes = self.volatility_model.analyze_multiple_assets(
                self.returns_data, use_garch=True
            )
        
        # Update correlation manager
        if self.enable_correlation_optimization and self.correlation_manager:
            self.correlation_status = self.correlation_manager.update_correlations(
                self.returns_data
            )
        
        logger.debug("Risk components updated with new market data")
    
    def _extract_returns(self, market_data: Dict[str, Dict[str, pd.Series]]) -> Dict[str, pd.Series]:
        """
        Extract returns series from market data.
        
        Args:
            market_data: Dictionary with market data
            
        Returns:
            Dictionary mapping symbols to return series
        """
        returns_data = {}
        
        for symbol, data in market_data.items():
            if "prices" in data and len(data["prices"]) > 2:
                returns_data[symbol] = data["prices"].pct_change().dropna()
            elif "returns" in data:
                returns_data[symbol] = data["returns"]
        
        return returns_data
    
    def adapt_to_market_regime(
        self,
        market_regime: str,
        volatility_regime: Optional[str] = None,
        drawdown: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Adapt all risk parameters to current market regime.
        
        Args:
            market_regime: Current market regime
            volatility_regime: Current volatility regime (optional)
            drawdown: Current drawdown as decimal (optional)
            
        Returns:
            Dictionary with updated risk parameters
        """
        # Adjust risk parameters using adaptive risk manager
        risk_params = self.adaptive_risk_manager.adjust_for_market_regime(
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            drawdown=drawdown
        )
        
        # Store the current risk metrics
        self.risk_metrics.update({
            "portfolio_risk": risk_params["portfolio_risk"],
            "max_position_size": risk_params["max_position_size"],
            "risk_multiplier": risk_params["risk_multiplier"],
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "in_stress_mode": self.adaptive_risk_manager.in_stress_mode,
            "last_updated": datetime.now()
        })
        
        # Generate alerts for significant changes
        if self.adaptive_risk_manager.in_stress_mode:
            self._add_alert(
                level="warning",
                message="Market stress detected - defensive risk posture activated",
                details={
                    "stress_indicators": self.adaptive_risk_manager.stress_indicators
                }
            )
        
        logger.info(f"Risk adapted to {market_regime} regime - "
                  f"portfolio risk: {risk_params['portfolio_risk']:.1%}, "
                  f"max position: {risk_params['max_position_size']:.1%}")
        
        return self.risk_metrics
    
    def calculate_position_sizes(
        self,
        portfolio_value: float,
        target_allocations: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate position sizes with adaptive risk management.
        
        Args:
            portfolio_value: Current portfolio value
            target_allocations: Target allocation weights
            
        Returns:
            Dictionary with optimized position sizes for each asset
        """
        # Extract volatility estimates for all assets
        volatilities = {}
        for symbol in target_allocations.keys():
            if self.enable_volatility_clustering and symbol in self.volatility_regimes:
                volatilities[symbol] = self.volatility_regimes[symbol]["current_volatility"]
            elif symbol in self.adaptive_risk_manager.volatility_estimates:
                volatilities[symbol] = self.adaptive_risk_manager.volatility_estimates[symbol]["used"]
            else:
                # Default volatility if not available
                volatilities[symbol] = 0.20  # 20% annualized
        
        # Apply correlation optimization if enabled
        optimized_allocations = target_allocations.copy()
        
        if self.enable_correlation_optimization and self.correlation_manager and self.returns_data:
            # Optimize to reduce correlation
            optimized_allocations = self.correlation_manager.optimize_portfolio_correlation(
                self.returns_data,
                target_allocations,
                max_portfolio_correlation=0.6
            )
            
            # If risk parity is also enabled, blend with risk parity weights
            if self.enable_risk_parity:
                risk_parity_weights = self.correlation_manager.calculate_risk_parity_weights(
                    self.returns_data,
                    optimized_allocations,
                    volatilities
                )
                
                # Use risk parity weights as the optimized allocations
                optimized_allocations = risk_parity_weights
        
        # Calculate final position sizes using adaptive risk manager
        position_sizes = self.adaptive_risk_manager.calculate_position_sizes(
            portfolio_value=portfolio_value,
            target_allocations=optimized_allocations,
            correlation_aware=self.enable_correlation_optimization
        )
        
        # Store position limits for reference
        self.position_limits = {
            symbol: {
                "min_weight": 0.0,
                "max_weight": self.adaptive_risk_manager.current_max_position,
                "current_weight": info["adjusted_weight"]
            }
            for symbol, info in position_sizes.items()
            if symbol != "_portfolio"
        }
        
        return position_sizes
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive risk metrics for monitoring.
        
        Returns:
            Dictionary with current risk metrics
        """
        metrics = self.risk_metrics.copy()
        
        # Add volatility information
        if self.enable_volatility_clustering and self.volatility_regimes:
            metrics["volatility"] = {
                symbol: {
                    "current": data["current_volatility"],
                    "regime": data["regime"],
                    "increasing": data.get("volatility_increasing", False)
                }
                for symbol, data in self.volatility_regimes.items()
                if symbol != "_market"
            }
            
            # Add market-wide volatility if available
            if "_market" in self.volatility_regimes:
                metrics["volatility"]["_market"] = self.volatility_regimes["_market"]
        
        # Add correlation information
        if self.enable_correlation_optimization and self.correlation_status and self.correlation_status.get("success", False):
            metrics["correlation"] = {
                "avg_correlation": self.correlation_status.get("avg_correlation", 0),
                "high_correlation_pairs": len(self.correlation_status.get("high_correlation_pairs", [])),
                "in_regime_shift": self.correlation_status.get("in_regime_shift", False)
            }
        
        # Add stress indicators
        if self.enable_stress_detection:
            metrics["stress"] = {
                "in_stress_mode": self.adaptive_risk_manager.in_stress_mode,
                "stress_indicators": self.adaptive_risk_manager.stress_indicators
            }
        
        return metrics
    
    def get_position_limits(self) -> Dict[str, Dict[str, float]]:
        """
        Get position limits for all assets.
        
        Returns:
            Dictionary with position limit information for each asset
        """
        return self.position_limits
    
    def get_alerts(self, max_alerts: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent risk alerts.
        
        Args:
            max_alerts: Maximum number of alerts to return
            
        Returns:
            List of risk alerts
        """
        if max_alerts < 1:
            return []
            
        return self.alert_history[-max_alerts:]
    
    def _add_alert(self, level: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a risk alert to the history.
        
        Args:
            level: Alert level (info, warning, critical)
            message: Alert message
            details: Additional alert details
        """
        self.alert_history.append({
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "details": details or {}
        })
        
        # Log the alert
        if level == "warning":
            logger.warning(message)
        elif level == "critical":
            logger.error(message)
        else:
            logger.info(message)
        
        # Trim history if needed
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
