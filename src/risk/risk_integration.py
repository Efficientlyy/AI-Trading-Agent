#!/usr/bin/env python3
"""
Risk Integration Module

This module integrates the risk budget management system with the portfolio manager
and other components of the trading system. It provides:

1. Risk limit checking for new positions
2. Risk utilization tracking and updates
3. Performance-based risk allocation
4. Risk alerts generation
5. Integration with dashboard for risk monitoring
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any
from decimal import Decimal
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.risk.risk_budget_manager import RiskBudgetManager, RiskAllocationMethod
from src.models.signals import Signal, SignalType
from src.portfolio.portfolio_manager import Position, PositionType, PositionStatus
from src.alerts.alert import AlertType, AlertSeverity, AlertCategory, Alert

# Setup logging
logger = logging.getLogger(__name__)


class RiskIntegration:
    """
    Integrates risk management with portfolio management and other system components.
    
    This class acts as a bridge between the risk budget management system and
    other components of the trading system, providing risk checks before trades
    and updating risk utilization based on actual positions.
    """
    
    def __init__(self, 
                 risk_budget_manager: RiskBudgetManager,
                 config_path: Optional[str] = None):
        """
        Initialize the risk integration module.
        
        Args:
            risk_budget_manager: The risk budget manager instance
            config_path: Path to the risk configuration file (optional)
        """
        self.risk_budget_manager = risk_budget_manager
        self.config_path = config_path
        
        # Map from position IDs to risk paths
        self.position_risk_map = {}
        
        # Map from symbols to their asset paths
        self.symbol_path_map = {}
        
        # Last time we calculated performance metrics
        self.last_performance_calc = datetime.now()
        
        # Performance metrics for strategies and assets
        self.performance_metrics = {}
        
        # Cache of position sizes allowed for different risk amounts
        self.position_size_cache = {}
        
        logger.info("Risk integration module initialized")
    
    def build_symbol_path_map(self):
        """
        Build a mapping from symbols to their asset paths in the risk hierarchy.
        
        This mapping helps quickly locate the risk budget for a symbol.
        """
        # Start with an empty map
        self.symbol_path_map = {}
        
        # Traverse the risk budget hierarchy
        for strategy_name, strategy_budget in self.risk_budget_manager.root_budget.children.items():
            for market_name, market_budget in strategy_budget.children.items():
                for asset_name, asset_budget in market_budget.children.items():
                    # Create a full path for this asset
                    path = [strategy_name, market_name, asset_name]
                    
                    # Map the asset name to its path
                    if asset_name not in self.symbol_path_map:
                        self.symbol_path_map[asset_name] = []
                    
                    self.symbol_path_map[asset_name].append(path)
        
        logger.debug(f"Built symbol path map with {len(self.symbol_path_map)} symbols")
    
    def get_paths_for_symbol(self, symbol: str, strategy_id: Optional[str] = None) -> List[List[str]]:
        """
        Get all risk budget paths for a symbol.
        
        Args:
            symbol: The trading symbol
            strategy_id: Optional strategy ID to filter by
            
        Returns:
            List of paths to the risk budgets for this symbol
        """
        if not self.symbol_path_map:
            self.build_symbol_path_map()
        
        # Get all paths for this symbol
        paths = self.symbol_path_map.get(symbol, [])
        
        # Filter by strategy if provided
        if strategy_id and paths:
            filtered_paths = [path for path in paths if path[0] == strategy_id]
            if filtered_paths:
                return filtered_paths
        
        return paths
    
    def check_risk_for_signal(self, 
                             signal: Signal, 
                             current_price: Decimal, 
                             portfolio_value: Decimal) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a trading signal would breach risk limits.
        
        Args:
            signal: The trading signal
            current_price: Current price of the asset
            portfolio_value: Current portfolio value
            
        Returns:
            (is_allowed, risk_info) tuple
        """
        symbol = signal.symbol
        strategy_id = signal.strategy_id
        
        # Get all risk budget paths for this symbol and strategy
        paths = self.get_paths_for_symbol(symbol, strategy_id)
        
        if not paths:
            logger.warning(f"No risk budget found for {symbol} with strategy {strategy_id}")
            return False, {"error": "No risk budget found"}
        
        # Calculate the position size implied by the signal
        position_size = self._calculate_position_size(signal, current_price)
        
        # Calculate the risk value of this position
        risk_value = self._estimate_position_risk(signal, position_size, current_price, portfolio_value)
        
        # Check risk limit for each path
        results = []
        for path in paths:
            is_breached, available = self.risk_budget_manager.check_risk_breach(path, risk_value)
            
            result = {
                "path": "/".join(path),
                "risk_value": risk_value,
                "available_risk": available,
                "is_breached": is_breached
            }
            results.append(result)
        
        # Signal is allowed if at least one path doesn't breach limits
        allowed_paths = [r for r in results if not r["is_breached"]]
        is_allowed = len(allowed_paths) > 0
        
        # Return the detailed risk information
        return is_allowed, {
            "symbol": symbol,
            "strategy_id": strategy_id,
            "position_size": str(position_size),
            "risk_value": risk_value,
            "is_allowed": is_allowed,
            "path_results": results,
            "chosen_path": allowed_paths[0]["path"] if allowed_paths else None
        }
    
    def register_position(self, position: Position, path: List[str]) -> None:
        """
        Register a new position with the risk management system.
        
        Args:
            position: The position that was opened
            path: The risk budget path used for this position
        """
        # Store the mapping from position ID to risk path
        self.position_risk_map[position.position_id] = path
        
        # Calculate the risk value of this position
        position_size = Decimal(position.size) * Decimal(position.entry_price)
        portfolio_value = self._get_portfolio_value()
        
        # Convert position to a signal-like object for risk estimation
        signal_type = SignalType.LONG if position.position_type == PositionType.LONG else SignalType.SHORT
        signal = Signal(
            symbol=position.symbol,
            signal_type=signal_type,
            strategy_id=position.strategy_id,
            timestamp=position.timestamp
        )
        
        risk_value = self._estimate_position_risk(
            signal, position_size, position.entry_price, portfolio_value
        )
        
        # Update the risk utilization
        self.risk_budget_manager.update_risk_utilization(path, risk_value)
        
        logger.info(f"Registered position {position.position_id} with risk path {'/'.join(path)}")
    
    def update_position_risk(self, position: Position, current_price: Optional[Decimal] = None) -> None:
        """
        Update the risk utilization for an existing position.
        
        Args:
            position: The position to update
            current_price: Optional current price for the position, if not already set
        """
        if position.position_id not in self.position_risk_map:
            logger.warning(f"Position {position.position_id} not registered with risk management")
            return
        
        # Get the risk path for this position
        path = self.position_risk_map[position.position_id]
        
        # If position is closed, remove its risk
        if position.status == PositionStatus.CLOSED:
            self.risk_budget_manager.update_risk_utilization(path, 0.0)
            self.position_risk_map.pop(position.position_id, None)
            logger.info(f"Removed risk utilization for closed position {position.position_id}")
            return
        
        # Use provided current price or get it from position if available
        # Otherwise default to entry price
        price_to_use = current_price
        if not price_to_use and hasattr(position, 'current_price'):
            price_to_use = position.current_price
        if not price_to_use:
            price_to_use = position.entry_price
        
        # Calculate the current risk value of this position
        position_size = Decimal(position.size) * price_to_use
        portfolio_value = self._get_portfolio_value()
        
        # Convert position to a signal-like object for risk estimation
        direction = "long" if position.position_type == PositionType.LONG else "short"
        signal = Signal(
            symbol=position.symbol,
            signal_type=SignalType.ENTRY,
            direction=direction,
            price=float(price_to_use),
            confidence=1.0,
            strategy_id=position.strategy_id,
            timestamp=position.timestamp
        )
        
        risk_value = self._estimate_position_risk(
            signal, position_size, price_to_use, portfolio_value
        )
        
        # Update the risk utilization
        self.risk_budget_manager.update_risk_utilization(path, risk_value)
        
        logger.debug(f"Updated risk for position {position.position_id} to {risk_value:.4f}%")
    
    def _calculate_position_size(self, signal: Signal, current_price: Decimal) -> Decimal:
        """
        Calculate the position size implied by a signal.
        
        Args:
            signal: The trading signal
            current_price: Current price of the asset
            
        Returns:
            Position size in base currency
        """
        # In a real implementation, this would use signal attributes to determine size
        # For now, we'll use a simple approach based on signal strength
        
        # Default to a medium-sized position
        position_value = Decimal("10000.0")
        
        # Adjust based on signal strength if available
        if hasattr(signal, "strength") and signal.strength is not None:
            strength = Decimal(str(signal.strength))
            position_value = position_value * (Decimal("0.5") + strength / Decimal("2.0"))
        
        return position_value
    
    def _estimate_position_risk(self, 
                              signal: Signal, 
                              position_size: Decimal, 
                              current_price: Decimal,
                              portfolio_value: Decimal) -> float:
        """
        Estimate the risk value of a potential position.
        
        This calculates what percentage of the risk budget would be consumed
        by taking this position.
        
        Args:
            signal: The trading signal
            position_size: Size of the position in base currency
            current_price: Current price of the asset
            portfolio_value: Current portfolio value
            
        Returns:
            Risk value as a percentage of portfolio
        """
        # In a real implementation, this would use volatility and other metrics
        # For this example, we'll use a simple approximation
        
        # Convert to float for calculations
        position_size_float = float(position_size)
        portfolio_value_float = float(portfolio_value)
        
        # Basic risk is position size as percentage of portfolio
        basic_risk = position_size_float / portfolio_value_float * 100.0
        
        # Apply a risk multiplier based on the asset
        # In reality, this would come from volatility calculations
        risk_multipliers = {
            "BTC": 1.5,
            "ETH": 1.3,
            "SOL": 1.8,
            "LINK": 1.6,
            "AVAX": 1.9,
            "EUR/USD": 0.7,
            "GBP/USD": 0.8,
            "GOLD": 0.9,
            "SILVER": 1.1
        }
        
        # Default risk multiplier if symbol not found
        risk_multiplier = risk_multipliers.get(signal.symbol, 1.0)
        
        # Apply signal type multiplier (shorts might be more risky)
        signal_type_multiplier = 1.2 if signal.signal_type == SignalType.SHORT else 1.0
        
        # Calculate final risk value
        risk_value = basic_risk * risk_multiplier * signal_type_multiplier
        
        return risk_value
    
    def _get_portfolio_value(self) -> Decimal:
        """
        Get the current portfolio value.
        
        In a real implementation, this would come from the portfolio manager.
        
        Returns:
            Current portfolio value
        """
        # Placeholder implementation
        return Decimal("1000000.0")
    
    def update_performance_metrics(self, performance_data: Dict[str, float]) -> None:
        """
        Update performance metrics and optimize risk allocation if needed.
        
        Args:
            performance_data: Dictionary mapping paths to performance metrics
        """
        # Update our stored metrics
        self.performance_metrics.update(performance_data)
        self.last_performance_calc = datetime.now()
        
        # Optimize risk allocation based on performance
        self.risk_budget_manager.optimize_allocations(
            performance_data=self.performance_metrics
        )
        
        logger.info("Updated performance metrics and optimized risk allocation")
    
    def generate_risk_alerts(self) -> List[Alert]:
        """
        Generate alerts for risk limit breaches.
        
        Returns:
            List of alerts for risk issues
        """
        alerts = []
        
        # Get the latest risk report
        risk_report = self.risk_budget_manager.risk_report()
        
        # Check system-level risk
        system_risk_util = risk_report["risk_utilization_percent"]
        if system_risk_util > 95:
            alerts.append(Alert(
                title="CRITICAL: System Risk Budget Nearly Exhausted",
                message=f"System risk utilization at {system_risk_util:.1f}%",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.RISK,
                timestamp=datetime.now()
            ))
        elif system_risk_util > 85:
            alerts.append(Alert(
                title="WARNING: High System Risk Utilization",
                message=f"System risk utilization at {system_risk_util:.1f}%",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.RISK,
                timestamp=datetime.now()
            ))
        
        # Check strategy-level risk
        for strategy_name, strategy_data in risk_report["strategies"].items():
            util_pct = strategy_data["risk_utilization_percent"]
            
            if util_pct > 95:
                alerts.append(Alert(
                    title=f"CRITICAL: Strategy Risk Budget Nearly Exhausted",
                    message=f"Strategy '{strategy_name}' risk utilization at {util_pct:.1f}%",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.RISK,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": strategy_name,
                        "utilization": util_pct,
                        "max_risk": strategy_data["max_risk"]
                    }
                ))
            elif util_pct > 85:
                alerts.append(Alert(
                    title=f"WARNING: High Strategy Risk Utilization",
                    message=f"Strategy '{strategy_name}' risk utilization at {util_pct:.1f}%",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.RISK,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": strategy_name,
                        "utilization": util_pct,
                        "max_risk": strategy_data["max_risk"]
                    }
                ))
        
        return alerts
    
    async def scheduled_tasks(self):
        """
        Run scheduled risk management tasks.
        
        This method is intended to be run periodically by the system scheduler.
        """
        # Save risk budget configuration if modified
        if self.risk_budget_manager.modified and self.config_path:
            self.risk_budget_manager.save_to_file(self.config_path)
        
        # Generate risk alerts
        alerts = self.generate_risk_alerts()
        
        # In a real implementation, we would publish these alerts to the alert system
        for alert in alerts:
            logger.warning(f"Risk Alert: {alert.title} - {alert.message}")
        
        logger.info("Completed scheduled risk management tasks")
    
    def get_risk_limits_for_dashboard(self) -> Dict[str, Any]:
        """
        Get risk information for the dashboard.
        
        Returns:
            Dictionary with risk information formatted for the dashboard
        """
        # Get the risk report
        risk_report = self.risk_budget_manager.risk_report()
        
        # Format for the dashboard
        dashboard_data = {
            "timestamp": risk_report["timestamp"],
            "system_risk": {
                "current": risk_report["current_system_risk"],
                "maximum": risk_report["total_risk_percent"],
                "utilization": risk_report["risk_utilization_percent"]
            },
            "strategies": {}
        }
        
        # Add strategy data
        for strategy_name, strategy_data in risk_report["strategies"].items():
            dashboard_data["strategies"][strategy_name] = {
                "current": strategy_data["current_risk"],
                "maximum": strategy_data["max_risk"],
                "utilization": strategy_data["risk_utilization_percent"],
                "markets": {}
            }
            
            # Add market data
            for market_name, market_data in strategy_data["markets"].items():
                dashboard_data["strategies"][strategy_name]["markets"][market_name] = {
                    "current": market_data["current_risk"],
                    "maximum": market_data["max_risk"],
                    "utilization": market_data["risk_utilization_percent"]
                }
        
        return dashboard_data 