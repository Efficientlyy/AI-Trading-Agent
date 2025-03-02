# Risk Management System Implementation Guide

*AI Crypto Trading Agent - Phase 1 Implementation*  
*Version: 1.0*  
*Last Updated: March 2, 2025*

This guide provides detailed implementation instructions for Phase 1 of the Risk Management System, focusing on the Risk Manager core and Position Risk Calculator components. It includes code examples, configuration details, and testing scenarios to jumpstart the development process.

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Component Implementation](#2-component-implementation)
3. [Configuration Files](#3-configuration-files)
4. [Event Definitions](#4-event-definitions)
5. [Integration Points](#5-integration-points)
6. [Testing Scenarios](#6-testing-scenarios)
7. [Development Workflow](#7-development-workflow)

## 1. Project Structure

The Risk Management System will be organized in the following directory structure:

```
src/
└── risk/
    ├── __init__.py                    # Package initialization
    ├── component.py                   # Component base classes
    ├── events.py                      # Risk event definitions
    ├── models.py                      # Risk data models
    ├── exceptions.py                  # Risk-specific exceptions
    ├── utils.py                       # Risk utility functions
    ├── config.py                      # Risk configuration loader
    ├── manager.py                     # Risk Manager implementation
    ├── position_risk_calculator.py    # Position Risk Calculator
    ├── position_risk_analyzer.py      # Existing risk analyzer (to be extended)
    ├── dynamic_risk_limits.py         # Existing dynamic limits (to be extended)
    ├── portfolio_risk_controller.py   # Portfolio Risk Controller (Phase 2)
    ├── circuit_breakers.py            # Circuit Breakers (Phase 3)
    ├── risk_limits_manager.py         # Risk Limits Manager
    └── visualization/                 # Risk visualization tools
        ├── __init__.py
        ├── dashboards.py
        └── plots.py
tests/
└── risk/
    ├── __init__.py
    ├── test_manager.py
    ├── test_position_risk_calculator.py
    ├── test_risk_limits_manager.py
    └── fixtures/                      # Test fixtures
        ├── __init__.py
        ├── position_fixtures.py
        └── market_data_fixtures.py
config/
└── risk.yaml                          # Risk configuration file
```

## 2. Component Implementation

### 2.1 Risk Manager (src/risk/manager.py)

The Risk Manager is the central component that coordinates all risk management activities. It will be implemented as follows:

```python
"""
Risk Manager implementation for the AI Trading System.

This module implements the central Risk Manager component that coordinates
all risk management activities, validates signals and orders, and manages
the risk state.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from src.common.component import Component
from src.common.events import event_bus
from src.common.config import config
from src.common.logging import get_logger
from src.models.events import ErrorEvent, SystemStatusEvent, SignalEvent, OrderEvent
from src.risk.events import (
    RiskStatusEvent, RiskLimitEvent, ValidationPassedEvent, ValidationRejectedEvent
)
from src.risk.models import RiskLimit, RiskStatus
from src.risk.position_risk_calculator import PositionRiskCalculator
from src.risk.risk_limits_manager import RiskLimitsManager


class RiskManager(Component):
    """
    Central Risk Manager component.
    
    The Risk Manager coordinates all risk management activities, validates
    trading signals and orders, and manages the overall risk state.
    """
    
    def __init__(self):
        """Initialize the Risk Manager component."""
        super().__init__("risk_manager")
        self.logger = get_logger("risk", "manager")
        
        # Component references
        self.position_risk_calculator = None
        self.risk_limits_manager = None
        
        # Risk state
        self.risk_status = RiskStatus.NORMAL
        self.last_status_update = datetime.utcnow()
        self.validation_counts = {
            "total": 0,
            "passed": 0,
            "rejected": 0,
            "by_reason": {}
        }
        
        # Configuration
        self.enabled = config.get("risk.enabled", True)
        self.max_validation_delay_ms = config.get("risk.max_validation_delay_ms", 200)
    
    async def _initialize(self) -> None:
        """Initialize the Risk Manager component."""
        if not self.enabled:
            self.logger.info("Risk Manager is disabled")
            return
        
        self.logger.info("Initializing Risk Manager")
        
        # Initialize subcomponents
        self.position_risk_calculator = PositionRiskCalculator()
        await self.position_risk_calculator.initialize()
        
        self.risk_limits_manager = RiskLimitsManager()
        await self.risk_limits_manager.initialize()
        
        self.logger.info("Risk Manager initialized")
    
    async def _start(self) -> None:
        """Start the Risk Manager component."""
        if not self.enabled:
            return
        
        self.logger.info("Starting Risk Manager")
        
        # Start subcomponents
        await self.position_risk_calculator.start()
        await self.risk_limits_manager.start()
        
        # Register event handlers
        event_bus.subscribe("SignalEvent", self._handle_signal_event)
        event_bus.subscribe("OrderEvent", self._handle_order_event)
        
        # Publish initial status
        await self.publish_status("Risk Manager started")
        
        # Create task for periodic status updates
        self.create_task(self._periodic_status_update())
    
    async def _stop(self) -> None:
        """Stop the Risk Manager component."""
        if not self.enabled:
            return
        
        self.logger.info("Stopping Risk Manager")
        
        # Unsubscribe from events
        event_bus.unsubscribe("SignalEvent", self._handle_signal_event)
        event_bus.unsubscribe("OrderEvent", self._handle_order_event)
        
        # Stop subcomponents
        await self.position_risk_calculator.stop()
        await self.risk_limits_manager.stop()
        
        # Publish final status
        await self.publish_status("Risk Manager stopped")
    
    async def _handle_signal_event(self, event: SignalEvent) -> None:
        """
        Handle a signal event.
        
        Args:
            event: The signal event to handle
        """
        signal = event.signal
        
        self.logger.debug(
            "Validating signal",
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            direction=signal.direction
        )
        
        # Validate the signal against risk parameters
        is_valid, reason = await self.validate_signal(signal)
        
        # Update validation counts
        self.validation_counts["total"] += 1
        if is_valid:
            self.validation_counts["passed"] += 1
            await self.publish_event(ValidationPassedEvent(
                source=self.name,
                signal=signal
            ))
        else:
            self.validation_counts["rejected"] += 1
            if reason not in self.validation_counts["by_reason"]:
                self.validation_counts["by_reason"][reason] = 0
            self.validation_counts["by_reason"][reason] += 1
            
            await self.publish_event(ValidationRejectedEvent(
                source=self.name,
                signal=signal,
                reason=reason
            ))
            
            self.logger.info(
                "Signal rejected by risk validation",
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                direction=signal.direction,
                reason=reason
            )
    
    async def _handle_order_event(self, event: OrderEvent) -> None:
        """
        Handle an order event.
        
        Args:
            event: The order event to handle
        """
        order = event.order
        
        self.logger.debug(
            "Validating order",
            symbol=order.symbol,
            order_type=order.order_type,
            side=order.side,
            quantity=order.quantity
        )
        
        # Validate the order against risk parameters
        is_valid, reason = await self.validate_order(order)
        
        if not is_valid:
            self.logger.info(
                "Order rejected by risk validation",
                symbol=order.symbol,
                order_type=order.order_type,
                side=order.side,
                reason=reason
            )
            
            # Publish order rejected event (handled by execution service)
            await self.publish_event(OrderRejectedEvent(
                source=self.name,
                order=order,
                reason=reason
            ))
    
    async def validate_signal(self, signal) -> Tuple[bool, str]:
        """
        Validate a trading signal against risk parameters.
        
        Args:
            signal: The trading signal to validate
            
        Returns:
            A tuple containing:
              - bool: Whether the signal passed validation
              - str: The reason for rejection, if applicable
        """
        # Check if signal validation is enabled
        if not self.enabled:
            return True, ""
        
        # Check risk limits for the signal
        is_valid, reason = await self.risk_limits_manager.validate_signal(signal)
        if not is_valid:
            return False, reason
        
        # Check position risk for the signal
        is_valid, reason = await self.position_risk_calculator.validate_signal(signal)
        if not is_valid:
            return False, reason
        
        return True, ""
    
    async def validate_order(self, order) -> Tuple[bool, str]:
        """
        Validate an order against risk parameters.
        
        Args:
            order: The order to validate
            
        Returns:
            A tuple containing:
              - bool: Whether the order passed validation
              - str: The reason for rejection, if applicable
        """
        # Check if order validation is enabled
        if not self.enabled:
            return True, ""
        
        # Check risk limits for the order
        is_valid, reason = await self.risk_limits_manager.validate_order(order)
        if not is_valid:
            return False, reason
        
        # Check position risk for the order
        is_valid, reason = await self.position_risk_calculator.validate_order(order)
        if not is_valid:
            return False, reason
        
        return True, ""
    
    async def get_risk_status(self) -> Dict[str, Any]:
        """
        Get the current risk status.
        
        Returns:
            A dictionary containing the current risk status
        """
        return {
            "status": self.risk_status.value,
            "last_update": self.last_status_update,
            "validation_counts": self.validation_counts,
            "position_risk": await self.position_risk_calculator.get_risk_status(),
            "risk_limits": await self.risk_limits_manager.get_risk_status()
        }
    
    async def _periodic_status_update(self) -> None:
        """Periodically publish risk status updates."""
        update_interval = config.get("risk.status_update_interval_sec", 60)
        
        while self.running:
            try:
                # Update risk status based on subcomponent status
                await self._update_risk_status()
                
                # Publish risk status event
                await self.publish_event(RiskStatusEvent(
                    source=self.name,
                    status=self.risk_status.value,
                    message=f"Risk status: {self.risk_status.value}",
                    metrics=await self.get_risk_metrics()
                ))
                
                # Sleep until next update
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in periodic status update",
                    error=str(e)
                )
                await asyncio.sleep(5)  # Short sleep before retry
    
    async def _update_risk_status(self) -> None:
        """Update the overall risk status based on subcomponent status."""
        position_risk_status = await self.position_risk_calculator.get_risk_status()
        risk_limits_status = await self.risk_limits_manager.get_risk_status()
        
        # Determine the most severe status
        if (position_risk_status.get("status") == RiskStatus.CRITICAL.value or
            risk_limits_status.get("status") == RiskStatus.CRITICAL.value):
            new_status = RiskStatus.CRITICAL
        elif (position_risk_status.get("status") == RiskStatus.WARNING.value or
              risk_limits_status.get("status") == RiskStatus.WARNING.value):
            new_status = RiskStatus.WARNING
        else:
            new_status = RiskStatus.NORMAL
        
        # Update status if changed
        if new_status != self.risk_status:
            self.logger.info(
                f"Risk status changed from {self.risk_status.value} to {new_status.value}"
            )
            self.risk_status = new_status
            self.last_status_update = datetime.utcnow()
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            A dictionary containing current risk metrics
        """
        position_metrics = await self.position_risk_calculator.get_risk_metrics()
        limit_metrics = await self.risk_limits_manager.get_risk_metrics()
        
        # Combine metrics
        metrics = {
            "validation_rate": {
                "total": self.validation_counts["total"],
                "passed": self.validation_counts["passed"],
                "rejected": self.validation_counts["rejected"],
                "pass_rate": (self.validation_counts["passed"] / self.validation_counts["total"] 
                             if self.validation_counts["total"] > 0 else 1.0)
            },
            "position_metrics": position_metrics,
            "limit_metrics": limit_metrics,
            "overall_status": self.risk_status.value
        }
        
        return metrics
```

### 2.2 Position Risk Calculator (src/risk/position_risk_calculator.py)

The Position Risk Calculator will be implemented as follows, building on the existing position_risk_analyzer.py:

```python
"""
Position Risk Calculator for the AI Trading System.

This module extends the existing Position Risk Analyzer to provide position-level
risk management with support for stop-loss, take-profit, and trailing stop functionality.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from src.common.component import Component
from src.common.events import event_bus
from src.common.config import config
from src.common.logging import get_logger
from src.models.events import PositionUpdateEvent
from src.risk.events import PositionRiskEvent, StopLossUpdatedEvent, TakeProfitUpdatedEvent
from src.risk.models import PositionRisk, ExitRecommendation, RiskStatus
from src.risk.position_risk_analyzer import PositionRiskAnalyzer


class PositionRiskCalculator(Component):
    """
    Position Risk Calculator component.
    
    Tracks and manages risk for individual trading positions, including
    stop-loss, take-profit, and trailing stop functionality.
    """
    
    def __init__(self):
        """Initialize the Position Risk Calculator component."""
        super().__init__("position_risk_calculator")
        self.logger = get_logger("risk", "position_risk")
        
        # Reference to the position risk analyzer
        self.risk_analyzer = None
        
        # Position risk state
        self.position_risks: Dict[str, PositionRisk] = {}
        self.historical_data: Dict[str, List[float]] = {}
        
        # Configuration
        self.enabled = config.get("risk.position_risk.enabled", True)
        self.default_stop_loss_pct = config.get("risk.position_risk.default_stop_loss_pct", 0.02)
        self.default_take_profit_pct = config.get("risk.position_risk.default_take_profit_pct", 0.06)
        self.enable_trailing_stops = config.get("risk.position_risk.enable_trailing_stops", True)
        self.default_trailing_pct = config.get("risk.position_risk.default_trailing_pct", 0.015)
        self.min_risk_reward_ratio = config.get("risk.position_risk.min_risk_reward_ratio", 2.0)
    
    async def _initialize(self) -> None:
        """Initialize the Position Risk Calculator component."""
        if not self.enabled:
            self.logger.info("Position Risk Calculator is disabled")
            return
        
        self.logger.info("Initializing Position Risk Calculator")
        
        # Initialize the risk analyzer with a risk-free rate
        risk_free_rate = config.get("risk.position_risk.risk_free_rate", 0.04)
        self.risk_analyzer = PositionRiskAnalyzer(risk_free_rate=risk_free_rate)
        
        # Set default trailing stop configuration
        self.trailing_stop_config = {
            "enabled": self.enable_trailing_stops,
            "default_pct": self.default_trailing_pct
        }
        
        self.logger.info("Position Risk Calculator initialized")
    
    async def _start(self) -> None:
        """Start the Position Risk Calculator component."""
        if not self.enabled:
            return
        
        self.logger.info("Starting Position Risk Calculator")
        
        # Register event handlers
        event_bus.subscribe("PositionUpdateEvent", self._handle_position_update)
        event_bus.subscribe("TradeDataEvent", self._handle_trade_data)
        
        # Create task for periodic position risk updates
        update_interval = config.get("risk.position_risk.update_interval_sec", 30)
        self.create_task(self._periodic_position_risk_update(update_interval))
        
        self.logger.info("Position Risk Calculator started")
    
    async def _stop(self) -> None:
        """Stop the Position Risk Calculator component."""
        if not self.enabled:
            return
        
        self.logger.info("Stopping Position Risk Calculator")
        
        # Unsubscribe from events
        event_bus.unsubscribe("PositionUpdateEvent", self._handle_position_update)
        event_bus.unsubscribe("TradeDataEvent", self._handle_trade_data)
        
        self.logger.info("Position Risk Calculator stopped")
    
    async def _handle_position_update(self, event: PositionUpdateEvent) -> None:
        """
        Handle a position update event.
        
        Args:
            event: The position update event
        """
        position = event.position
        position_id = position.id
        
        # Update or create position risk entry
        await self.update_position_risk(position)
        
        # Check if we should generate exit recommendations
        if position_id in self.position_risks:
            position_risk = self.position_risks[position_id]
            if position_risk.exit_recommendation != ExitRecommendation.HOLD:
                self.logger.info(
                    "Exit recommendation generated",
                    position_id=position_id,
                    symbol=position.symbol,
                    recommendation=position_risk.exit_recommendation,
                    reason=position_risk.exit_reason
                )
                
                # Publish position risk event with exit recommendation
                await self.publish_event(PositionRiskEvent(
                    source=self.name,
                    position_id=position_id,
                    symbol=position.symbol,
                    risk_metrics=position_risk.to_dict(),
                    exit_recommendation=position_risk.exit_recommendation,
                    exit_reason=position_risk.exit_reason
                ))
    
    async def _handle_trade_data(self, event) -> None:
        """
        Handle a trade data event to update price information.
        
        Args:
            event: The trade data event
        """
        trade = event.trade
        symbol = trade.symbol
        price = trade.price
        
        # Update historical data for the symbol
        if symbol not in self.historical_data:
            self.historical_data[symbol] = []
        
        self.historical_data[symbol].append(price)
        
        # Limit the size of historical data
        max_data_points = config.get("risk.position_risk.max_historical_data_points", 1000)
        if len(self.historical_data[symbol]) > max_data_points:
            self.historical_data[symbol] = self.historical_data[symbol][-max_data_points:]
        
        # Update position risk for positions with this symbol
        for position_id, position_risk in list(self.position_risks.items()):
            if position_risk.symbol == symbol:
                # Update with new price
                position_risk.current_price = price
                position_risk.last_updated = datetime.utcnow()
                
                # Update unrealized P&L
                direction = 1 if position_risk.direction == "long" else -1
                price_diff = (price - position_risk.entry_price) * direction
                position_risk.unrealized_pnl = price_diff * position_risk.size
                position_risk.unrealized_pnl_pct = price_diff / position_risk.entry_price
                
                # Check trailing stop if enabled
                if position_risk.trailing_stop_enabled:
                    await self._check_trailing_stop(position_id, position_risk, price)
                
                # Check exit conditions
                await self._check_exit_conditions(position_id, position_risk)
    
    async def _check_trailing_stop(self, position_id: str, position_risk: PositionRisk, price: float) -> None:
        """
        Check and update trailing stop if needed.
        
        Args:
            position_id: The position ID
            position_risk: The position risk data
            price: The current price
        """
        if not position_risk.trailing_stop_enabled or not position_risk.trailing_stop_distance_pct:
            return
        
        # For long positions
        if position_risk.direction == "long":
            # If price has moved up, adjust trailing stop
            if price > position_risk.max_favorable_excursion:
                position_risk.max_favorable_excursion = price
                new_stop = price * (1 - position_risk.trailing_stop_distance_pct)
                
                # Only move the stop loss up, never down
                if position_risk.stop_loss is None or new_stop > position_risk.stop_loss:
                    position_risk.stop_loss = new_stop
                    
                    # Publish stop loss updated event
                    await self.publish_event(StopLossUpdatedEvent(
                        source=self.name,
                        position_id=position_id,
                        symbol=position_risk.symbol,
                        new_stop_loss=new_stop,
                        reason="Trailing stop adjustment"
                    ))
                    
                    self.logger.debug(
                        "Trailing stop updated for long position",
                        position_id=position_id,
                        symbol=position_risk.symbol,
                        new_stop_loss=new_stop,
                        price=price
                    )
        
        # For short positions
        elif position_risk.direction == "short":
            # If price has moved down, adjust trailing stop
            if price < position_risk.max_favorable_excursion or position_risk.max_favorable_excursion == 0:
                if position_risk.max_favorable_excursion == 0:
                    position_risk.max_favorable_excursion = price
                else:
                    position_risk.max_favorable_excursion = price
                
                new_stop = price * (1 + position_risk.trailing_stop_distance_pct)
                
                # Only move the stop loss down, never up
                if position_risk.stop_loss is None or new_stop < position_risk.stop_loss:
                    position_risk.stop_loss = new_stop
                    
                    # Publish stop loss updated event
                    await self.publish_event(StopLossUpdatedEvent(
                        source=self.name,
                        position_id=position_id,
                        symbol=position_risk.symbol,
                        new_stop_loss=new_stop,
                        reason="Trailing stop adjustment"
                    ))
                    
                    self.logger.debug(
                        "Trailing stop updated for short position",
                        position_id=position_id,
                        symbol=position_risk.symbol,
                        new_stop_loss=new_stop,
                        price=price
                    )
    
    async def _check_exit_conditions(self, position_id: str, position_risk: PositionRisk) -> None:
        """
        Check if any exit conditions are met.
        
        Args:
            position_id: The position ID
            position_risk: The position risk data
        """
        price = position_risk.current_price
        exit_recommendation = ExitRecommendation.HOLD
        exit_reason = None
        
        # For long positions
        if position_risk.direction == "long":
            # Check stop loss
            if position_risk.stop_loss and price <= position_risk.stop_loss:
                exit_recommendation = ExitRecommendation.STOP_LOSS
                exit_reason = f"Price {price} below stop loss {position_risk.stop_loss}"
            
            # Check take profit
            elif position_risk.take_profit and price >= position_risk.take_profit:
                exit_recommendation = ExitRecommendation.TAKE_PROFIT
                exit_reason = f"Price {price} above take profit {position_risk.take_profit}"
        
        # For short positions
        elif position_risk.direction == "short":
            # Check stop loss
            if position_risk.stop_loss and price >= position_risk.stop_loss:
                exit_recommendation = ExitRecommendation.STOP_LOSS
                exit_reason = f"Price {price} above stop loss {position_risk.stop_loss}"
            
            # Check take profit
            elif position_risk.take_profit and price <= position_risk.take_profit:
                exit_recommendation = ExitRecommendation.TAKE_PROFIT
                exit_reason = f"Price {price} below take profit {position_risk.take_profit}"
        
        # Update position risk with exit recommendation
        if exit_recommendation != ExitRecommendation.HOLD:
            position_risk.exit_recommendation = exit_recommendation
            position_risk.exit_reason = exit_reason
            
            # Publish position risk event with exit recommendation
            await self.publish_event(PositionRiskEvent(
                source=self.name,
                position_id=position_id,
                symbol=position_risk.symbol,
                risk_metrics=position_risk.to_dict(),
                exit_recommendation=exit_recommendation.value,
                exit_reason=exit_reason
            ))
    
    async def update_position_risk(self, position) -> PositionRisk:
        """
        Update or create position risk data for a position.
        
        Args:
            position: The position data
            
        Returns:
            The updated position risk data
        """
        position_id = position.id
        
        # Create new position risk entry if needed
        if position_id not in self.position_risks:
            # Initialize with default values
            position_risk = PositionRisk(
                position_id=position_id,
                symbol=position.symbol,
                strategy_id=position.strategy_id if hasattr(position, "strategy_id") else "unknown",
                entry_price=position.entry_price,
                current_price=position.current_price,
                size=position.quantity,
                direction=position.side.lower(),  # Normalize to lowercase
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                max_favorable_excursion=position.current_price,
                max_adverse_excursion=0.0,
                position_started=position.opened_at if hasattr(position, "opened_at") else datetime.utcnow(),
                last_updated=datetime.utcnow(),
                exit_recommendation=ExitRecommendation.HOLD
            )
            
            # Set default stop loss and take profit
            if self.default_stop_loss_pct > 0:
                if position.side.lower() == "long":
                    position_risk.stop_loss = position.entry_price * (1 - self.default_stop_loss_pct)
                else:
                    position_risk.stop_loss = position.entry_price * (1 + self.default_stop_loss_pct)
            
            if self.default_take_profit_pct > 0:
                if position.side.lower() == "long":
                    position_risk.take_profit = position.entry_price * (1 + self.default_take_profit_pct)
                else:
                    position_risk.take_profit = position.entry_price * (1 - self.default_take_profit_pct)
            
            # Calculate risk-reward ratio
            if position_risk.stop_loss and position_risk.take_profit:
                stop_distance = abs(position.entry_price - position_risk.stop_loss)
                target_distance = abs(position_risk.take_profit - position.entry_price)
                position_risk.risk_reward_ratio = target_distance / stop_distance if stop_distance > 0 else None
            
            # Enable trailing stop if configured
            position_risk.trailing_stop_enabled = self.trailing_stop_config["enabled"]
            position_risk.trailing_stop_distance_pct = self.trailing_stop_config["default_pct"]
            position_risk.trailing_stop_activation_price = None  # Will be set when trail starts
            
            self.position_risks[position_id] = position_risk
            
            self.logger.info(
                "Created position risk entry",
                position_id=position_id,
                symbol=position.symbol,
                direction=position.side,
                entry_price=position.entry_price,
                stop_loss=position_risk.stop_loss,
                take_profit=position_risk.take_profit
            )
        else:
            # Update existing position risk entry
            position_risk = self.position_risks[position_id]
            position_risk.current_price = position.current_price
            position_risk.size = position.quantity
            position_risk.last_updated = datetime.utcnow()
            
            # Calculate days held
            days_held = (datetime.utcnow() - position_risk.position_started).total_seconds() / 86400
            position_risk.days_held = days_held
            
            # Update unrealized P&L
            direction = 1 if position_risk.direction == "long" else -1
            price_diff = (position.current_price - position_risk.entry_price) * direction
            position_risk.unrealized_pnl = price_diff * position_risk.size
            position_risk.unrealized_pnl_pct = price_diff / position_risk.entry_price
            
            # Track maximum favorable and adverse excursions
            if price_diff > 0 and price_diff > position_risk.max_favorable_excursion:
                position_risk.max_favorable_excursion = price_diff
            elif price_diff < 0 and (position_risk.max_adverse_excursion == 0 or abs(price_diff) > position_risk.max_adverse_excursion):
                position_risk.max_adverse_excursion = abs(price_diff)
        
        return position_risk
    
    async def update_stop_loss(self, position_id: str, stop_loss: float) -> bool:
        """
        Update the stop loss for a position.
        
        Args:
            position_id: The position ID
            stop_loss: The new stop loss price
            
        Returns:
            Whether the update was successful
        """
        if position_id not in self.position_risks:
            self.logger.warning(
                "Cannot update stop loss for unknown position",
                position_id=position_id
            )
            return False
        
        position_risk = self.position_risks[position_id]
        
        # Validate stop loss direction
        if position_risk.direction == "long" and stop_loss >= position_risk.entry_price:
            self.logger.warning(
                "Invalid stop loss for long position (above entry price)",
                position_id=position_id,
                stop_loss=stop_loss,
                entry_price=position_risk.entry_price
            )
            return False
        
        if position_risk.direction == "short" and stop_loss <= position_risk.entry_price:
            self.logger.warning(
                "Invalid stop loss for short position (below entry price)",
                position_id=position_id,
                stop_loss=stop_loss,
                entry_price=position_risk.entry_price
            )
            return False
        
        # Update stop loss
        position_risk.stop_loss = stop_loss
        
        # Recalculate risk-reward ratio
        if position_risk.take_profit:
