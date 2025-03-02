"""
Portfolio Management System.

This module implements a comprehensive portfolio manager responsible for:
- Position sizing and allocation
- Risk management
- Performance tracking
- Trading decision execution

The portfolio manager translates signals from strategies into actual positions
while enforcing risk controls and position limits.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import uuid

from src.common.logging import get_logger
from src.models.signals import Signal, SignalType


class PositionStatus(Enum):
    """Status of a trading position."""
    
    PENDING = "pending"  # Position has been created but not yet filled
    OPEN = "open"  # Position is currently open
    CLOSING = "closing"  # Position is in the process of being closed
    CLOSED = "closed"  # Position has been closed
    CANCELED = "canceled"  # Position was canceled before being filled
    REJECTED = "rejected"  # Position was rejected by the exchange


class PositionType(Enum):
    """Type of trading position."""
    
    LONG = "long"  # Long position (profit when price increases)
    SHORT = "short"  # Short position (profit when price decreases)


class Position:
    """Represents a trading position.
    
    A position is created in response to a trading signal and includes
    information about entry, size, risk parameters, and performance.
    """
    
    def __init__(
        self,
        symbol: str,
        position_type: PositionType,
        size: Decimal,
        entry_price: Decimal,
        take_profit_price: Optional[Decimal] = None,
        stop_loss_price: Optional[Decimal] = None,
        strategy_id: Optional[str] = None,
        position_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a new trading position.
        
        Args:
            symbol: The trading pair symbol
            position_type: Type of position (LONG or SHORT)
            size: Size of the position in base currency units
            entry_price: Entry price of the position
            take_profit_price: Optional price at which to take profit
            stop_loss_price: Optional price at which to stop loss
            strategy_id: ID of the strategy that generated this position
            position_id: Unique ID for this position (generated if None)
            timestamp: Time when position was created (now if None)
            metadata: Additional metadata about this position
        """
        self.symbol = symbol
        self.position_type = position_type
        self.size = size
        self.entry_price = entry_price
        self.take_profit_price = take_profit_price
        self.stop_loss_price = stop_loss_price
        self.strategy_id = strategy_id
        self.position_id = position_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        
        # Position tracking
        self.status = PositionStatus.PENDING
        self.exit_price: Optional[Decimal] = None
        self.exit_timestamp: Optional[datetime] = None
        self.realized_pnl: Optional[Decimal] = None
        self.fee_paid: Decimal = Decimal("0")
        
        # Order tracking
        self.entry_order_id: Optional[str] = None
        self.exit_order_id: Optional[str] = None
        
        # Additional tracking
        self.max_profit: Decimal = Decimal("0")  # Maximum unrealized profit reached
        self.max_loss: Decimal = Decimal("0")  # Maximum unrealized loss reached
        self.last_price: Optional[Decimal] = None  # Last known price
        self.last_price_timestamp: Optional[datetime] = None  # Time of last price update
        
        # Updates history
        self.updates: List[Dict[str, Any]] = []
        
    def update_status(self, status: PositionStatus, timestamp: Optional[datetime] = None) -> None:
        """Update the status of this position.
        
        Args:
            status: New status of the position
            timestamp: Timestamp of the status update (now if None)
        """
        if self.status == status:
            return  # No change
            
        old_status = self.status
        self.status = status
        update_time = timestamp or datetime.now()
        
        # Record the update
        self.updates.append({
            "timestamp": update_time,
            "type": "status_change",
            "old_status": old_status.value,
            "new_status": status.value
        })
    
    def update_price(self, price: Decimal, timestamp: Optional[datetime] = None) -> None:
        """Update the current price for this position.
        
        Args:
            price: Current market price
            timestamp: Timestamp of the price update (now if None)
        """
        self.last_price = price
        self.last_price_timestamp = timestamp or datetime.now()
        
        # Calculate unrealized profit/loss
        if self.status == PositionStatus.OPEN:
            unrealized_pnl = self.calculate_unrealized_pnl(price)
            
            # Update max profit/loss
            if unrealized_pnl > self.max_profit:
                self.max_profit = unrealized_pnl
            elif unrealized_pnl < self.max_loss:
                self.max_loss = unrealized_pnl
    
    def close(
        self, 
        exit_price: Decimal, 
        timestamp: Optional[datetime] = None,
        fee_paid: Optional[Decimal] = None
    ) -> None:
        """Close this position with the specified exit price.
        
        Args:
            exit_price: Exit price for this position
            timestamp: Timestamp of the position close (now if None)
            fee_paid: Additional fees paid for this transaction
        """
        if self.status == PositionStatus.CLOSED:
            return  # Already closed
            
        self.exit_price = exit_price
        self.exit_timestamp = timestamp or datetime.now()
        self.realized_pnl = self.calculate_pnl(exit_price)
        
        if fee_paid:
            self.fee_paid += fee_paid
            
        self.update_status(PositionStatus.CLOSED, self.exit_timestamp)
    
    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate the unrealized profit/loss for this position.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L in quote currency
        """
        if not self.entry_price:
            return Decimal("0")
            
        if self.position_type == PositionType.LONG:
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size
    
    def calculate_pnl(self, exit_price: Decimal) -> Decimal:
        """Calculate the realized profit/loss for this position.
        
        Args:
            exit_price: Exit price for this position
            
        Returns:
            Realized P&L in quote currency
        """
        if not self.entry_price:
            return Decimal("0")
            
        if self.position_type == PositionType.LONG:
            return (exit_price - self.entry_price) * self.size - self.fee_paid
        else:
            return (self.entry_price - exit_price) * self.size - self.fee_paid
    
    def calculate_pnl_pct(self, exit_price: Optional[Decimal] = None) -> Decimal:
        """Calculate the profit/loss percentage for this position.
        
        Args:
            exit_price: Optional exit price (uses last_price if None)
            
        Returns:
            P&L as a percentage of the initial position value
        """
        if not self.entry_price:
            return Decimal("0")
            
        price = exit_price or self.last_price or self.entry_price
        position_value = self.entry_price * self.size
        
        if position_value == Decimal("0"):
            return Decimal("0")
            
        pnl = self.calculate_unrealized_pnl(price)
        return (pnl / position_value) * Decimal("100")
    
    def should_take_profit(self, current_price: Decimal) -> bool:
        """Check if the take-profit level has been reached.
        
        Args:
            current_price: Current market price
            
        Returns:
            True if take-profit level is reached, False otherwise
        """
        if not self.take_profit_price:
            return False
            
        if self.position_type == PositionType.LONG:
            return current_price >= self.take_profit_price
        else:
            return current_price <= self.take_profit_price
    
    def should_stop_loss(self, current_price: Decimal) -> bool:
        """Check if the stop-loss level has been reached.
        
        Args:
            current_price: Current market price
            
        Returns:
            True if stop-loss level is reached, False otherwise
        """
        if not self.stop_loss_price:
            return False
            
        if self.position_type == PositionType.LONG:
            return current_price <= self.stop_loss_price
        else:
            return current_price >= self.stop_loss_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to a dictionary.
        
        Returns:
            Dictionary representation of this position
        """
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "position_type": self.position_type.value,
            "size": str(self.size),
            "entry_price": str(self.entry_price),
            "take_profit_price": str(self.take_profit_price) if self.take_profit_price else None,
            "stop_loss_price": str(self.stop_loss_price) if self.stop_loss_price else None,
            "strategy_id": self.strategy_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl else None,
            "fee_paid": str(self.fee_paid),
            "entry_order_id": self.entry_order_id,
            "exit_order_id": self.exit_order_id,
            "max_profit": str(self.max_profit),
            "max_loss": str(self.max_loss),
            "last_price": str(self.last_price) if self.last_price else None,
            "last_price_timestamp": self.last_price_timestamp.isoformat() if self.last_price_timestamp else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create a Position object from a dictionary.
        
        Args:
            data: Dictionary containing position data
            
        Returns:
            Position object
        """
        position = cls(
            symbol=data["symbol"],
            position_type=PositionType(data["position_type"]),
            size=Decimal(data["size"]),
            entry_price=Decimal(data["entry_price"]),
            take_profit_price=Decimal(data["take_profit_price"]) if data.get("take_profit_price") else None,
            stop_loss_price=Decimal(data["stop_loss_price"]) if data.get("stop_loss_price") else None,
            strategy_id=data.get("strategy_id"),
            position_id=data.get("position_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None,
            metadata=data.get("metadata", {})
        )
        
        position.status = PositionStatus(data["status"]) if "status" in data else PositionStatus.PENDING
        position.exit_price = Decimal(data["exit_price"]) if data.get("exit_price") else None
        position.exit_timestamp = datetime.fromisoformat(data["exit_timestamp"]) if data.get("exit_timestamp") else None
        position.realized_pnl = Decimal(data["realized_pnl"]) if data.get("realized_pnl") else None
        position.fee_paid = Decimal(data["fee_paid"]) if data.get("fee_paid") else Decimal("0")
        position.entry_order_id = data.get("entry_order_id")
        position.exit_order_id = data.get("exit_order_id")
        position.max_profit = Decimal(data["max_profit"]) if data.get("max_profit") else Decimal("0")
        position.max_loss = Decimal(data["max_loss"]) if data.get("max_loss") else Decimal("0")
        position.last_price = Decimal(data["last_price"]) if data.get("last_price") else None
        
        if data.get("last_price_timestamp"):
            position.last_price_timestamp = datetime.fromisoformat(data["last_price_timestamp"])
        
        return position


class RiskParameters:
    """Risk management parameters for portfolio management."""
    
    def __init__(
        self,
        max_position_size: Decimal,
        max_risk_per_trade_pct: Decimal,
        max_risk_per_day_pct: Decimal,
        max_open_positions: int,
        max_open_positions_per_symbol: int,
        max_drawdown_pct: Decimal
    ):
        """Initialize risk parameters.
        
        Args:
            max_position_size: Maximum position size as a percentage of portfolio
            max_risk_per_trade_pct: Maximum risk per trade as percentage of portfolio
            max_risk_per_day_pct: Maximum risk per day as percentage of portfolio
            max_open_positions: Maximum number of open positions allowed
            max_open_positions_per_symbol: Maximum open positions per symbol
            max_drawdown_pct: Maximum allowed drawdown percentage
        """
        self.max_position_size = max_position_size
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_risk_per_day_pct = max_risk_per_day_pct
        self.max_open_positions = max_open_positions
        self.max_open_positions_per_symbol = max_open_positions_per_symbol
        self.max_drawdown_pct = max_drawdown_pct
    
    @classmethod
    def default(cls) -> 'RiskParameters':
        """Create default risk parameters.
        
        Returns:
            Default risk parameters
        """
        return cls(
            max_position_size=Decimal("0.1"),  # 10% of portfolio
            max_risk_per_trade_pct=Decimal("0.01"),  # 1% risk per trade
            max_risk_per_day_pct=Decimal("0.05"),  # 5% risk per day
            max_open_positions=5,
            max_open_positions_per_symbol=2,
            max_drawdown_pct=Decimal("0.2")  # 20% max drawdown
        )


class PortfolioManager:
    """Main class for portfolio management.
    
    The PortfolioManager is responsible for:
    - Tracking portfolio value and allocation
    - Managing positions (opening, closing, tracking)
    - Enforcing risk management rules
    - Calculating performance metrics
    - Rebalancing and optimizing the portfolio
    """
    
    def __init__(
        self,
        initial_balance: Decimal,
        risk_parameters: Optional[RiskParameters] = None,
        name: str = "main"
    ):
        """Initialize the portfolio manager.
        
        Args:
            initial_balance: Initial portfolio balance in base currency
            risk_parameters: Risk management parameters
            name: Name of this portfolio
        """
        self.name = name
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_parameters = risk_parameters or RiskParameters.default()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.open_positions: Set[str] = set()  # Set of open position_ids
        self.open_positions_by_symbol: Dict[str, Set[str]] = {}  # symbol -> Set of position_ids
        self.open_positions_by_strategy: Dict[str, Set[str]] = {}  # strategy_id -> Set of position_ids
        
        # Performance tracking
        self.highest_balance = initial_balance
        self.lowest_balance = initial_balance
        self.total_realized_pnl = Decimal("0")
        self.daily_pnl: Dict[str, Decimal] = {}  # date string -> daily P&L
        
        # Logger
        self.logger = get_logger("portfolio", f"portfolio_manager_{name}")
        self.logger.info(f"Initialized portfolio manager '{name}' with balance: {initial_balance}")
    
    async def process_signal(self, signal: Signal) -> Optional[Position]:
        """Process a trading signal and potentially open a position.
        
        Args:
            signal: Trading signal from a strategy
            
        Returns:
            Position object if a position was opened, None otherwise
        """
        if signal.signal_type != SignalType.ENTRY:
            self.logger.debug(f"Ignoring non-entry signal: {signal.signal_type}")
            return None
            
        # Check if we can open a new position
        if not self._can_open_position(signal.symbol, signal.strategy_id):
            self.logger.info(f"Cannot open position for {signal.symbol}: risk limits reached")
            return None
        
        # Determine position size
        position_size = self._calculate_position_size(signal)
        if position_size <= Decimal("0"):
            self.logger.info(f"Calculated position size too small for {signal.symbol}")
            return None
        
        # Determine position direction
        position_type = PositionType.LONG if signal.direction == "long" else PositionType.SHORT
        
        # Calculate stop loss and take profit levels
        stop_loss, take_profit = self._calculate_exit_levels(signal)
        
        # Create the position
        position = Position(
            symbol=signal.symbol,
            position_type=position_type,
            size=position_size,
            entry_price=Decimal(str(signal.price)),
            take_profit_price=take_profit,
            stop_loss_price=stop_loss,
            strategy_id=signal.strategy_id,
            timestamp=signal.timestamp,
            metadata={
                "signal_confidence": signal.confidence,
                "signal_reason": signal.reason,
                "signal_metadata": signal.metadata
            }
        )
        
        # Record the position
        self.positions[position.position_id] = position
        self._add_to_open_positions(position)
        
        self.logger.info(
            f"Opened {position_type.value} position {position.position_id} for {signal.symbol} "
            f"at {position.entry_price} with size {position_size}"
        )
        
        return position
    
    def update_position(
        self, 
        position_id: str, 
        status: Optional[PositionStatus] = None,
        current_price: Optional[Decimal] = None,
        exit_price: Optional[Decimal] = None,
        fee_paid: Optional[Decimal] = None
    ) -> None:
        """Update a position's status, price, or other attributes.
        
        Args:
            position_id: ID of the position to update
            status: New status for the position
            current_price: Current market price
            exit_price: Exit price for closing positions
            fee_paid: Additional fees paid for this transaction
        """
        if position_id not in self.positions:
            self.logger.warning(f"Position not found: {position_id}")
            return
            
        position = self.positions[position_id]
        
        # Update status if provided
        if status:
            position.update_status(status)
            
            # Handle status changes
            if status == PositionStatus.CLOSED or status == PositionStatus.CANCELED:
                self._remove_from_open_positions(position)
        
        # Update current price if provided
        if current_price:
            position.update_price(current_price)
            
        # Close position if exit price is provided
        if exit_price:
            position.close(exit_price, fee_paid=fee_paid)
            self._remove_from_open_positions(position)
            
            # Update portfolio balance
            if position.realized_pnl:
                self.current_balance += position.realized_pnl
                self.total_realized_pnl += position.realized_pnl
                
                # Update high/low water marks
                if self.current_balance > self.highest_balance:
                    self.highest_balance = self.current_balance
                elif self.current_balance < self.lowest_balance:
                    self.lowest_balance = self.current_balance
                
                # Update daily P&L
                date_str = position.exit_timestamp.strftime("%Y-%m-%d")
                if date_str not in self.daily_pnl:
                    self.daily_pnl[date_str] = Decimal("0")
                self.daily_pnl[date_str] += position.realized_pnl
                
                self.logger.info(
                    f"Closed position {position_id} with P&L: {position.realized_pnl} "
                    f"({position.calculate_pnl_pct(exit_price):.2f}%)"
                )
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """Check if any risk limits have been breached.
        
        Returns:
            Dictionary of risk limit status
        """
        # Calculate current drawdown
        if self.highest_balance == Decimal("0"):
            current_drawdown_pct = Decimal("0")
        else:
            current_drawdown_pct = (
                (self.highest_balance - self.current_balance) / self.highest_balance
            ) * Decimal("100")
        
        # Check drawdown limit
        drawdown_limit_breached = current_drawdown_pct > self.risk_parameters.max_drawdown_pct
        
        # Check position count limits
        total_positions_limit_breached = len(self.open_positions) >= self.risk_parameters.max_open_positions
        
        # Check per-symbol position limits
        symbol_limits_breached = []
        for symbol, positions in self.open_positions_by_symbol.items():
            if len(positions) > self.risk_parameters.max_open_positions_per_symbol:
                symbol_limits_breached.append(symbol)
        
        # Calculate daily risk used
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_risk_used = self.daily_pnl.get(today_str, Decimal("0"))
        daily_risk_pct = (daily_risk_used / self.initial_balance) * Decimal("100")
        daily_risk_limit_breached = abs(daily_risk_pct) > self.risk_parameters.max_risk_per_day_pct
        
        return {
            "drawdown_pct": current_drawdown_pct,
            "drawdown_limit_breached": drawdown_limit_breached,
            "total_positions": len(self.open_positions),
            "total_positions_limit_breached": total_positions_limit_breached,
            "symbol_limits_breached": symbol_limits_breached,
            "daily_risk_pct": daily_risk_pct,
            "daily_risk_limit_breached": daily_risk_limit_breached
        }
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get the current state of the portfolio.
        
        Returns:
            Dictionary containing portfolio state information
        """
        # Calculate unrealized P&L
        unrealized_pnl = Decimal("0")
        for position_id in self.open_positions:
            position = self.positions[position_id]
            if position.last_price:
                unrealized_pnl += position.calculate_unrealized_pnl(position.last_price)
        
        # Calculate total value
        total_value = self.current_balance + unrealized_pnl
        
        # Calculate total return
        if self.initial_balance == Decimal("0"):
            total_return_pct = Decimal("0")
        else:
            total_return_pct = ((total_value - self.initial_balance) / self.initial_balance) * Decimal("100")
        
        # Calculate drawdown
        if self.highest_balance == Decimal("0"):
            current_drawdown_pct = Decimal("0")
        else:
            current_drawdown_pct = (
                (self.highest_balance - total_value) / self.highest_balance
            ) * Decimal("100")
        
        return {
            "name": self.name,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "unrealized_pnl": unrealized_pnl,
            "total_value": total_value,
            "total_return_pct": total_return_pct,
            "realized_pnl": self.total_realized_pnl,
            "open_positions_count": len(self.open_positions),
            "total_positions_count": len(self.positions),
            "drawdown_pct": current_drawdown_pct,
            "highest_balance": self.highest_balance,
            "lowest_balance": self.lowest_balance
        }
    
    def get_open_positions(self) -> List[Position]:
        """Get all currently open positions.
        
        Returns:
            List of open Position objects
        """
        return [self.positions[position_id] for position_id in self.open_positions]
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of Position objects for the specified symbol
        """
        position_ids = self.open_positions_by_symbol.get(symbol, set())
        return [self.positions[position_id] for position_id in position_ids]
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Position]:
        """Get all positions created by a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of Position objects created by the specified strategy
        """
        position_ids = self.open_positions_by_strategy.get(strategy_id, set())
        return [self.positions[position_id] for position_id in position_ids]
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by its ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position object if found, None otherwise
        """
        return self.positions.get(position_id)
    
    def _add_to_open_positions(self, position: Position) -> None:
        """Add a position to the open positions tracking.
        
        Args:
            position: Position to add
        """
        # Add to open positions set
        self.open_positions.add(position.position_id)
        
        # Add to per-symbol tracking
        if position.symbol not in self.open_positions_by_symbol:
            self.open_positions_by_symbol[position.symbol] = set()
        self.open_positions_by_symbol[position.symbol].add(position.position_id)
        
        # Add to per-strategy tracking
        if position.strategy_id:
            if position.strategy_id not in self.open_positions_by_strategy:
                self.open_positions_by_strategy[position.strategy_id] = set()
            self.open_positions_by_strategy[position.strategy_id].add(position.position_id)
    
    def _remove_from_open_positions(self, position: Position) -> None:
        """Remove a position from the open positions tracking.
        
        Args:
            position: Position to remove
        """
        # Remove from open positions set
        if position.position_id in self.open_positions:
            self.open_positions.remove(position.position_id)
        
        # Remove from per-symbol tracking
        if position.symbol in self.open_positions_by_symbol:
            if position.position_id in self.open_positions_by_symbol[position.symbol]:
                self.open_positions_by_symbol[position.symbol].remove(position.position_id)
        
        # Remove from per-strategy tracking
        if position.strategy_id:
            if position.strategy_id in self.open_positions_by_strategy:
                if position.position_id in self.open_positions_by_strategy[position.strategy_id]:
                    self.open_positions_by_strategy[position.strategy_id].remove(position.position_id)
    
    def _can_open_position(self, symbol: str, strategy_id: Optional[str] = None) -> bool:
        """Check if a new position can be opened based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            strategy_id: Optional strategy identifier
            
        Returns:
            True if a new position can be opened, False otherwise
        """
        # Check overall open position limit
        if len(self.open_positions) >= self.risk_parameters.max_open_positions:
            self.logger.debug(f"Maximum open positions limit reached: {len(self.open_positions)}")
            return False
        
        # Check per-symbol position limit
        symbol_positions = self.open_positions_by_symbol.get(symbol, set())
        if len(symbol_positions) >= self.risk_parameters.max_open_positions_per_symbol:
            self.logger.debug(f"Maximum positions per symbol reached for {symbol}: {len(symbol_positions)}")
            return False
        
        # Check daily risk limit
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_risk_used = self.daily_pnl.get(today_str, Decimal("0"))
        if daily_risk_used < Decimal("0"):  # Only consider losses
            daily_risk_pct = abs(daily_risk_used) / self.initial_balance
            if daily_risk_pct >= self.risk_parameters.max_risk_per_day_pct:
                self.logger.debug(f"Daily risk limit reached: {daily_risk_pct:.2%}")
                return False
        
        # Check drawdown limit
        if self.highest_balance > Decimal("0"):
            current_drawdown = (self.highest_balance - self.current_balance) / self.highest_balance
            if current_drawdown >= self.risk_parameters.max_drawdown_pct:
                self.logger.debug(f"Maximum drawdown reached: {current_drawdown:.2%}")
                return False
        
        return True
    
    def _calculate_position_size(self, signal: Signal) -> Decimal:
        """Calculate the appropriate position size based on the signal and risk parameters.
        
        Args:
            signal: Trading signal from a strategy
            
        Returns:
            Position size in base currency units
        """
        # If signal includes a suggested position size, use it as a starting point
        suggested_size = Decimal(str(signal.metadata.get("suggested_size", 0)))
        
        # Calculate maximum size based on portfolio value and risk parameters
        max_size_by_portfolio = self.current_balance * self.risk_parameters.max_position_size
        
        # Calculate size based on risk per trade
        # This would typically use the stop loss to determine the maximum size
        # that would risk max_risk_per_trade_pct of the portfolio
        risk_amount = self.current_balance * self.risk_parameters.max_risk_per_trade_pct
        
        stop_loss = signal.metadata.get("stop_loss_price")
        if stop_loss is not None and signal.price != 0:
            stop_loss = Decimal(str(stop_loss))
            price = Decimal(str(signal.price))
            
            # Calculate risk per unit
            if signal.direction == "long":
                risk_per_unit = price - stop_loss
            else:
                risk_per_unit = stop_loss - price
            
            # If stop loss is invalid, use a default 1% risk
            if risk_per_unit <= Decimal("0"):
                risk_per_unit = price * Decimal("0.01")
            
            # Calculate size based on risk amount
            size_by_risk = risk_amount / risk_per_unit
        else:
            # If no stop loss provided, use a default 1% risk
            size_by_risk = risk_amount / (Decimal(str(signal.price)) * Decimal("0.01"))
        
        # Use the smallest of the calculated sizes
        if suggested_size > Decimal("0"):
            return min(suggested_size, max_size_by_portfolio, size_by_risk)
        else:
            return min(max_size_by_portfolio, size_by_risk)
    
    def _calculate_exit_levels(self, signal: Signal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate stop loss and take profit levels based on the signal.
        
        Args:
            signal: Trading signal from a strategy
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        price = Decimal(str(signal.price))
        
        # Check if signal includes suggested levels
        stop_loss = signal.metadata.get("stop_loss_price")
        take_profit = signal.metadata.get("take_profit_price")
        
        if stop_loss is not None:
            stop_loss = Decimal(str(stop_loss))
        else:
            # Default stop loss at 1% from entry
            if signal.direction == "long":
                stop_loss = price * Decimal("0.99")
            else:
                stop_loss = price * Decimal("1.01")
        
        if take_profit is not None:
            take_profit = Decimal(str(take_profit))
        else:
            # Default take profit at 2% from entry
            if signal.direction == "long":
                take_profit = price * Decimal("1.02")
            else:
                take_profit = price * Decimal("0.98")
        
        return stop_loss, take_profit
    
    async def update_prices(self, symbol: str, price: Decimal) -> None:
        """Update all open positions with the latest price.
        
        Args:
            symbol: Trading pair symbol
            price: Current market price
        """
        position_ids = self.open_positions_by_symbol.get(symbol, set())
        for position_id in position_ids:
            position = self.positions[position_id]
            position.update_price(price)
            
            # Check for stop loss or take profit triggers
            if position.status == PositionStatus.OPEN:
                if position.should_stop_loss(price):
                    self.logger.info(f"Stop loss triggered for position {position_id} at {price}")
                    position.close(price)
                    self._remove_from_open_positions(position)
                    
                    # Update portfolio balance
                    if position.realized_pnl:
                        self.current_balance += position.realized_pnl
                        self.total_realized_pnl += position.realized_pnl
                
                elif position.should_take_profit(price):
                    self.logger.info(f"Take profit triggered for position {position_id} at {price}")
                    position.close(price)
                    self._remove_from_open_positions(position)
                    
                    # Update portfolio balance
                    if position.realized_pnl:
                        self.current_balance += position.realized_pnl
                        self.total_realized_pnl += position.realized_pnl 