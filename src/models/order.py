"""Order models for the trading system.

This module defines the Order class and related enums for representing trading orders.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class OrderType(str, Enum):
    """Type of order."""
    MARKET = "market"  # Market order (executed immediately at market price)
    LIMIT = "limit"    # Limit order (executed at specified price or better)
    STOP = "stop"      # Stop order (converted to market order when price reaches stop price)
    STOP_LIMIT = "stop_limit"  # Stop-limit order (converted to limit order when price reaches stop price)
    TRAILING_STOP = "trailing_stop"  # Trailing stop order (stop price moves with market)


class OrderSide(str, Enum):
    """Side of an order."""
    BUY = "buy"      # Buy order
    SELL = "sell"    # Sell order


class TimeInForce(str, Enum):
    """Time in force for an order."""
    GTC = "gtc"    # Good Till Cancelled
    IOC = "ioc"    # Immediate Or Cancel
    FOK = "fok"    # Fill Or Kill
    DAY = "day"    # Day order (valid until end of day)


class OrderStatus(str, Enum):
    """Status of an order."""
    CREATED = "created"       # Order created but not yet submitted
    PENDING = "pending"       # Order submitted but not yet acknowledged
    OPEN = "open"             # Order is open on the exchange
    PARTIALLY_FILLED = "partially_filled"  # Order is partially filled
    FILLED = "filled"         # Order is completely filled
    CANCELLED = "cancelled"   # Order was cancelled
    REJECTED = "rejected"     # Order was rejected by the exchange
    EXPIRED = "expired"       # Order expired according to time in force


class Order(BaseModel):
    """Model representing a trading order."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    exchange: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None  # Required for limit orders
    stop_price: Optional[float] = None  # Required for stop orders
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.CREATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    fees: Dict[str, float] = Field(default_factory=dict)  # Currency -> amount
    is_post_only: bool = False
    is_reduce_only: bool = False
    position_id: Optional[str] = None  # Reference to the position this order is for
    strategy_id: Optional[str] = None  # Reference to the strategy that created this order
    metadata: Dict = Field(default_factory=dict)
    error_message: Optional[str] = None  # Error message if order was rejected
    
    def is_active(self) -> bool:
        """Check if the order is still active on the exchange.
        
        Returns:
            bool: True if the order is active, False otherwise
        """
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING)
    
    def is_complete(self) -> bool:
        """Check if the order is complete (filled or cancelled).
        
        Returns:
            bool: True if the order is complete, False otherwise
        """
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)
    
    def remaining_quantity(self) -> float:
        """Calculate the remaining quantity to be filled.
        
        Returns:
            float: The remaining quantity
        """
        return max(0.0, self.quantity - self.filled_quantity)
    
    def fill_percent(self) -> float:
        """Calculate the percentage of the order that has been filled.
        
        Returns:
            float: The fill percentage (0-100)
        """
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100.0
    
    def update_status(self, new_status: OrderStatus, filled_qty: Optional[float] = None,
                     avg_price: Optional[float] = None, exchange_id: Optional[str] = None,
                     error: Optional[str] = None) -> None:
        """Update the order status and fill information.
        
        Args:
            new_status: The new order status
            filled_qty: The total filled quantity (if available)
            avg_price: The average fill price (if available)
            exchange_id: The exchange order ID (if available)
            error: Error message (if order was rejected)
        """
        self.status = new_status
        self.updated_at = datetime.utcnow()
        
        if filled_qty is not None:
            self.filled_quantity = filled_qty
            
        if avg_price is not None:
            self.average_fill_price = avg_price
            
        if exchange_id is not None:
            self.exchange_order_id = exchange_id
            
        if error is not None:
            self.error_message = error
            
        # Update status based on fill quantity
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0 and self.filled_quantity < self.quantity:
            self.status = OrderStatus.PARTIALLY_FILLED 