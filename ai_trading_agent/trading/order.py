"""
Order class and related enums for AI Trading Agent.

This module defines the Order class and related enums for use
throughout the trading system. It provides a standardized interface
for creating and managing orders.
"""

from enum import Enum
from typing import Dict, Any, Optional
import uuid
from datetime import datetime


class OrderSide(str, Enum):
    """Order side enum (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP = "stop"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Order status enum."""
    NEW = "new"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order:
    """
    Order class for trading operations.
    
    This class represents an order in the trading system, with all
    relevant information such as symbol, side, type, quantity, price, etc.
    """
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        exchange_order_id: Optional[str] = None,
        status: OrderStatus = OrderStatus.NEW,
        filled_quantity: float = 0.0,
        average_price: Optional[float] = None,
        create_time: Optional[datetime] = None,
        update_time: Optional[datetime] = None,
        time_in_force: str = "GTC",  # Good Till Cancel
        additional_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an Order.
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): Order side (buy or sell)
            order_type (OrderType): Order type
            quantity (float): Order quantity
            price (float, optional): Order price. Required for limit orders.
            stop_price (float, optional): Stop price for stop orders.
            client_order_id (str, optional): Client order ID. Generated if not provided.
            exchange_order_id (str, optional): Exchange order ID. Filled after submission.
            status (OrderStatus, optional): Order status. Defaults to NEW.
            filled_quantity (float, optional): Filled quantity. Defaults to 0.
            average_price (float, optional): Average fill price. Defaults to None.
            create_time (datetime, optional): Creation time. Defaults to now.
            update_time (datetime, optional): Last update time. Defaults to now.
            time_in_force (str, optional): Time in force. Defaults to "GTC".
            additional_params (Dict[str, Any], optional): Additional parameters.
        """
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.client_order_id = client_order_id or f"algo_{uuid.uuid4().hex[:16]}"
        self.exchange_order_id = exchange_order_id
        self.status = status
        self.filled_quantity = filled_quantity
        self.average_price = average_price
        self.create_time = create_time or datetime.now()
        self.update_time = update_time or datetime.now()
        self.time_in_force = time_in_force
        self.additional_params = additional_params or {}
        
        # Calculate remaining quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
    
    def update(self, data: Dict[str, Any]) -> None:
        """
        Update order with new data.
        
        Args:
            data (Dict[str, Any]): New order data
        """
        # Update order data
        if "status" in data:
            self.status = OrderStatus(data["status"])
        if "filled_quantity" in data:
            self.filled_quantity = float(data["filled_quantity"])
            self.remaining_quantity = self.quantity - self.filled_quantity
        if "average_price" in data:
            self.average_price = float(data["average_price"])
        if "exchange_order_id" in data:
            self.exchange_order_id = data["exchange_order_id"]
        if "update_time" in data:
            self.update_time = data["update_time"]
        
        # Update additional parameters
        for key, value in data.items():
            if key not in [
                "status", "filled_quantity", "average_price",
                "exchange_order_id", "update_time"
            ]:
                self.additional_params[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order to dictionary.
        
        Returns:
            Dict[str, Any]: Order data as dictionary
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "create_time": self.create_time.isoformat() if self.create_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
            "time_in_force": self.time_in_force,
            **self.additional_params
        }
    
    def is_active(self) -> bool:
        """
        Check if order is active.
        
        Returns:
            bool: True if order is active, False otherwise
        """
        return self.status in [OrderStatus.NEW, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
    
    def is_complete(self) -> bool:
        """
        Check if order is complete.
        
        Returns:
            bool: True if order is complete, False otherwise
        """
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]