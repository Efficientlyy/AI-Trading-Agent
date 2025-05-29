"""
Execution Handler for AI Trading Agent.

This module provides an execution handler that manages order execution
through different exchange connectors and simulates market conditions
for backtesting.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import uuid

from ..common import get_logger
from .order import Order, OrderStatus, OrderType, OrderSide

logger = get_logger(__name__)


class ExecutionHandler:
    """
    Execution Handler that manages order execution.
    
    This class handles sending orders to exchanges, tracking order status,
    and simulating market conditions for backtesting.
    
    Attributes:
        exchange_connectors (Dict[str, Any]): Exchange connectors
        open_orders (Dict[str, Order]): Open orders by client_order_id
        filled_orders (Dict[str, Order]): Filled orders by client_order_id
        rejected_orders (Dict[str, Order]): Rejected orders by client_order_id
        order_history (List[Order]): Complete order history
    """
    
    def __init__(self, exchange_connectors: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution handler.
        
        Args:
            exchange_connectors (Dict[str, Any], optional): Map of exchange name to connector
        """
        self.exchange_connectors = exchange_connectors or {}
        self.open_orders = {}
        self.filled_orders = {}
        self.rejected_orders = {}
        self.order_history = []
        
        # Callbacks for order events
        self.callbacks = {
            "order_created": [],
            "order_filled": [],
            "order_rejected": [],
            "order_canceled": [],
        }
        
        logger.info("ExecutionHandler initialized")
    
    def add_exchange_connector(self, name: str, connector: Any) -> None:
        """
        Add an exchange connector.
        
        Args:
            name (str): Exchange name
            connector (Any): Exchange connector instance
        """
        self.exchange_connectors[name] = connector
        logger.info(f"Added {name} exchange connector")
    
    async def submit_order(self, order: Order, exchange: str = "default") -> Order:
        """
        Submit an order to an exchange.
        
        Args:
            order (Order): Order to submit
            exchange (str, optional): Exchange to submit to. Defaults to "default".
            
        Returns:
            Order: Updated order with exchange response
        """
        if exchange not in self.exchange_connectors:
            order.status = OrderStatus.REJECTED
            order.additional_params["reject_reason"] = f"Exchange {exchange} not found"
            self.rejected_orders[order.client_order_id] = order
            self._notify_callbacks("order_rejected", order)
            logger.error(f"Order rejected: Exchange {exchange} not found")
            return order
        
        try:
            # Submit order to exchange
            connector = self.exchange_connectors[exchange]
            response = await connector.create_order(order)
            
            # Update order with exchange response
            updated_order = self._process_order_response(order, response)
            
            # Add to open orders if not immediately filled or rejected
            if updated_order.status in [OrderStatus.NEW, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                self.open_orders[updated_order.client_order_id] = updated_order
                self._notify_callbacks("order_created", updated_order)
            
            # Add to order history
            self.order_history.append(updated_order)
            
            return updated_order
            
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.additional_params["reject_reason"] = str(e)
            self.rejected_orders[order.client_order_id] = order
            self._notify_callbacks("order_rejected", order)
            return order
    
    def _process_order_response(self, original_order: Order, response: Dict[str, Any]) -> Order:
        """
        Process exchange response for an order.
        
        Args:
            original_order (Order): Original order
            response (Dict[str, Any]): Exchange response
            
        Returns:
            Order: Updated order
        """
        # Make a copy of the original order
        order = original_order
        
        # Update order with exchange response
        if "orderId" in response:
            order.exchange_order_id = response["orderId"]
        
        if "status" in response:
            status_map = {
                "NEW": OrderStatus.NEW,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "FILLED": OrderStatus.FILLED,
                "CANCELED": OrderStatus.CANCELED,
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.EXPIRED
            }
            status = response["status"].upper()
            if status in status_map:
                order.status = status_map[status]
        
        if "executedQty" in response:
            order.filled_quantity = float(response["executedQty"])
            order.remaining_quantity = order.quantity - order.filled_quantity
        
        if "avgPrice" in response:
            order.average_price = float(response["avgPrice"])
        
        return order
    
    async def cancel_order(self, order: Order, exchange: str = "default") -> Order:
        """
        Cancel an order.
        
        Args:
            order (Order): Order to cancel
            exchange (str, optional): Exchange to cancel on. Defaults to "default".
            
        Returns:
            Order: Updated order
        """
        if exchange not in self.exchange_connectors:
            order.additional_params["cancel_error"] = f"Exchange {exchange} not found"
            logger.error(f"Cannot cancel order: Exchange {exchange} not found")
            return order
        
        try:
            # Cancel order on exchange
            connector = self.exchange_connectors[exchange]
            response = await connector.cancel_order(order)
            
            # Update order with exchange response
            updated_order = self._process_order_response(order, response)
            
            # Remove from open orders if canceled
            if updated_order.status == OrderStatus.CANCELED:
                if updated_order.client_order_id in self.open_orders:
                    del self.open_orders[updated_order.client_order_id]
                self._notify_callbacks("order_canceled", updated_order)
            
            return updated_order
            
        except Exception as e:
            logger.error(f"Error canceling order: {str(e)}")
            order.additional_params["cancel_error"] = str(e)
            return order
    
    def get_open_orders(self) -> List[Order]:
        """
        Get all open orders.
        
        Returns:
            List[Order]: Open orders
        """
        return list(self.open_orders.values())
    
    def get_filled_orders(self) -> List[Order]:
        """
        Get all filled orders.
        
        Returns:
            List[Order]: Filled orders
        """
        return list(self.filled_orders.values())
    
    def get_rejected_orders(self) -> List[Order]:
        """
        Get all rejected orders.
        
        Returns:
            List[Order]: Rejected orders
        """
        return list(self.rejected_orders.values())
    
    def get_order_history(self) -> List[Order]:
        """
        Get complete order history.
        
        Returns:
            List[Order]: Order history
        """
        return self.order_history
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for order events.
        
        Args:
            event (str): Event type (order_created, order_filled, order_rejected, order_canceled)
            callback (Callable): Callback function
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.info(f"Registered callback for {event} event")
        else:
            logger.warning(f"Unknown event type: {event}")
    
    def _notify_callbacks(self, event: str, order: Order) -> None:
        """
        Notify callbacks of an order event.
        
        Args:
            event (str): Event type
            order (Order): Order object
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in {event} callback: {str(e)}")
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        time_in_force: str = "GTC",
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create a new order object.
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): Order side
            order_type (OrderType): Order type
            quantity (float): Order quantity
            price (float, optional): Order price. Required for limit orders.
            stop_price (float, optional): Stop price for stop orders.
            client_order_id (str, optional): Client order ID. Generated if not provided.
            time_in_force (str, optional): Time in force. Defaults to "GTC".
            additional_params (Dict[str, Any], optional): Additional parameters.
            
        Returns:
            Order: New order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            client_order_id=client_order_id,
            status=OrderStatus.NEW,
            time_in_force=time_in_force,
            additional_params=additional_params
        )
        
        return order