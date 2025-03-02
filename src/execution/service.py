"""Execution service for the AI Crypto Trading System.

This module defines the ExecutionService class, which is responsible for
executing orders on exchanges and managing their lifecycle.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.common.security import get_api_key_manager, ApiCredential
from src.models.events import ErrorEvent, SystemStatusEvent, OrderEvent
from src.models.order import Order, OrderStatus, OrderType, OrderSide
from src.models.position import Position, PositionSide, PositionStatus
from src.execution.exchange import BaseExchangeConnector, MockExchangeConnector


class ExecutionService(Component):
    """Service for executing orders on exchanges.
    
    This component handles the order lifecycle, from creation to submission
    to monitoring for fills and cancellations.
    """
    
    def __init__(self):
        """Initialize the execution service."""
        super().__init__("execution")
        self.logger = get_logger("execution", "service")
        self.orders: Dict[str, Order] = {}  # All orders (active and inactive)
        self.active_orders: Set[str] = set()  # Set of active order IDs
        self.exchange_connectors: Dict[str, BaseExchangeConnector] = {}  # Exchange ID to connector
        
        # Configuration values
        self.retry_attempts = 3
        self.retry_delay = 1.0  # seconds
        self.order_update_interval = 1.0  # seconds
        
    async def _initialize(self) -> None:
        """Initialize the execution service."""
        self.logger.info("Initializing execution service")
        
        # Load configuration
        self.retry_attempts = self.get_config("retry_attempts", 3)
        self.retry_delay = self.get_config("retry_delay", 1.0)
        self.order_update_interval = self.get_config("order_update_interval", 1.0)
        
        # Initialize exchange connectors
        await self._initialize_exchange_connectors()
        
        self.logger.info("Execution service configuration loaded",
                       retry_attempts=self.retry_attempts,
                       retry_delay=self.retry_delay,
                       order_update_interval=self.order_update_interval)
    
    async def _initialize_exchange_connectors(self) -> None:
        """Initialize exchange connectors based on configuration."""
        exchanges_config = self.get_config("exchanges", {})
        
        # Get the API key manager
        api_key_manager = get_api_key_manager()
        
        for exchange_id, exchange_config in exchanges_config.items():
            connector_type = exchange_config.get("type", "mock")
            
            # Try to get credentials from the API key manager first
            credential = api_key_manager.get_credential(exchange_id)
            
            # If no credentials found, fall back to config
            if credential is None:
                api_key = exchange_config.get("api_key")
                api_secret = exchange_config.get("api_secret")
                
                # Log info about where we're getting credentials
                self.logger.info(f"Using API credentials from config for {exchange_id}")
            else:
                api_key = credential.key
                api_secret = credential.secret
                
                # Log info about where we're getting credentials
                self.logger.info(f"Using API credentials from secure storage for {exchange_id}")
            
            self.logger.info(f"Initializing connector for exchange: {exchange_id}")
            
            try:
                # Create the appropriate connector based on type
                if connector_type == "mock":
                    # Use mock connector for testing
                    connector = MockExchangeConnector(
                        exchange_id=exchange_id,
                        api_key=api_key,
                        api_secret=api_secret,
                        is_paper_trading=exchange_config.get("paper_trading", True),
                        latency_ms=exchange_config.get("simulated_latency_ms", 50),
                        fill_probability=exchange_config.get("fill_probability", 0.9),
                        price_volatility=exchange_config.get("price_volatility", 0.002)
                    )
                elif connector_type == "binance":
                    # Future implementation will add real exchange connectors
                    self.logger.warning(f"Binance connector not yet implemented, using mock for {exchange_id}")
                    connector = MockExchangeConnector(exchange_id=exchange_id, api_key=api_key, api_secret=api_secret)
                else:
                    self.logger.error(f"Unknown connector type: {connector_type}")
                    continue
                
                # Initialize the connector
                success = await connector.initialize()
                if success:
                    self.exchange_connectors[exchange_id] = connector
                    self.logger.info(f"Successfully initialized connector for {exchange_id}")
                else:
                    self.logger.error(f"Failed to initialize connector for {exchange_id}")
            
            except Exception as e:
                self.logger.error(f"Error initializing connector for {exchange_id}: {str(e)}")
    
    async def _start(self) -> None:
        """Start the execution service."""
        self.logger.info("Starting execution service")
        
        # Register event handlers
        event_bus.subscribe("OrderEvent", self._handle_order_event)
        
        # Start the order monitoring task
        self.create_task(self._monitor_orders())
        
        self.logger.info("Execution service started")
    
    async def _stop(self) -> None:
        """Stop the execution service."""
        self.logger.info("Stopping execution service")
        
        # Unregister event handlers
        event_bus.unsubscribe("OrderEvent", self._handle_order_event)
        
        # Cancel all active orders if configured to do so
        if self.get_config("cancel_orders_on_shutdown", True):
            await self._cancel_all_active_orders("System shutdown")
            
        # Shutdown exchange connectors
        for exchange_id, connector in self.exchange_connectors.items():
            try:
                self.logger.info(f"Shutting down connector for {exchange_id}")
                await connector.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down connector for {exchange_id}: {str(e)}")
        
        self.logger.info("Execution service stopped")
    
    async def _handle_order_event(self, event: 'OrderEvent') -> None:
        """Handle an order event.
        
        Args:
            event: The order event
        """
        self.logger.debug("Received order event", 
                        order_id=event.order_id, 
                        event_type=event.event_type)
        
        # Process the order event based on its type
        if event.event_type == "create":
            if event.order is not None:
                await self.create_order(event.order)
            else:
                self.logger.error("Cannot create order: order is None", order_id=event.order_id)
        elif event.event_type == "cancel":
            await self.cancel_order(event.order_id)
        elif event.event_type == "update":
            # Handle order updates (e.g., changing order parameters)
            pass
    
    async def create_order(self, order: Order) -> bool:
        """Create and submit a new order.
        
        Args:
            order: The order to create
            
        Returns:
            bool: True if the order was created successfully, False otherwise
        """
        if order.id in self.orders:
            self.logger.warning("Order already exists", order_id=order.id)
            return False
        
        # Store the order
        self.orders[order.id] = order
        
        # Submit the order to the exchange
        success = await self._submit_order(order)
        if not success:
            return False
        
        # Add to active orders
        self.active_orders.add(order.id)
        
        self.logger.info("Order created and submitted", 
                       order_id=order.id,
                       exchange=order.exchange,
                       symbol=order.symbol,
                       side=order.side.value,
                       order_type=order.order_type.value,
                       quantity=order.quantity,
                       price=order.price)
        
        return True
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            bool: True if the order was cancelled successfully, False otherwise
        """
        if order_id not in self.orders:
            self.logger.warning("Order not found", order_id=order_id)
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active():
            self.logger.warning("Order is not active", 
                              order_id=order_id, 
                              status=order.status.value)
            return False
        
        # Cancel the order on the exchange
        success = await self._cancel_order_on_exchange(order)
        if not success:
            return False
        
        # Remove from active orders
        if order_id in self.active_orders:
            self.active_orders.remove(order_id)
        
        self.logger.info("Order cancelled", 
                       order_id=order_id,
                       exchange=order.exchange,
                       symbol=order.symbol)
        
        return True
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by its ID.
        
        Args:
            order_id: The order ID
            
        Returns:
            The order, or None if not found
        """
        return self.orders.get(order_id)
    
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders.
        
        Returns:
            List of active orders
        """
        return [self.orders[order_id] for order_id in self.active_orders]
    
    async def _submit_order(self, order: Order) -> bool:
        """Submit an order to the exchange.
        
        Args:
            order: Order object to submit
            
        Returns:
            bool: True if the order was submitted successfully, False otherwise
        """
        # Store the order (in case it wasn't already)
        self.orders[order.id] = order
        
        # Submit the order to the exchange
        success = await self._submit_order_to_exchange(order)
        
        if success:
            # Add to active orders
            self.active_orders.add(order.id)
            
            # Publish an order event
            await event_bus.publish(OrderEvent(
                order_id=order.id,
                event_type="submitted",
                order=order,
                exchange=order.exchange,
                symbol=order.symbol,
                source="execution_service"
            ))
            
            return True
        else:
            # If submission failed, order should be in REJECTED status
            if order.id in self.active_orders:
                self.active_orders.remove(order.id)
                
            # Publish an order event
            await event_bus.publish(OrderEvent(
                order_id=order.id,
                event_type="rejected",
                order=order,
                exchange=order.exchange,
                symbol=order.symbol,
                source="execution_service"
            ))
            
            return False
    
    async def _cancel_order_on_exchange(self, order: Order) -> bool:
        """Cancel an order on the exchange.
        
        Args:
            order: Order to cancel
            
        Returns:
            bool: True if the order was cancelled successfully, False otherwise
        """
        if not order.exchange in self.exchange_connectors:
            self.logger.error("Exchange not found", exchange=order.exchange)
            return False
        
        if not order.exchange_order_id:
            self.logger.error("Cannot cancel order without exchange_order_id", order_id=order.id)
            return False
        
        connector = self.exchange_connectors[order.exchange]
        
        # Try to cancel the order with retries
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self.logger.debug(f"Cancelling order on {order.exchange}, attempt {attempt}",
                               order_id=order.id,
                               exchange_order_id=order.exchange_order_id)
                
                # Cancel the order on the exchange
                success, error = await connector.cancel_order(order.exchange_order_id, order.symbol)
                
                if success:
                    order.update_status(OrderStatus.CANCELLED)
                    
                    self.logger.info(f"Order cancelled on {order.exchange}",
                                   order_id=order.id,
                                   exchange_order_id=order.exchange_order_id)
                    
                    return True
                else:
                    self.logger.warning(f"Failed to cancel order on {order.exchange}",
                                      order_id=order.id,
                                      error=error,
                                      attempt=attempt)
            
            except Exception as e:
                self.logger.error(f"Error cancelling order on {order.exchange}",
                                order_id=order.id,
                                error=str(e),
                                attempt=attempt)
            
            # Wait before retrying
            if attempt < self.retry_attempts:
                await asyncio.sleep(self.retry_delay * attempt)  # Incremental backoff
        
        # All attempts failed
        self.logger.error(f"Failed to cancel order on {order.exchange} after multiple attempts",
                        order_id=order.id)
        
        return False
    
    async def _monitor_orders(self) -> None:
        """Monitor active orders and update their status."""
        while True:
            try:
                # Check all active orders
                for order_id in list(self.active_orders):
                    if order_id not in self.orders:
                        self.active_orders.remove(order_id)
                        continue
                    
                    order = self.orders[order_id]
                    
                    # Skip if already complete
                    if order.is_complete():
                        self.active_orders.remove(order_id)
                        continue
                    
                    # Update order status from exchange
                    await self._update_order_status(order)
                
                # Sleep for a short interval
                await asyncio.sleep(self.order_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in order monitoring", error=str(e))
                await asyncio.sleep(self.order_update_interval)
    
    async def _update_order_status(self, order: Order) -> None:
        """Update the status of an order from the exchange.
        
        Args:
            order: The order to update
        """
        if not order.exchange_order_id:
            self.logger.debug("Order has no exchange_order_id, skipping update", order_id=order.id)
            return
            
        if not order.exchange in self.exchange_connectors:
            self.logger.error("Exchange not found", exchange=order.exchange, order_id=order.id)
            return
            
        try:
            # Get order status from exchange
            new_status, error_msg = await self._get_order_status_from_exchange(order)
            
            if error_msg:
                self.logger.warning("Error getting order status", 
                                  order_id=order.id, 
                                  error=error_msg)
                return
                
            # If status changed, update the order
            if new_status != order.status:
                old_status = order.status
                order.update_status(new_status)
                
                self.logger.info("Order status updated", 
                               order_id=order.id,
                               old_status=old_status.value,
                               new_status=new_status.value,
                               exchange=order.exchange)
                
                # If order is now complete, remove from active orders
                if order.is_complete() and order.id in self.active_orders:
                    self.active_orders.remove(order.id)
                    
                    # Publish an order update event
                    await event_bus.publish(OrderEvent(
                        order_id=order.id,
                        event_type="status_update",
                        order=order,
                        exchange=order.exchange,
                        symbol=order.symbol,
                        source="execution_service"
                    ))
            
        except Exception as e:
            self.logger.error("Error updating order status", 
                           order_id=order.id,
                           error=str(e))
    
    async def _cancel_all_active_orders(self, reason: str) -> int:
        """Cancel all active orders.
        
        Args:
            reason: The reason for cancellation
            
        Returns:
            int: The number of orders cancelled
        """
        cancelled_count = 0
        
        for order_id in list(self.active_orders):
            success = await self.cancel_order(order_id)
            if success:
                cancelled_count += 1
        
        self.logger.info("Cancelled all active orders", 
                       count=cancelled_count,
                       reason=reason)
        
        return cancelled_count
    
    async def create_market_order(
        self,
        exchange: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        position_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Order]:
        """Create a market order.
        
        Args:
            exchange: The exchange to use
            symbol: The trading pair symbol
            side: Buy or sell
            quantity: The quantity to trade
            position_id: The position ID (optional)
            strategy_id: The strategy ID (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            The created order, or None if creation failed
        """
        order = Order(
            exchange=exchange,
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            position_id=position_id,
            strategy_id=strategy_id,
            metadata=metadata or {}
        )
        
        success = await self.create_order(order)
        if not success:
            return None
        
        return order
    
    async def create_limit_order(
        self,
        exchange: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        position_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        post_only: bool = False,
        reduce_only: bool = False
    ) -> Optional[Order]:
        """Create a limit order.
        
        Args:
            exchange: The exchange to use
            symbol: The trading pair symbol
            side: Buy or sell
            quantity: The quantity to trade
            price: The limit price
            position_id: The position ID (optional)
            strategy_id: The strategy ID (optional)
            metadata: Additional metadata (optional)
            post_only: Whether the order should be post-only
            reduce_only: Whether the order should be reduce-only
            
        Returns:
            The created order, or None if creation failed
        """
        order = Order(
            exchange=exchange,
            symbol=symbol,
            order_type=OrderType.LIMIT,
            side=side,
            quantity=quantity,
            price=price,
            position_id=position_id,
            strategy_id=strategy_id,
            metadata=metadata or {},
            is_post_only=post_only,
            is_reduce_only=reduce_only
        )
        
        success = await self.create_order(order)
        if not success:
            return None
        
        return order
    
    async def create_stop_order(
        self,
        exchange: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        position_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        reduce_only: bool = True
    ) -> Optional[Order]:
        """Create a stop or stop-limit order.
        
        Args:
            exchange: The exchange to use
            symbol: The trading pair symbol
            side: Buy or sell
            quantity: The quantity to trade
            stop_price: The stop price
            limit_price: The limit price (optional, if provided creates a stop-limit order)
            position_id: The position ID (optional)
            strategy_id: The strategy ID (optional)
            metadata: Additional metadata (optional)
            reduce_only: Whether the order should be reduce-only (default True for stops)
            
        Returns:
            The created order, or None if creation failed
        """
        order_type = OrderType.STOP_LIMIT if limit_price is not None else OrderType.STOP
        
        order = Order(
            exchange=exchange,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price,
            position_id=position_id,
            strategy_id=strategy_id,
            metadata=metadata or {},
            is_reduce_only=reduce_only
        )
        
        success = await self.create_order(order)
        if not success:
            return None
        
        return order
    
    async def _get_order_status_from_exchange(self, order: Order) -> Tuple[OrderStatus, Optional[str]]:
        """Get the current status of an order from the exchange.
        
        Args:
            order: Order to check
            
        Returns:
            Tuple[OrderStatus, Optional[str]]: The order status and any error message
        """
        if not order.exchange in self.exchange_connectors:
            error_msg = f"Exchange {order.exchange} not found"
            self.logger.error(error_msg)
            return OrderStatus.REJECTED, error_msg
        
        if not order.exchange_order_id:
            error_msg = "Cannot check order without exchange_order_id"
            self.logger.error(error_msg, order_id=order.id)
            return OrderStatus.REJECTED, error_msg
        
        connector = self.exchange_connectors[order.exchange]
        
        try:
            self.logger.debug(f"Checking order status on {order.exchange}",
                           order_id=order.id,
                           exchange_order_id=order.exchange_order_id)
            
            # Get order status from exchange
            order_info = await connector.get_order(order.exchange_order_id, order.symbol)
            
            if not order_info:
                error_msg = "Failed to get order info from exchange"
                self.logger.warning(f"Failed to get order status from {order.exchange}",
                                  order_id=order.id,
                                  error=error_msg)
                return OrderStatus.REJECTED, error_msg
            
            # Map exchange status to our OrderStatus enum
            exchange_status = order_info.get('status', '').upper()
            
            if exchange_status == 'FILLED':
                return OrderStatus.FILLED, None
            elif exchange_status in ['CANCELED', 'CANCELLED']:
                return OrderStatus.CANCELLED, None
            elif exchange_status == 'REJECTED':
                return OrderStatus.REJECTED, None
            elif exchange_status in ['NEW', 'PARTIALLY_FILLED']:
                return OrderStatus.OPEN, None
            else:
                self.logger.warning(f"Unknown order status from {order.exchange}",
                                  order_id=order.id,
                                  exchange_status=exchange_status)
                return OrderStatus.REJECTED, f"Unknown status: {exchange_status}"
            
        except Exception as e:
            error_msg = f"Error checking order status: {str(e)}"
            self.logger.error(error_msg,
                            order_id=order.id)
            return OrderStatus.REJECTED, error_msg
            
    async def _submit_order_to_exchange(self, order: Order) -> bool:
        """Submit an order to the exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            bool: True if the order was submitted successfully, False otherwise
        """
        if not order.exchange in self.exchange_connectors:
            self.logger.error("Exchange not found", exchange=order.exchange)
            return False
            
        connector = self.exchange_connectors[order.exchange]
        
        # Set order status to pending and record submission time
        order.update_status(OrderStatus.PENDING)
        order.submitted_at = datetime.now(timezone.utc)
        
        # Try to submit the order with retries
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self.logger.debug(f"Submitting order to {order.exchange}, attempt {attempt}",
                                order_id=order.id,
                                symbol=order.symbol,
                                side=order.side,
                                order_type=order.order_type,
                                quantity=order.quantity,
                                price=order.price)
                
                # Submit the order to the exchange
                success, exchange_order_id, error = await connector.create_order(order)
                
                if success and exchange_order_id:
                    # Update order with exchange ID
                    order.exchange_order_id = exchange_order_id
                    order.update_status(OrderStatus.OPEN)
                    
                    self.logger.info(f"Order submitted to {order.exchange}",
                                    order_id=order.id,
                                    exchange_order_id=order.exchange_order_id)
                    
                    return True
                else:
                    self.logger.warning(f"Failed to submit order to {order.exchange}",
                                    order_id=order.id,
                                    error=error,
                                    attempt=attempt)
            
            except Exception as e:
                self.logger.error(f"Error submitting order to {order.exchange}",
                                order_id=order.id,
                                error=str(e),
                                attempt=attempt)
            
            # Wait before retrying
            if attempt < self.retry_attempts:
                await asyncio.sleep(self.retry_delay * attempt)  # Incremental backoff
        
        # All attempts failed
        order.update_status(OrderStatus.REJECTED)
        self.logger.error(f"Failed to submit order to {order.exchange} after multiple attempts",
                        order_id=order.id)
        
        return False 