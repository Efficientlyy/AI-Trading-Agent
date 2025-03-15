"""Unified Exchange Interface

This module provides a high-level interface for interacting with exchanges,
abstracting away the details of specific exchange connectors.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union

from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.execution.exchange import BaseExchangeConnector
from src.models.order import Order, OrderStatus, OrderType, OrderSide, TimeInForce

logger = get_logger("execution", "interface")


class ExchangeInterface:
    """Unified interface for interacting with exchanges.
    
    This class provides a high-level, exchange-agnostic API for executing
    trades and accessing market data across multiple exchanges.
    """
    
    def __init__(self, exchange_connectors: Dict[str, BaseExchangeConnector]):
        """Initialize the exchange interface.
        
        Args:
            exchange_connectors: Dictionary mapping exchange IDs to connectors
        """
        self.exchange_connectors = exchange_connectors
    
    def get_available_exchanges(self) -> List[str]:
        """Get the list of available exchanges.
        
        Returns:
            List of exchange IDs that are available for trading
        """
        return list(self.exchange_connectors.keys())
    
    def get_connector(self, exchange_id: str) -> Optional[BaseExchangeConnector]:
        """Get the connector for a specific exchange.
        
        Args:
            exchange_id: The ID of the exchange
            
        Returns:
            The exchange connector, or None if not found
        """
        return self.exchange_connectors.get(exchange_id)
    
    async def get_market_price(self, exchange_id: str, symbol: str) -> Optional[float]:
        """Get the current market price for a symbol.
        
        Args:
            exchange_id: Exchange ID
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Current market price or None if not available
        """
        connector = self.get_connector(exchange_id)
        if not connector:
            logger.error(f"Exchange connector not found: {exchange_id}")
            return None
        
        try:
            ticker = await connector.get_ticker(symbol)
            return float(ticker.get("last", 0))
        except Exception as e:
            logger.error(f"Error getting market price for {symbol} on {exchange_id}: {str(e)}")
            return None
    
    async def get_orderbook_snapshot(self, exchange_id: str, symbol: str, depth: int = 10) -> Optional[Dict]:
        """Get a snapshot of the current orderbook.
        
        Args:
            exchange_id: Exchange ID
            symbol: Trading pair symbol
            depth: Number of levels to retrieve
            
        Returns:
            Dict containing bids and asks or None if not available
        """
        connector = self.get_connector(exchange_id)
        if not connector:
            logger.error(f"Exchange connector not found: {exchange_id}")
            return None
        
        try:
            return await connector.get_orderbook(symbol, limit=depth)
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol} on {exchange_id}: {str(e)}")
            return None
    
    async def get_account_balance(self, exchange_id: str, asset: Optional[str] = None) -> Optional[Union[Dict[str, float], float]]:
        """Get account balance for all assets or a specific asset.
        
        Args:
            exchange_id: Exchange ID
            asset: Optional asset to get balance for
            
        Returns:
            Dict mapping asset to balance amount, or a single float if asset specified
        """
        connector = self.get_connector(exchange_id)
        if not connector:
            logger.error(f"Exchange connector not found: {exchange_id}")
            return None
        
        try:
            balances = await connector.get_account_balance()
            
            # Convert Decimal to float for easier handling
            float_balances = {k: float(v) for k, v in balances.items()}
            
            if asset:
                return float_balances.get(asset, 0.0)
            return float_balances
        except Exception as e:
            logger.error(f"Error getting account balance on {exchange_id}: {str(e)}")
            return None if asset else {}
    
    async def submit_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submit an order to the appropriate exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            Tuple of (success, exchange_order_id, error_message)
        """
        connector = self.get_connector(order.exchange)
        if not connector:
            return False, None, f"Exchange connector not found: {order.exchange}"
        
        try:
            return await connector.create_order(order)
        except Exception as e:
            logger.error(f"Error submitting order to {order.exchange}: {str(e)}")
            return False, None, str(e)
    
    async def cancel_order(self, exchange_id: str, order_id: str, symbol: str) -> Tuple[bool, Optional[str]]:
        """Cancel an order on the specified exchange.
        
        Args:
            exchange_id: Exchange ID
            order_id: Exchange order ID to cancel
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (success, error_message)
        """
        connector = self.get_connector(exchange_id)
        if not connector:
            return False, f"Exchange connector not found: {exchange_id}"
        
        try:
            return await connector.cancel_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} on {exchange_id}: {str(e)}")
            return False, str(e)
    
    async def get_order_status(self, exchange_id: str, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the current status of an order.
        
        Args:
            exchange_id: Exchange ID
            order_id: Exchange order ID
            symbol: Trading pair symbol
            
        Returns:
            Dict containing order status information or None if not found
        """
        connector = self.get_connector(exchange_id)
        if not connector:
            logger.error(f"Exchange connector not found: {exchange_id}")
            return None
        
        try:
            return await connector.get_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Error getting order status for {order_id} on {exchange_id}: {str(e)}")
            return None
    
    async def get_open_orders(self, exchange_id: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders on the specified exchange.
        
        Args:
            exchange_id: Exchange ID
            symbol: Optional trading pair symbol to filter by
            
        Returns:
            List of dictionaries containing order information
        """
        connector = self.get_connector(exchange_id)
        if not connector:
            logger.error(f"Exchange connector not found: {exchange_id}")
            return []
        
        try:
            return await connector.get_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error getting open orders on {exchange_id}: {str(e)}")
            return []
            
    async def create_market_order(
        self,
        exchange_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        client_order_id: Optional[str] = None,
        position_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Order], Optional[str]]:
        """Create and submit a market order.
        
        Args:
            exchange_id: Exchange ID
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            client_order_id: Optional client-specified order ID
            position_id: Optional position ID for this order
            strategy_id: Optional strategy ID that created this order
            metadata: Optional metadata dict
            
        Returns:
            Tuple of (success, order object if successful, error message if failed)
        """
        order = Order(
            exchange=exchange_id,
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            client_order_id=client_order_id,
            position_id=position_id,
            strategy_id=strategy_id,
            metadata=metadata or {},
            time_in_force=TimeInForce.FOK  # Use FOK for market orders to ensure full execution
        )
        
        success, exchange_order_id, error = await self.submit_order(order)
        
        if success and exchange_order_id:
            order.exchange_order_id = exchange_order_id
            order.status = OrderStatus.PENDING
            order.submitted_at = utc_now()
            return True, order, None
        
        return False, None, error or "Unknown error"
    
    async def create_limit_order(
        self,
        exchange_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        client_order_id: Optional[str] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        post_only: bool = False,
        reduce_only: bool = False,
        position_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, Optional[Order], Optional[str]]:
        """Create and submit a limit order.
        
        Args:
            exchange_id: Exchange ID
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Limit price
            client_order_id: Optional client-specified order ID
            time_in_force: Time in force
            post_only: Whether order should be post-only
            reduce_only: Whether order should be reduce-only
            position_id: Optional position ID for this order
            strategy_id: Optional strategy ID that created this order
            metadata: Optional metadata dict
            
        Returns:
            Tuple of (success, order object if successful, error message if failed)
        """
        order = Order(
            exchange=exchange_id,
            symbol=symbol,
            order_type=OrderType.LIMIT,
            side=side,
            quantity=quantity,
            price=price,
            client_order_id=client_order_id,
            time_in_force=time_in_force,
            is_post_only=post_only,
            is_reduce_only=reduce_only,
            position_id=position_id,
            strategy_id=strategy_id,
            metadata=metadata or {}
        )
        
        success, exchange_order_id, error = await self.submit_order(order)
        
        if success and exchange_order_id:
            order.exchange_order_id = exchange_order_id
            order.status = OrderStatus.PENDING
            order.submitted_at = utc_now()
            return True, order, None
        
        return False, None, error or "Unknown error" 