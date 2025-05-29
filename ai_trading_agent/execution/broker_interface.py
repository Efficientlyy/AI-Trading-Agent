"""
Broker Interface Module

Defines the common interface for broker integrations, allowing the trading
system to interact with different brokers in a unified way.
"""

import abc
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from datetime import datetime

from ..common.enums import OrderType, OrderSide, OrderStatus, TimeInForce
from ..common.models import Order, Trade, Position, Balance, Portfolio


class BrokerInterface(abc.ABC):
    """
    Abstract base class defining the interface for broker integrations.
    
    All broker implementations must implement these methods to provide
    a consistent interface for the trading system.
    """
    
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the broker connection and authenticate.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def place_order(self, order: Order) -> Tuple[bool, str, Optional[str]]:
        """
        Place an order with the broker.
        
        Args:
            order: Order details
            
        Returns:
            Tuple of (success, message, order_id)
        """
        pass
    
    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Tuple of (success, message)
        """
        pass
    
    @abc.abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order details by ID.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            Order object if found, None otherwise
        """
        pass
    
    @abc.abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of open orders
        """
        pass
    
    @abc.abstractmethod
    async def get_order_history(self, 
                          symbol: Optional[str] = None,
                          limit: int = 100) -> List[Order]:
        """
        Get order history.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of orders to return
            
        Returns:
            List of historical orders
        """
        pass
    
    @abc.abstractmethod
    async def get_trade_history(self,
                          symbol: Optional[str] = None,
                          limit: int = 100) -> List[Trade]:
        """
        Get trade execution history.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of trades to return
            
        Returns:
            List of trade objects
        """
        pass
    
    @abc.abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Position]:
        """
        Get current positions.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dictionary mapping symbols to position objects
        """
        pass
    
    @abc.abstractmethod
    async def get_balances(self) -> Dict[str, Balance]:
        """
        Get account balances.
        
        Returns:
            Dictionary mapping currencies to balance objects
        """
        pass
    
    @abc.abstractmethod
    async def get_portfolio(self) -> Portfolio:
        """
        Get full portfolio information.
        
        Returns:
            Portfolio object with balances, positions, and trade history
        """
        pass
    
    @abc.abstractmethod
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if unavailable
        """
        pass
    
    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the broker connection and clean up resources.
        """
        pass
