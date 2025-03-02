"""Base exchange connector interface.

This module defines the BaseExchangeConnector abstract base class, which
serves as an interface for all exchange connectors.
"""

import abc
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple

from src.models.order import Order, OrderType, OrderStatus, OrderSide, TimeInForce


class BaseExchangeConnector(abc.ABC):
    """Abstract base class for exchange connectors.
    
    All exchange connectors must implement the methods defined in this class
    to provide a standardized interface for interacting with exchanges.
    """
    
    def __init__(self, exchange_id: str, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize the exchange connector.
        
        Args:
            exchange_id: Unique identifier for this exchange
            api_key: API key for authenticated requests (optional)
            api_secret: API secret for authenticated requests (optional)
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """Initialize the exchange connector.
        
        This method should establish any necessary connections,
        validate API credentials, and set up internal state.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the exchange connector.
        
        This method should close any open connections and perform
        necessary cleanup.
        """
        pass
    
    @abc.abstractmethod
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information and trading rules.
        
        Returns:
            Dict containing exchange information
        """
        pass
    
    @abc.abstractmethod
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """Get account balances for all assets.
        
        Returns:
            Dict mapping asset symbol to balance amount
        """
        pass
    
    @abc.abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dict containing ticker information
        """
        pass
    
    @abc.abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            limit: Maximum number of bids/asks to return
            
        Returns:
            Dict containing order book data
        """
        pass
    
    @abc.abstractmethod
    async def create_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submit an order to the exchange.
        
        Args:
            order: Order object containing order details
            
        Returns:
            Tuple of (success, exchange_order_id, error_message)
        """
        pass
    
    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Tuple[bool, Optional[str]]:
        """Cancel an existing order.
        
        Args:
            order_id: Exchange order ID to cancel
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (success, error_message)
        """
        pass
    
    @abc.abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific order.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            
        Returns:
            Dict containing order information or None if not found
        """
        pass
    
    @abc.abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders.
        
        Args:
            symbol: Optional trading pair symbol to filter by
            
        Returns:
            List of dictionaries containing order information
        """
        pass
    
    @abc.abstractmethod
    async def get_trade_history(
        self, 
        symbol: str, 
        limit: int = 100,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get trade history for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades to return
            from_time: Optional start time for trades
            to_time: Optional end time for trades
            
        Returns:
            List of dictionaries containing trade information
        """
        pass
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize a symbol to exchange format.
        
        Different exchanges use different symbol formats (e.g., BTC/USDT vs BTCUSDT).
        This method converts from the standard format (BTC/USDT) to the exchange's format.
        
        Args:
            symbol: Symbol in standard format (e.g., "BTC/USDT")
            
        Returns:
            Symbol in exchange format
        """
        # Default implementation (to be overridden if needed)
        return symbol.replace("/", "")
    
    def standardize_symbol(self, symbol: str) -> str:
        """Convert a symbol from exchange format to standard format.
        
        This is the reverse of normalize_symbol.
        
        Args:
            symbol: Symbol in exchange format
            
        Returns:
            Symbol in standard format (e.g., "BTC/USDT")
        """
        # Default implementation (to be overridden if needed)
        # This implementation assumes symbols are in format like "BTCUSDT"
        # and tries to find common quote currencies
        common_quote_currencies = ["USDT", "USD", "BTC", "ETH", "BNB", "BUSD"]
        
        for quote in sorted(common_quote_currencies, key=len, reverse=True):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f"{base}/{quote}"
        
        # If no common quote currency found, default to a simple split
        # This is just a guess and should be overridden in real implementations
        if len(symbol) > 3:
            base = symbol[:-4]
            quote = symbol[-4:]
            return f"{base}/{quote}"
        
        return symbol 