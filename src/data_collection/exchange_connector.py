"""Base exchange connector for cryptocurrency exchanges.

This module defines the base class for all exchange connectors, providing
a consistent interface for interacting with different cryptocurrency exchanges.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

from src.common.component import Component
from src.common.events import event_bus
from src.common.logging import get_logger
from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData
from src.models.events import (
    CandleDataEvent, OrderBookEvent, SymbolListEvent, TradeDataEvent
)


class ExchangeConnector(Component, ABC):
    """Base class for all exchange connectors.
    
    This class defines the interface that all exchange connectors must implement,
    ensuring consistent behavior across different exchanges.
    """
    
    def __init__(self, exchange_id: str):
        """Initialize the exchange connector.
        
        Args:
            exchange_id: Unique identifier for the exchange
        """
        super().__init__(f"exchange_connector.{exchange_id}")
        self.exchange_id = exchange_id
        self.logger = get_logger("data_collection", f"exchange.{exchange_id}")
        self.available_symbols: Set[str] = set()
        self.subscribed_symbols: Set[str] = set()
        self.subscribed_timeframes: Dict[str, Set[TimeFrame]] = {}
        self.subscribed_orderbooks: Set[str] = set()
        self.subscribed_trades: Set[str] = set()
        self.ws_client = None
        self.rest_client = None
        self.last_candle_update: Dict[str, Dict[TimeFrame, datetime]] = {}
    
    async def initialize(self) -> bool:
        """Initialize the exchange connector.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing exchange connector", exchange=self.exchange_id)
        
        # Load exchange-specific configuration
        exchange_config = self.get_config(f"exchanges.{self.exchange_id}", {})
        if not exchange_config:
            self.logger.error("No configuration found for exchange", exchange=self.exchange_id)
            return False
        
        # Initialize REST client
        try:
            self._init_rest_client()
            self.logger.debug("REST client initialized")
        except Exception as e:
            self.logger.error("Failed to initialize REST client", error=str(e))
            return False
        
        # Fetch available symbols
        try:
            self.available_symbols = self.fetch_available_symbols()
            self.logger.info("Fetched available symbols", 
                            exchange=self.exchange_id, 
                            symbol_count=len(self.available_symbols))
            
            # Publish available symbols event
            await self.publish_event(SymbolListEvent(
                source=self.component_id,
                exchange=self.exchange_id,
                symbols=list(self.available_symbols)
            ))
        except Exception as e:
            self.logger.error("Failed to fetch available symbols", error=str(e))
            return False
        
        # Auto-subscribe to symbols if configured
        auto_subscribe = self.get_config(f"exchanges.{self.exchange_id}.auto_subscribe", [])
        if auto_subscribe:
            for symbol_config in auto_subscribe:
                symbol = symbol_config.get("symbol")
                if not symbol:
                    continue
                
                # Check if symbol is available
                if symbol not in self.available_symbols:
                    self.logger.warning("Cannot auto-subscribe to unavailable symbol", 
                                      symbol=symbol, exchange=self.exchange_id)
                    continue
                
                # Subscribe to candles if configured
                timeframes = symbol_config.get("timeframes", [])
                if timeframes:
                    for tf_str in timeframes:
                        try:
                            tf = TimeFrame(tf_str)
                            await self.subscribe_candles(symbol, tf)
                        except ValueError:
                            self.logger.warning("Invalid timeframe in auto-subscribe config", 
                                              timeframe=tf_str, symbol=symbol)
                
                # Subscribe to order book if configured
                if symbol_config.get("orderbook", False):
                    await self.subscribe_orderbook(symbol)
                
                # Subscribe to trades if configured
                if symbol_config.get("trades", False):
                    await self.subscribe_trades(symbol)
        
        return await super().initialize()
    
    async def start(self) -> bool:
        """Start the exchange connector.
        
        Returns:
            bool: True if start was successful, False otherwise
        """
        self.logger.info("Starting exchange connector", exchange=self.exchange_id)
        
        # Initialize WebSocket client if needed
        if self.subscribed_symbols:
            try:
                self._init_websocket_client()
                self.logger.debug("WebSocket client initialized")
            except Exception as e:
                self.logger.error("Failed to initialize WebSocket client", error=str(e))
                return False
        
        # Start data polling tasks for REST-based data
        self._start_polling_tasks()
        
        return await super().start()
    
    async def stop(self) -> bool:
        """Stop the exchange connector.
        
        Returns:
            bool: True if stop was successful, False otherwise
        """
        self.logger.info("Stopping exchange connector", exchange=self.exchange_id)
        
        # Close WebSocket connection if open
        if self.ws_client:
            try:
                self._close_websocket_client()
                self.logger.debug("WebSocket client closed")
            except Exception as e:
                self.logger.error("Error closing WebSocket client", error=str(e))
        
        return await super().stop()
    
    async def subscribe_candles(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Subscribe to candle data for a symbol and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if symbol not in self.available_symbols:
            self.logger.warning("Cannot subscribe to unavailable symbol", 
                              symbol=symbol, timeframe=timeframe.value)
            return False
        
        # Add to subscribed symbols
        self.subscribed_symbols.add(symbol)
        
        # Add to subscribed timeframes for this symbol
        if symbol not in self.subscribed_timeframes:
            self.subscribed_timeframes[symbol] = set()
        
        if timeframe in self.subscribed_timeframes[symbol]:
            self.logger.debug("Already subscribed to candles", 
                            symbol=symbol, timeframe=timeframe.value)
            return True
        
        self.subscribed_timeframes[symbol].add(timeframe)
        
        # Initialize last candle update time
        if symbol not in self.last_candle_update:
            self.last_candle_update[symbol] = {}
        self.last_candle_update[symbol][timeframe] = datetime.utcnow()
        
        self.logger.info("Subscribed to candles", 
                       symbol=symbol, timeframe=timeframe.value)
        
        # If already running, update subscriptions
        if self.is_running:
            self._update_subscriptions()
        
        return True
    
    async def subscribe_orderbook(self, symbol: str) -> bool:
        """Subscribe to order book data for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if symbol not in self.available_symbols:
            self.logger.warning("Cannot subscribe to unavailable symbol", symbol=symbol)
            return False
        
        # Add to subscribed symbols
        self.subscribed_symbols.add(symbol)
        
        if symbol in self.subscribed_orderbooks:
            self.logger.debug("Already subscribed to orderbook", symbol=symbol)
            return True
        
        self.subscribed_orderbooks.add(symbol)
        self.logger.info("Subscribed to orderbook", symbol=symbol)
        
        # If already running, update subscriptions
        if self.is_running:
            self._update_subscriptions()
        
        return True
    
    async def subscribe_trades(self, symbol: str) -> bool:
        """Subscribe to trade data for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if symbol not in self.available_symbols:
            self.logger.warning("Cannot subscribe to unavailable symbol", symbol=symbol)
            return False
        
        # Add to subscribed symbols
        self.subscribed_symbols.add(symbol)
        
        if symbol in self.subscribed_trades:
            self.logger.debug("Already subscribed to trades", symbol=symbol)
            return True
        
        self.subscribed_trades.add(symbol)
        self.logger.info("Subscribed to trades", symbol=symbol)
        
        # If already running, update subscriptions
        if self.is_running:
            self._update_subscriptions()
        
        return True
    
    async def unsubscribe_candles(self, symbol: str, timeframe: TimeFrame) -> bool:
        """Unsubscribe from candle data for a symbol and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if symbol not in self.subscribed_timeframes:
            return True
        
        if timeframe not in self.subscribed_timeframes[symbol]:
            return True
        
        self.subscribed_timeframes[symbol].remove(timeframe)
        
        # Remove from last candle update time
        if symbol in self.last_candle_update and timeframe in self.last_candle_update[symbol]:
            del self.last_candle_update[symbol][timeframe]
        
        self.logger.info("Unsubscribed from candles", 
                       symbol=symbol, timeframe=timeframe.value)
        
        # Check if we need to remove the symbol entirely
        self._check_remove_symbol(symbol)
        
        # If already running, update subscriptions
        if self.is_running:
            self._update_subscriptions()
        
        return True
    
    async def unsubscribe_orderbook(self, symbol: str) -> bool:
        """Unsubscribe from order book data for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if symbol not in self.subscribed_orderbooks:
            return True
        
        self.subscribed_orderbooks.remove(symbol)
        self.logger.info("Unsubscribed from orderbook", symbol=symbol)
        
        # Check if we need to remove the symbol entirely
        self._check_remove_symbol(symbol)
        
        # If already running, update subscriptions
        if self.is_running:
            self._update_subscriptions()
        
        return True
    
    async def unsubscribe_trades(self, symbol: str) -> bool:
        """Unsubscribe from trade data for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if symbol not in self.subscribed_trades:
            return True
        
        self.subscribed_trades.remove(symbol)
        self.logger.info("Unsubscribed from trades", symbol=symbol)
        
        # Check if we need to remove the symbol entirely
        self._check_remove_symbol(symbol)
        
        # If already running, update subscriptions
        if self.is_running:
            self._update_subscriptions()
        
        return True
    
    def _check_remove_symbol(self, symbol: str) -> None:
        """Check if a symbol should be removed from subscribed symbols.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Check if we're still subscribed to any data for this symbol
        has_candles = symbol in self.subscribed_timeframes and self.subscribed_timeframes[symbol]
        has_orderbook = symbol in self.subscribed_orderbooks
        has_trades = symbol in self.subscribed_trades
        
        if not (has_candles or has_orderbook or has_trades):
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
                self.logger.debug("Removed symbol from subscriptions", symbol=symbol)
    
    async def _start_polling_tasks(self) -> None:
        """Start polling tasks for REST-based data."""
        # This method should be implemented by subclasses if they need
        # to poll for data that isn't available via WebSocket
        pass
    
    async def _update_subscriptions(self) -> None:
        """Update WebSocket subscriptions based on current subscription state."""
        # This method should be implemented by subclasses to handle
        # dynamic subscription changes
        pass
    
    @abstractmethod
    async def _init_rest_client(self) -> None:
        """Initialize the REST API client."""
        pass
    
    @abstractmethod
    async def _init_websocket_client(self) -> None:
        """Initialize the WebSocket client."""
        pass
    
    @abstractmethod
    async def _close_websocket_client(self) -> None:
        """Close the WebSocket client."""
        pass
    
    @abstractmethod
    async def fetch_available_symbols(self) -> Set[str]:
        """Fetch the list of available trading pairs from the exchange.
        
        Returns:
            Set[str]: Set of available symbols
        """
        pass
    
    @abstractmethod
    async def fetch_candles(
        self, symbol: str, timeframe: TimeFrame, 
        since: Optional[datetime] = None, limit: int = 100
    ) -> List[CandleData]:
        """Fetch historical candle data for a symbol and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
            since: Optional start time for the candles
            limit: Maximum number of candles to fetch
            
        Returns:
            List[CandleData]: List of candle data objects
        """
        pass
    
    @abstractmethod
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> OrderBookData:
        """Fetch the current order book for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            limit: Maximum number of orders to fetch on each side
            
        Returns:
            OrderBookData: Order book data object
        """
        pass
    
    @abstractmethod
    async def fetch_trades(
        self, symbol: str, since: Optional[datetime] = None, limit: int = 100
    ) -> List[TradeData]:
        """Fetch recent trades for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            since: Optional start time for the trades
            limit: Maximum number of trades to fetch
            
        Returns:
            List[TradeData]: List of trade data objects
        """
        pass
    
    async def publish_candle_data(self, candle: CandleData) -> None:
        """Publish candle data to the event bus.
        
        Args:
            candle: The candle data to publish
        """
        event = CandleDataEvent(
            source=self.component_id,
            event_type="CandleDataEvent",
            symbol=candle.symbol,
            exchange=candle.exchange,
            candle=candle,
            timeframe=candle.timeframe
        )
        await self.publish_event(event)
        
        # Update last candle time
        if candle.symbol not in self.last_candle_update:
            self.last_candle_update[candle.symbol] = {}
        self.last_candle_update[candle.symbol][candle.timeframe] = datetime.utcnow()
    
    async def publish_orderbook_data(self, orderbook: OrderBookData) -> None:
        """Publish order book data to the event bus.
        
        Args:
            orderbook: The order book data to publish
        """
        event = OrderBookEvent(
            source=self.component_id,
            event_type="OrderBookEvent",
            symbol=orderbook.symbol,
            exchange=orderbook.exchange,
            orderbook=orderbook
        )
        await self.publish_event(event)
    
    async def publish_trade_data(self, trade: TradeData) -> None:
        """Publish trade data to the event bus.
        
        Args:
            trade: The trade data to publish
        """
        event = TradeDataEvent(
            source=self.component_id,
            event_type="TradeDataEvent",
            symbol=trade.symbol,
            exchange=trade.exchange,
            trade=trade
        )
        await self.publish_event(event) 