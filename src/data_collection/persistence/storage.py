"""Base storage implementation for the AI Crypto Trading System.

This module defines the base class for all storage backends, providing
a consistent interface for storing and retrieving market data.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union

from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData


class Storage(ABC):
    """Base class for all storage backends."""
    
    def __init__(self, name: str):
        """Initialize the storage backend.
        
        Args:
            name: The name of the storage backend
        """
        self.name = name
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the storage backend.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend."""
        pass
    
    @abstractmethod
    async def store_candle(self, candle: CandleData) -> bool:
        """Store a single candle.
        
        Args:
            candle: The candle data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def store_candles(self, candles: List[CandleData]) -> bool:
        """Store multiple candles in batch.
        
        Args:
            candles: The list of candle data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_candles(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None, 
        limit: Optional[int] = None
    ) -> List[CandleData]:
        """Retrieve candles for a symbol, exchange, and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            start_time: Optional start time for the query
            end_time: Optional end time for the query
            limit: Optional limit for the number of candles to retrieve
            
        Returns:
            List[CandleData]: List of candle data objects
        """
        pass
    
    @abstractmethod
    async def store_orderbook(self, orderbook: OrderBookData) -> bool:
        """Store an order book snapshot.
        
        Args:
            orderbook: The order book data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_orderbook(
        self, 
        symbol: str, 
        exchange: str, 
        timestamp: Optional[datetime] = None
    ) -> Optional[OrderBookData]:
        """Retrieve the latest (or specific) order book for a symbol and exchange.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timestamp: Optional specific timestamp to retrieve
            
        Returns:
            Optional[OrderBookData]: Order book data object, or None if not found
        """
        pass
    
    @abstractmethod
    async def store_trade(self, trade: TradeData) -> bool:
        """Store a single trade.
        
        Args:
            trade: The trade data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def store_trades(self, trades: List[TradeData]) -> bool:
        """Store multiple trades in batch.
        
        Args:
            trades: The list of trade data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_trades(
        self, 
        symbol: str, 
        exchange: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None, 
        limit: Optional[int] = None
    ) -> List[TradeData]:
        """Retrieve trades for a symbol and exchange.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            start_time: Optional start time for the query
            end_time: Optional end time for the query
            limit: Optional limit for the number of trades to retrieve
            
        Returns:
            List[TradeData]: List of trade data objects
        """
        pass
    
    @abstractmethod
    async def get_latest_candle(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame
    ) -> Optional[CandleData]:
        """Retrieve the latest candle for a symbol, exchange, and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            
        Returns:
            Optional[CandleData]: Latest candle data object, or None if not found
        """
        pass
    
    @abstractmethod
    async def purge_old_data(self, max_age_days: Dict[str, int]) -> bool:
        """Purge data older than the specified age.
        
        Args:
            max_age_days: Dictionary of data types to maximum age in days
            
        Returns:
            bool: True if purge was successful, False otherwise
        """
        pass 