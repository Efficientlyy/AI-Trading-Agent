"""Storage manager for the AI Crypto Trading System.

This module implements the storage manager, which coordinates the persistence
of market data across different storage backends.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type, Union

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.data_collection.persistence.file_storage import FileStorage
from src.data_collection.persistence.storage import Storage
from src.models.events import CandleDataEvent, OrderBookEvent, TradeDataEvent
from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData


class StorageManager(Component):
    """Manager for market data storage.
    
    This component manages the persistence of market data across different
    storage backends, subscribes to market data events, and handles data
    retention policies.
    """
    
    def __init__(self):
        """Initialize the storage manager."""
        super().__init__("storage_manager")
        self.logger = get_logger("data_collection", "storage_manager")
        self.storage_backends: Dict[str, Storage] = {}
        self.enabled = config.get("data_collection.persistence.enabled", True)
        self.retention_task = None
        
        # Initialize available storage backends
        storage_type = config.get("data_collection.persistence.storage_type", "file")
        
        if storage_type == "file":
            self.storage_backends["file"] = FileStorage()
        # Add additional storage backends here as they are implemented
        
        # Batch processing
        self.candle_batch: Dict[str, List[CandleData]] = {}
        self.trade_batch: Dict[str, List[TradeData]] = {}
        self.batch_size = config.get("data_collection.persistence.batch_size", 100)
        self.batch_interval = config.get("data_collection.persistence.batch_interval", 60)
        self.batch_flush_task = None
        
        # Cache for recent data
        self.cache_enabled = config.get("data_collection.persistence.cache_enabled", True)
        self.candle_cache: Dict[str, Dict[str, Dict[TimeFrame, List[CandleData]]]] = {}
        self.orderbook_cache: Dict[str, Dict[str, OrderBookData]] = {}
        self.trade_cache: Dict[str, Dict[str, List[TradeData]]] = {}
        self.cache_size = config.get("data_collection.persistence.cache_size", 1000)
    
    async def _initialize(self) -> None:
        """Initialize the storage manager."""
        if not self.enabled:
            self.logger.info("Storage manager is disabled")
            return
        
        self.logger.info("Initializing storage manager")
        
        # Initialize all storage backends
        for name, backend in self.storage_backends.items():
            try:
                success = backend.initialize()
                if not success:
                    self.logger.error("Failed to initialize storage backend", backend=name)
                else:
                    self.logger.info("Initialized storage backend", backend=name)
            except Exception as e:
                self.logger.error("Error initializing storage backend", 
                                backend=name, error=str(e))
    
    async def _start(self) -> None:
        """Start the storage manager."""
        if not self.enabled:
            return
        
        self.logger.info("Starting storage manager")
        
        # Subscribe to market data events
        event_bus.subscribe("CandleDataEvent", self._handle_candle_event)
        event_bus.subscribe("OrderBookEvent", self._handle_orderbook_event)
        event_bus.subscribe("TradeDataEvent", self._handle_trade_event)
        
        # Start the periodic batch flush task
        self.batch_flush_task = self.create_task(self._flush_batches_periodically())
        
        # Start the periodic data retention task
        retention_interval = config.get("data_collection.persistence.retention_check_interval", 86400)
        self.retention_task = self.create_task(self._run_retention_periodically(retention_interval))
    
    async def _stop(self) -> None:
        """Stop the storage manager."""
        if not self.enabled:
            return
        
        self.logger.info("Stopping storage manager")
        
        # Unsubscribe from market data events
        event_bus.unsubscribe("CandleDataEvent", self._handle_candle_event)
        event_bus.unsubscribe("OrderBookEvent", self._handle_orderbook_event)
        event_bus.unsubscribe("TradeDataEvent", self._handle_trade_event)
        
        # Cancel the batch flush task
        if self.batch_flush_task and not self.batch_flush_task.done():
            self.batch_flush_task.cancel()
            try:
                await self.batch_flush_task
            except asyncio.CancelledError:
                pass
        
        # Cancel the retention task
        if self.retention_task and not self.retention_task.done():
            self.retention_task.cancel()
            try:
                await self.retention_task
            except asyncio.CancelledError:
                pass
        
        # Flush any remaining batches
        self._flush_all_batches()
        
        # Close all storage backends
        for name, backend in self.storage_backends.items():
            try:
                backend.close()
                self.logger.info("Closed storage backend", backend=name)
            except Exception as e:
                self.logger.error("Error closing storage backend", 
                                backend=name, error=str(e))
    
    async def _handle_candle_event(self, event: CandleDataEvent) -> None:
        """Handle a candle data event.
        
        Args:
            event: The candle data event
        """
        if not self.enabled:
            return
        
        candle = event.candle
        
        # Add to cache if enabled
        if self.cache_enabled:
            self._add_to_candle_cache(candle)
        
        # Add to batch for storage
        key = f"{candle.exchange}_{candle.symbol}_{candle.timeframe.value}"
        if key not in self.candle_batch:
            self.candle_batch[key] = []
        
        self.candle_batch[key].append(candle)
        
        # Flush batch if it reaches the threshold
        if len(self.candle_batch[key]) >= self.batch_size:
            await self._flush_candle_batch(key)
    
    async def _handle_orderbook_event(self, event: OrderBookEvent) -> None:
        """Handle an order book event.
        
        Args:
            event: The order book event
        """
        if not self.enabled:
            return
        
        orderbook = event.orderbook
        
        # Add to cache if enabled
        if self.cache_enabled:
            self._add_to_orderbook_cache(orderbook)
        
        # Store order book directly (no batching for order books)
        store_orderbooks = config.get("data_collection.persistence.orderbooks.enabled", False)
        if store_orderbooks:
            for backend in self.storage_backends.values():
                try:
                    await backend.store_orderbook(orderbook)
                except Exception as e:
                    self.logger.error("Error storing orderbook", 
                                    exchange=orderbook.exchange, 
                                    symbol=orderbook.symbol, 
                                    error=str(e))
    
    async def _handle_trade_event(self, event: TradeDataEvent) -> None:
        """Handle a trade data event.
        
        Args:
            event: The trade data event
        """
        if not self.enabled:
            return
        
        trade = event.trade
        
        # Add to cache if enabled
        if self.cache_enabled:
            self._add_to_trade_cache(trade)
        
        # Add to batch for storage
        key = f"{trade.exchange}_{trade.symbol}"
        if key not in self.trade_batch:
            self.trade_batch[key] = []
        
        self.trade_batch[key].append(trade)
        
        # Flush batch if it reaches the threshold
        if len(self.trade_batch[key]) >= self.batch_size:
            await self._flush_trade_batch(key)
    
    async def _flush_candle_batch(self, key: str) -> None:
        """Flush a batch of candles to storage.
        
        Args:
            key: The batch key (exchange_symbol_timeframe)
        """
        if key not in self.candle_batch or not self.candle_batch[key]:
            return
        
        candles = self.candle_batch[key]
        self.candle_batch[key] = []
        
        store_candles = config.get("data_collection.persistence.candles.enabled", True)
        if store_candles:
            for backend in self.storage_backends.values():
                try:
                    await backend.store_candles(candles)
                except Exception as e:
                    self.logger.error("Error storing candles", 
                                    batch_key=key, 
                                    count=len(candles), 
                                    error=str(e))
    
    async def _flush_trade_batch(self, key: str) -> None:
        """Flush a batch of trades to storage.
        
        Args:
            key: The batch key (exchange_symbol)
        """
        if key not in self.trade_batch or not self.trade_batch[key]:
            return
        
        trades = self.trade_batch[key]
        self.trade_batch[key] = []
        
        store_trades = config.get("data_collection.persistence.trades.enabled", True)
        if store_trades:
            for backend in self.storage_backends.values():
                try:
                    await backend.store_trades(trades)
                except Exception as e:
                    self.logger.error("Error storing trades", 
                                    batch_key=key, 
                                    count=len(trades), 
                                    error=str(e))
    
    async def _flush_all_batches(self) -> None:
        """Flush all batches to storage."""
        # Flush all candle batches
        for key in list(self.candle_batch.keys()):
            await self._flush_candle_batch(key)
        
        # Flush all trade batches
        for key in list(self.trade_batch.keys()):
            await self._flush_trade_batch(key)
    
    async def _flush_batches_periodically(self) -> None:
        """Periodically flush all batches to storage."""
        try:
            while True:
                await asyncio.sleep(self.batch_interval)
                self._flush_all_batches()
                
        except asyncio.CancelledError:
            self.logger.debug("Batch flush task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in batch flush task", error=str(e))
    
    async def _run_retention_periodically(self, interval: int) -> None:
        """Periodically run the data retention policy.
        
        Args:
            interval: The interval in seconds between retention runs
        """
        try:
            while True:
                await asyncio.sleep(interval)
                self._run_retention()
                
        except asyncio.CancelledError:
            self.logger.debug("Retention task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in retention task", error=str(e))
    
    async def _run_retention(self) -> None:
        """Run the data retention policy."""
        self.logger.info("Running data retention")
        
        # Get retention settings
        retention_days = {}
        
        candle_retention = config.get("data_collection.persistence.candles.retention_days", {})
        if candle_retention:
            retention_days["candles"] = candle_retention.get("all", 365)
        
        orderbook_retention = config.get("data_collection.persistence.orderbooks.retention_days")
        if orderbook_retention:
            retention_days["orderbooks"] = orderbook_retention
        
        trade_retention = config.get("data_collection.persistence.trades.retention_days")
        if trade_retention:
            retention_days["trades"] = trade_retention
        
        if not retention_days:
            self.logger.info("No retention policy configured")
            return
        
        # Run retention on all storage backends
        for name, backend in self.storage_backends.items():
            try:
                success = await backend.purge_old_data(retention_days)
                if success:
                    self.logger.info("Retention completed successfully", backend=name)
                else:
                    self.logger.error("Retention completed with errors", backend=name)
            except Exception as e:
                self.logger.error("Error running retention", 
                                backend=name, error=str(e))
    
    def _add_to_candle_cache(self, candle: CandleData) -> None:
        """Add a candle to the cache.
        
        Args:
            candle: The candle data to add
        """
        exchange = candle.exchange
        symbol = candle.symbol
        timeframe = candle.timeframe
        
        # Initialize cache structure if needed
        if exchange not in self.candle_cache:
            self.candle_cache[exchange] = {}
        
        if symbol not in self.candle_cache[exchange]:
            self.candle_cache[exchange][symbol] = {}
        
        if timeframe not in self.candle_cache[exchange][symbol]:
            self.candle_cache[exchange][symbol][timeframe] = []
        
        # Add candle to cache
        cache = self.candle_cache[exchange][symbol][timeframe]
        cache.append(candle)
        
        # Enforce cache size limit
        if len(cache) > self.cache_size:
            cache.pop(0)
    
    def _add_to_orderbook_cache(self, orderbook: OrderBookData) -> None:
        """Add an order book to the cache.
        
        Args:
            orderbook: The order book data to add
        """
        exchange = orderbook.exchange
        symbol = orderbook.symbol
        
        # Initialize cache structure if needed
        if exchange not in self.orderbook_cache:
            self.orderbook_cache[exchange] = {}
        
        # Add orderbook to cache (only keep the latest)
        self.orderbook_cache[exchange][symbol] = orderbook
    
    def _add_to_trade_cache(self, trade: TradeData) -> None:
        """Add a trade to the cache.
        
        Args:
            trade: The trade data to add
        """
        exchange = trade.exchange
        symbol = trade.symbol
        
        # Initialize cache structure if needed
        if exchange not in self.trade_cache:
            self.trade_cache[exchange] = {}
        
        if symbol not in self.trade_cache[exchange]:
            self.trade_cache[exchange][symbol] = []
        
        # Add trade to cache
        cache = self.trade_cache[exchange][symbol]
        cache.append(trade)
        
        # Enforce cache size limit
        if len(cache) > self.cache_size:
            cache.pop(0)
    
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
        if not self.enabled or not self.storage_backends:
            return []
        
        # Use the first available storage backend
        backend = next(iter(self.storage_backends.values()))
        return await backend.get_candles(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
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
        if not self.enabled or not self.storage_backends:
            return None
        
        # Use the first available storage backend
        backend = next(iter(self.storage_backends.values()))
        return await backend.get_orderbook(
            symbol=symbol,
            exchange=exchange,
            timestamp=timestamp
        )
    
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
        if not self.enabled or not self.storage_backends:
            return []
        
        # Use the first available storage backend
        backend = next(iter(self.storage_backends.values()))
        return await backend.get_trades(
            symbol=symbol,
            exchange=exchange,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
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
        # Check cache first
        if self.cache_enabled:
            try:
                cache = self.candle_cache.get(exchange, {}).get(symbol, {}).get(timeframe, [])
                if cache:
                    return cache[-1]
            except Exception:
                pass
        
        if not self.enabled or not self.storage_backends:
            return None
        
        # Use the first available storage backend
        backend = next(iter(self.storage_backends.values()))
        return await backend.get_latest_candle(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe
        ) 