"""
Public API Data Provider

Provides market data from public cryptocurrency APIs without requiring exchange accounts.
Uses multiple data sources with fallback mechanisms for reliability.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import asyncio
import time
from datetime import datetime, timezone

from .base_provider import BaseDataProvider
from .api_clients.coingecko_client import CoinGeckoClient
from .api_clients.cryptocompare_client import CryptoCompareClient
from ..common import logger

class PublicApiDataProvider(BaseDataProvider):
    """
    Data provider implementation using public cryptocurrency APIs.
    Provides real-time and historical market data without requiring exchange accounts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PublicApiDataProvider with configuration."""
        super().__init__(config)
        self.provider_name = "public_api"
        
        # Configuration
        self.primary_source = self.config.get('primary_source', 'coingecko')
        self.backup_sources = self.config.get('backup_sources', ['cryptocompare'])
        self.update_interval = self.config.get('update_interval', 10)  # seconds
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        
        # Initialize API clients
        self.api_clients = {}
        self._initialize_api_clients()
        
        # Data cache
        self.data_cache = {}
        self.last_update_time = {}
        
        # Real-time streaming
        self._realtime_connected = False
        self._subscribed_symbols_realtime = set()
        self._realtime_tasks = {}
        self.realtime_queue = asyncio.Queue(maxsize=1000)
        
        logger.info(f"Initialized PublicApiDataProvider with primary source: {self.primary_source}")
    
    def _initialize_api_clients(self):
        """Initialize API clients for each configured data source."""
        sources = [self.primary_source] + self.backup_sources
        for source in set(sources):  # Use set to avoid duplicates
            if source == 'coingecko':
                self.api_clients[source] = CoinGeckoClient()
            elif source == 'cryptocompare':
                self.api_clients[source] = CryptoCompareClient()
            else:
                logger.warning(f"Unknown API source: {source}")
    
    async def fetch_historical_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical OHLCV data using public APIs."""
        logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        results = {}
        client = self.api_clients.get(self.primary_source)
        
        if not client:
            logger.error(f"Primary source {self.primary_source} not available")
            return results
        
        try:
            for symbol in symbols:
                data = await client.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                if data is not None and not data.empty:
                    results[symbol] = data
                else:
                    # Try backup sources
                    for backup_source in self.backup_sources:
                        backup_client = self.api_clients.get(backup_source)
                        if not backup_client:
                            continue
                            
                        data = await backup_client.get_historical_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if data is not None and not data.empty:
                            results[symbol] = data
                            logger.info(f"Used backup source {backup_source} for {symbol}")
                            break
                    
                    if symbol not in results:
                        logger.warning(f"Could not fetch historical data for {symbol} from any source")
        
        except Exception as e:
            logger.error(f"Error fetching historical data from primary source: {e}")
            # Try backup sources
            for backup_source in self.backup_sources:
                try:
                    backup_client = self.api_clients.get(backup_source)
                    if not backup_client:
                        continue
                        
                    for symbol in symbols:
                        if symbol in results:
                            continue  # Already have data for this symbol
                            
                        data = await backup_client.get_historical_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if data is not None and not data.empty:
                            results[symbol] = data
                            logger.info(f"Used backup source {backup_source} for {symbol}")
                except Exception as backup_e:
                    logger.error(f"Error with backup source {backup_source}: {backup_e}")
        
        return results
    
    async def connect_realtime(self):
        """Establish connection for real-time data streaming."""
        if self._realtime_connected:
            logger.warning("Real-time connection already established")
            return
        
        logger.info(f"Establishing real-time connection using {self.primary_source}")
        
        try:
            # Some sources like CoinGecko don't have true WebSocket APIs
            # We'll simulate real-time with periodic REST API calls
            self._realtime_connected = True
            
            # Start a background task to periodically fetch data
            task = asyncio.create_task(self._periodic_update_loop())
            self._realtime_tasks['periodic_update'] = task
            
            logger.info(f"Successfully established simulated real-time connection")
        except Exception as e:
            logger.error(f"Failed to establish real-time connection: {e}")
            self._realtime_connected = False
    
    async def disconnect_realtime(self):
        """Disconnect from real-time data stream."""
        if not self._realtime_connected:
            return
        
        logger.info("Disconnecting from real-time data stream")
        
        # Cancel all real-time tasks
        for name, task in self._realtime_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._realtime_tasks = {}
        self._realtime_connected = False
        self._subscribed_symbols_realtime.clear()
        
        logger.info("Successfully disconnected from real-time data stream")
    
    async def _periodic_update_loop(self):
        """Periodically fetch data to simulate real-time updates."""
        logger.info(f"Starting periodic update loop with interval: {self.update_interval}s")
        
        while self._realtime_connected:
            try:
                # Only fetch data for subscribed symbols
                if self._subscribed_symbols_realtime:
                    symbols = list(self._subscribed_symbols_realtime)
                    client = self.api_clients.get(self.primary_source)
                    
                    if not client:
                        logger.error(f"Primary source {self.primary_source} not available")
                        await asyncio.sleep(self.update_interval)
                        continue
                    
                    # Get current market data
                    data = await client.get_current_prices(symbols)
                    
                    if data:
                        # Update cache
                        timestamp = pd.Timestamp.now(tz='UTC')
                        for symbol, price_data in data.items():
                            # Create OHLCV-like structure
                            if symbol not in self.data_cache:
                                self.data_cache[symbol] = {
                                    'open': price_data['price'],
                                    'high': price_data['price'],
                                    'low': price_data['price'],
                                    'close': price_data['price'],
                                    'volume': price_data.get('volume_24h', 0),
                                    'timestamp': timestamp,
                                    'source': self.primary_source
                                }
                            else:
                                # Update existing data
                                current = self.data_cache[symbol]
                                current['close'] = price_data['price']
                                current['high'] = max(current['high'], price_data['price'])
                                current['low'] = min(current['low'], price_data['price'])
                                current['volume'] = price_data.get('volume_24h', current['volume'])
                                current['timestamp'] = timestamp
                                current['source'] = self.primary_source
                            
                            # If it's been more than the timeframe interval, reset OHLC
                            last_update = self.last_update_time.get(symbol)
                            if last_update is None or (timestamp - last_update).total_seconds() >= 60:  # 1-minute timeframe
                                self.data_cache[symbol]['open'] = price_data['price']
                                self.data_cache[symbol]['high'] = price_data['price']
                                self.data_cache[symbol]['low'] = price_data['price']
                                self.last_update_time[symbol] = timestamp
                        
                        # Put data in the queue for consumers
                        await self.realtime_queue.put(self.data_cache.copy())
                    else:
                        logger.warning(f"No data returned from {self.primary_source}")
                        
                        # Try backup sources
                        for backup_source in self.backup_sources:
                            backup_client = self.api_clients.get(backup_source)
                            if not backup_client:
                                continue
                                
                            backup_data = await backup_client.get_current_prices(symbols)
                            if backup_data:
                                # Update cache with backup data
                                timestamp = pd.Timestamp.now(tz='UTC')
                                for symbol, price_data in backup_data.items():
                                    if symbol in self.data_cache:
                                        self.data_cache[symbol]['close'] = price_data['price']
                                        self.data_cache[symbol]['timestamp'] = timestamp
                                        self.data_cache[symbol]['source'] = backup_source
                                    else:
                                        self.data_cache[symbol] = {
                                            'open': price_data['price'],
                                            'high': price_data['price'],
                                            'low': price_data['price'],
                                            'close': price_data['price'],
                                            'volume': price_data.get('volume_24h', 0),
                                            'timestamp': timestamp,
                                            'source': backup_source
                                        }
                                
                                # Put data in the queue for consumers
                                await self.realtime_queue.put(self.data_cache.copy())
                                logger.info(f"Used backup source {backup_source} for real-time data")
                                break
            
            except Exception as e:
                logger.error(f"Error in periodic update loop: {e}")
                
                # Try backup sources
                for backup_source in self.backup_sources:
                    try:
                        backup_client = self.api_clients.get(backup_source)
                        if not backup_client:
                            continue
                            
                        symbols = list(self._subscribed_symbols_realtime)
                        data = await backup_client.get_current_prices(symbols)
                        
                        if data:
                            # Update cache with backup data
                            timestamp = pd.Timestamp.now(tz='UTC')
                            for symbol, price_data in data.items():
                                if symbol in self.data_cache:
                                    self.data_cache[symbol]['close'] = price_data['price']
                                    self.data_cache[symbol]['timestamp'] = timestamp
                                    self.data_cache[symbol]['source'] = backup_source
                                else:
                                    self.data_cache[symbol] = {
                                        'open': price_data['price'],
                                        'high': price_data['price'],
                                        'low': price_data['price'],
                                        'close': price_data['price'],
                                        'volume': price_data.get('volume_24h', 0),
                                        'timestamp': timestamp,
                                        'source': backup_source
                                    }
                            
                            # Put data in the queue for consumers
                            await self.realtime_queue.put(self.data_cache.copy())
                            logger.info(f"Used backup source {backup_source} for real-time data")
                            break  # Successfully got data from backup
                    except Exception as backup_e:
                        logger.error(f"Error with backup source {backup_source}: {backup_e}")
            
            # Wait for next update
            await asyncio.sleep(self.update_interval)
    
    async def subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to real-time updates for specific symbols."""
        if not self._realtime_connected:
            logger.warning("Cannot subscribe symbols, real-time connection not active")
            return
        
        for symbol in symbols:
            if symbol not in self._subscribed_symbols_realtime:
                logger.info(f"Subscribing to real-time data for {symbol}")
                self._subscribed_symbols_realtime.add(symbol)
    
    async def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """Fetch the latest real-time data update from the queue."""
        try:
            latest_update = await asyncio.wait_for(self.realtime_queue.get(), timeout=0.1)
            self.realtime_queue.task_done()
            return latest_update
        except asyncio.TimeoutError:
            return None  # No new data in the queue
    
    def get_supported_timeframes(self) -> List[str]:
        """Get a list of timeframes supported by the provider."""
        # Common timeframes supported by most public APIs
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the data provider."""
        base_info = super().get_info()
        base_info.update({
            "provider_name": self.provider_name,
            "primary_source": self.primary_source,
            "backup_sources": self.backup_sources,
            "update_interval": self.update_interval,
            "realtime_connected": self._realtime_connected,
            "subscribed_symbols": list(self._subscribed_symbols_realtime),
            "cache_size": len(self.data_cache),
        })
        return base_info
    
    async def close(self):
        """Cleanly close connections and stop tasks."""
        logger.info(f"Closing PublicApiDataProvider")
        await self.disconnect_realtime()
        
        # Close API clients if they have close methods
        for source, client in self.api_clients.items():
            if hasattr(client, 'close') and callable(client.close):
                await client.close()
