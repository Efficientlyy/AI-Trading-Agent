# Public API Data Provider Implementation

This document outlines the implementation details for the `PublicApiDataProvider` class that will power our realistic paper trading system without requiring exchange accounts.

## Class Structure

```python
from typing import List, Dict, Any, Optional
import pandas as pd
import asyncio
import aiohttp
import time
from datetime import datetime, timezone

from .base_provider import BaseDataProvider
from ..common import logger

class PublicApiDataProvider(BaseDataProvider):
    """
    Data provider implementation using public cryptocurrency APIs.
    Provides real-time and historical market data without requiring exchange accounts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
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
        for source in sources:
            if source == 'coingecko':
                from .api_clients.coingecko_client import CoinGeckoClient
                self.api_clients[source] = CoinGeckoClient()
            elif source == 'cryptocompare':
                from .api_clients.cryptocompare_client import CryptoCompareClient
                self.api_clients[source] = CryptoCompareClient()
            elif source == 'coinapi':
                from .api_clients.coinapi_client import CoinAPIClient
                self.api_clients[source] = CoinAPIClient()
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
        client = self.api_clients[self.primary_source]
        
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
                        backup_client = self.api_clients[backup_source]
                        data = await backup_client.get_historical_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if data is not None and not data.empty:
                            results[symbol] = data
                            break
                    
                    if symbol not in results:
                        logger.warning(f"Could not fetch historical data for {symbol}")
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            # Try backup sources
            for backup_source in self.backup_sources:
                try:
                    backup_client = self.api_clients[backup_source]
                    for symbol in symbols:
                        data = await backup_client.get_historical_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if data is not None and not data.empty:
                            results[symbol] = data
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
                    client = self.api_clients[self.primary_source]
                    
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
                                    'timestamp': timestamp
                                }
                            else:
                                # Update existing data
                                current = self.data_cache[symbol]
                                current['close'] = price_data['price']
                                current['high'] = max(current['high'], price_data['price'])
                                current['low'] = min(current['low'], price_data['price'])
                                current['volume'] = price_data.get('volume_24h', current['volume'])
                                current['timestamp'] = timestamp
                            
                            # If it's been more than the timeframe interval, reset OHLC
                            last_update = self.last_update_time.get(symbol)
                            if last_update is None or (timestamp - last_update).total_seconds() >= 60:  # 1-minute timeframe
                                self.data_cache[symbol]['open'] = price_data['price']
                                self.data_cache[symbol]['high'] = price_data['price']
                                self.data_cache[symbol]['low'] = price_data['price']
                                self.last_update_time[symbol] = timestamp
                        
                        # Put data in the queue for consumers
                        await self.realtime_queue.put(self.data_cache.copy())
            
            except Exception as e:
                logger.error(f"Error in periodic update loop: {e}")
                
                # Try backup sources
                for backup_source in self.backup_sources:
                    try:
                        backup_client = self.api_clients[backup_source]
                        symbols = list(self._subscribed_symbols_realtime)
                        data = await backup_client.get_current_prices(symbols)
                        
                        if data:
                            # Update cache with backup data
                            timestamp = pd.Timestamp.now(tz='UTC')
                            for symbol, price_data in data.items():
                                if symbol in self.data_cache:
                                    self.data_cache[symbol]['close'] = price_data['price']
                                    self.data_cache[symbol]['timestamp'] = timestamp
                            
                            # Put data in the queue for consumers
                            await self.realtime_queue.put(self.data_cache.copy())
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
            "primary_source": self.primary_source,
            "backup_sources": self.backup_sources,
            "update_interval": self.update_interval,
            "realtime_connected": self._realtime_connected,
            "subscribed_symbols": list(self._subscribed_symbols_realtime),
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
```

## API Client Implementations

### CoinGecko Client

```python
import aiohttp
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from ...common import logger

class CoinGeckoClient:
    """Client for the CoinGecko API."""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = None
        self.rate_limit_remaining = 50
        self.rate_limit_reset = 0
        self.coin_map = {}  # Maps trading pairs to CoinGecko IDs
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def _get_coin_map(self):
        """Get mapping of trading pairs to CoinGecko IDs."""
        if not self.coin_map:
            await self._ensure_session()
            
            try:
                async with self.session.get(f"{self.base_url}/coins/list") as response:
                    if response.status == 200:
                        coins = await response.json()
                        # Create a map of symbol to ID
                        for coin in coins:
                            self.coin_map[coin['symbol'].upper()] = coin['id']
                    else:
                        logger.error(f"Failed to get coin list: {response.status}")
            except Exception as e:
                logger.error(f"Error getting coin map: {e}")
    
    async def _convert_pair_to_id(self, pair: str) -> Optional[str]:
        """Convert trading pair (e.g., BTC/USDT) to CoinGecko coin ID."""
        await self._get_coin_map()
        
        # Split the pair
        base, quote = pair.split('/')
        
        # Return the ID for the base currency
        return self.coin_map.get(base.upper())
    
    async def _handle_rate_limit(self):
        """Handle API rate limiting."""
        if self.rate_limit_remaining <= 5:
            wait_time = max(0, self.rate_limit_reset - time.time())
            if wait_time > 0:
                logger.warning(f"Rate limit approaching, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data from CoinGecko."""
        await self._ensure_session()
        await self._handle_rate_limit()
        
        coin_id = await self._convert_pair_to_id(symbol)
        if not coin_id:
            logger.warning(f"Could not find CoinGecko ID for {symbol}")
            return None
        
        # Convert timeframe to days
        days = 1
        if timeframe == '1d':
            days = 1
        elif timeframe == '1h':
            days = max(1, int((end_date - start_date).total_seconds() / 3600))
        
        # CoinGecko uses Unix timestamps in milliseconds
        from_timestamp = int(start_date.timestamp() * 1000)
        to_timestamp = int(end_date.timestamp() * 1000)
        
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': from_timestamp // 1000,
                'to': to_timestamp // 1000
            }
            
            async with self.session.get(url, params=params) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 50))
                self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Process the data
                    prices = data.get('prices', [])
                    volumes = data.get('total_volumes', [])
                    
                    if not prices:
                        return None
                    
                    # Create a DataFrame
                    df_data = []
                    for i, (timestamp, price) in enumerate(prices):
                        volume = volumes[i][1] if i < len(volumes) else 0
                        df_data.append({
                            'timestamp': pd.Timestamp(timestamp, unit='ms', tz='UTC'),
                            'open': price,  # CoinGecko doesn't provide OHLC, just prices
                            'high': price,
                            'low': price,
                            'close': price,
                            'volume': volume
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    
                    return df
                else:
                    logger.error(f"Failed to get historical data: {response.status}")
                    return None
        
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get current prices for multiple symbols."""
        await self._ensure_session()
        await self._handle_rate_limit()
        
        # Convert symbols to CoinGecko IDs
        coin_ids = []
        symbol_to_id = {}
        for symbol in symbols:
            coin_id = await self._convert_pair_to_id(symbol)
            if coin_id:
                coin_ids.append(coin_id)
                symbol_to_id[coin_id] = symbol
        
        if not coin_ids:
            logger.warning("No valid coin IDs found")
            return {}
        
        try:
            url = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': ','.join(coin_ids),
                'order': 'market_cap_desc',
                'per_page': 100,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h'
            }
            
            async with self.session.get(url, params=params) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 50))
                self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                
                if response.status == 200:
                    data = await response.json()
                    
                    result = {}
                    for coin in data:
                        coin_id = coin['id']
                        if coin_id in symbol_to_id:
                            symbol = symbol_to_id[coin_id]
                            result[symbol] = {
                                'price': coin['current_price'],
                                'volume_24h': coin['total_volume'],
                                'market_cap': coin['market_cap'],
                                'price_change_24h': coin['price_change_percentage_24h']
                            }
                    
                    return result
                else:
                    logger.error(f"Failed to get current prices: {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
```

### CryptoCompare Client

```python
import aiohttp
import pandas as pd
from typing import List, Dict, Any, Optional
import time

from ...common import logger

class CryptoCompareClient:
    """Client for the CryptoCompare API."""
    
    def __init__(self):
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.session = None
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data from CryptoCompare."""
        await self._ensure_session()
        
        # Split the pair
        base, quote = symbol.split('/')
        
        # Map timeframe to CryptoCompare format
        timeframe_map = {
            '1m': 'histominute',
            '5m': 'histominute',
            '15m': 'histominute',
            '30m': 'histominute',
            '1h': 'histohour',
            '4h': 'histohour',
            '1d': 'histoday',
            '1w': 'histoday'
        }
        
        # Calculate limit based on timeframe
        limit_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080
        }
        
        endpoint = timeframe_map.get(timeframe, 'histoday')
        limit = min(2000, int((end_date - start_date).total_seconds() / 60 / limit_map.get(timeframe, 1)))
        
        try:
            url = f"{self.base_url}/{endpoint}"
            params = {
                'fsym': base.upper(),
                'tsym': quote.upper(),
                'limit': limit,
                'toTs': int(end_date.timestamp())
            }
            
            # Add aggregation for timeframes
            if timeframe in ['5m', '15m', '30m']:
                params['aggregate'] = int(timeframe[:-1])
            elif timeframe == '4h':
                params['aggregate'] = 4
            elif timeframe == '1w':
                params['aggregate'] = 7
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data['Response'] == 'Success':
                        # Process the data
                        ohlcv_data = data['Data']
                        
                        # Create a DataFrame
                        df_data = []
                        for item in ohlcv_data:
                            df_data.append({
                                'timestamp': pd.Timestamp(item['time'], unit='s', tz='UTC'),
                                'open': item['open'],
                                'high': item['high'],
                                'low': item['low'],
                                'close': item['close'],
                                'volume': item['volumefrom']
                            })
                        
                        df = pd.DataFrame(df_data)
                        df.set_index('timestamp', inplace=True)
                        
                        return df
                    else:
                        logger.error(f"API error: {data['Message']}")
                        return None
                else:
                    logger.error(f"Failed to get historical data: {response.status}")
                    return None
        
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get current prices for multiple symbols."""
        await self._ensure_session()
        
        # Extract base and quote currencies
        fsyms = []
        tsyms = []
        symbol_map = {}
        
        for symbol in symbols:
            base, quote = symbol.split('/')
            fsyms.append(base.upper())
            tsyms.append(quote.upper())
            symbol_map[(base.upper(), quote.upper())] = symbol
        
        try:
            url = f"{self.base_url}/pricemultifull"
            params = {
                'fsyms': ','.join(set(fsyms)),
                'tsyms': ','.join(set(tsyms))
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'RAW' in data:
                        result = {}
                        raw_data = data['RAW']
                        
                        for base in raw_data:
                            for quote in raw_data[base]:
                                if (base, quote) in symbol_map:
                                    symbol = symbol_map[(base, quote)]
                                    coin_data = raw_data[base][quote]
                                    result[symbol] = {
                                        'price': coin_data['PRICE'],
                                        'volume_24h': coin_data['VOLUME24HOUR'],
                                        'market_cap': coin_data.get('MKTCAP', 0),
                                        'price_change_24h': coin_data.get('CHANGEPCT24HOUR', 0)
                                    }
                        
                        return result
                    else:
                        logger.error("No RAW data in response")
                        return {}
                else:
                    logger.error(f"Failed to get current prices: {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
```

## Integration with Data Service

To integrate the new `PublicApiDataProvider` with the existing `DataService`, we need to:

1. Add the new provider to the factory method in `DataService._create_provider()`
2. Update the configuration to use the new provider

```python
def _create_provider(self) -> BaseDataProvider:
    """
    Factory method to create the configured data provider instance.
    """
    provider_name = self.active_provider_name.lower()

    if provider_name == 'mock':
        logger.info(f"Creating MockDataProvider with config: {self.provider_config}")
        return MockDataProvider(config=self.provider_config)
    elif provider_name == 'ccxt':
        logger.info(f"Creating CcxtProvider with config: {self.provider_config}")
        return CcxtProvider(config=self.provider_config)
    elif provider_name == 'public_api':
        logger.info(f"Creating PublicApiDataProvider with config: {self.provider_config}")
        return PublicApiDataProvider(config=self.provider_config)
    else:
        logger.error(f"Unsupported data provider specified: {self.active_provider_name}")
        raise ValueError(f"Unsupported data provider: {self.active_provider_name}")
```

## Configuration Example

```yaml
# config/trading_config.yaml
data_sources:
  active_provider: public_api
  public_api:
    primary_source: coingecko
    backup_sources: [cryptocompare]
    update_interval: 10
    symbols: [BTC/USDT, ETH/USDT, SOL/USDT]
    realtime_timeframe: 1m
```

This implementation provides a robust solution for realistic paper trading without requiring exchange accounts. It leverages free public APIs with fallback mechanisms for reliability and implements a simulated real-time data stream that closely mimics exchange behavior.
