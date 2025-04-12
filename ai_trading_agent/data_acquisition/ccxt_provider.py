"""
CCXT Data Provider

Fetches historical and real-time data using the CCXT library,
supporting various cryptocurrency exchanges.
"""

import ccxt.async_support as ccxt
import pandas as pd
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from .base_provider import BaseDataProvider
from ..common import logger, get_config_value

# CCXT uses milliseconds for timestamps
def datetime_to_milliseconds(dt: pd.Timestamp) -> int:
    """Convert pandas Timestamp (UTC) to milliseconds since epoch."""
    # Ensure timestamp is timezone-aware (UTC)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.tz_localize('UTC')
    elif dt.tzinfo != timezone.utc:
        dt = dt.tz_convert('UTC')
    return int(dt.timestamp() * 1000)

def milliseconds_to_datetime(ms: int) -> pd.Timestamp:
    """Convert milliseconds since epoch to pandas Timestamp (UTC)."""
    return pd.Timestamp(ms, unit='ms', tz='UTC')

class CcxtProvider(BaseDataProvider):
    """
    Data provider implementation using the CCXT library.
    Handles fetching data from configured cryptocurrency exchanges.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.exchange_id = self.config.get('exchange_id', 'binance') # Default to binance
        self.api_key = self.config.get('api_key')
        self.secret_key = self.config.get('secret_key')
        self.password = self.config.get('password') # For exchanges like KuCoin
        self.exchange_options = self.config.get('options', {}) # Custom exchange options

        self.exchange: ccxt.Exchange = self._initialize_exchange()
        self.rate_limit_delay = self.exchange.rateLimit / 1000 # Seconds
        self.markets = None
        self._realtime_connected = False
        self._subscribed_symbols_realtime = set()
        self._realtime_tasks = {}

        logger.info(f"Initialized CcxtProvider for exchange: {self.exchange_id}")

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize the CCXT exchange instance."""
        if not hasattr(ccxt, self.exchange_id):
            logger.error(f"Exchange ID '{self.exchange_id}' not found in CCXT.")
            raise ValueError(f"Invalid exchange ID: {self.exchange_id}")

        exchange_class = getattr(ccxt, self.exchange_id)
        options = {
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'password': self.password,
            'options': self.exchange_options,
            # Enable rate limiting by default
            'enableRateLimit': True,
        }
        # Filter out None values
        options = {k: v for k, v in options.items() if v is not None}

        try:
            return exchange_class(options)
        except ccxt.AuthenticationError as e:
            logger.error(f"CCXT Authentication Error for {self.exchange_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CCXT exchange {self.exchange_id}: {e}")
            raise

    async def _load_markets(self, reload: bool = False):
        """Load markets from the exchange if not already loaded."""
        if self.markets is None or reload:
            try:
                logger.debug(f"Loading markets for {self.exchange_id}...")
                self.markets = await self.exchange.load_markets()
                logger.info(f"Successfully loaded {len(self.markets)} markets for {self.exchange_id}.")
            except ccxt.NetworkError as e:
                logger.error(f"Network error loading markets for {self.exchange_id}: {e}")
                raise
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error loading markets for {self.exchange_id}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error loading markets for {self.exchange_id}: {e}")
                raise

    async def fetch_historical_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical OHLCV data using CCXT."""
        await self._load_markets()
        if not self.exchange.has['fetchOHLCV']:
            logger.error(f"Exchange {self.exchange_id} does not support fetchOHLCV.")
            return {symbol: pd.DataFrame() for symbol in symbols}

        if timeframe not in self.exchange.timeframes:
             logger.warning(f"Timeframe '{timeframe}' might not be supported by {self.exchange_id}. Available: {list(self.exchange.timeframes.keys())}")
             # Allow attempting anyway, CCXT might handle aliases

        results = {}
        start_ms = datetime_to_milliseconds(start_date)
        end_ms = datetime_to_milliseconds(end_date) # CCXT usually fetches *up to* this timestamp

        for symbol in symbols:
            if symbol not in self.markets:
                logger.warning(f"Symbol '{symbol}' not found in {self.exchange_id} markets. Skipping.")
                results[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']).set_index(pd.DatetimeIndex([]))
                continue

            logger.info(f"Fetching historical data for {symbol} ({timeframe}) from {start_date} to {end_date} on {self.exchange_id}")
            all_ohlcv = []
            current_start_ms = start_ms
            limit = self.config.get('fetch_limit', 1000) # Max candles per request

            while current_start_ms < end_ms:
                try:
                    fetch_since_ms = current_start_ms # Store the timestamp used for this fetch request
                    logger.debug(f"Fetching {symbol} {timeframe} from {milliseconds_to_datetime(fetch_since_ms)} (limit: {limit})")
                    # Use asyncio.sleep for rate limiting if exchange.rateLimit is enabled
                    # CCXT handles rate limiting internally if enableRateLimit=True
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since_ms, limit=limit)

                    if not ohlcv:
                        logger.debug(f"No more data returned for {symbol} starting from {milliseconds_to_datetime(fetch_since_ms)}.")
                        break # No more data available for this period

                    all_ohlcv.extend(ohlcv)
                    last_timestamp_ms = ohlcv[-1][0]
                    
                    # Check if the last received timestamp is not later than the requested start time
                    # This prevents infinite loops if the exchange returns overlapping or non-advancing data.
                    if last_timestamp_ms <= fetch_since_ms:
                         logger.warning(
                             f"Timestamp did not advance for {symbol}. Last timestamp ({milliseconds_to_datetime(last_timestamp_ms)}) "
                             f"<= requested 'since' timestamp ({milliseconds_to_datetime(fetch_since_ms)}). Breaking fetch loop."
                         )
                         break

                    # Calculate next start time for the subsequent request
                    current_start_ms = last_timestamp_ms + self.exchange.parse_timeframe(timeframe) * 1000

                    # Optional: Small sleep even with internal rate limiting, just in case
                    # await asyncio.sleep(self.rate_limit_delay * 0.1)

                except ccxt.RateLimitExceeded as e:
                    logger.warning(f"Rate limit exceeded for {self.exchange_id}: {e}. Retrying after delay...")
                    await asyncio.sleep(self.rate_limit_delay * 1.5) # Wait longer on explicit rate limit
                except ccxt.NetworkError as e:
                    logger.warning(f"Network error fetching {symbol} ({self.exchange_id}): {e}. Retrying after delay...")
                    await asyncio.sleep(5) # Wait on network errors
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error fetching {symbol} ({self.exchange_id}): {e}. Skipping symbol.")
                    all_ohlcv = [] # Discard partial data on exchange error
                    break
                except Exception as e:
                    logger.error(f"Unexpected error fetching {symbol} ({self.exchange_id}): {e}. Skipping symbol.")
                    all_ohlcv = []
                    break

            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp')
                # Filter out data outside the requested range (fetch_ohlcv might return more)
                df = df[(df.index >= start_date.tz_convert('UTC')) & (df.index <= end_date.tz_convert('UTC'))]
                df = df[~df.index.duplicated(keep='first')] # Remove potential duplicates
                logger.info(f"Fetched {len(df)} data points for {symbol} ({timeframe}) on {self.exchange_id}")
                results[symbol] = df
            else:
                logger.warning(f"No data fetched for {symbol} ({timeframe}) on {self.exchange_id}")
                results[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']).set_index(pd.DatetimeIndex([]))

        return results

    # --- Real-time Data Methods (Basic Implementation - More complex in practice) ---

    async def connect_realtime(self):
        """Establish connection for real-time data streaming (if supported)."""
        if not self.exchange.has.get('watchOHLCV') and not self.exchange.has.get('watchTicker'):
            logger.warning(f"Exchange {self.exchange_id} does not support real-time OHLCV or Ticker watching via CCXT.")
            return

        logger.info(f"Connecting to real-time feed for {self.exchange_id} (simulated connection setup). Actual connection happens on subscription.")
        # In CCXT's async model, connection often happens implicitly when you start watching
        self._realtime_connected = True # Mark as conceptually connected
        # Ensure exchange is closed properly on application exit
        # Consider adding cleanup logic elsewhere (e.g., in main application shutdown)

    async def disconnect_realtime(self):
        """Disconnect from real-time data stream."""
        logger.info(f"Disconnecting from real-time feed for {self.exchange_id}. Stopping watch tasks.")
        self._realtime_connected = False
        tasks_to_cancel = list(self._realtime_tasks.values())
        self._realtime_tasks = {}
        self._subscribed_symbols_realtime = set()
        for task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("Real-time watch task cancelled successfully.")
        # Explicitly close the exchange connection ONLY if no other operations are pending
        # This can be tricky; often better managed at a higher application level.
        # try:
        #     logger.info(f"Closing CCXT exchange connection for {self.exchange_id}")
        #     await self.exchange.close()
        # except Exception as e:
        #     logger.error(f"Error closing CCXT exchange connection: {e}")

    async def _watch_loop(self, symbol: str, timeframe: str = '1m'):
        """
        Internal loop to watch for real-time data for a specific symbol.
        NOTE: This is a simplified example. Real-world handling needs robust error management,
              reconnection logic, and potentially a queue to pass data back to the DataService.
        """
        if not self.exchange.has['watchOHLCV']:
            logger.error(f"{self.exchange_id} does not support watchOHLCV.")
            # Potentially fallback to watchTicker if needed
            return

        logger.info(f"Starting real-time watch loop for {symbol} ({timeframe}) on {self.exchange_id}")
        while self._realtime_connected and symbol in self._subscribed_symbols_realtime:
            try:
                # watchOHLCV returns a list of candles since the last call (or initial state)
                ohlcv_updates = await self.exchange.watch_ohlcv(symbol, timeframe)
                if symbol not in self._subscribed_symbols_realtime or not self._realtime_connected:
                     break # Stop if unsubscribed or disconnected while awaiting

                if ohlcv_updates:
                    logger.debug(f"Received {len(ohlcv_updates)} real-time OHLCV update(s) for {symbol}:")
                    # TODO: Process these updates - e.g., put them onto an async queue
                    # for the DataService.get_realtime_data() to retrieve.
                    # For now, just logging the latest one.
                    latest_candle = ohlcv_updates[-1]
                    timestamp = milliseconds_to_datetime(latest_candle[0])
                    logger.info(f"Latest {symbol} {timeframe} candle: {timestamp}, O:{latest_candle[1]}, H:{latest_candle[2]}, L:{latest_candle[3]}, C:{latest_candle[4]}, V:{latest_candle[5]}")
                    # --- Placeholder for sending data back --- 
                    # await self.realtime_queue.put({ 'symbol': symbol, 'data': latest_candle }) 
                    # ---------------------------------------

            except asyncio.CancelledError:
                logger.info(f"Watch loop for {symbol} cancelled.")
                break
            except ccxt.NetworkError as e:
                logger.warning(f"Network error in watch loop for {symbol}: {e}. Retrying after delay...")
                if not self._realtime_connected: break
                await asyncio.sleep(10)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error in watch loop for {symbol}: {e}. Stopping watch.")
                self._subscribed_symbols_realtime.discard(symbol)
                break
            except Exception as e:
                logger.error(f"Unexpected error in watch loop for {symbol}: {e}. Stopping watch.")
                self._subscribed_symbols_realtime.discard(symbol)
                break
        logger.info(f"Exited real-time watch loop for {symbol} ({timeframe}) on {self.exchange_id}")

    async def subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to real-time updates for specific symbols."""
        if not self._realtime_connected:
            logger.warning("Cannot subscribe symbols, real-time connection not active.")
            return

        await self._load_markets()
        default_tf = get_config_value('data_sources.realtime_timeframe', '1m')

        for symbol in symbols:
            if symbol not in self.markets:
                logger.warning(f"Cannot subscribe to real-time data for unknown symbol: {symbol}")
                continue
            if symbol not in self._subscribed_symbols_realtime:
                logger.info(f"Subscribing to real-time data for {symbol} (Timeframe: {default_tf})")
                self._subscribed_symbols_realtime.add(symbol)
                # Start the watch loop task for this symbol
                task = asyncio.create_task(self._watch_loop(symbol, default_tf))
                self._realtime_tasks[symbol] = task
            else:
                logger.debug(f"Already subscribed to real-time data for {symbol}")

    async def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest real-time data update (requires a more sophisticated approach).
        This basic version doesn't directly return data from watch loops.
        A better implementation would use an asyncio.Queue populated by the _watch_loop.
        """
        # TODO: Implement queue-based retrieval
        logger.warning("get_realtime_data() with CcxtProvider currently relies on watch loop logging. Implement queue for proper data retrieval.")
        # Example (if queue existed):
        # try:
        #     latest_update = await asyncio.wait_for(self.realtime_queue.get(), timeout=0.1)
        #     self.realtime_queue.task_done()
        #     return latest_update
        # except asyncio.TimeoutError:
        #     return None # No new data in the queue
        return None

    def get_supported_timeframes(self) -> List[str]:
        """
        Get a list of timeframes supported by the exchange.
        May require markets to be loaded first.
        """
        if not self.exchange.timeframes:
            logger.warning(f"Timeframes not loaded for {self.exchange_id}. Attempting to load markets first. Call fetch_historical_data or connect_realtime first.")
            # We avoid calling an async method (_load_markets) in a sync method.
            # Rely on prior calls to load markets.
            return []
        return list(self.exchange.timeframes.keys())

    def get_info(self) -> Dict[str, Any]:
        """Get information about the data provider."""
        base_info = super().get_info()
        base_info.update({
            "exchange_id": self.exchange_id,
            "realtime_connected": self._realtime_connected,
            "realtime_subscriptions": list(self._subscribed_symbols_realtime),
            "loaded_markets_count": len(self.markets) if self.markets else 0,
        })
        return base_info

    async def close(self):
        """
        Cleanly close the exchange connection and stop tasks.
        Should be called during application shutdown.
        """
        logger.info(f"Closing CcxtProvider for {self.exchange_id}...")
        await self.disconnect_realtime() # Ensure loops are stopped
        try:
            if self.exchange and hasattr(self.exchange, 'close'):
                logger.info(f"Closing CCXT exchange connection for {self.exchange_id}")
                await self.exchange.close()
        except Exception as e:
            logger.error(f"Error closing CCXT exchange connection: {e}")
