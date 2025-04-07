"""
Mock Data Provider for generating synthetic market data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import timedelta

from .base_provider import BaseDataProvider
from src.common import logger

class MockDataProvider(BaseDataProvider):
    """
    Provides mock historical and real-time data for testing and development.
    Generates predictable, synthetic OHLCV data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info(f"Initialized MockDataProvider with config: {self.config}")
        self._realtime_connected = False
        self._subscribed_symbols = []
        self._last_generated_time = {}

    def _generate_ohlcv(self, start_date: pd.Timestamp, end_date: pd.Timestamp, timeframe: str, symbol: str) -> pd.DataFrame:
        """
        Generates a synthetic OHLCV DataFrame.
        """
        freq_map = {
            '1m': 'T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': 'H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '12h': '12H',
            '1d': 'D',
            '1w': 'W'
        }
        freq = freq_map.get(timeframe)
        if not freq:
            raise ValueError(f"Unsupported timeframe for mock generation: {timeframe}")

        # Generate DatetimeIndex
        dt_index = pd.date_range(start=start_date, end=end_date, freq=freq, tz='UTC')
        if dt_index.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']).set_index(pd.DatetimeIndex([]))

        n_periods = len(dt_index)

        # Base price (symbol dependent for slight variation)
        base_price = 50000 + hash(symbol) % 10000

        # Generate random walks for price changes
        price_changes = np.random.randn(n_periods) * (base_price * 0.01) # 1% std dev
        close_prices = base_price + np.cumsum(price_changes)

        # Generate OHLC based on close
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0] - price_changes[0] # Estimate first open

        high_prices = np.maximum(open_prices, close_prices) + np.random.rand(n_periods) * (base_price * 0.005) # Add small random positive amount
        low_prices = np.minimum(open_prices, close_prices) - np.random.rand(n_periods) * (base_price * 0.005) # Subtract small random positive amount

        # Ensure OHLC consistency
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

        # Generate volume
        volume = np.random.poisson(lam=100, size=n_periods) + np.random.rand(n_periods) * 50

        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dt_index)

        df.index.name = 'timestamp'

        logger.debug(f"Generated mock OHLCV data for {symbol} ({timeframe}) from {start_date} to {end_date}, shape: {df.shape}")
        return df

    async def fetch_historical_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """Fetch mock historical data."""
        logger.info(f"Fetching mock historical data for {symbols} ({timeframe}) between {start_date} and {end_date}")
        results = {}
        for symbol in symbols:
            results[symbol] = self._generate_ohlcv(start_date, end_date, timeframe, symbol)
            self._last_generated_time[symbol] = results[symbol].index[-1] if not results[symbol].empty else start_date

        return results

    async def connect_realtime(self):
        """Simulate connecting to real-time feed."""
        logger.info("MockDataProvider: Connecting to real-time feed (simulated).")
        self._realtime_connected = True

    async def disconnect_realtime(self):
        """Simulate disconnecting from real-time feed."""
        logger.info("MockDataProvider: Disconnecting from real-time feed (simulated).")
        self._realtime_connected = False
        self._subscribed_symbols = []

    async def subscribe_to_symbols(self, symbols: List[str]):
        """Simulate subscribing to symbols."""
        if not self._realtime_connected:
            logger.warning("Cannot subscribe, mock real-time feed not connected.")
            return
        logger.info(f"MockDataProvider: Subscribing to symbols: {symbols}")
        self._subscribed_symbols = list(set(self._subscribed_symbols + symbols))
        # Initialize last generated time if not already set by historical fetch
        for symbol in symbols:
            if symbol not in self._last_generated_time:
                 self._last_generated_time[symbol] = pd.Timestamp.utcnow().floor('min') - timedelta(minutes=1)


    async def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """
        Generate the next mock data point if connected and subscribed.
        Simulates receiving a new Kline/tick.
        """
        if not self._realtime_connected or not self._subscribed_symbols:
            return None

        # Simulate data for one subscribed symbol per call (round-robin or random)
        symbol_to_update = np.random.choice(self._subscribed_symbols)

        last_time = self._last_generated_time.get(symbol_to_update)
        if not last_time:
             logger.warning(f"No last generated time found for {symbol_to_update}")
             return None # Should have been initialized in subscribe

        # Determine the next timestamp (e.g., 1 minute later for simplicity)
        # In a real scenario, this depends on the subscribed timeframe/tick stream
        next_time = last_time + timedelta(minutes=1)

        # Limit generation to current time
        now = pd.Timestamp.utcnow()
        if next_time > now:
            # logger.debug(f"MockDataProvider: No new data for {symbol_to_update} yet (next time {next_time} > now {now}).")
            return None # Simulate no new data yet

        # Generate a single new data point (simulating a closed kline or tick)
        # Reuse _generate_ohlcv for simplicity, just taking the first row
        # Note: Using a fixed timeframe here for simplicity, real mock might need more logic
        new_data_df = self._generate_ohlcv(next_time, next_time, '1m', symbol_to_update)

        if new_data_df.empty:
             return None

        new_data_point = new_data_df.iloc[0].to_dict()
        new_data_point['timestamp'] = new_data_df.index[0]
        new_data_point['symbol'] = symbol_to_update

        # Update last generated time
        self._last_generated_time[symbol_to_update] = new_data_point['timestamp']

        logger.debug(f"MockDataProvider: Generated real-time data point for {symbol_to_update}: {new_data_point}")

        return {
            'type': 'kline', # Or 'tick' depending on simulation
            'data': new_data_point
        }

    def get_supported_timeframes(self) -> List[str]:
        """Return timeframes mock data can be generated for."""
        return ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']

