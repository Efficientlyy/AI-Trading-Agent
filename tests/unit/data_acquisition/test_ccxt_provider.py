"""
Unit tests for the CcxtProvider.
Uses mocking to avoid actual network calls.
"""

import pytest
import pandas as pd
from datetime import timezone
from unittest.mock import patch, AsyncMock, MagicMock
import ccxt.async_support as ccxt

from ai_trading_agent.data_acquisition.ccxt_provider import CcxtProvider, datetime_to_milliseconds

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# Sample config for tests
@pytest.fixture
def ccxt_config():
    return {
        'exchange_id': 'binance',
        'fetch_limit': 5 # Smaller limit for tests
    }

# Mock CCXT Exchange object
@pytest.fixture
def mock_ccxt_exchange():
    mock_exchange = AsyncMock(spec=ccxt.Exchange)
    mock_exchange.id = 'binance'
    mock_exchange.rateLimit = 100 # ms
    mock_exchange.timeframes = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
        '1d': '1d', '1w': '1w', '1M': '1M'
    }
    mock_exchange.has = {'fetchOHLCV': True, 'watchOHLCV': True}
    mock_exchange.load_markets = AsyncMock(return_value={
        'BTC/USDT': {'symbol': 'BTC/USDT', 'base': 'BTC', 'quote': 'USDT'},
        'ETH/USDT': {'symbol': 'ETH/USDT', 'base': 'ETH', 'quote': 'USDT'},
    })
    # Configure fetch_ohlcv mock
    async def mock_fetch_ohlcv(symbol, timeframe, since, limit):
        # Simulate returning data based on 'since' to test pagination
        start_dt = pd.Timestamp(since, unit='ms', tz='UTC')
        if start_dt.hour >= 3: # Stop returning data after 3 AM for test
            return []

        # Generate 5 candles (limit)
        timestamps = pd.date_range(start=start_dt, periods=limit, freq=timeframe, tz='UTC')
        data = []
        base_price = 60000 if symbol == 'BTC/USDT' else 4000
        for i, ts in enumerate(timestamps):
            ms = datetime_to_milliseconds(ts.to_pydatetime())
            price = base_price + i * 10
            data.append([ms, price, price + 5, price - 5, price + 2, 100 + i])
        return data

    mock_exchange.fetch_ohlcv = AsyncMock(side_effect=mock_fetch_ohlcv)
    mock_exchange.close = AsyncMock()
    # Mock parse_timeframe to return seconds (e.g., 3600 for '1h')
    mock_exchange.parse_timeframe = MagicMock(return_value=3600) # Assumes '1h' tests primarily
    return mock_exchange

# Patch ccxt library to return the mock exchange
@pytest.fixture
def patched_ccxt_provider(ccxt_config, mock_ccxt_exchange):
    with patch('ccxt.async_support.binance', return_value=mock_ccxt_exchange) as mock_init:
        provider = CcxtProvider(config=ccxt_config)
        # Manually assign the already mocked exchange instance after init
        provider.exchange = mock_ccxt_exchange
        yield provider # Yield the provider instance for the test
        # Fixture teardown: Try closing the provider cleanly
        # This helps ensure the mocked exchange.close() is awaited if needed
        # Note: This assumes the test itself doesn't explicitly call close.
        # If tests call close, this might be redundant or cause issues.
        # A better approach might involve a context manager fixture.
        async def close_provider():
            if hasattr(provider, 'close') and asyncio.iscoroutinefunction(provider.close):
                await provider.close()
        
        # Schedule the cleanup using pytest-asyncio's event loop management
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # Schedule cleanup but don't necessarily wait here if loop is closing
            loop.create_task(close_provider())
        except RuntimeError: # Loop might already be closed
            pass 

async def test_initialize_ccxt_provider(patched_ccxt_provider):
    """Test provider initialization and market loading."""
    provider = patched_ccxt_provider
    assert provider.exchange_id == 'binance'
    assert provider.exchange is not None
    # Test lazy loading of markets
    assert provider.markets is None
    await provider._load_markets() # Call explicitly for test
    assert provider.markets is not None
    assert "BTC/USDT" in provider.markets
    provider.exchange.load_markets.assert_called_once()

async def test_fetch_historical_data_ccxt(patched_ccxt_provider):
    """Test fetching historical data with pagination."""
    provider = patched_ccxt_provider
    symbols = ["BTC/USDT"]
    timeframe = "1h"
    start_date = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    # End date will cause multiple fetches because limit is 5 and we stop generating mock data at 3 AM
    end_date = pd.Timestamp("2023-01-01 05:00:00", tz="UTC")

    # Pre-load markets
    await provider._load_markets()

    data = await provider.fetch_historical_data(symbols, timeframe, start_date, end_date)

    assert "BTC/USDT" in data
    df = data["BTC/USDT"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # The loop runs once: fetch starts at 00:00, mock returns 5 candles (00:00 to 04:00).
    # The next loop start would be 05:00, which == end_date, so loop stops.
    assert len(df) == 5
    # The mock fetch_ohlcv generates 5 candles starting from 00:00.
    assert df.index.min() == start_date
    assert df.index.max() == pd.Timestamp("2023-01-01 04:00:00", tz="UTC") # Max timestamp is the 5th candle (04:00)
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']

    # Check that fetch_ohlcv was called exactly once
    assert provider.exchange.fetch_ohlcv.call_count == 1 # Only called once for 00:00

async def test_fetch_historical_data_ccxt_unsupported_symbol(patched_ccxt_provider):
    """Test fetching data for a symbol not in loaded markets."""
    provider = patched_ccxt_provider
    symbols = ["XYZ/ABC"] # Does not exist in mock markets
    timeframe = "1h"
    start_date = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    end_date = pd.Timestamp("2023-01-01 05:00:00", tz="UTC")
    await provider._load_markets()

    data = await provider.fetch_historical_data(symbols, timeframe, start_date, end_date)
    assert "XYZ/ABC" in data
    assert data["XYZ/ABC"].empty

async def test_fetch_historical_data_ccxt_exchange_error(patched_ccxt_provider):
    """Test handling of exchange errors during fetch."""
    provider = patched_ccxt_provider
    await provider._load_markets()
    # Configure mock to raise an error on the first call
    provider.exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("Test exchange error")

    symbols = ["BTC/USDT"]
    timeframe = "1h"
    start_date = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    end_date = pd.Timestamp("2023-01-01 05:00:00", tz="UTC")

    data = await provider.fetch_historical_data(symbols, timeframe, start_date, end_date)
    assert "BTC/USDT" in data
    assert data["BTC/USDT"].empty # Should return empty df on error

def test_get_supported_timeframes_ccxt(patched_ccxt_provider):
    """Test getting timeframes from the mocked exchange."""
    provider = patched_ccxt_provider
    # Timeframes are set directly on the mock exchange in the fixture
    timeframes = provider.get_supported_timeframes()
    assert timeframes == ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']

async def test_close_ccxt_provider(patched_ccxt_provider):
    """Test the close method calls exchange.close."""
    provider = patched_ccxt_provider
    await provider.close()
    provider.exchange.close.assert_called_once()
