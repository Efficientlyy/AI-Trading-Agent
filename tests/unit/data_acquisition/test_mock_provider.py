"""
Unit tests for the MockDataProvider.
"""

import pytest
import pandas as pd
from datetime import timedelta, timezone

from src.data_acquisition.mock_provider import MockDataProvider

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_provider():
    """Fixture to create a MockDataProvider instance."""
    return MockDataProvider(config={"generation_seed": 123})

async def test_fetch_historical_data_mock(mock_provider):
    """Test fetching historical data from MockDataProvider."""
    symbols = ["BTC/USD", "ETH/USD"]
    timeframe = "1h"
    start_date = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    end_date = pd.Timestamp("2023-01-01 10:00:00", tz="UTC")

    data = await mock_provider.fetch_historical_data(symbols, timeframe, start_date, end_date)

    assert isinstance(data, dict)
    assert len(data) == 2
    assert "BTC/USD" in data
    assert "ETH/USD" in data

    for symbol, df in data.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.index.name == 'timestamp'
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz == timezone.utc
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        # Check if data is within the requested date range (or close)
        assert df.index.min() >= start_date
        assert df.index.max() <= end_date
        # Check number of rows (11 hours including start and end)
        assert len(df) == 11

async def test_fetch_historical_data_mock_empty_range(mock_provider):
    """Test fetching with a start date after end date."""
    symbols = ["BTC/USD"]
    timeframe = "1d"
    start_date = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    end_date = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")

    data = await mock_provider.fetch_historical_data(symbols, timeframe, start_date, end_date)
    assert isinstance(data["BTC/USD"], pd.DataFrame)
    assert data["BTC/USD"].empty

async def test_realtime_simulation_mock(mock_provider):
    """Test basic real-time simulation flow."""
    symbols = ["SOL/USD"]

    # Should return None initially
    rt_data_before_connect = await mock_provider.get_realtime_data()
    assert rt_data_before_connect is None

    await mock_provider.connect_realtime()
    assert mock_provider._realtime_connected

    # Should still return None before subscription
    rt_data_before_subscribe = await mock_provider.get_realtime_data()
    assert rt_data_before_subscribe is None

    await mock_provider.subscribe_to_symbols(symbols)
    assert "SOL/USD" in mock_provider._subscribed_symbols
    assert "SOL/USD" in mock_provider._last_generated_time # Should be initialized

    # Allow some time to pass virtually for generation
    mock_provider._last_generated_time["SOL/USD"] = pd.Timestamp.now(tz=timezone.utc) - timedelta(minutes=5)

    rt_data = await mock_provider.get_realtime_data()
    assert rt_data is not None
    assert isinstance(rt_data, dict)
    assert rt_data['type'] == 'kline'
    assert isinstance(rt_data['data'], dict)
    assert rt_data['data']['symbol'] == "SOL/USD"
    assert 'timestamp' in rt_data['data']
    assert 'close' in rt_data['data']

    await mock_provider.disconnect_realtime()
    assert not mock_provider._realtime_connected
    assert not mock_provider._subscribed_symbols

    # Should return None after disconnect
    rt_data_after_disconnect = await mock_provider.get_realtime_data()
    assert rt_data_after_disconnect is None

def test_get_supported_timeframes_mock(mock_provider):
    """Test retrieving supported timeframes."""
    timeframes = mock_provider.get_supported_timeframes()
    assert isinstance(timeframes, list)
    assert "1m" in timeframes
    assert "1h" in timeframes
    assert "1d" in timeframes

def test_get_info_mock(mock_provider):
    """Test retrieving provider info."""
    info = mock_provider.get_info()
    assert info["provider_name"] == "MockDataProvider"
    assert isinstance(info["config"], dict)
    assert isinstance(info["supported_timeframes"], list)
