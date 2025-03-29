"""Tests for the Binance exchange connector."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import aiohttp
import pytest
from aiohttp import WSMsgType

from src.data_collection.connectors.binance import BinanceConnector
from src.models.market_data import TimeFrame


@pytest.fixture
def connector():
    """Fixture to provide a Binance connector instance."""
    return BinanceConnector()


@pytest.mark.asyncio
async def test_convert_symbols(connector):
    """Test symbol format conversion."""
    # Test standard to Binance format
    assert connector._convert_to_binance_symbol("BTC/USDT") == "BTCUSDT"
    assert connector._convert_to_binance_symbol("ETH/BTC") == "ETHBTC"
    assert connector._convert_to_binance_symbol("invalid") is None
    
    # Test Binance to standard format
    assert connector._convert_from_binance_symbol("BTCUSDT") == "BTC/USDT"
    assert connector._convert_from_binance_symbol("ETHBTC") == "ETH/BTC"
    
    # Edge cases for the reverse conversion should still work
    assert connector._convert_from_binance_symbol("ABCDEF") == "ABCDE/F"  # Best guess


@pytest.mark.asyncio
async def test_fetch_available_symbols(connector):
    """Test fetching available symbols."""
    # Mock response data
    mock_response = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT"
            },
            {
                "symbol": "ETHBTC",
                "status": "TRADING",
                "baseAsset": "ETH",
                "quoteAsset": "BTC"
            },
            {
                "symbol": "LTCUSDT",
                "status": "BREAK",  # Not trading
                "baseAsset": "LTC",
                "quoteAsset": "USDT"
            }
        ]
    }
    
    # Mock the session get method to properly handle async context manager
    mock_response_obj = MagicMock()
    mock_response_obj.status = 200
    mock_response_obj.json = AsyncMock(return_value=mock_response)
    
    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response_obj)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    
    with patch.object(connector, "session") as mock_session:
        mock_session.get = MagicMock(return_value=mock_context_manager)
        
        # Call the method
        symbols = connector.fetch_available_symbols()
        
        # Check results
        assert len(symbols) == 2
        assert "BTC/USDT" in symbols
        assert "ETH/BTC" in symbols
        assert "LTC/USDT" not in symbols  # Not trading


@pytest.mark.asyncio
async def test_fetch_candles(connector):
    """Test fetching candles."""
    # Mock response data
    mock_candles = [
        [
            1625097600000,  # Open time
            "35000.0",      # Open
            "35100.0",      # High
            "34900.0",      # Low
            "35050.0",      # Close
            "10.5",         # Volume
            1625097899999,  # Close time
            "367500.0",     # Quote asset volume
            100,            # Number of trades
            "5.25",         # Taker buy base asset volume
            "183750.0",     # Taker buy quote asset volume
            "0"             # Ignore
        ],
        [
            1625097900000,  # Open time
            "35050.0",      # Open
            "35200.0",      # High
            "35000.0",      # Low
            "35150.0",      # Close
            "8.2",          # Volume
            1625098199999,  # Close time
            "287900.0",     # Quote asset volume
            80,             # Number of trades
            "4.1",          # Taker buy base asset volume
            "143950.0",     # Taker buy quote asset volume
            "0"             # Ignore
        ]
    ]
    
    # Mock the session get method
    mock_response_obj = MagicMock()
    mock_response_obj.status = 200
    mock_response_obj.json = AsyncMock(return_value=mock_candles)
    
    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response_obj)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    
    with patch.object(connector, "session") as mock_session:
        mock_session.get = MagicMock(return_value=mock_context_manager)
        
        # Call the method
        candles = await connector.fetch_candles("BTC/USDT", TimeFrame.MINUTE_1)
        
        # Check results
        assert len(candles) == 2
        assert candles[0].symbol == "BTC/USDT"
        assert candles[0].exchange == "binance"
        assert candles[0].timeframe == TimeFrame.MINUTE_1
        assert candles[0].open == 35000.0
        assert candles[0].high == 35100.0
        assert candles[0].low == 34900.0
        assert candles[0].close == 35050.0
        assert candles[0].volume == 10.5
        assert isinstance(candles[0].timestamp, datetime)


@pytest.mark.asyncio
async def test_fetch_orderbook(connector):
    """Test fetching order book."""
    # Mock response data
    mock_orderbook = {
        "lastUpdateId": 1027024,
        "bids": [
            ["35000.0", "3.5"],
            ["34990.0", "2.8"]
        ],
        "asks": [
            ["35010.0", "2.1"],
            ["35020.0", "4.3"]
        ]
    }
    
    # Mock the session get method
    mock_response_obj = MagicMock()
    mock_response_obj.status = 200
    mock_response_obj.json = AsyncMock(return_value=mock_orderbook)
    
    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response_obj)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    
    with patch.object(connector, "session") as mock_session:
        mock_session.get = MagicMock(return_value=mock_context_manager)
        
        # Call the method
        orderbook = await connector.fetch_orderbook("BTC/USDT")
        
        # Check results
        assert orderbook is not None
        assert orderbook.symbol == "BTC/USDT"
        assert orderbook.exchange == "binance"
        assert isinstance(orderbook.timestamp, datetime)
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.bids[0]["price"] == 35000.0
        assert orderbook.bids[0]["size"] == 3.5
        assert orderbook.asks[0]["price"] == 35010.0
        assert orderbook.asks[0]["size"] == 2.1


@pytest.mark.asyncio
async def test_fetch_trades(connector):
    """Test fetching trades."""
    # Mock response data
    mock_trades = [
        {
            "id": 28457,
            "price": "35000.0",
            "qty": "0.5",
            "time": 1625097600000,
            "isBuyerMaker": True,
            "isBestMatch": True
        },
        {
            "id": 28458,
            "price": "35010.0",
            "qty": "0.8",
            "time": 1625097610000,
            "isBuyerMaker": False,
            "isBestMatch": True
        }
    ]
    
    # Mock the session get method
    mock_response_obj = MagicMock()
    mock_response_obj.status = 200
    mock_response_obj.json = AsyncMock(return_value=mock_trades)
    
    mock_context_manager = MagicMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response_obj)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    
    with patch.object(connector, "session") as mock_session:
        mock_session.get = MagicMock(return_value=mock_context_manager)
        
        # Call the method
        trades = await connector.fetch_trades("BTC/USDT")
        
        # Check results
        assert len(trades) == 2
        assert trades[0].symbol == "BTC/USDT"
        assert trades[0].exchange == "binance"
        assert trades[0].price == 35000.0
        assert trades[0].size == 0.5
        assert trades[0].side == "buy"  # isBuyerMaker=True means sell order was filled
        assert trades[1].side == "sell"  # isBuyerMaker=False means buy order was filled
        assert isinstance(trades[0].timestamp, datetime)


@pytest.mark.asyncio
async def test_websocket_message_processing(connector):
    """Test WebSocket message processing."""
    # Mock WebSocket and session
    mock_ws = MagicMock()
    mock_session = MagicMock()
    connector.ws = mock_ws
    connector.session = mock_session
    
    # Mock publish methods
    connector.publish_trade_data = AsyncMock()
    
    # Create a trade message
    trade_msg = {
        "e": "trade",
        "s": "BTCUSDT",
        "t": 123456,
        "p": "35000.0",
        "q": "0.5",
        "T": 1625097600000,
        "m": False  # Is buyer maker
    }
    
    # Call the handler
    await connector._handle_ws_message(trade_msg)
    
    # Check that publish_trade_data was called
    assert connector.publish_trade_data.called
    call_args = connector.publish_trade_data.call_args[0]
    trade = call_args[0]
    assert trade.symbol == "BTC/USDT"
    assert trade.price == 35000.0
    assert trade.size == 0.5
    assert trade.side == "buy"  # m=False means buyer is taker


@pytest.mark.asyncio
async def test_get_poll_interval(connector):
    """Test getting the poll interval for different timeframes."""
    assert connector._get_poll_interval(TimeFrame.MINUTE_1) == 30.0
    assert connector._get_poll_interval(TimeFrame.MINUTE_5) == 60.0
    assert connector._get_poll_interval(TimeFrame.HOUR_1) == 15 * 60.0
    assert connector._get_poll_interval(TimeFrame.DAY_1) == 60 * 60.0
    
    # Test unknown timeframe defaults to 60 seconds
    class UnknownTimeFrame:
        value = "unknown"
    
    unknown = UnknownTimeFrame()
    assert connector._get_poll_interval(unknown) == 60.0


@pytest.mark.asyncio
async def test_update_subscriptions(connector):
    """Test updating WebSocket subscriptions."""
    # Setup mock WebSocket
    connector.ws = MagicMock()
    connector.ws.send_json = AsyncMock()
    
    # Setup subscriptions
    connector.subscribed_orderbooks = {"BTC/USDT", "ETH/USDT"}
    connector.subscribed_trades = {"BTC/USDT"}
    connector.ws_subscriptions = {"btcusdt@depth"}  # Already subscribed to one
    
    # Call the method
    connector._update_subscriptions()
    
    # Check that ws.send_json was called with correct parameters
    assert connector.ws.send_json.called
    
    # There should be two calls - one to subscribe to new streams
    call_args = connector.ws.send_json.call_args_list
    
    # Should have called subscribe for ethusdt@depth and btcusdt@trade
    subscribed_params = None
    for call in call_args:
        args = call[0][0]
        if args["method"] = = "SUBSCRIBE":
            subscribed_params = args["params"]
    
    assert subscribed_params is not None
    assert set(subscribed_params) == {"ethusdt@depth", "btcusdt@trade"}
    
    # Check that ws_subscriptions was updated
    assert connector.ws_subscriptions == {"btcusdt@depth", "ethusdt@depth", "btcusdt@trade"}