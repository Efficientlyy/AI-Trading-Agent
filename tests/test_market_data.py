"""Tests for market data models."""

import unittest
from datetime import datetime, timezone

from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData


class TestMarketData(unittest.TestCase):
    """Test cases for market data models."""

    def test_candle_data_validation(self):
        """Test validation for CandleData."""
        # Valid candle
        valid_candle = CandleData(
            symbol="BTC/USDT",
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.HOUR_1,
            open=50000.0,
            high=51000.0,
            low=49500.0,
            close=50500.0,
            volume=100.5
        )
        self.assertEqual(valid_candle.symbol, "BTC/USDT")
        self.assertEqual(valid_candle.exchange, "binance")
        
        # Test range property
        self.assertEqual(valid_candle.range, 1500.0)
        
        # Test body property
        self.assertEqual(valid_candle.body, 500.0)
        
        # Test is_bullish property
        self.assertTrue(valid_candle.is_bullish)
        
        # Invalid symbol
        with self.assertRaises(ValueError) as context:
            CandleData(
                symbol="BTCUSDT",  # Missing /
                exchange="binance",
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.HOUR_1,
                open=50000.0,
                high=51000.0,
                low=49500.0,
                close=50500.0,
                volume=100.5
            )
        self.assertIn("Symbol must be in format", str(context.exception))
        
        # Invalid price (negative)
        with self.assertRaises(ValueError) as context:
            CandleData(
                symbol="BTC/USDT",
                exchange="binance",
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.HOUR_1,
                open=50000.0,
                high=51000.0,
                low=-100.0,  # Negative price
                close=50500.0,
                volume=100.5
            )
        self.assertIn("Price must be positive", str(context.exception))
        
        # Invalid volume (negative)
        with self.assertRaises(ValueError) as context:
            CandleData(
                symbol="BTC/USDT",
                exchange="binance",
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.HOUR_1,
                open=50000.0,
                high=51000.0,
                low=49500.0,
                close=50500.0,
                volume=-10.0  # Negative volume
            )
        self.assertIn("Volume must be non-negative", str(context.exception))

    def test_order_book_data_validation(self):
        """Test validation for OrderBookData."""
        # Valid order book
        valid_order_book = OrderBookData(
            symbol="BTC/USDT",
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
            bids=[{"price": 50000.0, "size": 1.5}, {"price": 49900.0, "size": 2.0}],
            asks=[{"price": 50100.0, "size": 1.0}, {"price": 50200.0, "size": 3.0}]
        )
        
        # Test best_bid property
        self.assertEqual(valid_order_book.best_bid, 50000.0)
        
        # Test best_ask property
        self.assertEqual(valid_order_book.best_ask, 50100.0)
        
        # Test spread property
        self.assertEqual(valid_order_book.spread, 100.0)
        
        # Test spread_percent property
        self.assertEqual(valid_order_book.spread_percent, 0.2)
        
        # Invalid bids (missing size)
        with self.assertRaises(ValueError) as context:
            OrderBookData(
                symbol="BTC/USDT",
                exchange="binance",
                timestamp=datetime.now(timezone.utc),
                bids=[{"price": 50000.0}],  # Missing size
                asks=[{"price": 50100.0, "size": 1.0}]
            )
        error_msg = str(context.exception)
        self.assertIn("bids", error_msg)
        self.assertIn("Each order must have", error_msg)
        
        # Invalid asks (negative price)
        with self.assertRaises(ValueError) as context:
            OrderBookData(
                symbol="BTC/USDT",
                exchange="binance",
                timestamp=datetime.now(timezone.utc),
                bids=[{"price": 50000.0, "size": 1.5}],
                asks=[{"price": -50100.0, "size": 1.0}]  # Negative price
            )
        error_msg = str(context.exception)
        self.assertIn("asks", error_msg)
        self.assertIn("Price and size must be positive", error_msg)

    def test_trade_data_validation(self):
        """Test validation for TradeData."""
        # Valid trade
        valid_trade = TradeData(
            symbol="BTC/USDT",
            exchange="binance",
            timestamp=datetime.now(timezone.utc),
            price=50000.0,
            size=1.5,
            side="buy"
        )
        self.assertEqual(valid_trade.symbol, "BTC/USDT")
        self.assertEqual(valid_trade.side, "buy")
        
        # Invalid side
        with self.assertRaises(ValueError) as context:
            TradeData(
                symbol="BTC/USDT",
                exchange="binance",
                timestamp=datetime.now(timezone.utc),
                price=50000.0,
                size=1.5,
                side="invalid"  # Invalid side
            )
        self.assertIn("Side must be 'buy' or 'sell'", str(context.exception))
        
        # Invalid price (zero)
        with self.assertRaises(ValueError) as context:
            TradeData(
                symbol="BTC/USDT",
                exchange="binance",
                timestamp=datetime.now(timezone.utc),
                price=0.0,  # Zero price
                size=1.5,
                side="buy"
            )
        self.assertIn("Value must be positive", str(context.exception))


if __name__ == "__main__":
    unittest.main()
