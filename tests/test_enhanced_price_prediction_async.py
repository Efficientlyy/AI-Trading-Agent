"""Async unit tests for enhanced price prediction strategy."""

import unittest
import asyncio
from datetime import datetime, timedelta
from src.ml.models.enhanced_price_prediction import EnhancedPricePredictionStrategy
from src.models.market_data import CandleData, OrderBookData, TimeFrame
import numpy as np

def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

class TestEnhancedPricePredictionStrategy(unittest.TestCase):
    @async_test
    async def setUp(self):
        """Set up test environment before each test case."""
        self.symbol = "BTC/USD"
        self.exchange = "binance"
        self.timeframe = TimeFrame.ONE_HOUR
        self.strategy = EnhancedPricePredictionStrategy(
            strategy_id="test_strategy",
            trading_symbols=[self.symbol],
            lookback_window=100,
            prediction_horizon=24,
            confidence_threshold=0.7,
            timeframe=self.timeframe
        )
        await self.strategy._strategy_initialize()

    @async_test
    async def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.trading_symbols, [self.symbol])
        self.assertEqual(self.strategy.lookback_window, 100)
        self.assertEqual(self.strategy.prediction_horizon, 24)
        self.assertEqual(self.strategy.confidence_threshold, 0.7)

    @async_test
    async def test_candle_processing(self):
        """Test candle data processing."""
        # Create sample candle data
        candle = CandleData(
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=100.0
        )
        
        # Process candle
        await self.strategy.process_candle(candle)
        
        # Check if candle was added to buffer
        self.assertTrue(len(self.strategy.candle_buffer[self.symbol]) > 0)
        self.assertEqual(
            self.strategy.candle_buffer[self.symbol][-1].close,
            candle.close
        )

    @async_test
    async def test_orderbook_processing(self):
        """Test orderbook data processing."""
        # Create sample orderbook data with proper dictionary format
        orderbook = OrderBookData(
            symbol=self.symbol,
            exchange=self.exchange,
            timestamp=datetime.now(),
            bids=[{"price": 49900.0, "amount": 1.0}, {"price": 49800.0, "amount": 2.0}],
            asks=[{"price": 50100.0, "amount": 1.0}, {"price": 50200.0, "amount": 2.0}]
        )
        
        # Process orderbook
        await self.strategy.process_orderbook(orderbook)
        
        # Check if orderbook was added to buffer
        self.assertTrue(len(self.strategy.orderbook_buffer[self.symbol]) > 0)

    @async_test
    async def test_liquidity_score_calculation(self):
        """Test liquidity score calculation."""
        # Create sample orderbook with proper dictionary format
        orderbook = OrderBookData(
            symbol=self.symbol,
            exchange=self.exchange,
            timestamp=datetime.now(),
            bids=[{"price": 49900.0, "amount": 1.0}, {"price": 49800.0, "amount": 2.0}],
            asks=[{"price": 50100.0, "amount": 1.0}, {"price": 50200.0, "amount": 2.0}]
        )
        
        # Process orderbook
        await self.strategy.process_orderbook(orderbook)
        
        # Calculate liquidity score
        score = self.strategy._calculate_liquidity_score(self.symbol)
        
        # Score should be non-negative
        self.assertGreaterEqual(float(score), 0.0)

    @async_test
    async def test_feature_preparation(self):
        """Test feature preparation."""
        # Add sample data
        for i in range(100):
            candle = CandleData(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                timestamp=datetime.now() - timedelta(hours=i),
                open=50000.0 + i,
                high=51000.0 + i,
                low=49000.0 + i,
                close=50500.0 + i,
                volume=100.0
            )
            await self.strategy.process_candle(candle)
        
        # Prepare features
        features = await self.strategy._prepare_features(self.symbol)
        
        # Check feature vector
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)

if __name__ == "__main__":
    unittest.main() 