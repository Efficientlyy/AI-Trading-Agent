"""
Tests for the EnhancedSentimentStrategy class.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from ai_trading_agent.strategies.enhanced_sentiment_strategy import EnhancedSentimentStrategy
from ai_trading_agent.signal_processing.sentiment_processor import SentimentSignal
from ai_trading_agent.signal_processing.regime import MarketRegime
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, Portfolio
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager


class TestEnhancedSentimentStrategy(unittest.TestCase):
    """Test cases for the EnhancedSentimentStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock sentiment analyzer
        self.mock_sentiment_analyzer = MagicMock()
        
        # Create mock sentiment data
        dates = [datetime.now() - timedelta(days=i) for i in range(30)]
        dates.reverse()  # Earliest date first
        
        self.mock_sentiment_data = pd.DataFrame({
            'timestamp': dates,
            'compound_score': [0.2 + (i * 0.01) for i in range(30)],  # Increasing sentiment
            'positive_score': [0.6 + (i * 0.01) for i in range(30)],
            'negative_score': [0.2 - (i * 0.005) for i in range(30)],
            'neutral_score': [0.2 for i in range(30)],
            'volume': [100 + i for i in range(30)]
        })
        
        # Mock the analyze_sentiment method
        self.mock_sentiment_analyzer.analyze_sentiment.return_value = self.mock_sentiment_data
        
        # Create strategy with mocked dependencies
        with patch('ai_trading_agent.strategies.enhanced_sentiment_strategy.SentimentAnalyzer', 
                  return_value=self.mock_sentiment_analyzer):
            self.strategy = EnhancedSentimentStrategy({
                'sentiment_threshold': 0.2,
                'window_size': 3,
                'sentiment_weight': 0.4,
                'min_confidence': 0.6,
                'enable_regime_detection': True,
                'timeframe': '1d',
                'assets': ['BTC', 'ETH']
            })
        
        # Mock the signal processor
        self.mock_signal_processor = MagicMock()
        self.strategy.signal_processor = self.mock_signal_processor
        
        # Create mock sentiment signals
        self.mock_signals = [
            SentimentSignal(
                symbol='BTC',
                timestamp=pd.Timestamp(datetime.now()),
                signal_type='buy',
                strength=0.8,
                confidence=0.7,
                timeframe='1d',
                metadata={'raw_score': 0.5}
            ),
            SentimentSignal(
                symbol='ETH',
                timestamp=pd.Timestamp(datetime.now()),
                signal_type='sell',
                strength=0.6,
                confidence=0.8,
                timeframe='1d',
                metadata={'raw_score': -0.4}
            )
        ]
        
        # Mock the process_sentiment_data method
        self.mock_signal_processor.process_sentiment_data.return_value = self.mock_signals
        
        # Mock the regime detector
        self.mock_regime_detector = MagicMock()
        self.strategy.regime_detector = self.mock_regime_detector
        
        # Mock the detect_regime method
        self.mock_regime_detector.detect_regime.return_value = MarketRegime.TRENDING
        
        # Mock regime parameters
        self.mock_regime_detector.get_regime_parameters.return_value = {
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15,
            'position_size_pct': 1.0,
            'sentiment_weight': 0.6,
            'technical_weight': 0.8,
            'trailing_stop': True
        }
        
        # Create mock portfolio manager
        self.mock_portfolio_manager = MagicMock(spec=PortfolioManager)
        self.mock_portfolio_manager.calculate_position_size.return_value = Decimal('0.1')
        self.mock_portfolio_manager.get_portfolio_value.return_value = Decimal('10000')
        
        # Create mock market prices
        self.market_prices = {
            'BTC': Decimal('30000'),
            'ETH': Decimal('2000')
        }

    def test_get_sentiment_data(self):
        """Test getting sentiment data."""
        # Call the method
        result = self.strategy.get_sentiment_data('BTC', is_topic=False)
        
        # Check that the sentiment analyzer was called correctly
        self.mock_sentiment_analyzer.analyze_sentiment.assert_called_once_with(
            crypto_ticker='BTC', 
            days_back=self.strategy.days_back
        )
        
        # Check that the result is correct
        pd.testing.assert_frame_equal(result, self.mock_sentiment_data)
        
        # Check that the data was cached
        self.assertIn('asset:BTC', self.strategy.sentiment_cache)
        self.assertIn('asset:BTC', self.strategy.sentiment_cache_expiry)

    def test_process_sentiment_signals(self):
        """Test processing sentiment signals."""
        # Mock get_price_data
        self.strategy.get_price_data = MagicMock(return_value=pd.DataFrame({
            'open': [30000],
            'high': [31000],
            'low': [29000],
            'close': [30500],
            'volume': [1000]
        }))
        
        # Call the method
        result = self.strategy.process_sentiment_signals('BTC')
        
        # Check that the signal processor was called correctly
        self.mock_signal_processor.process_sentiment_data.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result, self.mock_signals)

    def test_detect_market_regime(self):
        """Test detecting market regime."""
        # Mock get_price_data
        self.strategy.get_price_data = MagicMock(return_value=pd.DataFrame({
            'open': [30000],
            'high': [31000],
            'low': [29000],
            'close': [30500],
            'volume': [1000]
        }))
        
        # Call the method
        result = self.strategy.detect_market_regime('BTC')
        
        # Check that the regime detector was called correctly
        self.mock_regime_detector.detect_regime.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result, MarketRegime.TRENDING)

    def test_generate_orders_from_signals(self):
        """Test generating orders from signals."""
        # Call the method
        result = self.strategy.generate_orders_from_signals(
            self.mock_signals,
            self.mock_portfolio_manager,
            self.market_prices
        )
        
        # Check that the result is a list of orders
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(order, Order) for order in result))
        
        # Check that the correct number of orders was generated
        self.assertEqual(len(result), 2)
        
        # Check the first order (BUY)
        buy_order = result[0]
        self.assertEqual(buy_order.symbol, 'BTC')
        self.assertEqual(buy_order.side, OrderSide.BUY)
        self.assertEqual(buy_order.order_type, OrderType.MARKET)
        
        # Check the second order (SELL)
        sell_order = result[1]
        self.assertEqual(sell_order.symbol, 'ETH')
        self.assertEqual(sell_order.side, OrderSide.SELL)
        self.assertEqual(sell_order.order_type, OrderType.MARKET)
        
        # Check that the signal history was updated
        self.assertEqual(len(self.strategy.signal_history), 2)
        self.assertEqual(self.strategy.signal_history[0]['symbol'], 'BTC')
        self.assertEqual(self.strategy.signal_history[0]['signal_type'], 'buy')
        self.assertEqual(self.strategy.signal_history[1]['symbol'], 'ETH')
        self.assertEqual(self.strategy.signal_history[1]['signal_type'], 'sell')

    def test_run_strategy(self):
        """Test running the strategy."""
        # Mock process_sentiment_signals
        self.strategy.process_sentiment_signals = MagicMock(return_value=self.mock_signals)
        
        # Mock generate_orders_from_signals
        mock_orders = [
            Order(symbol='BTC', side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1, price=30000),
            Order(symbol='ETH', side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.0, price=2000)
        ]
        self.strategy.generate_orders_from_signals = MagicMock(return_value=mock_orders)
        
        # Call the method
        result = self.strategy.run_strategy(
            self.mock_portfolio_manager,
            self.market_prices
        )
        
        # Check that process_sentiment_signals was called for each asset
        self.assertEqual(self.strategy.process_sentiment_signals.call_count, 2)
        
        # Check that generate_orders_from_signals was called once with all signals
        self.strategy.generate_orders_from_signals.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result, mock_orders)

    def test_evaluate_signal_performance(self):
        """Test evaluating signal performance."""
        # Add some signals to the history
        self.strategy.signal_history = [
            {
                "timestamp": datetime.now(),
                "symbol": "BTC",
                "signal_type": "buy",
                "strength": 0.8,
                "confidence": 0.7,
                "regime": "trending",
                "order": Order(symbol='BTC', side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1, price=30000)
            },
            {
                "timestamp": datetime.now() - timedelta(days=10),
                "symbol": "ETH",
                "signal_type": "sell",
                "strength": 0.6,
                "confidence": 0.8,
                "regime": "volatile",
                "order": Order(symbol='ETH', side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=1.0, price=2000)
            }
        ]
        
        # Call the method
        result = self.strategy.evaluate_signal_performance(days_back=30)
        
        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that the metrics are correct
        self.assertEqual(result['total_signals'], 2)
        self.assertEqual(result['buy_signals'], 1)
        self.assertEqual(result['sell_signals'], 1)
        self.assertEqual(result['regime_distribution']['trending'], 1)
        self.assertEqual(result['regime_distribution']['volatile'], 1)
        self.assertAlmostEqual(result['avg_strength'], 0.7)
        self.assertAlmostEqual(result['avg_confidence'], 0.75)


if __name__ == '__main__':
    unittest.main()
