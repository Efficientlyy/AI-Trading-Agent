"""
Unit tests for the advanced sentiment strategy.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime

from ai_trading_agent.strategies.advanced_sentiment_strategy import AdvancedSentimentStrategy
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, OrderStatus, Position, Portfolio
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager


class TestAdvancedSentimentStrategy(unittest.TestCase):
    """Test cases for the advanced sentiment strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "sentiment_threshold": 0.2,
            "position_sizing_method": "risk_based",
            "risk_per_trade": 0.02,
            "max_position_size": 0.1,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "topics": ["blockchain"],
            "assets": {
                "blockchain": ["BTC", "ETH"]
            },
            "topic_asset_weights": {
                "blockchain": {"BTC": 0.7, "ETH": 0.3}
            },
            "signal_weights": {
                "sentiment": 0.6,
                "trend": 0.3,
                "volatility": 0.1
            }
        }
        self.strategy = AdvancedSentimentStrategy(config=self.config)
        
        # Create a mock portfolio manager
        self.portfolio = Portfolio(
            account_id="test_account",
            starting_balance=Decimal("100000"),
            current_balance=Decimal("100000"),
            positions={},
            orders=[],
            trades=[],
            timestamp=datetime.now()
        )
        self.portfolio_manager = PortfolioManager(
            initial_capital=Decimal("100000"),
            risk_per_trade=Decimal("0.02"),
            max_position_size=Decimal("0.1"),
            max_correlation=Decimal("0.7"),
            rebalance_frequency=7
        )
        self.portfolio_manager.portfolio = self.portfolio
        
        # Mock sentiment data
        self.mock_sentiment_df = pd.DataFrame({
            "time_published": pd.date_range(start="2023-01-01", periods=10),
            "overall_sentiment_score": [0.1, 0.2, 0.3, 0.25, 0.15, 0.05, -0.1, -0.2, 0.1, 0.3],
            "weighted_sentiment_score": [0.15, 0.25, 0.35, 0.3, 0.2, 0.1, -0.05, -0.15, 0.15, 0.35],
            "sentiment_trend": [0.05, 0.1, 0.15, 0.1, 0.05, -0.05, -0.1, -0.15, 0.05, 0.15],
            "sentiment_volatility": [0.02, 0.03, 0.04, 0.03, 0.02, 0.02, 0.03, 0.04, 0.03, 0.02],
            "signal": [0, 1, 1, 1, 0, 0, 0, -1, 0, 1]
        })

    @patch('ai_trading_agent.sentiment_analysis.sentiment_analyzer.SentimentAnalyzer.analyze_sentiment')
    def test_analyze_sentiment(self, mock_analyze_sentiment):
        """Test the analyze_sentiment method."""
        # Mock the sentiment analyzer response
        mock_analyze_sentiment.return_value = self.mock_sentiment_df
        
        # Call the method
        result = self.strategy.analyze_sentiment("blockchain")
        
        # Check the result
        self.assertEqual(result["topic"], "blockchain")
        self.assertIsInstance(result["sentiment_score"], Decimal)
        self.assertIsInstance(result["sentiment_trend"], Decimal)
        self.assertIsInstance(result["sentiment_volatility"], Decimal)
        self.assertIn(result["signal"], [-1, 0, 1])
        
        # Check that the sentiment analyzer was called with the correct arguments
        mock_analyze_sentiment.assert_called_once_with(topic="blockchain", days_back=14)
        
        # Test caching
        self.strategy.analyze_sentiment("blockchain")
        # The sentiment analyzer should still only be called once
        mock_analyze_sentiment.assert_called_once()

    def test_calculate_sentiment_trend(self):
        """Test the _calculate_sentiment_trend method."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        trend = self.strategy._calculate_sentiment_trend(empty_df)
        self.assertEqual(trend, Decimal("0"))
        
        # Test with actual data
        trend = self.strategy._calculate_sentiment_trend(self.mock_sentiment_df)
        self.assertIsInstance(trend, Decimal)

    def test_calculate_sentiment_volatility(self):
        """Test the _calculate_sentiment_volatility method."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        volatility = self.strategy._calculate_sentiment_volatility(empty_df)
        self.assertEqual(volatility, Decimal("0"))
        
        # Test with actual data
        volatility = self.strategy._calculate_sentiment_volatility(self.mock_sentiment_df)
        self.assertIsInstance(volatility, Decimal)

    def test_generate_signal(self):
        """Test the _generate_signal method."""
        # Test positive signal
        signal = self.strategy._generate_signal(
            sentiment_score=Decimal("0.3"),
            sentiment_trend=Decimal("0.1"),
            sentiment_volatility=Decimal("0.05")
        )
        self.assertEqual(signal, 1)
        
        # Test negative signal
        signal = self.strategy._generate_signal(
            sentiment_score=Decimal("-0.3"),
            sentiment_trend=Decimal("-0.1"),
            sentiment_volatility=Decimal("0.05")
        )
        self.assertEqual(signal, -1)
        
        # Test neutral signal
        signal = self.strategy._generate_signal(
            sentiment_score=Decimal("0.1"),
            sentiment_trend=Decimal("0.05"),
            sentiment_volatility=Decimal("0.1")
        )
        self.assertEqual(signal, 0)

    def test_calculate_position_size(self):
        """Test the _calculate_position_size method."""
        # Test with zero signal
        position_size = self.strategy._calculate_position_size(
            portfolio_manager=self.portfolio_manager,
            asset="BTC",
            price=Decimal("50000"),
            signal=0
        )
        self.assertEqual(position_size, Decimal("0"))
        
        # Test with risk-based sizing
        with patch.object(self.portfolio_manager, 'calculate_position_size', return_value=Decimal("0.05")):
            position_size = self.strategy._calculate_position_size(
                portfolio_manager=self.portfolio_manager,
                asset="BTC",
                price=Decimal("50000"),
                signal=1
            )
            self.assertEqual(position_size, Decimal("0.05"))
        
        # Test with fixed percentage sizing
        self.strategy.position_sizing_method = "fixed_pct"
        position_size = self.strategy._calculate_position_size(
            portfolio_manager=self.portfolio_manager,
            asset="BTC",
            price=Decimal("50000"),
            signal=1
        )
        expected_size = (Decimal("100000") * Decimal("0.1")) / Decimal("50000")
        self.assertEqual(position_size, expected_size)
        
        # Test with unknown method
        self.strategy.position_sizing_method = "unknown"
        position_size = self.strategy._calculate_position_size(
            portfolio_manager=self.portfolio_manager,
            asset="BTC",
            price=Decimal("50000"),
            signal=1
        )
        self.assertEqual(position_size, Decimal("0.01"))

    def test_calculate_stop_loss_take_profit(self):
        """Test the _calculate_stop_loss_take_profit method."""
        # Test for BUY order
        stop_loss, take_profit = self.strategy._calculate_stop_loss_take_profit(
            price=Decimal("50000"),
            side=OrderSide.BUY
        )
        self.assertEqual(stop_loss, Decimal("50000") * (Decimal("1") - Decimal("0.05")))
        self.assertEqual(take_profit, Decimal("50000") * (Decimal("1") + Decimal("0.1")))
        
        # Test for SELL order
        stop_loss, take_profit = self.strategy._calculate_stop_loss_take_profit(
            price=Decimal("50000"),
            side=OrderSide.SELL
        )
        self.assertEqual(stop_loss, Decimal("50000") * (Decimal("1") + Decimal("0.05")))
        self.assertEqual(take_profit, Decimal("50000") * (Decimal("1") - Decimal("0.1")))

    @patch('ai_trading_agent.strategies.advanced_sentiment_strategy.AdvancedSentimentStrategy.analyze_sentiment')
    def test_generate_orders(self, mock_analyze_sentiment):
        """Test the generate_orders method."""
        # Mock the analyze_sentiment method
        mock_analyze_sentiment.return_value = {
            "topic": "blockchain",
            "sentiment_score": Decimal("0.3"),
            "sentiment_trend": Decimal("0.1"),
            "sentiment_volatility": Decimal("0.05"),
            "signal": 1,
            "raw_data": self.mock_sentiment_df
        }
        
        # Call the method
        orders = self.strategy.generate_orders(self.portfolio_manager)
        
        # Check the result
        self.assertIsInstance(orders, list)
        self.assertGreater(len(orders), 0)
        
        # Check the first order
        order = orders[0]
        self.assertIsInstance(order, Order)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.MARKET)
        
        # Test with negative signal
        mock_analyze_sentiment.return_value = {
            "topic": "blockchain",
            "sentiment_score": Decimal("-0.3"),
            "sentiment_trend": Decimal("-0.1"),
            "sentiment_volatility": Decimal("0.05"),
            "signal": -1,
            "raw_data": self.mock_sentiment_df
        }
        
        orders = self.strategy.generate_orders(self.portfolio_manager)
        self.assertGreater(len(orders), 0)
        self.assertEqual(orders[0].side, OrderSide.SELL)
        
        # Test with neutral signal
        mock_analyze_sentiment.return_value = {
            "topic": "blockchain",
            "sentiment_score": Decimal("0.1"),
            "sentiment_trend": Decimal("0.05"),
            "sentiment_volatility": Decimal("0.1"),
            "signal": 0,
            "raw_data": self.mock_sentiment_df
        }
        
        orders = self.strategy.generate_orders(self.portfolio_manager)
        self.assertEqual(len(orders), 0)

    @patch('ai_trading_agent.strategies.advanced_sentiment_strategy.AdvancedSentimentStrategy.generate_orders')
    @patch('ai_trading_agent.strategies.advanced_sentiment_strategy.AdvancedSentimentStrategy._apply_portfolio_constraints')
    def test_run_strategy(self, mock_apply_constraints, mock_generate_orders):
        """Test the run_strategy method."""
        # Mock the generate_orders method
        mock_orders = [
            Order(
                symbol="BTC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                price=Decimal("50000")
            )
        ]
        mock_generate_orders.return_value = mock_orders
        
        # Mock the _apply_portfolio_constraints method
        mock_apply_constraints.return_value = mock_orders
        
        # Call the method
        orders = self.strategy.run_strategy(self.portfolio_manager)
        
        # Check the result
        self.assertEqual(orders, mock_orders)
        
        # Check that the methods were called with the correct arguments
        mock_generate_orders.assert_called_once_with(self.portfolio_manager)
        mock_apply_constraints.assert_called_once_with(self.portfolio_manager, mock_orders)

    def test_apply_portfolio_constraints(self):
        """Test the _apply_portfolio_constraints method."""
        # Create some test orders
        orders = [
            Order(
                symbol="BTC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                price=Decimal("50000")
            ),
            Order(
                symbol="ETH",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=Decimal("3000")
            )
        ]
        
        # Test with no existing positions and max_positions > len(orders)
        self.strategy.max_positions = 5
        filtered_orders = self.strategy._apply_portfolio_constraints(self.portfolio_manager, orders)
        self.assertEqual(len(filtered_orders), 2)
        
        # Test with no existing positions and max_positions < len(orders)
        self.strategy.max_positions = 1
        filtered_orders = self.strategy._apply_portfolio_constraints(self.portfolio_manager, orders)
        self.assertEqual(len(filtered_orders), 1)
        self.assertEqual(filtered_orders[0].symbol, "BTC")  # Higher value order should be kept
        
        # Test with existing positions
        self.portfolio.positions = {
            "BTC": Position(
                symbol="BTC",
                quantity=Decimal("0.2"),
                entry_price=Decimal("45000")
            )
        }
        self.strategy.max_positions = 2
        filtered_orders = self.strategy._apply_portfolio_constraints(self.portfolio_manager, orders)
        self.assertEqual(len(filtered_orders), 1)
        self.assertEqual(filtered_orders[0].symbol, "BTC")  # Higher value order should be kept
        
        # Test with empty orders list
        filtered_orders = self.strategy._apply_portfolio_constraints(self.portfolio_manager, [])
        self.assertEqual(len(filtered_orders), 0)

    def test_update_performance_history(self):
        """Test the update_performance_history method."""
        # Call the method
        self.strategy.update_performance_history(self.portfolio_manager)
        
        # Check the result
        self.assertEqual(len(self.strategy.performance_history), 1)
        self.assertIn("timestamp", self.strategy.performance_history[0])
        self.assertIn("portfolio_value", self.strategy.performance_history[0])
        self.assertEqual(self.strategy.performance_history[0]["portfolio_value"], Decimal("100000"))
        
        # Update the portfolio value and call the method again
        self.portfolio.current_balance = Decimal("110000")
        self.strategy.update_performance_history(self.portfolio_manager)
        
        # Check the result
        self.assertEqual(len(self.strategy.performance_history), 2)
        self.assertEqual(self.strategy.performance_history[1]["portfolio_value"], Decimal("110000"))

    def test_get_performance_metrics(self):
        """Test the get_performance_metrics method."""
        # Test with empty performance history
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics["total_return"], Decimal("0"))
        self.assertEqual(metrics["annualized_return"], Decimal("0"))
        self.assertEqual(metrics["sharpe_ratio"], Decimal("0"))
        self.assertEqual(metrics["max_drawdown"], Decimal("0"))
        
        # Add some performance history
        self.strategy.performance_history = [
            {"timestamp": datetime(2023, 1, 1), "portfolio_value": Decimal("100000")},
            {"timestamp": datetime(2023, 1, 2), "portfolio_value": Decimal("102000")},
            {"timestamp": datetime(2023, 1, 3), "portfolio_value": Decimal("101000")},
            {"timestamp": datetime(2023, 1, 4), "portfolio_value": Decimal("103000")},
            {"timestamp": datetime(2023, 1, 5), "portfolio_value": Decimal("105000")}
        ]
        
        # Test with actual performance history
        metrics = self.strategy.get_performance_metrics()
        self.assertIsInstance(metrics["total_return"], Decimal)
        self.assertIsInstance(metrics["annualized_return"], Decimal)
        self.assertIsInstance(metrics["sharpe_ratio"], Decimal)
        self.assertIsInstance(metrics["max_drawdown"], Decimal)
        self.assertEqual(metrics["total_return"], Decimal("0.05"))  # (105000 / 100000) - 1


if __name__ == '__main__':
    unittest.main()
