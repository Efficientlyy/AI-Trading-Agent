"""
Test module for the simple sentiment strategy.

This module tests the functionality of the simple sentiment strategy, including
generating signals and orders based on sentiment analysis.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

from ai_trading_agent.strategies.simple_sentiment_strategy import SimpleSentimentStrategy
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager


class TestSimpleSentimentStrategy:
    """Test class for the simple sentiment strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "sentiment_threshold": 0.2,
            "position_sizing_method": "risk_based",
            "risk_per_trade": 0.02,
            "max_position_size": 0.1,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "topics": ["blockchain"],
            "assets": ["BTC"]
        }
        self.strategy = SimpleSentimentStrategy(config=self.config)

    @patch('ai_trading_agent.sentiment_analysis.sentiment_analyzer.SentimentAnalyzer.analyze_sentiment')
    def test_analyze_sentiment(self, mock_analyze_sentiment):
        """Test analyzing sentiment for a topic."""
        # Mock the sentiment analyzer's analyze_sentiment method
        mock_df = pd.DataFrame({
            "time_published": [datetime.now()],
            "overall_sentiment_score": [0.8],
            "weighted_sentiment_score": [0.7],
            "signal": [1]
        })
        mock_analyze_sentiment.return_value = mock_df

        # Test analyzing sentiment for a topic
        result = self.strategy.analyze_sentiment("blockchain")
        
        # Verify the result
        assert result["topic"] == "blockchain"
        assert result["sentiment_score"] == 0.7
        assert result["signal"] == 1
        mock_analyze_sentiment.assert_called_with(topic="blockchain", days_back=7)

    @patch('ai_trading_agent.strategies.simple_sentiment_strategy.SimpleSentimentStrategy.analyze_sentiment')
    def test_generate_orders(self, mock_analyze_sentiment):
        """Test generating orders based on sentiment analysis."""
        # Mock the analyze_sentiment method
        mock_analyze_sentiment.return_value = {
            "topic": "blockchain",
            "sentiment_score": 0.7,
            "signal": 1
        }
        
        # Create a portfolio manager
        portfolio_manager = PortfolioManager(
            initial_capital=100000.0,
            risk_per_trade=0.02,
            max_position_size=0.1
        )
        
        # Test generating orders
        orders = self.strategy.generate_orders(portfolio_manager)
        
        # Verify the result
        assert len(orders) == 1
        
        # Check the order
        order = orders[0]
        assert order.symbol == "BTC"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 0.1
        assert order.price == 50000.0

    @patch('ai_trading_agent.strategies.simple_sentiment_strategy.SimpleSentimentStrategy.generate_orders')
    def test_run_strategy(self, mock_generate_orders):
        """Test running the sentiment strategy to generate orders."""
        # Mock the generate_orders method
        mock_orders = [
            Order(
                symbol="BTC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
                price=50000.0
            )
        ]
        mock_generate_orders.return_value = mock_orders
        
        # Create a portfolio manager
        portfolio_manager = PortfolioManager(
            initial_capital=100000.0,
            risk_per_trade=0.02,
            max_position_size=0.1
        )
        
        # Test running the strategy
        orders = self.strategy.run_strategy(portfolio_manager)
        
        # Verify the result
        assert orders == mock_orders
        mock_generate_orders.assert_called_with(portfolio_manager)
