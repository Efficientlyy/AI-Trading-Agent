"""
Test module for the sentiment strategy.

This module tests the functionality of the sentiment strategy, including
generating signals and orders based on sentiment analysis.
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import patch, MagicMock
from datetime import datetime

from ai_trading_agent.strategies.sentiment_strategy import SentimentStrategy
from ai_trading_agent.trading_engine.models import Portfolio, Order, OrderSide, OrderType
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager


class TestSentimentStrategy:
    """Test class for the sentiment strategy."""

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
            "assets": ["BTC"],
            "days_back": 3,
            "cache_expiry_hours": 24
        }
        self.strategy = SentimentStrategy(config=self.config)

    @patch('ai_trading_agent.sentiment_analysis.sentiment_analyzer.SentimentAnalyzer.analyze_sentiment')
    def test_get_sentiment_data(self, mock_analyze_sentiment):
        """Test getting sentiment data for a topic or asset."""
        # Mock the sentiment analyzer's analyze_sentiment method
        mock_df = pd.DataFrame({
            "time_published": [datetime.now()],
            "overall_sentiment_score": [Decimal("0.8")],
            "weighted_sentiment_score": [Decimal("0.7")],
            "signal": [1]
        })
        mock_analyze_sentiment.return_value = mock_df

        # Test getting sentiment data for a topic
        result_df = self.strategy.get_sentiment_data("blockchain", is_topic=True)
        
        # Verify the result
        assert not result_df.empty
        assert "weighted_sentiment_score" in result_df.columns
        assert "signal" in result_df.columns
        mock_analyze_sentiment.assert_called_with(topic="blockchain", days_back=3)
        
        # Test caching
        self.strategy.get_sentiment_data("blockchain", is_topic=True)
        assert mock_analyze_sentiment.call_count == 1  # Should use cached data

    @patch('ai_trading_agent.sentiment_analysis.sentiment_analyzer.SentimentAnalyzer.analyze_sentiment')
    def test_get_latest_signal(self, mock_analyze_sentiment):
        """Test getting the latest trading signal for a topic or asset."""
        # Mock the sentiment analyzer's analyze_sentiment method
        mock_df = pd.DataFrame({
            "time_published": [datetime.now()],
            "overall_sentiment_score": [Decimal("0.8")],
            "weighted_sentiment_score": [Decimal("0.7")],
            "signal": [1]
        })
        mock_analyze_sentiment.return_value = mock_df

        # Test getting the latest signal for a topic
        signal = self.strategy.get_latest_signal("blockchain", is_topic=True)
        
        # Verify the result
        assert signal["topic_or_asset"] == "blockchain"
        assert signal["is_topic"] is True
        assert signal["signal"] == 1
        assert signal["weighted_score"] == Decimal("0.7")
        assert isinstance(signal["timestamp"], datetime)

    @patch('ai_trading_agent.strategies.sentiment_strategy.SentimentStrategy.get_latest_signal')
    def test_generate_signals(self, mock_get_latest_signal):
        """Test generating trading signals for all topics and assets."""
        # Mock the get_latest_signal method
        mock_get_latest_signal.side_effect = [
            {
                "topic_or_asset": "blockchain",
                "is_topic": True,
                "signal": 1,
                "weighted_score": Decimal("0.7"),
                "timestamp": datetime.now()
            },
            {
                "topic_or_asset": "BTC",
                "is_topic": False,
                "signal": 0.5,
                "weighted_score": Decimal("0.3"),
                "timestamp": datetime.now()
            }
        ]

        # Test generating signals
        signals = self.strategy.generate_signals()
        
        # Verify the result
        assert len(signals) == 2
        assert signals[0]["topic_or_asset"] == "blockchain"
        assert signals[0]["is_topic"] is True
        assert signals[1]["topic_or_asset"] == "BTC"
        assert signals[1]["is_topic"] is False
        assert mock_get_latest_signal.call_count == 2

    def test_generate_orders(self):
        """Test generating orders based on sentiment signals."""
        # Create test signals
        signals = [
            {
                "topic_or_asset": "blockchain",
                "is_topic": True,
                "signal": 1,
                "weighted_score": Decimal("0.7"),
                "timestamp": datetime.now()
            },
            {
                "topic_or_asset": "BTC",
                "is_topic": False,
                "signal": 0.8,
                "weighted_score": Decimal("0.6"),
                "timestamp": datetime.now()
            },
            {
                "topic_or_asset": "ETH",
                "is_topic": False,
                "signal": -0.5,
                "weighted_score": Decimal("-0.4"),
                "timestamp": datetime.now()
            }
        ]
        
        # Create a portfolio manager
        portfolio_manager = PortfolioManager(
            initial_capital=Decimal("100000"),
            risk_per_trade=Decimal("0.02"),
            max_position_size=Decimal("0.1")
        )
        
        # Create market prices
        market_prices = {
            "BTC": Decimal("50000"),
            "ETH": Decimal("3000")
        }
        
        # Test generating orders
        orders = self.strategy.generate_orders(signals, portfolio_manager, market_prices)
        
        # Verify the result
        assert len(orders) == 2  # One for BTC (buy) and one for ETH (sell)
        
        # Check BTC order
        btc_order = next((o for o in orders if o.symbol == "BTC"), None)
        assert btc_order is not None
        assert btc_order.side == OrderSide.BUY
        assert btc_order.order_type == OrderType.MARKET  # Use order_type property
        assert isinstance(btc_order.price, float)  # Check type is float, not Decimal
        
        # Check ETH order
        eth_order = next((o for o in orders if o.symbol == "ETH"), None)
        assert eth_order is not None
        assert eth_order.side == OrderSide.SELL
        assert eth_order.order_type == OrderType.MARKET  # Use order_type property
        assert isinstance(eth_order.price, float)  # Check type is float, not Decimal

    @patch('ai_trading_agent.strategies.sentiment_strategy.SentimentStrategy.generate_signals')
    @patch('ai_trading_agent.strategies.sentiment_strategy.SentimentStrategy.generate_orders')
    def test_run_strategy(self, mock_generate_orders, mock_generate_signals):
        """Test running the sentiment strategy to generate orders."""
        # Mock the generate_signals and generate_orders methods
        mock_signals = [
            {
                "topic_or_asset": "blockchain",
                "is_topic": True,
                "signal": 1,
                "weighted_score": Decimal("0.7"),
                "timestamp": datetime.now()
            },
            {
                "topic_or_asset": "BTC",
                "is_topic": False,
                "signal": 0.8,
                "weighted_score": Decimal("0.6"),
                "timestamp": datetime.now()
            }
        ]
        mock_generate_signals.return_value = mock_signals
        
        mock_orders = [
            Order(
                symbol="BTC",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                quantity=0.1,
                price=50000.0,
                created_at=datetime.now()
            )
        ]
        mock_generate_orders.return_value = mock_orders
        
        # Create a portfolio manager
        portfolio_manager = PortfolioManager(
            initial_capital=Decimal("100000"),
            risk_per_trade=Decimal("0.02"),
            max_position_size=Decimal("0.1")
        )
        
        # Create market prices
        market_prices = {
            "BTC": Decimal("50000"),
            "ETH": Decimal("3000")
        }
        
        # Test running the strategy
        orders = self.strategy.run_strategy(portfolio_manager, market_prices)
        
        # Verify the result
        assert orders == mock_orders
        assert mock_generate_signals.called
        mock_generate_orders.assert_called_with(mock_signals, portfolio_manager, market_prices)
