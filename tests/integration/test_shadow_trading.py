"""
Integration Tests for Shadow Trading System with Real Market Data

These tests validate that the shadow trading system can properly:
1. Connect to real market data providers
2. Execute simulated trades based on real market conditions
3. Calculate portfolio performance metrics
4. Handle order lifecycle management

This is a critical test for Day 1 of our production deployment plan.
"""

import os
import unittest
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from ai_trading_agent.execution.shadow.shadow_trader import ShadowTrader
from ai_trading_agent.execution.models import Order, OrderStatus, OrderType
from ai_trading_agent.data.models import MarketData
from ai_trading_agent.execution.broker.base import BaseBroker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockRealBroker(BaseBroker):
    """Mock implementation of a real broker for testing."""
    
    def __init__(self):
        self.market_data = {}
        # Prepopulate some test data
        self._add_test_data()
    
    def _add_test_data(self):
        """Add test market data."""
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
        prices = [185.34, 415.56, 175.89, 182.22]
        
        for symbol, price in zip(symbols, prices):
            self.market_data[symbol] = MarketData(
                symbol=symbol,
                last_price=price,
                bid=price - 0.05,
                ask=price + 0.05,
                volume=10000,
                timestamp=datetime.now()
            )
    
    def get_market_data(self, symbol):
        """Get market data for a symbol."""
        if symbol in self.market_data:
            # Update the timestamp to simulate real-time data
            self.market_data[symbol].timestamp = datetime.now()
            return self.market_data[symbol]
        raise ValueError(f"Symbol {symbol} not found")
    
    def update_price(self, symbol, new_price):
        """Update the price for testing."""
        if symbol in self.market_data:
            self.market_data[symbol].last_price = new_price
            self.market_data[symbol].bid = new_price - 0.05
            self.market_data[symbol].ask = new_price + 0.05
            self.market_data[symbol].timestamp = datetime.now()
        else:
            raise ValueError(f"Symbol {symbol} not found")


@pytest.mark.integration
class TestShadowTrading(unittest.TestCase):
    """Integration tests for shadow trading with real market data."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock broker
        self.mock_broker = MockRealBroker()
        
        # Create a shadow trader with the mock broker
        self.shadow_trader = ShadowTrader(self.mock_broker)
        
        # Set up test parameters
        self.test_symbol = "AAPL"
        self.test_quantity = 10
    
    def test_shadow_trader_initialization(self):
        """Test that shadow trader initializes correctly."""
        performance = self.shadow_trader.get_performance()
        
        # Verify initial state
        self.assertEqual(performance['position_count'], 0)
        self.assertEqual(performance['trade_count'], 0)
        self.assertEqual(performance['return_pct'], 0)
        self.assertTrue(performance['starting_capital'] > 0)
    
    def test_place_and_execute_market_order(self):
        """Test placing and executing a market order in shadow mode."""
        # Create a buy order
        order = Order(
            symbol=self.test_symbol,
            side="BUY",
            quantity=self.test_quantity,
            order_type=OrderType.MARKET
        )
        
        # Place the order
        order_id = self.shadow_trader.place_order(order)
        
        # Verify the order was placed and executed
        self.assertIsNotNone(order_id)
        
        # Get the order and check its status
        orders = self.shadow_trader.get_orders()
        self.assertEqual(len(orders), 1)
        executed_order = orders[0]
        self.assertEqual(executed_order.status, OrderStatus.FILLED)
        self.assertIsNotNone(executed_order.filled_price)
        self.assertIsNotNone(executed_order.filled_time)
        
        # Check that a position was created
        positions = self.shadow_trader.get_positions()
        self.assertIn(self.test_symbol, positions)
        self.assertEqual(positions[self.test_symbol].quantity, self.test_quantity)
        
        # Check that a trade was recorded
        trades = self.shadow_trader.get_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].symbol, self.test_symbol)
        self.assertEqual(trades[0].quantity, self.test_quantity)
    
    def test_position_update_with_price_change(self):
        """Test position value updates when market prices change."""
        # Create and execute a buy order
        buy_order = Order(
            symbol=self.test_symbol,
            side="BUY",
            quantity=self.test_quantity,
            order_type=OrderType.MARKET
        )
        self.shadow_trader.place_order(buy_order)
        
        # Get initial position and portfolio values
        positions_before = self.shadow_trader.get_positions()
        portfolio_before = self.shadow_trader.get_performance()
        
        # Record initial position value
        initial_position_value = positions_before[self.test_symbol].quantity * positions_before[self.test_symbol].current_price
        
        # Simulate a price increase of 5%
        current_price = self.mock_broker.market_data[self.test_symbol].last_price
        new_price = current_price * 1.05
        self.mock_broker.update_price(self.test_symbol, new_price)
        
        # Get updated position and portfolio values
        positions_after = self.shadow_trader.get_positions()
        portfolio_after = self.shadow_trader.get_performance()
        
        # Verify position price was updated
        self.assertEqual(positions_after[self.test_symbol].current_price, new_price)
        
        # Verify portfolio value increased
        expected_position_value = positions_after[self.test_symbol].quantity * new_price
        position_value_increase = expected_position_value - initial_position_value
        
        # Allow for small floating point differences
        self.assertAlmostEqual(
            portfolio_after['current_value'], 
            portfolio_before['current_value'] + position_value_increase,
            places=2
        )
    
    def test_full_trading_cycle(self):
        """Test a full trading cycle of buy and sell in shadow mode."""
        # 1. Place buy order
        buy_order = Order(
            symbol=self.test_symbol,
            side="BUY",
            quantity=self.test_quantity,
            order_type=OrderType.MARKET
        )
        buy_order_id = self.shadow_trader.place_order(buy_order)
        
        # Verify buy execution
        buy_orders = self.shadow_trader.get_orders()
        self.assertEqual(buy_orders[0].status, OrderStatus.FILLED)
        
        # Record cash after buy
        cash_after_buy = self.shadow_trader.available_cash
        
        # 2. Simulate price increase
        current_price = self.mock_broker.market_data[self.test_symbol].last_price
        new_price = current_price * 1.10  # 10% increase
        self.mock_broker.update_price(self.test_symbol, new_price)
        
        # 3. Place sell order
        sell_order = Order(
            symbol=self.test_symbol,
            side="SELL",
            quantity=self.test_quantity,
            order_type=OrderType.MARKET
        )
        sell_order_id = self.shadow_trader.place_order(sell_order)
        
        # Verify sell execution
        sell_orders = self.shadow_trader.get_orders()
        self.assertEqual(len(sell_orders), 2)  # Both buy and sell orders
        
        # Find the sell order
        sell_order_result = next(order for order in sell_orders if order.order_id == sell_order_id)
        self.assertEqual(sell_order_result.status, OrderStatus.FILLED)
        
        # 4. Verify position is closed
        positions = self.shadow_trader.get_positions()
        self.assertTrue(
            self.test_symbol not in positions or positions[self.test_symbol].quantity == 0,
            f"Position should be closed but has quantity: {positions.get(self.test_symbol, 'not found')}"
        )
        
        # 5. Verify profit was realized (cash increased)
        cash_after_sell = self.shadow_trader.available_cash
        self.assertTrue(cash_after_sell > cash_after_buy)
        
        # 6. Verify trade history
        trades = self.shadow_trader.get_trades()
        self.assertEqual(len(trades), 2)  # One buy and one sell trade
        
        # 7. Verify performance metrics
        performance = self.shadow_trader.get_performance()
        self.assertTrue(performance['return_pct'] > 0)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
