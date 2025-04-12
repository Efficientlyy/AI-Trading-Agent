"""
Unit tests for the ExecutionHandler class.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

from ai_trading_agent.trading_engine.execution_handler import ExecutionHandler, SimulatedExchange
from ai_trading_agent.trading_engine.models import Order, Trade
from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus


class TestExecutionHandler:
    """Test cases for the ExecutionHandler class."""
    
    @pytest.fixture
    def execution_handler(self):
        """Create a basic execution handler for testing."""
        return ExecutionHandler(
            commission_rate=0.001,
            slippage_model="fixed",
            slippage_params={"fixed": 0.0},
            enable_partial_fills=False,
            rejection_probability=0.0,
        )
    
    @pytest.fixture
    def market_data(self):
        """Create mock market data for testing."""
        # Create a DataFrame with OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data = {
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }
        return pd.DataFrame(data, index=dates)
    
    def test_init(self):
        """Test initialization of ExecutionHandler."""
        handler = ExecutionHandler(
            commission_rate=0.002,
            slippage_model="normal",
            slippage_params={"normal": {"mean": 0.001, "std": 0.002}},
            enable_partial_fills=True,
            rejection_probability=0.05,
        )
        
        assert handler.commission_rate == 0.002
        assert handler.slippage_model == "normal"
        assert handler.slippage_params["normal"]["mean"] == 0.001
        assert handler.slippage_params["normal"]["std"] == 0.002
        assert handler.enable_partial_fills is True
        assert handler.rejection_probability == 0.05
    
    def test_execute_market_buy_order(self, execution_handler, market_data):
        """Test execution of a market buy order."""
        # Create a market buy order
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        timestamp = market_data.index[2]  # Use the third day
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that a trade was created
        assert len(trades) == 1
        trade = trades[0]
        
        # Check trade details
        assert trade.symbol == "BTC/USD"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 1.0
        assert trade.price == market_data.loc[timestamp, 'open']  # Market orders use open price
        assert trade.timestamp == timestamp
        
        # Check order status
        assert order.status == OrderStatus.FILLED
    
    def test_execute_market_sell_order(self, execution_handler, market_data):
        """Test execution of a market sell order."""
        # Create a market sell order
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        timestamp = market_data.index[2]  # Use the third day
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that a trade was created
        assert len(trades) == 1
        trade = trades[0]
        
        # Check trade details
        assert trade.symbol == "BTC/USD"
        assert trade.side == OrderSide.SELL
        assert trade.quantity == 1.0
        assert trade.price == market_data.loc[timestamp, 'open']  # Market orders use open price
        assert trade.timestamp == timestamp
        
        # Check order status
        assert order.status == OrderStatus.FILLED
    
    def test_execute_limit_buy_order_executed(self, execution_handler, market_data):
        """Test execution of a limit buy order that should be executed."""
        # Create a limit buy order with a price above the low
        timestamp = market_data.index[2]  # Use the third day
        limit_price = market_data.loc[timestamp, 'low'] + 0.5  # Set price above the low
        
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.LIMIT,
            price=limit_price,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that a trade was created
        assert len(trades) == 1
        trade = trades[0]
        
        # Check trade details
        assert trade.symbol == "BTC/USD"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 1.0
        assert trade.price == limit_price  # Limit orders use the limit price
        assert trade.timestamp == timestamp
        
        # Check order status
        assert order.status == OrderStatus.FILLED
    
    def test_execute_limit_buy_order_not_executed(self, execution_handler, market_data):
        """Test execution of a limit buy order that should not be executed."""
        # Create a limit buy order with a price below the low
        timestamp = market_data.index[2]  # Use the third day
        limit_price = market_data.loc[timestamp, 'low'] - 0.5  # Set price below the low
        
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.LIMIT,
            price=limit_price,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that no trades were created
        assert len(trades) == 0
        
        # Check order status
        assert order.status == OrderStatus.OPEN  # Order remains open
    
    def test_execute_limit_sell_order_executed(self, execution_handler, market_data):
        """Test execution of a limit sell order that should be executed."""
        # Create a limit sell order with a price below the high
        timestamp = market_data.index[2]  # Use the third day
        limit_price = market_data.loc[timestamp, 'high'] - 0.5  # Set price below the high
        
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            order_type=OrderType.LIMIT,
            price=limit_price,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that a trade was created
        assert len(trades) == 1
        trade = trades[0]
        
        # Check trade details
        assert trade.symbol == "BTC/USD"
        assert trade.side == OrderSide.SELL
        assert trade.quantity == 1.0
        assert trade.price == limit_price  # Limit orders use the limit price
        assert trade.timestamp == timestamp
        
        # Check order status
        assert order.status == OrderStatus.FILLED
    
    def test_execute_limit_sell_order_not_executed(self, execution_handler, market_data):
        """Test execution of a limit sell order that should not be executed."""
        # Create a limit sell order with a price above the high
        timestamp = market_data.index[2]  # Use the third day
        limit_price = market_data.loc[timestamp, 'high'] + 0.5  # Set price above the high
        
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            order_type=OrderType.LIMIT,
            price=limit_price,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that no trades were created
        assert len(trades) == 0
        
        # Check order status
        assert order.status == OrderStatus.OPEN  # Order remains open
    
    def test_execute_stop_buy_order_executed(self, execution_handler, market_data):
        """Test execution of a stop buy order that should be executed."""
        # Create a stop buy order with a price below the high
        timestamp = market_data.index[2]  # Use the third day
        price = market_data.loc[timestamp, 'high'] - 0.5  # Set price below the high
        
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.STOP,
            price=price,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that a trade was created
        assert len(trades) == 1
        trade = trades[0]
        
        # Check trade details
        assert trade.symbol == "BTC/USD"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 1.0
        assert trade.price == price  # Stop orders use stop price when triggered
        assert trade.timestamp == timestamp
        
        # Check order status
        assert order.status == OrderStatus.FILLED
    
    def test_execute_stop_buy_order_not_executed(self, execution_handler, market_data):
        """Test execution of a stop buy order that should not be executed."""
        # Create a stop buy order with a price above the high
        timestamp = market_data.index[2]  # Use the third day
        price = market_data.loc[timestamp, 'high'] + 0.5  # Set price above the high
        
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.STOP,
            price=price,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that no trades were created
        assert len(trades) == 0
        
        # Check order status
        assert order.status == OrderStatus.OPEN  # Order remains open
    
    def test_execute_stop_sell_order_executed(self, execution_handler, market_data):
        """Test execution of a stop sell order that should be executed."""
        # Create a stop sell order with a price above the low
        timestamp = market_data.index[2]  # Use the third day
        price = market_data.loc[timestamp, 'low'] + 0.5  # Set price above the low
        
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            order_type=OrderType.STOP,
            price=price,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that a trade was created
        assert len(trades) == 1
        trade = trades[0]
        
        # Check trade details
        assert trade.symbol == "BTC/USD"
        assert trade.side == OrderSide.SELL
        assert trade.quantity == 1.0
        assert trade.price == price  # Stop orders use stop price when triggered
        assert trade.timestamp == timestamp
        
        # Check order status
        assert order.status == OrderStatus.FILLED
    
    def test_execute_stop_sell_order_not_executed(self, execution_handler, market_data):
        """Test execution of a stop sell order that should not be executed."""
        # Create a stop sell order with a price below the low
        timestamp = market_data.index[2]  # Use the third day
        price = market_data.loc[timestamp, 'low'] - 0.5  # Set price below the low
        
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            order_type=OrderType.STOP,
            price=price,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        trades = execution_handler.execute_order(order, market_data, timestamp)
        
        # Check that no trades were created
        assert len(trades) == 0
        
        # Check order status
        assert order.status == OrderStatus.OPEN  # Order remains open
    
    def test_apply_slippage_fixed(self, execution_handler):
        """Test applying fixed slippage to an order."""
        # Set fixed slippage
        handler = execution_handler
        handler.slippage_model = "fixed"
        handler.slippage_params = {"fixed": 0.005}  # 0.5% slippage
        
        # Test buy order slippage (price increases)
        buy_order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        price = 100.0
        slipped_price = handler._apply_slippage(buy_order, price)
        assert slipped_price == price * 1.005  # Price increases by 0.5%
        
        # Test sell order slippage (price decreases)
        sell_order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        price = 100.0
        slipped_price = handler._apply_slippage(sell_order, price)
        assert slipped_price == price * 0.995  # Price decreases by 0.5%
    
    def test_apply_slippage_proportional(self, execution_handler):
        """Test applying proportional slippage to an order."""
        # Set proportional slippage
        handler = execution_handler
        handler.slippage_model = "proportional"
        handler.slippage_params = {"proportional": 0.01}  # 1% slippage
        
        # Test buy order slippage (price increases)
        buy_order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        price = 100.0
        slipped_price = handler._apply_slippage(buy_order, price)
        assert slipped_price == price * 1.01  # Price increases by 1%
        
        # Test sell order slippage (price decreases)
        sell_order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        price = 100.0
        slipped_price = handler._apply_slippage(sell_order, price)
        assert slipped_price == price * 0.99  # Price decreases by 1%
    
    def test_partial_fills(self):
        """Test partial fills for orders."""
        # Create an execution handler with partial fills enabled
        handler = ExecutionHandler(
            commission_rate=0.001,
            slippage_model="fixed",
            slippage_params={"fixed": 0.0},
            enable_partial_fills=True,
            rejection_probability=0.0,
        )
        
        # Mock the _determine_fill_quantity method to return a partial fill
        original_method = handler._determine_fill_quantity
        handler._determine_fill_quantity = lambda order: order.quantity * 0.5
        
        # Create a market data DataFrame
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data = {
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }
        market_data = pd.DataFrame(data, index=dates)
        
        # Create a market buy order
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute the order
        timestamp = market_data.index[2]  # Use the third day
        trades = handler.execute_order(order, market_data, timestamp)
        
        # Check that a trade was created with partial fill
        assert len(trades) == 1
        trade = trades[0]
        assert trade.quantity == 0.5  # Half of the original quantity
        
        # Check order status
        assert order.status == OrderStatus.PARTIALLY_FILLED
        
        # Restore the original method
        handler._determine_fill_quantity = original_method


class TestSimulatedExchange:
    """Test cases for the SimulatedExchange class."""
    
    @pytest.fixture
    def market_data(self):
        """Create mock market data for testing."""
        # Create DataFrames with OHLCV data for two symbols
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        
        btc_data = {
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }
        
        eth_data = {
            'open': [50.0, 51.0, 52.0, 53.0, 54.0],
            'high': [52.0, 53.0, 54.0, 55.0, 56.0],
            'low': [49.0, 50.0, 51.0, 52.0, 53.0],
            'close': [51.0, 52.0, 53.0, 54.0, 55.0],
            'volume': [2000, 2100, 2200, 2300, 2400]
        }
        
        return {
            "BTC/USD": pd.DataFrame(btc_data, index=dates),
            "ETH/USD": pd.DataFrame(eth_data, index=dates)
        }
    
    def test_init(self, market_data):
        """Test initialization of SimulatedExchange."""
        exchange = SimulatedExchange(
            market_data=market_data,
            commission_rate=0.002,
            slippage_model="normal",
            slippage_params={"normal": {"mean": 0.001, "std": 0.002}},
            enable_partial_fills=True,
            rejection_probability=0.05,
        )
        
        assert exchange.market_data == market_data
        assert exchange.symbols == ["BTC/USD", "ETH/USD"]
        assert exchange.execution_handler.commission_rate == 0.002
        assert exchange.execution_handler.slippage_model == "normal"
        assert exchange.execution_handler.slippage_params["normal"]["mean"] == 0.001
        assert exchange.execution_handler.slippage_params["normal"]["std"] == 0.002
        assert exchange.execution_handler.enable_partial_fills is True
        assert exchange.execution_handler.rejection_probability == 0.05
        
        # Check order book initialization
        assert "BTC/USD" in exchange.order_book
        assert "ETH/USD" in exchange.order_book
        assert exchange.order_book["BTC/USD"]["bids"] == []
        assert exchange.order_book["BTC/USD"]["asks"] == []
    
    def test_place_order_valid(self, market_data):
        """Test placing a valid order."""
        exchange = SimulatedExchange(market_data=market_data)
        
        # Create a valid order
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Place the order
        exchange.place_order(order)
        
        # Check order status
        assert order.status == OrderStatus.OPEN
    
    def test_place_order_invalid_symbol(self, market_data):
        """Test placing an order with an invalid symbol."""
        exchange = SimulatedExchange(market_data=market_data)
        
        # Create an order with an invalid symbol
        order = Order(
            symbol="XRP/USD",  # Not in market_data
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Place the order
        exchange.place_order(order)
        
        # Check order status
        assert order.status == OrderStatus.REJECTED
    
    def test_execute_orders(self, market_data):
        """Test executing multiple orders."""
        exchange = SimulatedExchange(market_data=market_data)
        
        # Create orders
        order1 = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        order2 = Order(
            symbol="ETH/USD",
            side=OrderSide.SELL,
            quantity=2.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.OPEN,  # Set initial status to OPEN
        )
        
        # Execute orders
        timestamp = market_data["BTC/USD"].index[2]  # Use the third day
        trades = exchange.execute_orders([order1, order2], timestamp)
        
        # Check that trades were created
        assert len(trades) == 2
        
        # Check trade details for first order
        trade1 = [t for t in trades if t.symbol == "BTC/USD"][0]
        assert trade1.side == OrderSide.BUY
        assert trade1.quantity == 1.0
        assert trade1.price == market_data["BTC/USD"].loc[timestamp, 'open']
        
        # Check trade details for second order
        trade2 = [t for t in trades if t.symbol == "ETH/USD"][0]
        assert trade2.side == OrderSide.SELL
        assert trade2.quantity == 2.0
        assert trade2.price == market_data["ETH/USD"].loc[timestamp, 'open']
        
        # Check order status
        assert order1.status == OrderStatus.FILLED
        assert order2.status == OrderStatus.FILLED
    
    def test_get_current_prices(self, market_data):
        """Test getting current prices."""
        exchange = SimulatedExchange(market_data=market_data)
        
        # Set current timestamp
        timestamp = market_data["BTC/USD"].index[2]  # Use the third day
        exchange.current_timestamp = timestamp
        
        # Get current prices
        prices = exchange.get_current_prices()
        
        # Check prices
        assert "BTC/USD" in prices
        assert "ETH/USD" in prices
        
        assert prices["BTC/USD"]["open"] == market_data["BTC/USD"].loc[timestamp, 'open']
        assert prices["BTC/USD"]["high"] == market_data["BTC/USD"].loc[timestamp, 'high']
        assert prices["BTC/USD"]["low"] == market_data["BTC/USD"].loc[timestamp, 'low']
        assert prices["BTC/USD"]["close"] == market_data["BTC/USD"].loc[timestamp, 'close']
        assert prices["BTC/USD"]["volume"] == market_data["BTC/USD"].loc[timestamp, 'volume']
        
        assert prices["ETH/USD"]["open"] == market_data["ETH/USD"].loc[timestamp, 'open']
        assert prices["ETH/USD"]["high"] == market_data["ETH/USD"].loc[timestamp, 'high']
        assert prices["ETH/USD"]["low"] == market_data["ETH/USD"].loc[timestamp, 'low']
        assert prices["ETH/USD"]["close"] == market_data["ETH/USD"].loc[timestamp, 'close']
        assert prices["ETH/USD"]["volume"] == market_data["ETH/USD"].loc[timestamp, 'volume']
