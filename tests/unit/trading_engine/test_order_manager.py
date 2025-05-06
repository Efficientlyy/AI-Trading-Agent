"""
Unit tests for the OrderManager.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ai_trading_agent.trading_engine.models import Portfolio, Order, Trade, OrderStatus
from ai_trading_agent.trading_engine.order_manager import OrderManager

@pytest.fixture
def mock_portfolio():
    """Creates a Portfolio instance for testing."""
    # Use a real Portfolio instance as OrderManager interacts directly with its dicts
    return Portfolio(starting_balance=10000.0, current_balance=10000.0)

@pytest.fixture
def order_manager(mock_portfolio): # Depends on the mock_portfolio fixture
    """Creates an OrderManager instance with the mock portfolio."""
    return OrderManager(portfolio=mock_portfolio)

# --- Test Order Creation --- 
def test_create_market_order_success(order_manager, mock_portfolio):
    order = order_manager.create_order(
        symbol="BTC/USDT",
        side='buy',
        order_type='market',
        quantity=0.1
    )
    assert order is not None
    assert isinstance(order, Order)
    assert order.order_id in mock_portfolio.orders
    assert mock_portfolio.orders[order.order_id] == order
    assert order.status == 'new'

def test_create_limit_order_success(order_manager, mock_portfolio):
    order = order_manager.create_order(
        symbol="ETH/USD",
        side='sell',
        order_type='limit',
        quantity=1.0,
        price=3000.0
    )
    assert order is not None
    assert isinstance(order, Order)
    assert order.order_id in mock_portfolio.orders
    assert order.price == 3000.0
    assert order.status == 'new'

def test_create_order_invalid_quantity(order_manager, mock_portfolio):
    order = order_manager.create_order(
        symbol="BTC/USDT",
        side='buy',
        order_type='market',
        quantity=0
    )
    assert order is None
    assert not mock_portfolio.orders # No order should be added

    order_neg = order_manager.create_order(
        symbol="BTC/USDT",
        side='buy',
        order_type='market',
        quantity=-0.1
    )
    assert order_neg is None
    assert not mock_portfolio.orders

def test_create_limit_order_missing_price(order_manager, mock_portfolio):
    order = order_manager.create_order(
        symbol="ETH/USD",
        side='sell',
        order_type='limit',
        quantity=1.0,
        price=None
    )
    assert order is None
    assert not mock_portfolio.orders

def test_create_limit_order_invalid_price(order_manager, mock_portfolio):
    order = order_manager.create_order(
        symbol="ETH/USD",
        side='sell',
        order_type='limit',
        quantity=1.0,
        price=0
    )
    assert order is None
    assert not mock_portfolio.orders

    order_neg = order_manager.create_order(
        symbol="ETH/USD",
        side='sell',
        order_type='limit',
        quantity=1.0,
        price=-3000
    )
    assert order_neg is None
    assert not mock_portfolio.orders

# --- Test Order Retrieval --- 
def test_get_order_found(order_manager, mock_portfolio):
    created_order = order_manager.create_order("BTC/USDT", 'buy', 'market', 0.1)
    retrieved_order = order_manager.get_order(created_order.order_id)
    assert retrieved_order == created_order

def test_get_order_not_found(order_manager):
    retrieved_order = order_manager.get_order("nonexistent_id")
    assert retrieved_order is None

# --- Test Order Status Update --- 
def test_update_order_status_success(order_manager):
    order = order_manager.create_order("BTC/USDT", 'buy', 'market', 0.1)
    order_id = order.order_id
    # Use string literal for OrderStatus
    order_manager.update_order_status(order_id, 'open')
    updated_order = order_manager.get_order(order_id)
    assert updated_order.status == 'open'

def test_update_order_status_not_found(order_manager):
    # Should log a warning, but not raise an error
    order_manager.update_order_status("nonexistent_id", 'filled')
    # No assertion needed, just check it doesn't crash

# --- Test Order Cancellation --- 
def test_cancel_order_success(order_manager):
    order = order_manager.create_order("BTC/USDT", 'buy', 'market', 0.1)
    order_id = order.order_id
    cancelled = order_manager.cancel_order(order_id)
    assert cancelled is True
    updated_order = order_manager.get_order(order_id)
    assert updated_order.status == 'canceled'

def test_cancel_partially_filled_order_success(order_manager):
    order = order_manager.create_order("BTC/USDT", 'buy', 'limit', 0.2, 50000)
    order_id = order.order_id
    # Simulate partial fill
    trade = Trade(order_id=order_id, symbol="BTC/USDT", side='buy', quantity=0.1, price=49990)
    current_prices = {"BTC/USDT": 50000.0}
    order_manager.process_trade(trade, current_prices)
    assert order.status == 'partially_filled'
    # Cancel remaining
    cancelled = order_manager.cancel_order(order_id)
    assert cancelled is True
    updated_order = order_manager.get_order(order_id)
    assert updated_order.status == 'canceled'

def test_cancel_order_already_final_status(order_manager):
    order = order_manager.create_order("BTC/USDT", 'buy', 'market', 0.1)
    order_id = order.order_id
    # Simulate fill
    trade = Trade(order_id=order_id, symbol="BTC/USDT", side='buy', quantity=0.1, price=50000)
    current_prices = {"BTC/USDT": 50000.0}
    order_manager.process_trade(trade, current_prices)
    assert order.status == 'filled'
    # Attempt cancel
    cancelled = order_manager.cancel_order(order_id)
    assert cancelled is False # Cannot cancel filled order
    assert order.status == 'filled' # Status remains filled

def test_cancel_order_not_found(order_manager):
    cancelled = order_manager.cancel_order("nonexistent_id")
    assert cancelled is False

# --- Test Trade Processing --- 
def test_process_trade_full_fill(order_manager):
    order = order_manager.create_order("BTC/USDT", 'buy', 'market', 0.1)
    order_id = order.order_id
    trade = Trade(order_id=order_id, symbol="BTC/USDT", side='buy', quantity=0.1, price=50000)
    current_prices = {"BTC/USDT": 50000.0}
    order_manager.process_trade(trade, current_prices)

    updated_order = order_manager.get_order(order_id)
    assert updated_order.status == 'filled'
    assert updated_order.filled_quantity == pytest.approx(0.1)
    assert updated_order.average_fill_price == pytest.approx(50000)

def test_process_trade_partial_fill(order_manager):
    order = order_manager.create_order("ETH/USD", 'sell', 'limit', 1.0, 3000)
    order_id = order.order_id
    trade = Trade(order_id=order_id, symbol="ETH/USD", side='sell', quantity=0.4, price=3010)
    current_prices = {"ETH/USD": 3010.0}
    order_manager.process_trade(trade, current_prices)

    updated_order = order_manager.get_order(order_id)
    assert updated_order.status == 'partially_filled'
    assert updated_order.filled_quantity == pytest.approx(0.4)
    assert updated_order.average_fill_price == pytest.approx(3010)

def test_process_trade_multiple_fills(order_manager):
    order = order_manager.create_order("ETH/USD", 'sell', 'limit', 1.0, 3000)
    order_id = order.order_id
    trade1 = Trade(order_id=order_id, symbol="ETH/USD", side='sell', quantity=0.4, price=3010)
    trade2 = Trade(order_id=order_id, symbol="ETH/USD", side='sell', quantity=0.6, price=3005)
    current_prices = {"ETH/USD": 3005.0}
    order_manager.process_trade(trade1, current_prices)
    order_manager.process_trade(trade2, current_prices)

    updated_order = order_manager.get_order(order_id)
    assert updated_order.status == 'filled'
    assert updated_order.filled_quantity == pytest.approx(1.0)
    expected_avg_price = ((0.4 * 3010) + (0.6 * 3005)) / 1.0 # (1204 + 1803) / 1 = 3007
    assert updated_order.average_fill_price == pytest.approx(expected_avg_price)

def test_process_trade_for_unknown_order(order_manager):
    trade = Trade(order_id="unknown_order", symbol="BTC/USDT", side='buy', quantity=0.1, price=50000)
    current_prices = {"BTC/USDT": 50000.0}
    # Should log an error but not raise exception
    order_manager.process_trade(trade, current_prices)
    # No assertion needed, just check no crash

def test_process_trade_for_finalized_order(order_manager):
    order = order_manager.create_order("BTC/USDT", 'buy', 'market', 0.1)
    order_id = order.order_id
    # Mark as filled first
    order_manager.update_order_status(order_id, 'filled')
    
    # Try to process a trade for it
    trade = Trade(order_id=order_id, symbol="BTC/USDT", side='buy', quantity=0.05, price=50000)
    current_prices = {"BTC/USDT": 50000.0}
    order_manager.process_trade(trade, current_prices)
    
    # Order should remain filled with original details (no change from new trade)
    updated_order = order_manager.get_order(order_id)
    assert updated_order.status == 'filled'
    assert updated_order.filled_quantity == 0 # Original fill details weren't added

# --- Test Get Open Orders --- 
def test_get_open_orders_none(order_manager):
    assert order_manager.get_open_orders() == {}

def test_get_open_orders_all(order_manager, mock_portfolio):
    o1 = order_manager.create_order("BTC/USDT", 'buy', 'market', 0.1)
    o2 = order_manager.create_order("ETH/USD", 'sell', 'limit', 1.0, 3000)
    # Mark o1 as filled
    mock_portfolio.orders[o1.order_id].status = 'filled'
    # Leave o2 as new (which is open)
    
    open_orders = order_manager.get_open_orders()
    assert len(open_orders) == 1
    assert o2.order_id in open_orders
    assert o1.order_id not in open_orders

def test_get_open_orders_by_symbol(order_manager):
    o1 = order_manager.create_order("BTC/USDT", 'buy', 'market', 0.1)
    o2 = order_manager.create_order("ETH/USD", 'sell', 'limit', 1.0, 3000)
    o3 = order_manager.create_order("BTC/USDT", 'sell', 'limit', 0.05, 60000)
    # o1, o2, o3 are all 'new', hence open
    
    btc_open = order_manager.get_open_orders(symbol="BTC/USDT")
    assert len(btc_open) == 2
    assert o1.order_id in btc_open
    assert o3.order_id in btc_open

    eth_open = order_manager.get_open_orders(symbol="ETH/USD")
    assert len(eth_open) == 1
    assert o2.order_id in eth_open

    xrp_open = order_manager.get_open_orders(symbol="XRP/USD")
    assert len(xrp_open) == 0
