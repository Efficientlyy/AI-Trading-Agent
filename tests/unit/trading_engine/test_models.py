"""
Unit tests for trading engine data models (Order, Trade, Position, Portfolio).
"""

import pytest
from datetime import datetime, timedelta, timezone
import uuid
from pydantic import ValidationError

from trading_engine.models import (
    Order, Trade, Position, Portfolio,
    OrderSide, OrderType, OrderStatus, PositionSide,
    utcnow
)

# --- Test Order Model --- 
def test_order_creation_market():
    order = Order(symbol="BTC/USDT", side='buy', type='market', quantity=0.1)
    assert order.order_id.startswith('ord_')
    assert order.symbol == "BTC/USDT"
    assert order.side == 'buy'
    assert order.type == 'market'
    assert order.quantity == 0.1
    assert order.price is None
    assert order.status == 'new'
    assert order.filled_quantity == 0.0
    assert (utcnow() - order.created_at) < timedelta(seconds=1)
    assert order.created_at == order.updated_at

def test_order_creation_limit():
    order = Order(symbol="ETH/USD", side='sell', type='limit', quantity=1.5, price=3000.0)
    assert order.price == 3000.0
    assert order.type == 'limit'

def test_order_validation_limit_price():
    # Price required for limit
    with pytest.raises(ValidationError, match='Price is required for limit orders'):
        Order(symbol="BTC/USDT", side='buy', type='limit', quantity=0.1)
    # Price must be positive for limit
    with pytest.raises(ValidationError, match='Limit price must be positive'):
        Order(symbol="BTC/USDT", side='buy', type='limit', quantity=0.1, price=0)
    with pytest.raises(ValidationError, match='Limit price must be positive'):
        Order(symbol="BTC/USDT", side='buy', type='limit', quantity=0.1, price=-100)

def test_order_validation_quantity():
    with pytest.raises(ValidationError, match='quantity'):
        Order(symbol="BTC/USDT", side='buy', type='market', quantity=0)
    with pytest.raises(ValidationError, match='quantity'):
        Order(symbol="BTC/USDT", side='buy', type='market', quantity=-0.1)

def test_order_update_status():
    order = Order(symbol="BTC/USDT", side='buy', type='market', quantity=0.1)
    initial_update_time = order.updated_at
    # Need to sleep briefly to ensure time difference
    import time; time.sleep(0.001)
    order.update_status('open')
    assert order.status == 'open'
    assert order.updated_at > initial_update_time

def test_order_add_fill_single():
    order = Order(symbol="BTC/USDT", side='buy', type='market', quantity=0.1)
    order.add_fill(fill_quantity=0.1, fill_price=50000, commission=1.0, commission_asset='USDT')
    assert order.filled_quantity == pytest.approx(0.1)
    assert order.get_average_fill_price() == pytest.approx(50000)
    assert order.status == 'filled'
    assert order.fills[0]['commission'] == pytest.approx(1.0)
    assert order.fills[0]['commission_asset'] == 'USDT'

def test_order_add_fill_partial():
    order = Order(symbol="BTC/USDT", side='buy', type='market', quantity=0.2)
    order.add_fill(fill_quantity=0.1, fill_price=50000)
    assert order.filled_quantity == pytest.approx(0.1)
    assert order.get_average_fill_price() == pytest.approx(50000)
    assert order.status == 'partially_filled'
    assert order.remaining_quantity == pytest.approx(0.1)
    
    # Add another fill
    order.add_fill(fill_quantity=0.1, fill_price=51000)
    assert order.filled_quantity == pytest.approx(0.2)
    # Average price should be weighted by quantity
    assert order.get_average_fill_price() == pytest.approx(50500)
    assert order.status == 'filled'
    assert order.remaining_quantity == pytest.approx(0.0)

def test_order_add_fill_zero_qty():
    order = Order(symbol="BTC/USDT", side='buy', type='market', quantity=0.1)
    
    # Should raise ValueError for zero or negative quantity
    with pytest.raises(ValueError, match="Fill quantity must be positive"):
        order.add_fill(fill_quantity=0, fill_price=50000)
        
    with pytest.raises(ValueError, match="Fill quantity must be positive"):
        order.add_fill(fill_quantity=-0.1, fill_price=50000)
        
    # Order should remain unchanged
    assert order.filled_quantity == 0
    assert order.get_average_fill_price() is None
    assert order.status == 'new'

def test_order_add_fill_mixed_commission_asset():
    order = Order(symbol="BTC/USDT", side='buy', type='market', quantity=0.2)
    order.add_fill(fill_quantity=0.1, fill_price=50000, commission=1.0, commission_asset='USDT')
    
    # Add another fill with different commission asset
    fill = order.add_fill(fill_quantity=0.1, fill_price=51000, commission=0.0001, commission_asset='BTC')
    
    # Check that both fills are stored correctly
    assert len(order.fills) == 2
    assert order.fills[0]['commission_asset'] == 'USDT'
    assert order.fills[1]['commission_asset'] == 'BTC'
    assert order.status == 'filled'

# --- Test Trade Model --- 
def test_trade_creation():
    order_id = f"ord_{uuid.uuid4()}"
    trade = Trade(
        order_id=order_id,
        symbol="BTC/USDT",
        side='buy',
        quantity=0.05,
        price=50100,
        commission=0.5,
        commission_asset='USDT',
        is_maker=False
    )
    assert trade.trade_id.startswith('trd_')
    assert trade.order_id == order_id
    assert trade.quantity == 0.05
    assert trade.price == 50100
    assert trade.commission == 0.5
    assert trade.commission_asset == 'USDT'
    assert not trade.is_maker
    assert (utcnow() - trade.timestamp) < timedelta(seconds=1)

def test_trade_validation():
    order_id = f"ord_{uuid.uuid4()}"
    with pytest.raises(ValueError): # Quantity must be > 0
        Trade(order_id=order_id, symbol="BTC/USDT", side='buy', quantity=0, price=50000)
    with pytest.raises(ValueError): # Price must be > 0
        Trade(order_id=order_id, symbol="BTC/USDT", side='buy', quantity=0.1, price=0)

# --- Test Position Model --- 
def test_position_creation():
    pos = Position(symbol="ETH/USD", side='long', quantity=2.0, entry_price=3000.0)
    assert pos.symbol == "ETH/USD"
    assert pos.side == 'long'
    assert pos.quantity == 2.0
    assert pos.entry_price == 3000.0
    assert pos.unrealized_pnl == 0.0
    assert pos.realized_pnl == 0.0
    assert (utcnow() - pos.last_update_time) < timedelta(seconds=1)

def test_position_validation():
     with pytest.raises(ValidationError, match="quantity"): # Quantity >= 0
         Position(symbol="ETH/USD", side='long', quantity=-1.0, entry_price=3000.0)
     with pytest.raises(ValueError): # Entry price > 0
         Position(symbol="ETH/USD", side='long', quantity=1.0, entry_price=0)

def test_position_update_unrealized_pnl():
    pos = Position(symbol="ETH/USD", side='long', quantity=2.0, entry_price=3000.0)
    pos.update_market_price(current_price=3100.0)
    assert pos.unrealized_pnl == pytest.approx((3100.0 - 3000.0) * 2.0) # 200

    pos_short = Position(symbol="BTC/USDT", side='short', quantity=0.1, entry_price=50000.0)
    pos_short.update_market_price(current_price=49000.0)
    assert pos_short.unrealized_pnl == pytest.approx((50000.0 - 49000.0) * 0.1) # 100
    pos_short.update_market_price(current_price=51000.0)
    assert pos_short.unrealized_pnl == pytest.approx((50000.0 - 51000.0) * 0.1) # -100

def test_position_update_open_new():
    pos = Position(symbol="SOL/USD", side='long', quantity=0, entry_price=0) # Start empty
    pos.update_position(trade_qty=10, trade_price=150, trade_side='buy', current_market_price=155)
    assert pos.side == 'long'
    assert pos.quantity == 10
    assert pos.entry_price == 150
    assert pos.realized_pnl == 0
    assert pos.unrealized_pnl == pytest.approx((155 - 150) * 10) # 50

def test_position_update_increase_long():
    pos = Position(symbol="SOL/USD", side='long', quantity=10, entry_price=150)
    pos.update_position(trade_qty=5, trade_price=160, trade_side='buy', current_market_price=165)
    assert pos.side == 'long'
    assert pos.quantity == 15
    new_entry = ((10 * 150) + (5 * 160)) / 15 # (1500 + 800) / 15 = 2300 / 15 = 153.333
    assert pos.entry_price == pytest.approx(153.33333333)
    assert pos.realized_pnl == 0
    assert pos.unrealized_pnl == pytest.approx((165 - pos.entry_price) * 15)

def test_position_update_increase_short():
    pos = Position(symbol="SOL/USD", side='short', quantity=10, entry_price=150)
    pos.update_position(trade_qty=5, trade_price=140, trade_side='sell', current_market_price=135)
    assert pos.side == 'short'
    assert pos.quantity == 15
    new_entry = ((10 * 150) + (5 * 140)) / 15 # (1500 + 700) / 15 = 2200 / 15 = 146.666
    assert pos.entry_price == pytest.approx(146.66666666)
    assert pos.realized_pnl == 0
    assert pos.unrealized_pnl == pytest.approx((pos.entry_price - 135) * 15)

def test_position_update_reduce_long():
    pos = Position(symbol="SOL/USD", side='long', quantity=10, entry_price=150)
    pos.update_position(trade_qty=4, trade_price=160, trade_side='sell', current_market_price=165)
    assert pos.side == 'long'
    assert pos.quantity == 6
    assert pos.entry_price == 150 # Entry price doesn't change on reduction
    assert pos.realized_pnl == pytest.approx((160 - 150) * 4) # 40
    assert pos.unrealized_pnl == pytest.approx((165 - 150) * 6) # 90

def test_position_update_reduce_short():
    pos = Position(symbol="SOL/USD", side='short', quantity=10, entry_price=150)
    pos.update_position(trade_qty=4, trade_price=140, trade_side='buy', current_market_price=135)
    assert pos.side == 'short'
    assert pos.quantity == 6
    assert pos.entry_price == 150
    assert pos.realized_pnl == pytest.approx((150 - 140) * 4) # 40
    assert pos.unrealized_pnl == pytest.approx((150 - 135) * 6) # 90

def test_position_update_close_long():
    pos = Position(symbol="SOL/USD", side='long', quantity=10, entry_price=150, realized_pnl=10)
    pos.update_position(trade_qty=10, trade_price=160, trade_side='sell', current_market_price=160)
    assert pos.side == 'long' # Side doesn't matter when closed
    assert pos.quantity == 0
    assert pos.entry_price == 0
    assert pos.realized_pnl == pytest.approx(10 + (160 - 150) * 10) # 10 + 100 = 110
    assert pos.unrealized_pnl == 0

def test_position_update_close_short():
    pos = Position(symbol="SOL/USD", side='short', quantity=10, entry_price=150, realized_pnl=20)
    pos.update_position(trade_qty=10, trade_price=140, trade_side='buy', current_market_price=140)
    assert pos.side == 'long' # Reset to default when closed
    assert pos.quantity == 0
    assert pos.entry_price == 0
    assert pos.realized_pnl == pytest.approx(20 + (150 - 140) * 10) # 20 + 100 = 120
    assert pos.unrealized_pnl == 0

def test_position_update_flip_long_to_short():
    pos = Position(symbol="SOL/USD", side='long', quantity=10, entry_price=150, realized_pnl=10)
    pos.update_position(trade_qty=15, trade_price=160, trade_side='sell', current_market_price=155)
    assert pos.side == 'short' # Flipped
    assert pos.quantity == 5 # Remaining short quantity
    assert pos.entry_price == 160 # Entry price of the new short position
    # PnL from closing the long part
    close_pnl = (160 - 150) * 10 # 100
    assert pos.realized_pnl == pytest.approx(10 + close_pnl) # 110
    # Unrealized PnL of the new short position
    assert pos.unrealized_pnl == pytest.approx((160 - 155) * 5) # 25

def test_position_update_flip_short_to_long():
    pos = Position(symbol="SOL/USD", side='short', quantity=10, entry_price=150, realized_pnl=20)
    pos.update_position(trade_qty=15, trade_price=140, trade_side='buy', current_market_price=145)
    assert pos.side == 'long' # Flipped
    assert pos.quantity == 5 # Remaining long quantity
    assert pos.entry_price == 140
    # PnL from closing the short part
    close_pnl = (150 - 140) * 10 # 100
    assert pos.realized_pnl == pytest.approx(20 + close_pnl) # 120
    # Unrealized PnL of the new long position
    assert pos.unrealized_pnl == pytest.approx((145 - 140) * 5) # 25

# --- Test Portfolio Model --- 
@pytest.fixture
def sample_portfolio():
    return Portfolio(starting_balance=10000.0, current_balance=10000.0)

def test_portfolio_creation(sample_portfolio):
    assert sample_portfolio.starting_balance == 10000.0
    assert sample_portfolio.current_balance == 10000.0
    assert sample_portfolio.positions == {}
    assert sample_portfolio.orders == {}
    assert sample_portfolio.trades == []
    assert (utcnow() - sample_portfolio.timestamp) < timedelta(seconds=1)

def test_portfolio_get_position(sample_portfolio):
    pos = Position(symbol="BTC/USDT", side='long', quantity=0.1, entry_price=50000)
    sample_portfolio.positions["BTC/USDT"] = pos
    assert sample_portfolio.get_position("BTC/USDT") == pos
    assert sample_portfolio.get_position("ETH/USD") is None

def test_portfolio_update_from_trade_new_pos(sample_portfolio):
    trade = Trade(order_id="o1", symbol="BTC/USDT", side='buy', quantity=0.1, price=50000)
    market_prices = {"BTC/USDT": 51000}
    sample_portfolio.update_from_trade(trade, market_prices)

    assert "BTC/USDT" in sample_portfolio.positions
    pos = sample_portfolio.positions["BTC/USDT"]
    assert pos.side == 'long'
    assert pos.quantity == 0.1
    assert pos.entry_price == 50000
    assert pos.realized_pnl == 0
    assert pos.unrealized_pnl == pytest.approx((51000 - 50000) * 0.1) # 100

    # Check balance update (commission ignored)
    expected_balance = 10000.0 - (0.1 * 50000) # 10000 - 5000 = 5000
    assert sample_portfolio.current_balance == pytest.approx(expected_balance)

    assert len(sample_portfolio.trades) == 1
    assert sample_portfolio.trades[0] == trade

def test_portfolio_update_from_trade_existing_pos(sample_portfolio):
    # Initial position
    initial_pos = Position(symbol="BTC/USDT", side='long', quantity=0.1, entry_price=50000)
    sample_portfolio.positions["BTC/USDT"] = initial_pos
    sample_portfolio.current_balance = 5000 # Balance after initial buy

    # Trade to close the position
    trade = Trade(order_id="o2", symbol="BTC/USDT", side='sell', quantity=0.1, price=52000, commission=5.0)
    market_prices = {"BTC/USDT": 52000} # Assume market price is trade price for simplicity now
    sample_portfolio.update_from_trade(trade, market_prices)

    # Position should be closed and removed
    assert "BTC/USDT" not in sample_portfolio.positions

    # Check balance update
    # Start: 5000. Sell value: 0.1 * 52000 = 5200. Commission: 5.0
    expected_balance = 5000 + 5200 - 5.0 # 10195
    assert sample_portfolio.current_balance == pytest.approx(expected_balance)

    assert len(sample_portfolio.trades) == 1 # Only the new trade
    assert sample_portfolio.trades[0] == trade

    # Check total equity (Cash + Unrealized PnL -> Cash + 0 since no positions)
    assert sample_portfolio.total_equity == pytest.approx(expected_balance)

def test_portfolio_update_from_trade_multiple_positions(sample_portfolio):
    # Pos 1: BTC Long
    trade1 = Trade(order_id="o1", symbol="BTC/USDT", side='buy', quantity=0.1, price=50000)
    sample_portfolio.update_from_trade(trade1, {"BTC/USDT": 51000, "ETH/USD": 3000})
    # Pos 2: ETH Short
    trade2 = Trade(order_id="o2", symbol="ETH/USD", side='sell', quantity=2.0, price=3000)
    sample_portfolio.update_from_trade(trade2, {"BTC/USDT": 52000, "ETH/USD": 2900})

    assert "BTC/USDT" in sample_portfolio.positions
    assert "ETH/USD" in sample_portfolio.positions

    btc_pos = sample_portfolio.positions["BTC/USDT"]
    eth_pos = sample_portfolio.positions["ETH/USD"]

    # Check BTC position (only updated on first trade, PnL updated on second)
    assert btc_pos.quantity == 0.1
    assert btc_pos.entry_price == 50000
    btc_pos.update_market_price(52000) # Manual update for check
    assert btc_pos.unrealized_pnl == pytest.approx((52000 - 50000) * 0.1) # 200

    # Check ETH position
    assert eth_pos.side == 'short'
    assert eth_pos.quantity == 2.0
    assert eth_pos.entry_price == 3000
    assert eth_pos.unrealized_pnl == pytest.approx((3000 - 2900) * 2.0) # 200

    # Check balance
    balance = 10000.0
    balance -= (0.1 * 50000) # trade1 buy
    balance += (2.0 * 3000) # trade2 sell
    assert sample_portfolio.current_balance == pytest.approx(balance) # 10000 - 5000 + 6000 = 11000

    # Check total equity
    expected_equity = balance + btc_pos.unrealized_pnl + eth_pos.unrealized_pnl
    # Need to call update_all_unrealized_pnl within update_from_trade or separately
    sample_portfolio.update_all_unrealized_pnl({"BTC/USDT": 52000, "ETH/USD": 2900}) # Ensure PnL is current
    assert sample_portfolio.total_equity == pytest.approx(11000 + 200 + 200) # 11400

def test_portfolio_total_realized_pnl(sample_portfolio):
    # Simulate a round trip
    trade_buy = Trade(order_id="o1", symbol="XRP/USD", side='buy', quantity=1000, price=0.50)
    sample_portfolio.update_from_trade(trade_buy, {"XRP/USD": 0.55})
    assert sample_portfolio.total_realized_pnl == 0 # PnL is unrealized
    
    # Store the position's realized PnL before it gets closed
    position_realized_pnl = sample_portfolio.positions["XRP/USD"].realized_pnl
    assert position_realized_pnl == 0  # Should be 0 before closing
    
    # Store the portfolio's total realized PnL before the closing trade
    portfolio_realized_pnl_before = sample_portfolio.total_realized_portfolio_pnl
    assert portfolio_realized_pnl_before == 0  # Should be 0 before closing

    trade_sell = Trade(order_id="o2", symbol="XRP/USD", side='sell', quantity=1000, price=0.60)
    sample_portfolio.update_from_trade(trade_sell, {"XRP/USD": 0.60})
    
    # Position should be closed and removed from positions
    assert "XRP/USD" not in sample_portfolio.positions
    
    # Now, the portfolio's total_realized_pnl should reflect the closed PnL.
    expected_pnl = (0.60 - 0.50) * 1000 # 100
    assert sample_portfolio.total_realized_pnl == pytest.approx(expected_pnl)
    # Check the accumulator directly
    assert sample_portfolio.total_realized_portfolio_pnl == pytest.approx(expected_pnl)

    # Test adding more PnL
    trade_buy2 = Trade(order_id="o3", symbol="XRP/USD", side='buy', quantity=500, price=0.58)
    sample_portfolio.update_from_trade(trade_buy2, {"XRP/USD": 0.59})
    trade_sell2 = Trade(order_id="o4", symbol="XRP/USD", side='sell', quantity=500, price=0.55)
    sample_portfolio.update_from_trade(trade_sell2, {"XRP/USD": 0.55})
    # PnL from second round trip: (0.55 - 0.58) * 500 = -15
    expected_pnl += -15
    assert sample_portfolio.total_realized_pnl == pytest.approx(expected_pnl) # 100 - 15 = 85
    assert sample_portfolio.total_realized_portfolio_pnl == pytest.approx(expected_pnl)
