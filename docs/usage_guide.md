# AI Trading Agent Usage Guide

This guide explains how to use the AI Trading Agent, including running tests, understanding core components, and eventually running backtests.

## Running Tests

As mentioned in the `README.md`, you can run the test suite using `pytest`:

```bash
# Run all tests
pytest

# Run tests in a specific directory
pytest tests/unit/trading_engine/

# Run with verbose output
pytest -v
```

## Core Trading Engine Models (`src/trading_engine/models.py`)

These Pydantic models form the foundation of the trading simulation. Understanding them is key to extending the engine or interpreting backtest results.

### `Order`
Represents a trading order. Key attributes:
*   `order_id`: Unique identifier.
*   `symbol`: Asset symbol (e.g., 'BTC/USDT').
*   `type`: 'market', 'limit', 'stop'.
*   `side`: 'buy', 'sell'.
*   `quantity`: Amount to trade (always positive).
*   `price`: Limit or stop price (required for non-market orders).
*   `status`: 'pending', 'open', 'filled', 'cancelled', 'rejected'.
*   `fills`: List of `Trade` objects representing partial or full fills.
*   `get_average_fill_price()`: Calculates the average price if filled.
*   `get_filled_quantity()`: Calculates the total filled quantity.

**Example:**
```python
from src.trading_engine.models import Order

limit_order = Order(
    symbol='ETH/USD',
    type='limit',
    side='buy',
    quantity=0.5,
    price=3000.0
)
print(limit_order.status) # Output: OrderStatus.PENDING
```

### `Trade`
Represents an executed trade.
*   `trade_id`: Unique identifier.
*   `order_id`: ID of the order that generated this trade.
*   `symbol`: Asset symbol.
*   `side`: 'buy', 'sell'.
*   `quantity`: Quantity executed in this trade.
*   `price`: Execution price.
*   `timestamp`: Execution time.
*   `fee`: Transaction fee (optional).

### `Position`
Represents the current holding of a single asset.
*   `symbol`: Asset symbol.
*   `quantity`: Current holding amount (absolute value).
*   `side`: 'long', 'short', or 'flat'.
*   `entry_price`: Average price at which the position was entered.
*   `unrealized_pnl`: Profit or loss based on the current market price (requires market price input).
*   `realized_pnl`: Profit or loss locked in from closing parts of the position.
*   `update_position(trade)`: Modifies the position based on an executed trade.
*   `get_position_value(current_price)`: Calculates the current market value.

**Example:**
```python
from src.trading_engine.models import Position, Trade, Side, utcnow

# Assume a buy trade occurred
trade = Trade(
    order_id="ord_123",
    symbol="BTC/USDT",
    side=Side.BUY,
    quantity=0.1,
    price=60000.0,
    timestamp=utcnow()
)

position = Position(symbol="BTC/USDT")
position.update_position(trade)

print(position.quantity)  # Output: 0.1
print(position.side)      # Output: PositionSide.LONG
print(position.entry_price) # Output: 60000.0
```

### `Portfolio`
Represents the entire trading account state.
*   `cash`: Available cash balance.
*   `positions`: Dictionary mapping symbols to `Position` objects.
*   `orders`: Dictionary mapping order IDs to `Order` objects (tracking open orders).
*   `update_from_trade(trade, current_market_prices)`: Updates cash, relevant position, and calculates realized PnL based on a trade.
*   `get_total_value(current_market_prices)`: Calculates the total portfolio value (cash + position values).
*   `get_exposure(symbol, current_price)`: Calculates the market exposure for a specific symbol.

## Running Backtests

(Details on configuring and running backtests using `src/backtesting/backtester.py` will be added here once the component is more developed.)

Refer to `docs/architecture.md` for a component overview and `docs/trading_engine_models_update.md` for details on recent model changes.
