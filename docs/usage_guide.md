# AI Trading Agent Usage Guide

This guide explains how to use the AI Trading Agent, including running tests, understanding core components, and running backtests.

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
*   `add_fill(fill_quantity, fill_price)`: Adds a new fill to the order.

**Example:**
```python
from src.trading_engine.models import Order
from src.trading_engine.enums import OrderType, OrderSide

limit_order = Order(
    symbol='ETH/USD',
    type=OrderType.LIMIT,
    side=OrderSide.BUY,
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
*   `market_price`: Current market price (updated via `update_market_price`).
*   `unrealized_pnl`: Profit or loss based on the current market price.
*   `realized_pnl`: Profit or loss locked in from closing parts of the position.
*   `update_position(trade)`: Modifies the position based on an executed trade.
*   `update_market_price(current_price)`: Updates the market price and recalculates unrealized P&L.
*   `get_position_value(current_price)`: Calculates the current market value.

**Example:**
```python
from src.trading_engine.models import Position, Trade
from src.trading_engine.enums import OrderSide
from datetime import datetime

# Assume a buy trade occurred
trade = Trade(
    order_id="ord_123",
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    quantity=0.1,
    price=60000.0,
    timestamp=datetime.utcnow()
)

position = Position(symbol="BTC/USDT")
position.update_position(trade)

print(position.quantity)  # Output: 0.1
print(position.side)      # Output: PositionSide.LONG
print(position.entry_price) # Output: 60000.0

# Update market price to calculate unrealized P&L
position.update_market_price(62000.0)
print(position.unrealized_pnl)  # Output: 200.0 (0.1 BTC * $2000 profit)
```

### `Portfolio`
Represents the entire trading account state.
*   `cash`: Available cash balance.
*   `positions`: Dictionary mapping symbols to `Position` objects.
*   `orders`: Dictionary mapping order IDs to `Order` objects (tracking open orders).
*   `update_from_trade(trade)`: Updates cash, relevant position, and calculates realized PnL based on a trade.
*   `update_total_value(current_market_prices)`: Calculates and updates the total portfolio value (cash + position values).
*   `get_exposure(symbol, current_price)`: Calculates the market exposure for a specific symbol.

## Running Backtests

The AI Trading Agent includes a comprehensive backtesting framework that supports multi-asset strategies with portfolio-level analysis.

### Backtesting Components

1. **Backtester** (`src/backtesting/backtester.py`): The main class for running backtests.
2. **Performance Metrics** (`src/backtesting/performance_metrics.py`): Calculates performance metrics like Sharpe ratio, Sortino ratio, drawdowns, etc.
3. **Rust Backtester** (`src/backtesting/rust_backtester.py`): A faster implementation using Rust for performance-critical components.

### Multi-Asset Backtesting Example

The repository includes a multi-asset backtesting example in `examples/multi_asset_backtest.py`. This example demonstrates how to:

1. Create a mock data provider for multiple assets
2. Initialize a moving average crossover strategy
3. Run a backtest with the strategy
4. Calculate and display performance metrics
5. Visualize the results

To run the example:

```bash
python examples/multi_asset_backtest.py
```

### Creating Your Own Backtest

To create your own backtest, follow these steps:

1. **Create a data provider**:
```python
from src.data_acquisition.mock_provider import MockDataProvider

data_provider = MockDataProvider()
symbols = ['BTC-USD', 'ETH-USD']
data = {}

for symbol in symbols:
    df = data_provider.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval='1d'
    )
    data[symbol] = df
```

2. **Create a strategy**:
```python
from src.strategies.ma_crossover_strategy import MACrossoverStrategy

strategy = MACrossoverStrategy(
    symbols=symbols,
    fast_period=10,
    slow_period=30,
    risk_pct=0.02,
    max_position_pct=0.2
)
```

3. **Initialize the backtester**:
```python
from src.backtesting.backtester import Backtester

backtester = Backtester(
    data=data,
    initial_capital=10000.0,
    commission_rate=0.001,
    slippage=0.001,
    enable_fractional=True
)
```

4. **Run the backtest**:
```python
results = backtester.run(strategy.generate_signals)
```

5. **Analyze the results**:
```python
# Print key metrics
print(f"Total Return: {results.total_return:.2%}")
print(f"Annualized Return: {results.annualized_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Sortino Ratio: {results.sortino_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")

# Plot equity curve
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(results.equity_curve)
plt.title('Portfolio Equity Curve')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.savefig('backtest_results.png')
plt.show()
```

Refer to `docs/architecture.md` for a component overview and `examples/multi_asset_backtest.py` for a complete working example.
