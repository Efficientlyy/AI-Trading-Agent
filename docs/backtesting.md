# Backtesting Engine Documentation

## Overview

The AI Crypto Trading System includes a high-performance backtesting engine implemented in Rust with Python bindings. This document explains how to use the backtesting engine, its architecture, features, and provides examples.

## Architecture

The backtesting engine is designed with the following components:

### Rust Implementation

![Backtesting Engine Architecture](./diagrams/backtest_architecture.png)

1. **Core Engine** (`BacktestEngine`): The central component responsible for processing market data, executing orders, tracking positions, and calculating performance statistics.

2. **Order Management**: Handles various order types (market, limit, stop), tracks their status, and executes them against market data.

3. **Position Management**: Tracks open positions, calculates unrealized P&L, and manages position closure.

4. **Performance Metrics**: Calculates various statistics like Sharpe ratio, drawdown, win rate, and profit factor.

### Python Interface

The Rust implementation is exposed to Python through PyO3 bindings, with the following components:

1. **Python Wrapper** (`BacktestEngine`): A high-level interface that handles Python-to-Rust data conversion and provides an intuitive API.

2. **Pure Python Fallback** (`PyBacktestEngine`): A pure Python implementation that mirrors the Rust functionality, used when Rust components are not available.

3. **Common Enums**: Shared enum values for order types, order sides, time frames, and backtest modes.

## Features

### High Performance

The Rust implementation provides significant performance improvements over pure Python:

- Processing of 1 million candles in under 500ms (60x+ faster than Python)
- Efficient memory usage for large datasets
- Optimized order matching algorithm

### Comprehensive Order Types

The backtest engine supports various order types:

- **Market Orders**: Executed immediately at the current market price
- **Limit Orders**: Executed when the price reaches a specified limit or better
- **Stop Market Orders**: Converted to market orders when price reaches the stop level
- **Stop Limit Orders**: Converted to limit orders when price reaches the stop level

### Detailed Performance Metrics

The engine calculates a wide range of performance metrics:

- Total/winning/losing trades
- Win rate and profit factor
- Maximum drawdown
- Sharpe and Sortino ratios
- Daily/monthly returns
- Return on investment (ROI)

### Additional Features

- Multi-asset backtesting
- Commission and slippage modeling
- Equity curve tracking
- Position sizing strategies
- Custom event handling

## Using the Backtesting Engine

### Basic Usage

```python
from src.backtesting import BacktestEngine, TimeFrame, OrderSide
from datetime import datetime

# Create a backtest engine
engine = BacktestEngine(
    initial_balance=10000.0,
    symbols=["BTCUSDT"],
    commission_rate=0.001  # 0.1%
)

# Process historical data
for timestamp, open_price, high, low, close, volume in historical_data:
    engine.process_candle(
        symbol="BTCUSDT",
        timestamp=timestamp,
        open_price=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timeframe=TimeFrame.HOUR_1
    )
    
    # Your strategy logic here
    if buy_condition:
        engine.submit_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0
        )
    elif sell_condition:
        engine.submit_market_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=1.0
        )

# Get backtest results
stats = engine.get_stats()
print(f"Total trades: {stats.total_trades}")
print(f"Win rate: {stats.win_rate:.2f}%")
print(f"Profit factor: {stats.profit_factor:.2f}")
print(f"Max drawdown: {stats.max_drawdown_percent:.2f}%")
print(f"Sharpe ratio: {stats.sharpe_ratio:.2f}")
```

### Implementing a Strategy

Here's how to implement a simple moving average crossover strategy:

```python
class MovingAverageCrossoverStrategy:
    def __init__(self, engine, symbol, fast_period=10, slow_period=30):
        self.engine = engine
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_ma = []
        self.slow_ma = []
        self.position_size = 0
        
    def on_candle(self, candle):
        # Update moving averages
        close = candle['close']
        self.fast_ma.append(close)
        self.slow_ma.append(close)
        
        if len(self.fast_ma) > self.fast_period:
            self.fast_ma.pop(0)
        if len(self.slow_ma) > self.slow_period:
            self.slow_ma.pop(0)
            
        # Calculate current values
        if len(self.fast_ma) < self.fast_period or len(self.slow_ma) < self.slow_period:
            return
            
        fast_value = sum(self.fast_ma) / len(self.fast_ma)
        slow_value = sum(self.slow_ma) / len(self.slow_ma)
        
        # Trading logic
        if fast_value > slow_value and self.position_size == 0:
            # Buy signal
            quantity = self.engine.get_balance() * 0.95 / close
            self.engine.submit_market_order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                quantity=quantity
            )
            self.position_size = quantity
        elif fast_value < slow_value and self.position_size > 0:
            # Sell signal
            self.engine.submit_market_order(
                symbol=self.symbol,
                side=OrderSide.SELL,
                quantity=self.position_size
            )
            self.position_size = 0
```

### Running a Backtest

```python
from src.backtesting import BacktestEngine, TimeFrame, OrderSide
import pandas as pd

# Load historical data
data = pd.read_csv('btc_hourly.csv')

# Initialize backtest engine
engine = BacktestEngine(
    initial_balance=10000.0,
    symbols=["BTCUSDT"],
    commission_rate=0.001
)

# Initialize strategy
strategy = MovingAverageCrossoverStrategy(
    engine=engine,
    symbol="BTCUSDT",
    fast_period=10,
    slow_period=30
)

# Run backtest
for _, row in data.iterrows():
    # Process candle
    engine.process_candle(
        symbol="BTCUSDT",
        timestamp=row['timestamp'],
        open_price=row['open'],
        high=row['high'],
        low=row['low'],
        close=row['close'],
        volume=row['volume'],
        timeframe=TimeFrame.HOUR_1
    )
    
    # Update strategy
    strategy.on_candle({
        'timestamp': row['timestamp'],
        'open': row['open'],
        'high': row['high'],
        'low': row['low'],
        'close': row['close'],
        'volume': row['volume']
    })

# Get backtest results
stats = engine.get_stats()
print(f"Final balance: ${engine.get_balance():.2f}")
print(f"Total trades: {stats.total_trades}")
print(f"Win rate: {stats.win_rate:.2f}%")
print(f"Max drawdown: {stats.max_drawdown_percent:.2f}%")
print(f"Sharpe ratio: {stats.sharpe_ratio:.2f}")
```

## Advanced Usage

### Using Different Order Types

```python
# Market order
engine.submit_market_order(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    quantity=1.0
)

# Limit order
engine.submit_limit_order(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    quantity=1.0,
    price=19500.0
)

# Stop market order
engine.submit_stop_market_order(
    symbol="BTCUSDT",
    side=OrderSide.SELL,
    quantity=1.0,
    stop_price=19000.0
)

# Stop limit order
engine.submit_stop_limit_order(
    symbol="BTCUSDT",
    side=OrderSide.SELL,
    quantity=1.0,
    stop_price=19000.0,
    limit_price=18900.0
)
```

### Tracking Equity Curve

```python
# Initialize an empty list to store equity values
equity_curve = []

# Run backtest
for timestamp, row in data.iterrows():
    # Process candle
    engine.process_candle(...)
    
    # Store equity value
    equity_curve.append({
        'timestamp': timestamp,
        'equity': engine.get_equity()
    })

# Convert to DataFrame
equity_df = pd.DataFrame(equity_curve)

# Plot equity curve
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(equity_df['timestamp'], equity_df['equity'])
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Multi-Asset Backtesting

```python
# Initialize backtest engine with multiple symbols
engine = BacktestEngine(
    initial_balance=10000.0,
    symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    commission_rate=0.001
)

# Process data for multiple assets
for _, row in btc_data.iterrows():
    engine.process_candle(symbol="BTCUSDT", ...)

for _, row in eth_data.iterrows():
    engine.process_candle(symbol="ETHUSDT", ...)

for _, row in ada_data.iterrows():
    engine.process_candle(symbol="ADAUSDT", ...)
```

## Performance Tips

1. **Pre-process Data**: Convert timestamps to integers before feeding them to the backtester.

2. **Batch Processing**: If possible, use vectorized operations for indicator calculations outside the backtest loop.

3. **Optimize Memory Usage**: For very large datasets, consider processing data in chunks or using memory-mapped files.

4. **Rust vs Python**: Use the Rust implementation for large backtests; the Python fallback is convenient for small tests and debugging.

5. **Profiling**: Monitor CPU and memory usage during backtests to identify bottlenecks.

## Troubleshooting

### Common Issues

1. **Rust Components Not Found**:
   - Ensure Rust is installed
   - Run `cargo build --release` in the `rust` directory
   - Check that the compiled library is in the correct location

2. **Memory Issues with Large Datasets**:
   - Process data in smaller chunks
   - Use a larger machine or cloud instance
   - Apply data filtering to reduce the dataset size

3. **Inconsistent Backtest Results**:
   - Verify the candle processing order (timestamp sorting)
   - Check for data gaps or outliers
   - Ensure commissions and slippage are properly accounted for

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Track specific order execution
order_id = engine.submit_market_order(...)
print(f"Order submitted: {order_id}")

# Check order status
order = engine.get_order(order_id)
print(f"Order status: {order.status}")

# Inspect position details
position = engine.get_position("BTCUSDT")
if position:
    print(f"Position size: {position.size}")
    print(f"Position entry price: {position.entry_price}")
    print(f"Unrealized P&L: {position.unrealized_pnl}")
```

## API Reference

### BacktestEngine

The main class for backtesting.

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| initial_balance | float | Starting account balance |
| symbols | List[str] | List of trading symbols |
| commission_rate | float | Trading fee percentage (e.g., 0.001 for 0.1%) |
| slippage | float | Slippage percentage (e.g., 0.0005 for 0.05%) |
| start_time | datetime | Backtest start time |
| end_time | datetime | Backtest end time |
| mode | str | Backtest mode ("candles" or "ticks") |

#### Methods

| Method | Description |
|--------|-------------|
| process_candle() | Process a single candle of market data |
| submit_market_order() | Submit a market order |
| submit_limit_order() | Submit a limit order |
| submit_stop_market_order() | Submit a stop market order |
| submit_stop_limit_order() | Submit a stop limit order |
| get_balance() | Get current account balance |
| get_equity() | Get total account equity including open positions |
| get_position() | Get details of a position for a specific symbol |
| get_stats() | Get performance statistics |
| get_order() | Get details of a specific order |
| get_orders() | Get all orders |
| get_trades() | Get all completed trades |

### BacktestStats

Class that holds performance statistics.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| initial_balance | float | Starting account balance |
| final_balance | float | Final account balance |
| total_trades | int | Total number of trades executed |
| winning_trades | int | Number of winning trades |
| losing_trades | int | Number of losing trades |
| break_even_trades | int | Number of break-even trades |
| win_rate | float | Percentage of winning trades |
| profit_factor | float | Ratio of gross profit to gross loss |
| max_drawdown_percent | float | Maximum drawdown as percentage |
| max_drawdown_duration | int | Maximum drawdown duration in candles |
| sharpe_ratio | float | Sharpe ratio (annualized) |
| sortino_ratio | float | Sortino ratio (annualized) |
| calmar_ratio | float | Calmar ratio |
| total_return_percent | float | Total return as percentage |
| annualized_return | float | Annualized return |
| daily_returns | List[float] | List of daily returns |

## Appendix

### Comparison with Other Backtesting Frameworks

| Feature | AI Crypto Trading | Backtrader | Zipline | PyAlgoTrade |
|---------|------------------|------------|---------|-------------|
| Performance | Very High (Rust) | Medium | Medium | Medium |
| Ease of Use | High | Medium | High | Medium |
| Multi-Asset | Yes | Yes | Yes | Yes |
| Order Types | Advanced | Advanced | Basic | Basic |
| Live Trading | Yes | Yes | Limited | Yes |
| Visualization | Basic | Advanced | Basic | Basic |
| Community | New | Large | Large | Medium |

### Future Roadmap

1. **Q4 2023**: 
   - Add Monte Carlo simulation capabilities
   - Improve visualization with interactive charts

2. **Q1 2024**:
   - Add portfolio optimization features
   - Implement walk-forward testing

3. **Q2 2024**:
   - Add machine learning integration
   - Expand to futures and options backtesting 