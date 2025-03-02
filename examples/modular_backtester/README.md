# Modular Backtesting Framework

A comprehensive framework for backtesting trading strategies with advanced features like risk management, position sizing, market regime detection, and parameter optimization.

## Features

- **Multiple Strategy Types**:
  - Moving Average Crossover
  - Enhanced MA with market regime detection
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Multi-strategy system with consensus signals

- **Risk Management**:
  - Stop Loss
  - Take Profit
  - Trailing Stop
  - Position sizing based on volatility

- **Advanced Analysis**:
  - Market regime detection (trending, ranging, volatile)
  - Parameter optimization
  - Detailed performance metrics
  - Equity curve visualization
  - Trade distribution analysis

- **Data Utilities**:
  - Synthetic data generation
  - CSV data loading/saving
  - Timeframe resampling
  - Data processing for indicators

## Project Structure

- `models.py` - Core data structures and enums
- `strategies.py` - Strategy implementations
- `backtester.py` - Backtesting engine and metrics
- `data_utils.py` - Data handling utilities
- `run_backtest.py` - Command-line interface

## Usage

### Basic Usage

Run a backtest with synthetic data:

```bash
python examples/modular_backtester/run_backtest.py --generate-data --strategy enhanced_ma --use-stop-loss --use-take-profit
```

Run a backtest with your own data:

```bash
python examples/modular_backtester/run_backtest.py --data-file data/historical/BTC-USD_1h.csv --symbol BTC-USD --timeframe 1h --strategy macd
```

### Parameter Optimization

Run parameter optimization:

```bash
python examples/modular_backtester/run_backtest.py --generate-data --strategy rsi --optimize --optimize-metric sharpe_ratio --max-iterations 20
```

### Complete Command-Line Options

```
usage: run_backtest.py [-h] [--data-file DATA_FILE] [--symbol SYMBOL]
                       [--timeframe {1m,5m,15m,30m,1h,4h,1d,1w}]
                       [--start-date START_DATE] [--end-date END_DATE]
                       [--generate-data]
                       [--strategy {ma_crossover,enhanced_ma,rsi,macd,multi}]
                       [--fast-period FAST_PERIOD] [--slow-period SLOW_PERIOD]
                       [--signal-period SIGNAL_PERIOD] [--rsi-period RSI_PERIOD]
                       [--overbought OVERBOUGHT] [--oversold OVERSOLD]
                       [--initial-capital INITIAL_CAPITAL]
                       [--position-size POSITION_SIZE] [--use-stop-loss]
                       [--stop-loss-pct STOP_LOSS_PCT] [--use-take-profit]
                       [--take-profit-pct TAKE_PROFIT_PCT] [--use-trailing-stop]
                       [--trailing-stop-pct TRAILING_STOP_PCT]
                       [--commission COMMISSION] [--optimize]
                       [--optimize-metric {sharpe_ratio,total_pnl,profit_factor,win_rate}]
                       [--max-iterations MAX_ITERATIONS]
                       [--output-dir OUTPUT_DIR] [--save-results]
```

## Programmatic Usage

```python
from examples.modular_backtester import (
    CandleData, TimeFrame, EnhancedMAStrategy, StrategyBacktester, generate_sample_data
)
from datetime import datetime, timedelta

# Generate sample data
start_time = datetime.now() - timedelta(days=365)
end_time = datetime.now()
candles = generate_sample_data(
    symbol="BTC-USD",
    timeframe=TimeFrame.HOUR_1,
    start_time=start_time,
    end_time=end_time,
    base_price=10000.0,
    volatility=0.015
)

# Create strategy
strategy = EnhancedMAStrategy(
    fast_ma_period=8,
    slow_ma_period=21,
    fast_ma_type="EMA",
    slow_ma_type="EMA",
    use_regime_filter=True
)

# Set up backtester
backtester = StrategyBacktester(
    initial_capital=10000.0,
    position_size=0.1,
    use_stop_loss=True,
    stop_loss_pct=0.05,
    use_take_profit=True,
    take_profit_pct=0.1
)

# Set strategy and add data
backtester.set_strategy(strategy)
backtester.add_historical_data("BTC-USD", candles)

# Run backtest
metrics = backtester.run_backtest()

# Generate report
report_path = backtester.generate_report("reports")
print(f"Report saved to: {report_path}")

# Print summary
metrics_dict = metrics.to_dict()
print(f"Total P&L: ${metrics_dict['total_pnl']:.2f}")
print(f"Win Rate: {metrics_dict['win_rate']:.2f}%")
print(f"Sharpe Ratio: {metrics_dict['sharpe_ratio']:.2f}")
```

## Extending the Framework

### Adding a New Strategy

Create a new strategy by extending the base `Strategy` class:

```python
from examples.modular_backtester import Strategy, Signal, SignalType, CandleData

class MyCustomStrategy(Strategy):
    def __init__(self, param1: float = 0.5, param2: int = 10):
        super().__init__("MyStrategy")
        self.param1 = param1
        self.param2 = param2
        
    def process_candle(self, candle: CandleData) -> Optional[Signal]:
        # Implement your strategy logic here
        # Return a Signal when a trade should be opened
        # Return None when no action should be taken
        pass
```

## Performance Metrics

The backtester provides the following performance metrics:

- **Total P&L**: Total profit/loss in base currency
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit divided by gross loss
- **Max Drawdown**: Maximum peak-to-trough decline in equity
- **Sharpe Ratio**: Risk-adjusted return measure
- **Volatility**: Annualized volatility of returns
- **Average Trade Duration**: Average duration of trades
- **Average Win P&L**: Average profit of winning trades
- **Average Loss P&L**: Average loss of losing trades
- **Max Consecutive Losses**: Maximum consecutive losing trades

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib

## License

MIT 