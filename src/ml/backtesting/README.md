# Market Regime Backtesting

A comprehensive backtesting framework for testing trading strategies based on market regime detection. This module provides a flexible and powerful set of tools to evaluate how different trading strategies perform across various market regimes.

## Features

- **Regime-based Trading Strategy**: Implement strategies that adapt to different market regimes
- **Performance Metrics**: Comprehensive set of performance metrics for strategy evaluation
- **Position Sizing**: Multiple position sizing algorithms (fixed, percent, Kelly, volatility-based)
- **Risk Management**: Various risk management techniques (stop-loss, trailing stops, volatility-based stops)
- **Visualization Tools**: Rich visualization capabilities for equity curves, drawdowns, and regime analysis
- **HTML Reports**: Generate detailed HTML reports for backtest results

## Quick Start

```python
from src.ml.backtesting import RegimeStrategy
import yfinance as yf

# Download market data
ticker = yf.Ticker('SPY')
df = ticker.history(period='5y')

# Prepare data for backtesting
data = {
    'symbol': 'SPY',
    'dates': df.index.tolist(),
    'prices': df['Close'].values.tolist(),
    'returns': df['Close'].pct_change().fillna(0).values.tolist(),
    'volumes': df['Volume'].values.tolist(),
    'highs': df['High'].values.tolist(),
    'lows': df['Low'].values.tolist()
}

# Create and backtest a regime-based strategy
strategy = RegimeStrategy(
    detector_method='trend',
    detector_params={'n_regimes': 3, 'trend_method': 'macd'},
    regime_rules={
        0: {'action': 'sell', 'allocation': 0.0},  # Downtrend
        1: {'action': 'hold', 'allocation': 0.5},  # Sideways
        2: {'action': 'buy', 'allocation': 1.0}    # Uptrend
    },
    initial_capital=10000.0,
    position_sizing='percent',
    stop_loss_pct=0.05
)

# Run backtest
results = strategy.backtest(data)

# Plot equity curve
strategy.plot_equity_curve()

# Print performance metrics
print("Performance Metrics:")
for metric, value in results['performance_metrics'].items():
    print(f"  {metric}: {value}")
```

## Example Script

The module includes a sample script `simple_example.py` that demonstrates how to use the backtesting framework with realistic data:

```python
from src.ml.backtesting.simple_example import run_simple_backtest

# Run a backtest on SPY with the trend detector
run_simple_backtest(
    symbol="SPY",
    period="5y",
    detector_method="trend",
    output_dir="./backtest_results"
)
```

## Module Components

### Core Components

- **RegimeStrategy**: Main class for implementing regime-based trading strategies
- **Backtester**: Low-level backtesting engine with realistic simulation of transactions

### Position Sizing

- **FixedPositionSizer**: Fixed fraction of equity
- **PercentPositionSizer**: Percentage of equity for each trade
- **KellyPositionSizer**: Position sizing based on Kelly criterion
- **VolatilityPositionSizer**: Adjust position size based on volatility

### Risk Management

- **BasicRiskManager**: Simple stop-loss and take-profit levels
- **TrailingStopManager**: Trailing stops that move with the price
- **VolatilityBasedRiskManager**: ATR-based stops that adapt to market volatility

### Performance Metrics

- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown
- Win Rate, Profit Factor
- Regime-specific metrics

### Visualization

- Equity curve with regime highlighting
- Drawdown analysis
- Regime performance comparison
- Trade analysis

## Advanced Usage

### Custom Position Sizing

```python
from src.ml.backtesting import RegimeStrategy, VolatilityPositionSizer

# Create a volatility-based position sizer
position_sizer = VolatilityPositionSizer(
    target_risk_pct=0.01,  # Target 1% daily risk
    volatility_lookback=20
)

# Create a strategy with custom position sizing
strategy = RegimeStrategy(
    detector_method='volatility',
    position_sizing=position_sizer
)
```

### Custom Risk Management

```python
from src.ml.backtesting import RegimeStrategy, TrailingStopManager

# Create a trailing stop manager
risk_manager = TrailingStopManager(
    initial_stop_pct=0.05,
    trailing_pct=0.02
)

# Create a strategy with custom risk management
strategy = RegimeStrategy(
    detector_method='hmm',
    risk_manager=risk_manager
)
```

## HTML Reports

The module can generate comprehensive HTML reports for backtest results:

```python
from src.ml.backtesting import save_html_report

# Create HTML report
report_path = save_html_report(
    metrics=results['performance_metrics'],
    trades=results['trades'],
    figures=figures,
    output_path="backtest_report.html"
)
```

## Performance Tips

- For faster backtests, use NumPy arrays instead of lists for price data
- Set `debug=False` in the backtest method to disable verbose logging
- Use the `max_lookback` parameter to limit the amount of historical data processed
- For large datasets, consider using the `sample_period` parameter to downsample the data

## Dependencies

- NumPy
- Pandas
- Matplotlib
- YFinance (for data retrieval in examples) 