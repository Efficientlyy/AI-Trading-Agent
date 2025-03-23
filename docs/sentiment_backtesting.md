# Sentiment Backtesting Framework

This document describes the sentiment backtesting framework implemented in the AI Trading Agent project. The framework allows for comprehensive backtesting of sentiment-based trading strategies using historical price and sentiment data from various sources.

## Overview

The sentiment backtesting framework consists of two main components:

1. **Sentiment Data Collector**: Collects, stores, and retrieves historical sentiment data from various sources including Fear & Greed Index, news sentiment, social media sentiment, and on-chain metrics.

2. **Sentiment Backtester**: Integrates sentiment data with price data to evaluate trading strategies based on sentiment signals.

## Sentiment Data Collector

The `SentimentCollector` class provides functionality to:

- Collect sentiment data from multiple sources
- Store sentiment data in a structured format (JSON)
- Load historical sentiment data for backtesting
- Provide timestamp-aligned sentiment data for strategy evaluation

### Key Features

- **Multi-source support**: Collects data from Fear & Greed Index, news, social media, and on-chain metrics
- **Standardized format**: Normalizes diverse sentiment data to a 0-1 scale
- **Scheduled collection**: Can collect data at regular intervals
- **Efficient storage**: Organizes data by source, symbol, and date range
- **Data aggregation**: Combines multiple sentiment sources with customizable resampling

### Usage

```python
collector = SentimentCollector()

# Collect historical Fear & Greed data
fear_greed_data = await collector.collect_historical_data(
    source="fear_greed",
    symbol="BTC",
    start_date=datetime.datetime(2023, 1, 1),
    end_date=datetime.datetime(2023, 3, 31),
    save=True
)

# Load sentiment data for backtesting
sentiment_df = collector.load_historical_data(
    source="fear_greed",
    symbol="BTC",
    start_date=datetime.datetime(2023, 1, 1),
    end_date=datetime.datetime(2023, 3, 31)
)

# Combine multiple sentiment sources
combined_df = collector.combine_sentiment_sources(
    symbol="BTC",
    start_date=datetime.datetime(2023, 1, 1),
    end_date=datetime.datetime(2023, 3, 31),
    sources=["fear_greed", "news", "social_media", "onchain"],
    resample_freq="1H"
)
```

## Sentiment Backtester

The `SentimentBacktester` class provides a comprehensive framework for evaluating sentiment-based trading strategies. It extends the modular backtester with sentiment-specific functionality.

### Key Features

- **Sentiment event integration**: Processes sentiment events with price data
- **Strategy evaluation**: Tests strategies in historical market conditions
- **Performance metrics**: Calculates standard and sentiment-specific metrics
- **Visualization**: Generates equity curves and trade visualizations
- **Parameter optimization**: Tunes strategy parameters using grid search

### Available Strategies

1. **SentimentStrategy**: Basic strategy that generates signals based on sentiment thresholds
   - Supports multiple sentiment sources with weighted averaging
   - Configurable entry/exit thresholds
   - Optional contrarian mode
   - Stop-loss and take-profit mechanisms

2. **AdvancedSentimentStrategy**: Extends basic strategy with advanced features
   - Sentiment trend analysis using linear regression
   - Market regime detection based on volatility
   - Adaptive parameters for different market conditions
   - Technical confirmation using RSI
   - More sophisticated signal generation logic

### Usage

```python
# Define backtest configuration
config = {
    'symbol': 'BTC-USD',
    'start_date': '2023-01-01',
    'end_date': '2023-03-31',
    'sources': ['fear_greed', 'news'],
    'price_data_path': 'data/historical/BTC-USD_1h.csv',
    'strategy': 'AdvancedSentimentStrategy',
    'strategy_config': {
        'sentiment_threshold_buy': 0.7,
        'sentiment_threshold_sell': 0.3,
        'trend_window': 14,
        'technical_confirmation': True
    },
    'initial_capital': 10000,
    'commission_rate': 0.001
}

# Initialize backtester
backtester = SentimentBacktester(config)

# Run backtest
results = backtester.run_backtest()

# Generate report
report = backtester.generate_report(results, 'reports/backtest_report.txt')

# Visualize results
backtester.visualize_results(results, 'reports/backtest_plot.png')

# Run parameter optimization
param_grid = {
    'sentiment_threshold_buy': [0.6, 0.7, 0.8],
    'sentiment_threshold_sell': [0.2, 0.3, 0.4],
    'trend_window': [7, 14, 21]
}
optimization_results = backtester.run_parameter_optimization(param_grid)
```

## Sentiment Events

The `SentimentEvent` class represents a sentiment data point with the following attributes:

- **timestamp**: When the sentiment was recorded
- **source**: Data source (e.g., "fear_greed", "news")
- **symbol**: Trading symbol (e.g., "BTC", "ETH")
- **sentiment_value**: Normalized sentiment value (0-1)
- **confidence**: Confidence in the sentiment value (0-1)
- **metadata**: Additional source-specific information

## Performance Metrics

The backtester calculates standard trading metrics:

- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor

Additionally, it calculates sentiment-specific metrics:

- Correlation between signals and sentiment values
- Average profit by sentiment level
- Effectiveness of contrarian vs. trend-following approaches

## Integration with Modular Backtester

The sentiment backtester integrates with the existing modular backtesting framework by:

1. Implementing compatible strategy classes
2. Reusing position and trade management logic
3. Extending the performance metrics calculation
4. Providing a consistent interface for strategy evaluation

## Example Workflow

1. **Collect sentiment data**:
   ```python
   collector = SentimentCollector()
   await collector.collect_historical_data("fear_greed", "BTC", start_date, end_date)
   ```

2. **Define strategy**:
   ```python
   strategy_config = {
       'sentiment_threshold_buy': 0.7,
       'sentiment_threshold_sell': 0.3,
       'contrarian': False,
       'source_weights': {'fear_greed': 0.7, 'news': 0.3}
   }
   ```

3. **Run backtest**:
   ```python
   backtester = SentimentBacktester(config)
   results = backtester.run_backtest()
   ```

4. **Analyze results**:
   ```python
   report = backtester.generate_report(results)
   backtester.visualize_results(results)
   ```

5. **Optimize parameters**:
   ```python
   param_grid = {'sentiment_threshold_buy': [0.6, 0.7, 0.8]}
   optimization_results = backtester.run_parameter_optimization(param_grid)
   ```

## Future Improvements

1. **More sentiment sources**: Integrate additional sentiment data providers
2. **Machine learning models**: Develop ML models for sentiment prediction
3. **Advanced optimization**: Implement genetic algorithms for parameter tuning
4. **Real-time backtesting**: Test strategies with streaming sentiment data
5. **Ensemble strategies**: Combine multiple sentiment strategies for robust performance
6. **Portfolio backtesting**: Test sentiment strategies on multiple assets simultaneously
7. **Regime-specific strategies**: Develop tailored strategies for different market regimes