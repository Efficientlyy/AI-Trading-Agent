# Sentiment-Based Trading Strategy

## Overview

The Sentiment-Based Trading Strategy is designed to generate trading signals based on sentiment analysis data from various sources. This strategy processes sentiment events, analyzes sentiment values, and generates buy or sell signals based on configured thresholds.

## Features

- **Multi-source Sentiment Analysis**: Combines sentiment data from various sources with configurable weights
- **Configurable Thresholds**: Customizable sentiment thresholds for bullish and bearish signals
- **Contrarian Mode**: Optional mode to reverse signals (buy on negative sentiment, sell on positive sentiment)
- **Confidence Filtering**: Minimum confidence thresholds to filter out low-confidence sentiment signals
- **Position Management**: Configurable stop-loss and take-profit settings

## Configuration

The strategy is configured in the `strategies.yaml` file. Here's an example configuration:

```yaml
SentimentStrategy:
  symbols:
    - BTC/USDT
    - ETH/USDT
  timeframes:
    - 1h
  parameters:
    sentiment_bull_threshold: 0.6
    sentiment_bear_threshold: -0.6
    min_confidence: 0.7
    contrarian_mode: false
    source_weights:
      twitter: 0.3
      news: 0.4
      reddit: 0.3
    position_management:
      stop_loss_pct: 0.05
      take_profit_pct: 0.15
```

### Configuration Parameters

- **sentiment_bull_threshold**: The threshold above which sentiment is considered bullish (0 to 1)
- **sentiment_bear_threshold**: The threshold below which sentiment is considered bearish (-1 to 0)
- **min_confidence**: Minimum confidence level required for a sentiment signal (0 to 1)
- **contrarian_mode**: When true, reverses the signal direction (buy on negative sentiment, sell on positive)
- **source_weights**: Weight factors for different sentiment sources (must sum to 1)
- **position_management**: Stop-loss and take-profit settings for the strategy

## Implementation Details

The strategy works by:

1. **Receiving Sentiment Events**: Processes sentiment data events from various sources
2. **Aggregating Sentiment**: Combines sentiment values from multiple sources using weighted averages
3. **Analyzing Sentiment**: Compares aggregated sentiment to thresholds to determine signal direction
4. **Generating Signals**: Creates buy/sell signals based on the analysis
5. **Managing Positions**: Tracks open positions and generates exit signals based on stop-loss/take-profit settings

## Usage

When deployed, the strategy automatically processes incoming sentiment events and generates trading signals that can be consumed by the execution components of the trading system.

## Testing

The strategy includes comprehensive test coverage to validate:
- Proper handling of sentiment events
- Accurate sentiment analysis and signal generation
- Correct behavior in contrarian mode
- Appropriate management of position exit signals

To run the tests:

```bash
python -m pytest tests/test_sentiment_strategy.py -v
```

## Future Enhancements

Potential improvements to the strategy include:

- **Sentiment Trend Analysis**: Consider trends in sentiment over time rather than just current values
- **Adaptive Thresholds**: Dynamically adjust thresholds based on market conditions
- **Source Weighting Optimization**: ML-based optimization of source weights
- **Cross-correlation with Market Data**: Combine sentiment analysis with price action
