# Fear & Greed Index Integration

## Overview

The Fear & Greed index is a market sentiment indicator that helps measure cryptocurrency market sentiment and emotions. The index represents market sentiment on a scale from 0 (Extreme Fear) to 100 (Extreme Greed). This document outlines the integration of the Fear & Greed index into the AI Trading Agent's sentiment analysis system.

## Implementation

The Fear & Greed index integration is implemented in the `FearGreedClient` class within the `src/analysis_agents/sentiment/market_sentiment.py` file. The implementation includes:

1. **Real API Integration**: Connecting to the Alternative.me Fear & Greed index API for cryptocurrency market sentiment data
2. **Caching Mechanism**: Minimizing API calls by caching responses with configurable expiry periods
3. **Error Handling**: Graceful degradation with fallback values when the API is unavailable
4. **Historical Data Support**: Fetching historical Fear & Greed index data for backtesting
5. **Session Management**: Proper HTTP session handling to efficiently manage connections

The client provides two primary methods:
- `get_current_index()`: Returns the current Fear & Greed value and classification
- `get_historical_index(days)`: Returns a time series of historical values

## Market Sentiment Analysis

The Fear & Greed index is incorporated into the market sentiment analysis process through the `MarketSentimentAgent`. The agent:

1. Fetches the current Fear & Greed index
2. Normalizes the index value to a 0-1 scale
3. Combines it with other market indicators (like long/short ratio)
4. Determines confidence based on agreement between indicators
5. Publishes sentiment events based on significant shifts or extreme values
6. Flags extreme values (≤20 or ≥80) as potential contrarian indicators

## Configuration

The Fear & Greed index integration is configured in the `sentiment_analysis.yaml` file under the `sentiment.apis.fear_greed` section:

```yaml
sentiment:
  apis:
    fear_greed:
      base_url: "https://api.alternative.me/fng/"
      cache_expiry: 3600  # 1 hour in seconds
```

## Interpreting Values

The Fear & Greed index values are interpreted as follows:

| Value Range | Classification   | Interpretation                                                  |
|-------------|------------------|----------------------------------------------------------------|
| 0-25        | Extreme Fear     | Investors are very fearful, potentially a contrarian buy signal |
| 26-40       | Fear             | Investors are worried, market sentiment is negative             |
| 41-60       | Neutral          | Market sentiment is balanced                                    |
| 61-80       | Greed            | Investors are optimistic, trending bullish                      |
| 81-100      | Extreme Greed    | Market is overheated, potentially a contrarian sell signal      |

## Integration with Market Data

When combined with market data, the Fear & Greed index provides additional context for trading decisions:

1. **Extreme Fear + High Volatility**: Often considered a contrarian buy signal
2. **Extreme Greed + Extended Rally**: Often considered a contrarian sell signal
3. **Rapid Changes in Index**: May indicate a sentiment shift and potential market move
4. **Divergence with Price Action**: When sentiment doesn't match price trends, suggests potential reversal

## Example Usage

See the `examples/fear_greed_index_demo.py` file for a comprehensive demonstration of the Fear & Greed index integration, including fetching current and historical data, visualization, and sentiment analysis.

```python
import asyncio
from src.analysis_agents.sentiment.market_sentiment import FearGreedClient

async def get_fear_greed_data():
    # Initialize the client
    client = FearGreedClient()
    
    # Get current Fear & Greed index
    current = await client.get_current_index()
    print(f"Current Fear & Greed Index: {current['value']} ({current['classification']})")
    
    # Get historical data (last 30 days)
    historical = await client.get_historical_index(days=30)
    for entry in historical[:5]:  # Show first 5 entries
        print(f"{entry['timestamp']}: {entry['value']} ({entry['classification']})")
    
    # Close the client
    await client.close()

# Run the example
asyncio.run(get_fear_greed_data())
```

## Market Sentiment Implementation

The Fear & Greed index is integrated into the `MarketSentimentAgent` as follows:

```python
# Fetch Fear & Greed Index
fear_greed_data = await self.fear_greed_client.get_current_index()
fear_greed = fear_greed_data.get("value", 50)

# Convert to 0-1 scale
fg_sentiment = fear_greed / 100.0

# Combine with other indicators
sentiment_value = (fg_sentiment + other_indicators) / num_indicators

# Check for extreme values
is_extreme = fear_greed <= 20 or fear_greed >= 80

# Publish event if significant shift or extreme values
if sentiment_shift > self.sentiment_shift_threshold or is_extreme:
    # ...publish sentiment event
```

## Backtesting Support

For backtesting, the Fear & Greed index can be used with historical data:

1. Fetch historical Fear & Greed data for the backtest period
2. Align with price data and other indicators
3. Generate signals based on sentiment values and extremes
4. Evaluate performance with different thresholds and signal types

## Next Steps

1. Add more market sentiment indicators to complement the Fear & Greed index
2. Implement adaptive weighting based on historical accuracy
3. Create specialized strategies for extreme sentiment conditions
4. Integrate with other sentiment sources for confirmation
5. Add dashboard visualization components for Fear & Greed index trends