# Impact Factor Analysis Engine

## Overview

The Impact Factor Analysis Engine identifies and measures which market factors have the strongest predictive power for cryptocurrency price movements. It forms the foundation of the system's ability to achieve high win rates by focusing on factors with proven impact.

## Key Responsibilities

- Define and calculate a wide range of market factors
- Measure the predictive power of each factor across different timeframes
- Track factor effectiveness over time and across market conditions
- Identify factor interactions and dependencies
- Provide weighted factor outputs to analysis agents

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Impact Factor Analysis Engine             │
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│  │ Factor      │   │ Factor      │   │ Impact      │    │
│  │ Definition  │──▶│ Calculation │──▶│ Measurement │    │
│  └─────────────┘   └─────────────┘   └──────┬──────┘    │
│         ▲                                    │          │
│         │                                    ▼          │
│  ┌─────────────┐                    ┌─────────────┐     │
│  │ Factor      │◀───────────────────│ Factor      │     │
│  │ Registry    │                    │ Weighting   │     │
│  └─────────────┘                    └─────────────┘     │
│                                            │            │
└────────────────────────────────────────────┼────────────┘
                                             │
                                             ▼
                                     ┌──────────────┐
                                     │ Factor Access│
                                     │ Service      │
                                     └──────────────┘
```

## Subcomponents

### 1. Factor Definition

Creates standardized definitions for market factors:

- Framework for defining different factor types
- Parameter specifications for each factor
- Factor categorization (technical, pattern, sentiment)
- Factor metadata (timeframes, references, etc.)

### 2. Factor Registry

Maintains the catalog of all available factors:

- Central repository of factor definitions
- Factory methods for factor instantiation
- Factor dependency management
- Factor discovery and registration

### 3. Factor Calculation

Computes factor values from market data:

- Efficient calculation algorithms
- Caching mechanisms for performance
- Multi-timeframe support
- Handling of missing or invalid data

### 4. Impact Measurement

Evaluates the predictive power of factors:

- Statistical correlation with future price movements
- Success rate measurement
- Significance testing
- Consistency analysis across market conditions

### 5. Factor Weighting

Assigns dynamic weights based on factor performance:

- Historical performance tracking
- Market regime adjustment
- Adaptive weighting algorithms
- Confidence score generation

### 6. Factor Access Service

Provides interfaces for other components to access factor data:

- Factor value retrieval API
- Factor subscription mechanism
- Impact measurement reporting
- Weight distribution information

## Factor Types

The engine supports several categories of market factors:

### Technical Factors

Derived from price and volume data:

- **Momentum Factors**: RSI, MACD, Momentum
- **Volatility Factors**: Bollinger Bands, ATR, Standard Deviation
- **Trend Factors**: Moving Averages, ADX, Trend Strength
- **Volume Factors**: OBV, Volume Profile, Volume Trends

### Pattern Factors

Based on chart pattern recognition:

- **Reversal Patterns**: Head & Shoulders, Double Top/Bottom
- **Continuation Patterns**: Flags, Pennants, Triangles
- **Candlestick Patterns**: Engulfing, Doji, Hammer
- **Harmonic Patterns**: Gartley, Butterfly, Crab

### Sentiment Factors

Derived from market sentiment data:

- **Social Sentiment**: Social Media Mentions, Sentiment Score
- **Market Sentiment**: Fear & Greed Index, Long/Short Ratio
- **Whale Activity**: Large Transactions, Wallet Movements
- **News Impact**: News Sentiment, Announcement Effects

## Factor Definition Format

Factors are defined using a standardized format:

```json
{
  "id": "rsi_14_oversold",
  "name": "RSI Oversold Condition",
  "category": "technical",
  "subcategory": "momentum",
  "description": "Identifies when 14-period RSI drops below oversold threshold (30)",
  "calculation": {
    "base_factor": "rsi",
    "parameters": {
      "period": 14,
      "threshold": 30
    },
    "condition": "less_than"
  },
  "timeframes": ["1h", "4h", "1d"],
  "signal_type": "reversal",
  "expected_direction": "up"
}
```

## Impact Measurement Metrics

The system uses several metrics to measure factor impact:

### Success Rate

Percentage of times a factor correctly predicts direction:

```
success_rate = correct_predictions / total_predictions
```

### Predictive Power

Statistical measure of correlation with future price movements:

```
predictive_power = correlation_coefficient(factor_value, future_return)
```

### Factor Consistency

Measures how consistently a factor performs across different market conditions:

```
consistency = standard_deviation(success_rate_across_regimes)
```

### Profit Factor

Ratio of profits to losses when trading based on the factor:

```
profit_factor = sum(profitable_trades) / sum(losing_trades)
```

## Factor Weight Calculation

Weights are assigned dynamically based on:

1. Historical success rate
2. Recent performance trend
3. Current market regime suitability
4. Factor consistency
5. Statistical significance of impact

The weighting algorithm balances long-term reliability with recent relevance.

## Configuration Options

The Impact Factor Analysis Engine is configurable through the `config/impact_factor.yaml` file:

```yaml
calculation:
  max_history_periods: 1000
  update_frequency: "1m"
  cache_size: 1000
  thread_pool_size: 4

impact_measurement:
  lookback_periods:
    short_term: 30  # days
    medium_term: 90  # days
    long_term: 365  # days
  significance_threshold: 0.05
  minimum_sample_size: 30

weighting:
  weight_update_frequency: "1h"
  regime_sensitivity: 0.7  # 0-1, higher = more sensitive to regime changes
  recency_bias: 0.3  # 0-1, higher = more weight to recent performance
  
factors:
  enabled_categories:
    - "technical"
    - "pattern"
    - "sentiment"
  default_timeframes:
    - "5m"
    - "15m"
    - "1h"
    - "4h"
    - "1d"
```

## Integration Points

### Input Interfaces
- Data Collection Framework for market data
- Configuration system for factor definitions and parameters

### Output Interfaces
- `get_factor_value(factor_id, symbol, timeframe, time)`: Get historical factor value
- `get_current_factor_value(factor_id, symbol, timeframe)`: Get latest factor value
- `get_factor_impact(factor_id, symbol, timeframe)`: Get impact measurements
- `get_factor_weights(category, symbol, timeframe)`: Get current factor weights
- `get_top_factors(symbol, timeframe, limit)`: Get highest-impact factors
- `subscribe_factor_updates(factor_ids, callback)`: Subscribe to factor updates

## Error Handling

The engine implements comprehensive error handling:

- Calculation errors: Fall back to default values, log error
- Missing data: Use interpolation or skip calculation
- Parameter validation: Enforce valid ranges, use defaults otherwise
- Performance issues: Implement timeouts, prioritize critical factors

## Metrics and Monitoring

The engine provides the following metrics for monitoring:

- Factor calculation performance (time per factor)
- Cache hit/miss rates
- Factor success rates over time
- Weight distribution changes
- Highest impact factors by timeframe
- Factor stability metrics

## Implementation Guidelines

- Use vectorized operations with NumPy/pandas for performance
- Implement proper caching strategies for frequent calculations
- Use statistical libraries for correlation and significance testing
- Create abstraction layers for different factor types
- Maintain comprehensive logs of impact measurements
- Implement thread safety for concurrent factor calculations
