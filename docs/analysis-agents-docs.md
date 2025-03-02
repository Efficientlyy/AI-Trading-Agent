# Analysis Agents

## Overview

Analysis Agents are specialized components that analyze market data from different perspectives to generate high-confidence predictions. The system uses three types of agents: Technical Analysis, Pattern Recognition, and Sentiment Analysis. Each agent produces standardized prediction outputs that are aggregated by the Decision Engine.

## Common Agent Architecture

All agents share a common architecture pattern:

```
┌─────────────────────────────────────────────────────────┐
│                  Analysis Agent (Generic)                │
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│  │ Data        │   │ Analysis    │   │ Signal      │    │
│  │ Collection  │──▶│ Modules     │──▶│ Generation  │    │
│  └─────────────┘   └─────────────┘   └──────┬──────┘    │
│         ▲                                    │          │
│         │                                    ▼          │
│  ┌─────────────┐                    ┌─────────────┐     │
│  │ Configuration│                   │ Performance  │     │
│  │ Manager     │                    │ Tracker     │     │
│  └─────────────┘                    └─────────────┘     │
│                                            │            │
└────────────────────────────────────────────┼────────────┘
                                             │
                                             ▼
                                     ┌──────────────┐
                                     │ Prediction   │
                                     │ Output       │
                                     └──────────────┘
```

## Agent Types

### 1. Technical Analysis Agent

Analyzes price and volume data using technical indicators to identify potential market movements.

#### Key Features

- Implements a comprehensive set of technical indicators
- Adapts indicator parameters based on market volatility
- Identifies support/resistance levels with confidence scores
- Performs multi-timeframe analysis for confirmation
- Calculates trend strength and momentum metrics
- Combines indicators for higher-confidence signals

#### Analysis Modules

- **Trend Analysis**: Moving averages, trend identification, trend strength
- **Momentum Analysis**: RSI, MACD, Stochastic, momentum oscillators
- **Volatility Analysis**: Bollinger Bands, ATR, volatility breakouts
- **Support/Resistance**: Key level identification, S/R strength calculation
- **Volume Analysis**: Volume profile, OBV, volume-price relationships

#### Signal Generation Logic

- Identify indicator convergence/divergence
- Require multi-indicator confirmation
- Apply strict confidence thresholds
- Weight signals based on historical indicator performance
- Consider market regime appropriateness

### 2. Pattern Recognition Agent

Identifies chart patterns and calculates the probability of pattern completion and price targets.

#### Key Features

- Detects common chart patterns with high reliability
- Calculates pattern quality and completion probability
- Validates patterns with volume confirmation
- Generates price targets based on pattern characteristics
- Identifies pattern confluence across timeframes
- Tracks historical reliability of pattern types

#### Analysis Modules

- **Reversal Pattern Detection**: Head & shoulders, double tops/bottoms
- **Continuation Pattern Detection**: Flags, pennants, triangles
- **Harmonic Pattern Recognition**: Gartley, butterfly, bat patterns
- **Candlestick Pattern Analysis**: Japanese candlestick patterns
- **Pattern Quality Assessment**: Measurements of pattern quality and reliability

#### Signal Generation Logic

- Calculate pattern completion probability
- Generate price targets with confidence intervals
- Require minimum pattern quality score
- Apply volume confirmation rules
- Consider pattern historical reliability

### 3. Sentiment Analysis Agent

Analyzes market sentiment from available sources to identify potential price movements.

#### Key Features

- Connects to free/low-cost sentiment sources
- Analyzes social media sentiment within budget constraints
- Identifies extreme sentiment conditions
- Detects sentiment divergences from price
- Tracks correlation between sentiment shifts and price movements
- Applies contrarian analysis at sentiment extremes

#### Analysis Modules

- **Social Sentiment Analysis**: Social media monitoring within rate limits
- **Market Sentiment Indicators**: Fear & Greed index, long/short ratio
- **News Sentiment Analysis**: News aggregation and sentiment scoring
- **On-chain Metrics**: Basic blockchain metrics where available
- **Sentiment Divergence Detection**: Price-sentiment relationship analysis

#### Signal Generation Logic

- Identify extreme sentiment conditions
- Generate contrarian signals at sentiment extremes
- Detect sentiment shifts with price implications
- Apply sentiment-specific confidence scoring
- Consider historical sentiment-price correlations

## Standardized Prediction Format

All agents produce predictions in a standardized format:

```json
{
  "agent_id": "technical_analysis_agent",
  "symbol": "BTCUSDT",
  "prediction_id": "pred_12345",
  "timestamp": 1645541585432,
  "direction": "up",
  "timeframe": "4h",
  "confidence": 87.5,
  "expected_magnitude": 3.2,
  "expected_duration": 28800,
  "entry_price": 38245.50,
  "stop_loss": 37950.00,
  "take_profit": 39125.00,
  "signals": [
    {
      "signal_id": "rsi_oversold_bounce",
      "confidence": 85.0,
      "description": "RSI(14) oversold condition with bullish divergence"
    },
    {
      "signal_id": "support_bounce",
      "confidence": 90.0,
      "description": "Strong bounce from key support level"
    }
  ],
  "reasoning": "Price bounced from key support level with RSI showing bullish divergence. Multiple timeframe confirmation with increased volume on bounce.",
  "market_context": {
    "trend": "sideways",
    "volatility": "medium",
    "volume": "increasing"
  }
}
```

## Performance Tracking

All agents track their prediction performance:

- Overall accuracy by timeframe and market condition
- Signal-specific accuracy metrics
- Confidence calibration assessment
- False positive/negative analysis
- Contribution to successful trades

Performance metrics are used to:
- Adjust signal weights
- Refine confidence calculations
- Improve agent parameters
- Identify strongest performing signals

## Configuration

Each agent has its own configuration file in the `config/` directory:

### Technical Analysis Agent

```yaml
# config/technical_analysis_agent.yaml
enabled: true
prediction:
  min_confidence: 85.0
  require_multi_indicator: true
  multi_timeframe_confirmation: true

indicators:
  moving_averages:
    enabled: true
    types: ["SMA", "EMA", "VWMA"]
    periods: [10, 20, 50, 100, 200]
  
  oscillators:
    enabled: true
    types: ["RSI", "MACD", "Stochastic"]
    rsi_periods: [14]
    rsi_overbought: 70
    rsi_oversold: 30
    
  volatility:
    enabled: true
    types: ["Bollinger", "ATR"]
    bollinger_periods: 20
    bollinger_deviations: 2.0
    
  support_resistance:
    enabled: true
    lookback_periods: 100
    strength_threshold: 3
    
timeframes:
  - "15m"
  - "1h"
  - "4h"
  - "1d"
```

### Pattern Recognition Agent

```yaml
# config/pattern_recognition_agent.yaml
enabled: true
prediction:
  min_confidence: 85.0
  require_volume_confirmation: true
  min_pattern_quality: 75.0

patterns:
  reversal:
    enabled: true
    types: ["HeadAndShoulders", "DoubleTop", "DoubleBottom"]
    min_formation_bars: 15
  
  continuation:
    enabled: true
    types: ["Flag", "Pennant", "Triangle"]
    min_formation_bars: 7
    
  harmonic:
    enabled: true
    types: ["Gartley", "Butterfly", "Bat"]
    precision_tolerance: 0.05
    
  candlestick:
    enabled: true
    types: ["Engulfing", "Hammer", "ShootingStar"]
    
timeframes:
  - "1h"
  - "4h"
  - "1d"
```

### Sentiment Analysis Agent

```yaml
# config/sentiment_analysis_agent.yaml
enabled: true
prediction:
  min_confidence: 85.0
  contrarian_threshold: 20.0
  sentiment_shift_threshold: 15.0

sources:
  social_media:
    enabled: true
    platforms: ["Twitter", "Reddit"]
    api_key: "${SOCIAL_API_KEY}"
    request_limit: 100
    
  market_sentiment:
    enabled: true
    indicators: ["FearGreedIndex", "LongShortRatio"]
    
  news:
    enabled: true
    sources: ["CryptoNews", "GeneralFinance"]
    
  onchain:
    enabled: true
    metrics: ["LargeTransactions", "ActiveAddresses"]
    
timeframes:
  - "4h"
  - "1d"
```

## Integration Points

### Input Interfaces
- Data Collection Framework for market data
- Impact Factor Analysis Engine for factor weights and values
- Configuration system for agent parameters

### Output Interfaces
- `generate_predictions(symbol, timeframe)`: Generate new predictions
- `get_active_predictions(symbol)`: Get currently active predictions
- `get_prediction_history(symbol, timeframe, start_time, end_time)`: Get historical predictions
- `get_performance_metrics()`: Get agent performance statistics

## Implementation Guidelines

- Create modular, maintainable code with clear separation of concerns
- Implement proper error handling for all calculations
- Use efficient algorithms for pattern detection
- Create comprehensive logging of all predictions and reasoning
- Implement parallel processing where appropriate
- Define clear interfaces between modules
- Maintain clean separation between analysis and signal generation
- Use proper abstractions for different indicator and pattern types
