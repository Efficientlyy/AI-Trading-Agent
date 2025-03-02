# Decision Engine

## Overview

The Decision Engine is the central decision-making component of the trading system. It aggregates predictions from all analysis agents, applies risk management rules, and generates high-confidence trading signals. It's responsible for achieving the target 75%+ win rate through selective trade execution.

## Key Responsibilities

- Aggregate and weight predictions from multiple analysis agents
- Apply confidence thresholds and quality filters
- Implement comprehensive risk management
- Generate precise trading signals with complete rationale
- Track open positions and manage trade lifecycle
- Maintain performance metrics for continuous improvement

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Decision Engine                      │
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│  │ Prediction  │   │ Decision    │   │ Risk        │    │
│  │ Aggregator  │──▶│ Rules       │──▶│ Management  │    │
│  └─────────────┘   └─────────────┘   └──────┬──────┘    │
│         ▲                                    │          │
│         │                                    ▼          │
│  ┌─────────────┐                    ┌─────────────┐     │
│  │ Performance │◀───────────────────│ Signal      │     │
│  │ Tracker     │                    │ Generator   │     │
│  └─────────────┘                    └─────────────┘     │
│                                            │            │
└────────────────────────────────────────────┼────────────┘
                                             │
                                             ▼
                                     ┌──────────────┐
                                     │ Position     │
                                     │ Manager      │
                                     └──────────────┘
```

## Subcomponents

### 1. Prediction Aggregator

Collects and combines predictions from all analysis agents:

- Gathers predictions from all agents
- Normalizes prediction formats
- Weights predictions based on agent performance
- Identifies consensus and disagreements
- Calculates aggregate confidence scores

### 2. Decision Rules

Applies rules to determine which predictions should generate trades:

- Enforces minimum confidence thresholds (85%+)
- Requires multi-agent confirmation
- Applies market condition filters
- Implements time-of-day restrictions
- Creates trade quality scoring

### 3. Risk Management

Implements comprehensive risk control:

- Position sizing using modified Kelly Criterion
- Maximum exposure limits per asset
- Portfolio correlation checks
- Drawdown protection rules
- Reward-risk ratio enforcement (minimum 2:1)

### 4. Signal Generator

Creates detailed trading signals:

- Precise entry price and method
- Stop-loss placement with rationale
- Take-profit targets with timeframes
- Detailed trade rationale
- Complete attribution to triggering factors

### 5. Position Manager

Tracks and manages open positions:

- Maintains current position inventory
- Monitors stop-loss and take-profit levels
- Generates adjustment signals when needed
- Tracks position performance
- Manages position lifecycle

### 6. Performance Tracker

Measures and analyzes trading performance:

- Tracks prediction accuracy
- Calculates win rate and profit metrics
- Analyzes performance by strategy type
- Identifies strengths and weaknesses
- Provides feedback for system improvement

## Trade Signal Format

The Decision Engine produces standardized trade signals:

```json
{
  "signal_id": "trade_signal_12345",
  "timestamp": 1645541585432,
  "symbol": "BTCUSDT",
  "direction": "long",
  "signal_type": "entry",
  "timeframe": "4h",
  "confidence": 92.5,
  "entry": {
    "price": 38245.50,
    "method": "limit",
    "valid_until": 1645545185432
  },
  "position_size": {
    "percentage": 2.5,
    "units": 0.05,
    "usd_value": 1912.28
  },
  "risk_management": {
    "stop_loss": {
      "price": 37950.00,
      "method": "market",
      "reason": "Below key support level"
    },
    "risk_percentage": 0.5,
    "reward_risk_ratio": 2.3
  },
  "targets": [
    {
      "price": 38725.00,
      "percentage": 50,
      "expected_time": 1645555185432
    },
    {
      "price": 39125.00,
      "percentage": 50,
      "expected_time": 1645570000000
    }
  ],
  "contributing_predictions": [
    {
      "agent_id": "technical_analysis_agent",
      "prediction_id": "pred_12345",
      "confidence": 87.5,
      "weight": 0.4
    },
    {
      "agent_id": "pattern_recognition_agent",
      "prediction_id": "pred_23456",
      "confidence": 92.0,
      "weight": 0.6
    }
  ],
  "reasoning": "Strong bullish reversal with multiple confirmations. Price bounced from key support level with increased volume. RSI shows bullish divergence and inverted head and shoulders pattern confirmed with neckline breakout.",
  "market_context": {
    "regime": "ranging",
    "volatility": "medium",
    "overall_trend": "neutral"
  }
}
```

## Risk Management Approach

The Decision Engine implements a sophisticated risk management framework:

### Position Sizing

Uses a modified Kelly Criterion:

```
position_size = (win_rate - ((1 - win_rate) / reward_risk_ratio)) * kelly_fraction
```

Where:
- `win_rate` is the historical win rate for similar setups
- `reward_risk_ratio` is the expected reward-to-risk ratio
- `kelly_fraction` is a safety factor (typically 0.3-0.5)

### Exposure Limits

Multiple limits to control risk:

- Maximum per-trade risk: 1% of account
- Maximum per-symbol exposure: 5% of account
- Maximum correlated asset exposure: 15% of account
- Maximum total exposure: 25% of account in high volatility, 50% in low volatility

### Stop-Loss Approach

Strategic stop-loss placement based on:

- Technical invalidation points
- Volatility-adjusted distance (using ATR)
- Maximum acceptable loss percentage
- Key support/resistance levels

## Performance Metrics

The Decision Engine tracks several key performance metrics:

- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profits divided by gross losses
- Average Reward/Risk: Average profit vs. average loss ratio
- Maximum Drawdown: Largest peak-to-trough decline
- Sharpe Ratio: Risk-adjusted return measurement
- Prediction Accuracy: Percentage of correct directional predictions

## Configuration Options

The Decision Engine is configurable through the `config/decision_engine.yaml` file:

```yaml
confidence:
  min_threshold: 85.0
  multi_agent_required: true
  min_agents_agreement: 2

risk_management:
  position_sizing:
    method: "kelly"
    kelly_fraction: 0.4
    min_position_size: 0.001
    max_position_size: 0.05
  
  exposure_limits:
    max_per_trade_risk: 1.0  # percentage of account
    max_per_symbol: 5.0  # percentage of account
    max_correlated_assets: 15.0  # percentage of account
    max_total_exposure: 50.0  # percentage of account
  
  reward_risk:
    min_ratio: 2.0
    preferred_ratio: 3.0
    
  stop_loss:
    method: "technical"  # technical, volatility, fixed
    volatility_multiplier: 1.5  # for ATR-based stops
    max_loss_percentage: 1.0
    
trade_execution:
  default_order_type: "limit"
  max_entry_validity: 1800  # seconds
  partial_fills_allowed: true
  
agent_weights:
  technical_analysis_agent: 0.4
  pattern_recognition_agent: 0.4
  sentiment_analysis_agent: 0.2
  
market_filters:
  min_volume: 1000000  # 24h USD volume
  max_spread: 0.1  # percentage
  restricted_hours: []  # e.g. ["00:00-01:00"]
```

## Integration Points

### Input Interfaces
- Agent predictions from all analysis agents
- Position updates from trading environment
- Market data from Data Collection Framework
- Configuration from configuration system

### Output Interfaces
- `generate_trade_signals()`: Generate new trade signals
- `get_active_signals()`: Get currently active trade signals
- `get_active_positions()`: Get currently open positions
- `get_position_updates(position_id)`: Get updates for specific position
- `get_performance_metrics()`: Get trading performance statistics

## Error Handling

The engine implements comprehensive error handling:

- Prediction aggregation errors: Fall back to conservative weights
- Risk calculation errors: Default to more conservative position sizing
- Signal generation errors: Skip trade opportunity
- Position tracking errors: Reconcile with exchange data

## Metrics and Monitoring

The engine provides the following metrics for monitoring:

- Trade signal generation rate
- Agent contribution to successful trades
- Win rate by strategy and market condition
- Risk utilization percentage
- Performance metrics over time

## Implementation Guidelines

- Create clear separation between prediction aggregation and decision rules
- Implement thread-safe position tracking
- Use proper abstractions for different risk management strategies
- Create comprehensive logging of all decisions and rationale
- Maintain clean transaction records for all trade activities
- Implement proper error handling for all operations
- Design for testability with dependency injection
