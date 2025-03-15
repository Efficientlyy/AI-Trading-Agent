# AI Trading Agent - Sentiment Analysis System

[![codecov](https://codecov.io/gh/Efficientlyy/AI-Trading-Agent/branch/main/graph/badge.svg)](https://codecov.io/gh/Efficientlyy/AI-Trading-Agent)

This repository contains a modular, high-performance sentiment analysis system for cryptocurrency trading. It's designed to analyze sentiment from multiple sources and provide actionable trading insights.

## Architecture

The sentiment analysis system is structured with a clear separation of concerns:

- **Base Components**: Common functionality shared across all sentiment analysis agents
- **Specialized Agents**: Individual components focused on specific sentiment sources
- **Aggregation Engine**: Combines signals from various sources for consolidated analysis
- **Manager**: Coordinates all sentiment components in the system

### Component Hierarchy

```
SentimentAnalysisManager
├── SocialMediaSentimentAgent (Twitter, Reddit, etc.)
├── NewsSentimentAgent (CryptoNews, CoinDesk, etc.)
├── MarketSentimentAgent (Fear & Greed Index, Long/Short Ratio)
├── OnchainSentimentAgent (Blockchain metrics, wallet activity)
└── SentimentAggregator (Combines and weights all signals)
```

## Features

- **Multi-source Analysis**: Processes sentiment data from social media, news, market indicators, and on-chain metrics
- **Configurable Weights**: Adjustable weighting of different sentiment sources
- **Contrarian Detection**: Identifies extreme sentiment that may indicate market reversals
- **Market Correlation**: Analyzes sentiment in relation to price action
- **Confidence Scoring**: Provides confidence levels for all sentiment signals
- **Aggregation Engine**: Combines signals with intelligent weighting based on reliability and recency

## Usage

Basic usage example to run the sentiment analysis system:

```python
import asyncio
from src.analysis_agents.sentiment_analysis_manager import SentimentAnalysisManager

async def run_system():
    # Create and start the sentiment analysis system
    manager = SentimentAnalysisManager()
    await manager.initialize()
    await manager.start()
    
    try:
        # Let it run for some time
        await asyncio.sleep(60)
        
        # Get all sentiment agents
        agents = manager.get_all_agents()
        
        # Get the aggregator specifically
        aggregator = manager.get_agent("aggregator")
        if aggregator:
            # Get the latest sentiment for BTC/USDT
            result = await aggregator.aggregate_sentiment("BTC/USDT")
            print(f"BTC/USDT Sentiment: {result['direction']} (confidence: {result['confidence']:.2f})")
            
    finally:
        # Properly shut down
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(run_system())
```

## Configuration

All sentiment analysis components are configurable via the `config/sentiment_analysis.yaml` file. Key configuration options include:

- Enabled/disabled status for each component
- Source weights for aggregation
- Update intervals for each data source
- Symbols to monitor
- Confidence thresholds
- Sentiment shift sensitivity

## Requirements

- Python 3.8+
- See requirements.txt for dependencies

## Development

This system follows a modular design with clear separation of concerns:

- Each sentiment source is isolated in its own module
- All files adhere to the project's 300-500 line limit
- Components use a consistent event-based communication system
- Error handling is robust with proper async patterns
- All modules include comprehensive documentation

## Test Coverage

The project enforces strict test coverage requirements to maintain code quality and ensure reliability across all components. Our coverage configuration follows the Component Isolation principle, with specific thresholds for each module.

### Coverage Thresholds

| Component | Threshold | Description |
|-----------|-----------|-------------|
| Market Regime Detection | ≥85% | Critical algorithms for market state identification |
| ML Models | ≥80% | Machine learning model implementations |
| Performance Evaluation | ≥75% | Metrics and evaluation tools |
| Rust Core | ≥80% | Performance-critical components written in Rust |
| Dashboard | ≥80% | React/TypeScript visualization components |

### CI Integration

Coverage is enforced through our CI pipeline:

- Each PR triggers component-specific coverage checks
- Coverage reports are generated for modified components
- PRs cannot be merged if coverage drops below thresholds
- Critical files receive extra scrutiny

### PR Comments

Codecov automatically comments on PRs with:

- Overall coverage changes
- Per-component coverage details
- Warnings for components that drop below thresholds
- Highlighting of changes to critical files

This system ensures that:
1. New code maintains high test quality
2. Contributors receive immediate feedback on coverage
3. Code quality remains consistent across all components
4. Critical components maintain the highest standards
