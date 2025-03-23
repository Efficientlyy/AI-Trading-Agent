# Sentiment Analysis System for AI Crypto Trading Agent

This document provides an overview of the sentiment analysis system integrated into the AI Crypto Trading Agent platform.

## Overview

The sentiment analysis system monitors and analyzes sentiment data from various sources to generate trading signals. By tracking social media, news, market indicators, and on-chain metrics, the system provides a comprehensive view of market sentiment.

## Architecture

The sentiment analysis system follows a modular design with the following components:

### NLP Service

The Natural Language Processing (NLP) service provides sentiment analysis functionality for text data. It:
- Uses transformer models for advanced sentiment analysis
- Falls back to lexicon-based analysis when needed
- Provides text processing utilities for sentiment scoring

### Sentiment Agents

The system includes specialized sentiment agents for different data sources:

#### Social Media Sentiment Agent
- Monitors platforms like Twitter and Reddit
- Analyzes crypto-related posts and discussions
- Detects extreme sentiment shifts

#### News Sentiment Agent
- Monitors crypto news sites and financial news
- Analyzes article text for sentiment signals
- Extracts relevant keywords

#### Market Sentiment Agent
- Monitors market indicators like Fear & Greed Index
- Tracks exchange-provided metrics like long/short ratio
- Identifies market extremes and trend shifts

#### Onchain Sentiment Agent
- Monitors blockchain metrics
- Tracks large transactions and wallet activity
- Analyzes exchange reserves and network health

### Sentiment Aggregator

The Sentiment Aggregator combines signals from various sources to provide a unified view:
- Applies configurable weights to different sources
- Measures agreement between different sources
- Generates high-confidence aggregated signals

### Sentiment Analysis Manager

The Sentiment Analysis Manager coordinates all sentiment components:
- Creates and initializes all sentiment agents
- Manages component lifecycle
- Provides a unified interface for the system

## Configuration

The sentiment analysis system is configured in `config/sentiment_analysis.yaml`. Key configuration options include:

```yaml
# Main sentiment analysis settings
sentiment:
  enabled: true
  
  # NLP settings
  nlp:
    sentiment_model: "distilbert-base-uncased-finetuned-sst-2-english"
    batch_size: 16
  
  # API keys 
  apis:
    twitter:
      api_key: "${TWITTER_API_KEY}"
      api_secret: "${TWITTER_API_SECRET}"
      access_token: "${TWITTER_ACCESS_TOKEN}"
      access_secret: "${TWITTER_ACCESS_SECRET}"
    
    reddit:
      client_id: "${REDDIT_CLIENT_ID}"
      client_secret: "${REDDIT_CLIENT_SECRET}"
      user_agent: "AI-Trading-Agent/1.0"
    
    # Other API keys...
  
  # Agent configurations
  social_media:
    enabled: true
    platforms:
      - Twitter
      - Reddit
    update_interval: 300  # 5 minutes
    min_confidence: 0.7
    # Other settings...
  
  # Other agent configurations...
```

### Environment Variables

The system uses environment variables for API keys. Create a `.env` file in the root directory with the following variables:

```
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret

REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

NEWS_API_KEY=your_news_api_key
CRYPTO_NEWS_API_KEY=your_crypto_news_api_key
EXCHANGE_DATA_API_KEY=your_exchange_data_api_key
BLOCKCHAIN_API_KEY=your_blockchain_api_key
```

## Usage

### Basic Usage

```python
import asyncio
from src.analysis_agents.sentiment_analysis_manager import SentimentAnalysisManager
from src.common.events import event_bus
from src.models.events import SentimentEvent

# Event handler for sentiment events
async def sentiment_event_handler(event: SentimentEvent):
    print(f"Received sentiment event for {event.symbol}:")
    print(f"  Source: {event.source}")
    print(f"  Direction: {event.sentiment_direction}")
    print(f"  Value: {event.sentiment_value:.2f}")
    print(f"  Confidence: {event.confidence:.2f}")

async def main():
    # Subscribe to sentiment events
    event_bus.subscribe("sentiment_event", sentiment_event_handler)
    
    # Create and initialize sentiment analysis manager
    manager = SentimentAnalysisManager()
    await manager.initialize()
    await manager.start()
    
    try:
        # Run your application
        await asyncio.sleep(60)  # Run for 60 seconds
    finally:
        # Stop the manager
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Creating a Sentiment-Based Strategy

```python
from src.strategy.base_strategy import Strategy
from src.models.events import SentimentEvent
from src.models.signals import Signal, SignalType

class SentimentStrategy(Strategy):
    def __init__(self, strategy_id: str = "sentiment_strategy"):
        super().__init__(strategy_id)
        
        # Subscribe to sentiment events
        event_bus.subscribe("sentiment_event", self.on_sentiment_event)
    
    async def on_sentiment_event(self, event: SentimentEvent):
        # Process sentiment event
        if event.sentiment_direction == "bullish" and event.confidence > 0.7:
            # Generate a buy signal
            signal = Signal(
                symbol=event.symbol,
                signal_type=SignalType.LONG,
                strategy=self.strategy_id,
                confidence=event.confidence
            )
            await self.publish_signal(signal)
```

## Examples

The repository includes several examples demonstrating the sentiment analysis system:

1. **Sentiment Analysis Demo**: Basic demonstration of the sentiment system
   ```
   python examples/sentiment_analysis_demo.py
   ```

2. **Sentiment Analysis Integration Example**: Complete example with strategy integration
   ```
   python examples/sentiment_analysis_integration_example.py
   ```

3. **Sentiment Backtest Example**: Backtesting with historical sentiment data
   ```
   python examples/sentiment_backtest_example.py
   ```

## Advanced Features

### Contrarian Analysis

The system can detect extreme sentiment levels that might indicate contrarian trading opportunities:

```python
# Example of checking for contrarian signals
if event.sentiment_value > 0.8 and "extreme" in event.details.get("tags", []):
    # This could be an extreme bullish sentiment
    # Consider a contrarian (bearish) signal
    pass
```

### Sentiment-Price Divergence

The system can detect divergences between sentiment and price action:

```python
# Example of price-sentiment divergence detection
if price_trend == "bullish" and event.sentiment_direction == "bearish":
    # Divergence detected
    pass
```

## Extending the System

### Adding a New Sentiment Source

To add a new sentiment source:

1. Create a new sentiment agent class that inherits from `BaseSentimentAgent`
2. Implement the required methods
3. Register the agent in the `SentimentAnalysisManager._create_agents` method

Example:

```python
class CustomSentimentAgent(BaseSentimentAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        
    async def _initialize(self) -> None:
        await super()._initialize()
        # Initialize custom resources
        
    async def _start(self) -> None:
        await super()._start()
        # Start periodic update task
        
    async def _analyze_custom_sentiment(self, symbol: str) -> None:
        # Implement custom sentiment analysis logic
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are properly set in environment variables
2. **Transformer Model Issues**: If you experience issues with the NLP model, the system will automatically fall back to lexicon-based analysis
3. **No Sentiment Events**: Check that the sentiment agents are enabled in the configuration

### Logging

The sentiment system uses structured logging. To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Dependencies

The sentiment analysis system requires the following dependencies:

- `transformers`: For advanced NLP models
- `pandas`: For data processing
- `matplotlib`: For visualization (examples only)
- API clients for data sources

You can install them with:

```
pip install transformers pandas matplotlib
```

## License

This sentiment analysis system is part of the AI Crypto Trading Agent platform and is subject to the same license terms.