# Sentiment Analysis System Guide

## Overview

The sentiment analysis system in the AI Trading Agent provides a robust framework for analyzing and incorporating market sentiment data into trading strategies. This guide explains how to use and extend the system.

## Key Components

The sentiment analysis system consists of several modular components:

1. **NLP Service**: Provides natural language processing capabilities for sentiment analysis
2. **Sentiment Agents**: Specialized agents for different data sources (social media, news, market, on-chain)
3. **Sentiment Aggregator**: Combines signals from different sources
4. **Sentiment Events**: Publishes signals to the trading system
5. **Sentiment Strategies**: Trading strategies that use sentiment signals

## System Architecture

The sentiment analysis system follows a modular, event-driven architecture:

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│                 │   │                 │   │                 │   │                 │
│  Data Sources   │──▶│ Sentiment Agents│──▶│    Aggregator   │──▶│  Event System   │
│                 │   │                 │   │                 │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘   └────────┬────────┘
                                                                           │
                                                                           ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│                 │   │                 │   │                 │   │                 │
│  Trading Signal │◀──┤ Risk Management│◀──┤    Strategy     │◀──┤ Sentiment Event │
│                 │   │                 │   │                 │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘
```

## Getting Started

### 1. Install Required Packages

Make sure you have the required packages installed:

```bash
pip install -r requirements.txt
```

The sentiment analysis system requires:
- transformers (for NLP models)
- tweepy (for Twitter/X integration)
- praw (for Reddit integration)
- beautifulsoup4 (for web scraping)
- spacy (for text processing)

### 2. Setting up API Credentials

To use real data sources, you need to set up API credentials for your data sources:

#### Twitter/X API
```bash
export TWITTER_API_KEY="your_api_key"
export TWITTER_API_SECRET="your_api_secret"
export TWITTER_ACCESS_TOKEN="your_access_token"
export TWITTER_ACCESS_SECRET="your_access_secret"
```

#### Reddit API
```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="AI-Trading-Agent/1.0"
```

### 3. Basic Usage

Here's a simple example of using the sentiment analysis system:

```python
import asyncio
from src.analysis_agents.sentiment.nlp_service import NLPService
from src.analysis_agents.sentiment.social_media_sentiment import TwitterClient, RedditClient

async def analyze_sentiment():
    # Initialize NLP service
    nlp_service = NLPService()
    await nlp_service.initialize()
    
    # Initialize Twitter client
    twitter_client = TwitterClient(
        api_key="your_api_key",
        api_secret="your_api_secret",
        access_token="your_access_token",
        access_secret="your_access_secret"
    )
    
    # Search for tweets
    tweets = await twitter_client.search_tweets(query="#BTC OR $BTC", count=100)
    
    # Analyze sentiment
    sentiment_scores = await nlp_service.analyze_sentiment(tweets)
    
    # Calculate overall sentiment
    sentiment_value = sum(sentiment_scores) / len(sentiment_scores)
    print(f"Bitcoin sentiment: {sentiment_value:.2f}")

# Run the analysis
asyncio.run(analyze_sentiment())
```

### 4. Running the Demo

For a complete demonstration of the sentiment analysis system, run the included demo:

```bash
python examples/sentiment_real_integration_demo.py
```

This demo:
1. Connects to social media APIs (or uses mock data if no credentials)
2. Fetches and analyzes sentiment for a specified cryptocurrency
3. Gets price data from Binance exchange
4. Visualizes the relationship between price and sentiment
5. Generates trading signals

## Using Sentiment in Trading

### Enhanced Sentiment Strategy

The framework includes an enhanced sentiment strategy that combines:
1. Sentiment analysis from multiple sources
2. Technical indicators for confirmation
3. Market regime detection

Key features:
- Sentiment thresholds for entry/exit
- Technical confirmation via RSI
- Contrarian signals for extreme sentiment
- Market regime awareness

To run the enhanced sentiment strategy:

```bash
python examples/enhanced_sentiment_trading_strategy.py
```

### Creating Custom Strategies

You can create your own sentiment-based strategies by:

1. Extending the base strategy classes
2. Subscribing to sentiment events
3. Implementing your custom logic

Example:

```python
class MySentimentStrategy(BaseTradingStrategy):
    async def initialize(self):
        # Subscribe to sentiment events
        self.event_bus.subscribe("sentiment_event", self.on_sentiment_event)
    
    async def on_sentiment_event(self, event):
        # Handle sentiment event
        if event.confidence > 0.7:
            if event.direction == "bullish":
                # Generate buy signal
                await self.generate_signal("BUY", event.symbol)
            elif event.direction == "bearish":
                # Generate sell signal
                await self.generate_signal("SELL", event.symbol)
```

## Extending the System

### Adding New Data Sources

To add a new sentiment data source:

1. Create a new agent class that extends `BaseSentimentAgent`
2. Implement the data fetching and processing logic
3. Register the agent with the sentiment manager

Example:

```python
class NewsSentimentAgent(BaseSentimentAgent):
    async def _initialize(self):
        await super()._initialize()
        # Setup your news API client
        
    async def _update_sentiment_periodically(self):
        while True:
            # Fetch news data
            # Process sentiment
            # Publish events
            await asyncio.sleep(self.update_interval)
```

### Custom NLP Models

To use a custom NLP model:

1. Fine-tune a transformer model on your specific domain (e.g., cryptocurrency news)
2. Update the NLP service to use your model:

```python
# In your configuration
config.set("nlp.sentiment_model", "path/to/your/model")
```

## Best Practices

1. **Multiple Sources**: Combine multiple sentiment sources for more robust signals
2. **Confidence Filtering**: Use confidence scores to filter out low-quality signals
3. **Technical Confirmation**: Confirm sentiment signals with technical indicators
4. **Contrarian Awareness**: Be cautious of extreme sentiment (may indicate market tops/bottoms)
5. **Regime Adaptation**: Adjust strategy based on market regime (trending vs. ranging)
6. **Calibration**: Regularly calibrate your sentiment thresholds based on performance

## Monitoring and Visualization

The system includes visualization tools to monitor sentiment and its relationship with price:

```python
# Create a sentiment visualization
from src.visualization.sentiment_visualizer import SentimentVisualizer

visualizer = SentimentVisualizer()
visualizer.plot_sentiment_price(symbol="BTC/USDT", timeframe="1h")
```

## Troubleshooting

Common issues and solutions:

1. **API Rate Limits**: Social media APIs have rate limits. Implement rate limiting and caching.
2. **NLP Model Loading**: Transformer models can be large. Use smaller models or offload to GPU.
3. **Mock Data**: If API credentials are missing, the system falls back to simulated data.
4. **Real-time Performance**: For production, optimize the update intervals based on your timeframe.

## Advanced Topics

### Market Regime Detection

The sentiment strategy performs better when aware of the current market regime:

- **Trending Markets**: Sentiment confirms trend direction
- **Ranging Markets**: Sentiment indicates potential breakouts
- **Volatile Markets**: Extreme sentiment may signal reversals

### Sentiment Divergence

Watch for divergence between sentiment and price:

- Price rising + sentiment falling = potential top
- Price falling + sentiment rising = potential bottom

### Custom Lexicons

For specialized domains, create custom sentiment lexicons:

```python
# Example: Add crypto-specific terms to lexicon
self.bullish_words.update({
    "hodl": 0.8,
    "moon": 0.9,
    "diamond_hands": 0.8
})

self.bearish_words.update({
    "fud": 0.7,
    "scam": 0.9,
    "rugpull": 1.0
})
```