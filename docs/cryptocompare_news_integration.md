# CryptoCompare News API Integration Guide

This guide explains how to use the CryptoCompare News API integration with the AI Trading Agent's news analysis system.

## Overview

The CryptoCompare News API integration allows the trading agent to fetch cryptocurrency-specific news articles from the CryptoCompare News service. This provides in-depth coverage of cryptocurrency markets, including trading analysis, technology developments, regulatory news, and more, enhancing the agent's sentiment analysis capabilities with specialized crypto content.

The integration consists of:

1. A `CryptoCompareNewsClient` class for API interaction
2. A mock client for testing and fallback
3. Integration with the existing `NewsAnalyzer` system
4. Helper methods for cryptocurrency-specific news searches
5. An example script demonstrating the integration

## Setup

### API Key

To use the CryptoCompare News API integration, you need to obtain an API key from [CryptoCompare](https://min-api.cryptocompare.com/). The free tier provides limited access, while paid tiers offer more requests and features.

Once you have an API key, you can provide it in one of these ways:

1. Set an environment variable:
   ```
   export CRYPTOCOMPARE_API_KEY=your_api_key_here
   ```

2. Add it to your configuration file:
   ```yaml
   apis:
     cryptocompare:
       key: your_api_key_here
   ```

3. Pass it directly when initializing the client:
   ```python
   from src.analysis_agents.news.cryptocompare_news_client import CryptoCompareNewsClient
   
   client = CryptoCompareNewsClient(api_key="your_api_key_here")
   ```

### Dependencies

The CryptoCompare client requires the `aiohttp` package, which should already be included in the project's requirements.

## Usage

### Basic Usage

```python
import asyncio
from src.analysis_agents.news.cryptocompare_news_client import CryptoCompareNewsClient

async def get_crypto_news():
    client = CryptoCompareNewsClient()
    
    # Get latest news
    articles = await client.get_latest_news(limit=10)
    
    for article in articles:
        print(f"Title: {article.get('title')}")
        print(f"Source: {article.get('source')}")
        print(f"Tags: {article.get('tags')}")
        print("-" * 30)

# Run the async function
asyncio.run(get_crypto_news())
```

### Search for Asset-Specific News

```python
async def get_bitcoin_news():
    client = CryptoCompareNewsClient()
    
    # Get Bitcoin news as NewsArticle objects
    btc_articles = await client.search_crypto_news(
        asset="BTC",
        days=7,
        limit=20
    )
    
    for article in btc_articles:
        print(f"Title: {article.title}")
        print(f"Source: {article.source}")
        print(f"Published: {article.published_at}")
        print(f"URL: {article.url}")
        print("-" * 30)
```

### Integration with NewsAnalyzer

The CryptoCompare News client is automatically integrated with the `NewsAnalyzer` system when an API key is available:

```python
from src.analysis_agents.news.news_analyzer import NewsAnalyzer

async def analyze_news():
    analyzer = NewsAnalyzer()
    await analyzer.initialize()
    
    # Collect articles from all configured sources, including CryptoCompare
    articles = await analyzer.collect_articles(
        timeframe="24h",
        assets=["BTC", "ETH", "SOL"]
    )
    
    # Analyze the articles
    await analyzer.analyze_articles(articles)
    
    # Generate a market brief
    brief = await analyzer.generate_market_brief(["BTC", "ETH", "SOL"])
    print(brief)
```

## API Reference

### CryptoCompareNewsClient

The main client class for interacting with the CryptoCompare News API.

#### Methods

- `get_latest_news(lang, feeds, categories, exclude_categories, limit_timestamp, sort_order, limit, use_cache)`:  
  Get latest cryptocurrency news articles

- `get_categories(lang, use_cache)`:  
  Get available news categories

- `get_feeds_list(lang, use_cache)`:  
  Get available news feeds

- `convert_to_news_articles(api_articles, prefix)`:  
  Convert raw API articles to internal NewsArticle objects

- `search_crypto_news(asset, categories, days, limit, use_cache)`:  
  Helper method for searching cryptocurrency news, returns NewsArticle objects
  
- `get_articles(timeframe, assets)`:  
  Get news articles for specified assets and timeframe (compatible with NewsAnalyzer interface)

### MockCryptoCompareNewsClient

A mock client for testing and fallback when no API key is available.

#### Methods

- `get_articles(timeframe, assets)`:  
  Generate mock cryptocurrency news articles

## Categories and Feeds

The CryptoCompare News API provides articles categorized into various topics relevant to cryptocurrency markets:

### Categories
- Trading: Technical analysis, market insights, price predictions
- Mining: Mining operations, hash rate, mining equipment
- ICO: Initial Coin Offerings, token sales
- Regulation: Legal developments, government policies
- Technology: Blockchain innovations, protocol upgrades
- Business: Partnerships, acquisitions, company news
- Exchanges: Exchange platforms, listings, trading fees
- Wallet: Cryptocurrency wallets, storage solutions
- Analysis: In-depth market and investment analysis

### Feeds
The API aggregates news from numerous cryptocurrency-specific sources, including CoinDesk, Cointelegraph, Decrypt, The Block, and many others.

## Example Script

An example script is provided to demonstrate the integration:

```bash
python examples/cryptocompare_news_example.py
```

This script demonstrates:
1. Fetching latest cryptocurrency news
2. Getting available categories and feeds
3. Searching for asset-specific news
4. Integration with the NewsAnalyzer for comprehensive news analysis
5. Generating a market brief with sentiment analysis

## Limitations and Considerations

- The free tier of CryptoCompare API is limited to about 50 calls per day
- Respect rate limits by implementing proper caching (already built into the client)
- News is primarily focused on cryptocurrencies, rather than general financial news
- The API provides specialized crypto-specific categories and tags

## Future Enhancements

Planned enhancements for the CryptoCompare News integration:

1. Better keyword filtering and relevance scoring for crypto-specific terminology
2. Enhanced analysis of technical indicators mentioned in articles
3. Specialized sentiment analysis for crypto market news
4. Historical news database for backtesting sentiment-based strategies