# NewsAPI Integration Guide

This guide explains how to use the NewsAPI integration with the AI Trading Agent's news analysis system.

## Overview

The NewsAPI integration allows the trading agent to fetch real-time news articles from thousands of sources worldwide via the [NewsAPI.org](https://newsapi.org/) service. This provides the agent with up-to-date information about cryptocurrency markets, allowing for more accurate sentiment analysis and market impact assessment.

The integration consists of:

1. A `NewsAPIClient` class that handles API authentication, requests, and error handling
2. Integration with the existing `NewsAnalyzer` system
3. Helper methods for cryptocurrency-specific news searches
4. An example script demonstrating the integration

## Setup

### API Key

To use the NewsAPI integration, you need to obtain an API key from [NewsAPI.org](https://newsapi.org/). The free tier provides limited access, while paid tiers offer more requests and features.

Once you have an API key, you can provide it in one of these ways:

1. Set an environment variable:
   ```
   export NEWS_API_KEY=your_api_key_here
   ```

2. Add it to your configuration file:
   ```yaml
   apis:
     news_api:
       key: your_api_key_here
   ```

3. Pass it directly when initializing the client:
   ```python
   from src.analysis_agents.news.news_api_client import NewsAPIClient
   
   client = NewsAPIClient(api_key="your_api_key_here")
   ```

### Dependencies

The NewsAPI client requires the `aiohttp` package for making asynchronous HTTP requests. Make sure it's installed:

```bash
pip install aiohttp
```

This should already be included in the project's `requirements.txt`.

## Usage

### Basic Usage

```python
import asyncio
from src.analysis_agents.news.news_api_client import NewsAPIClient

async def get_crypto_news():
    client = NewsAPIClient()
    
    # Get Bitcoin news
    articles = await client.get_everything(
        q="Bitcoin OR BTC",
        language="en",
        sort_by="publishedAt",
        page_size=10
    )
    
    for article in articles:
        print(f"Title: {article.get('title')}")
        print(f"Source: {article.get('source', {}).get('name')}")
        print(f"URL: {article.get('url')}")
        print("-" * 30)

# Run the async function
asyncio.run(get_crypto_news())
```

### Cryptocurrency-Specific Methods

The client includes helper methods for cryptocurrency-specific searches:

```python
async def get_asset_news():
    client = NewsAPIClient()
    
    # Get Ethereum news as NewsArticle objects
    eth_articles = await client.search_crypto_news(
        asset="ETH",
        days=7,
        page_size=20
    )
    
    for article in eth_articles:
        print(f"Title: {article.title}")
        print(f"Source: {article.source}")
        print(f"Published: {article.published_at}")
        print(f"URL: {article.url}")
        print("-" * 30)
```

### Integration with NewsAnalyzer

The NewsAPI client is automatically integrated with the `NewsAnalyzer` system when an API key is available:

```python
from src.analysis_agents.news.news_analyzer import NewsAnalyzer

async def analyze_news():
    analyzer = NewsAnalyzer()
    await analyzer.initialize()
    
    # Collect articles from all configured sources, including NewsAPI
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

### NewsAPIClient

The main client class for interacting with the NewsAPI.

#### Methods

- `get_everything(q, language, sort_by, page_size, page, from_date, to_date, domains, sources, use_cache)`:  
  Search for articles matching a query. Returns raw API response.

- `get_top_headlines(country, category, q, sources, page_size, page, use_cache)`:  
  Get top headlines. Returns raw API response.

- `get_sources(category, language, country, use_cache)`:  
  Get available news sources. Returns raw API response.

- `convert_to_news_articles(api_articles, prefix)`:  
  Convert raw API articles to internal NewsArticle objects.

- `search_crypto_news(asset, days, page_size, use_cache)`:  
  Helper method for searching cryptocurrency news. Returns NewsArticle objects.

## Example Script

An example script is provided to demonstrate the integration:

```bash
python examples/news_api_example.py
```

This script demonstrates:
1. Basic API usage
2. Cryptocurrency-specific searches
3. Integration with the NewsAnalyzer
4. Analysis of collected articles

## Limitations and Considerations

- The free tier of NewsAPI has a limited number of requests (500/day) and does not provide access to articles older than one month.
- Rate limiting is handled by the client, which will back off when limits are reached.
- Responses are cached to minimize API calls. The default cache duration is 1 hour.
- Error handling is built into the client, with appropriate logging of issues.

## Future Enhancements

Planned enhancements for the NewsAPI integration:

1. Integration with additional news sources like CryptoCompare and CoinDesk
2. Advanced filtering options for more targeted news collection
3. Improved sentiment analysis specifically tailored for financial news
4. Historical news database for backtesting sentiment-based strategies