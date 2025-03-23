"""Example script for demonstrating the NewsAPI integration.

This script demonstrates how to use the NewsAPI client to fetch
news articles about cryptocurrencies and analyze them using the
NewsAnalyzer.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pprint import pprint

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_agents.news.news_api_client import NewsAPIClient
from src.analysis_agents.news.news_analyzer import NewsAnalyzer, NewsArticle
from src.common.logging import get_logger, configure_logging


async def demo_news_api_client():
    """Demonstrate NewsAPI client functionality."""
    print("\n=== NewsAPI Client Demo ===\n")
    
    # Initialize client
    # You can set the API key in environment variable NEWS_API_KEY
    # or pass it directly here
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("No NewsAPI key found. Please set the NEWS_API_KEY environment variable.")
        print("For demo purposes, continuing with limited functionality.")
    
    client = NewsAPIClient(api_key=api_key)
    
    # Example 1: Get Bitcoin news
    print("\n--- Bitcoin News ---\n")
    btc_articles = await client.get_everything(
        q="Bitcoin OR BTC",
        language="en",
        sort_by="publishedAt",
        page_size=5
    )
    
    for i, article in enumerate(btc_articles):
        print(f"Article {i+1}: {article.get('title')}")
        print(f"Source: {article.get('source', {}).get('name')}")
        print(f"Published: {article.get('publishedAt')}")
        print(f"URL: {article.get('url')}")
        print(f"Description: {article.get('description', '')[:150]}...")
        print()
    
    # Example 2: Get top cryptocurrency headlines
    print("\n--- Top Cryptocurrency Headlines ---\n")
    headlines = await client.get_top_headlines(
        category="business",
        q="cryptocurrency",
        page_size=5
    )
    
    for i, article in enumerate(headlines):
        print(f"Headline {i+1}: {article.get('title')}")
        print(f"Source: {article.get('source', {}).get('name')}")
        print(f"Published: {article.get('publishedAt')}")
        print(f"URL: {article.get('url')}")
        print()
    
    # Example 3: Use the crypto news search helper
    print("\n--- Ethereum News (using helper) ---\n")
    eth_articles = await client.search_crypto_news(
        asset="ETH",
        days=3,
        page_size=5
    )
    
    for i, article in enumerate(eth_articles):
        print(f"Article {i+1}: {article.title}")
        print(f"Source: {article.source}")
        print(f"Published: {article.published_at}")
        print(f"URL: {article.url}")
        print(f"Content: {article.content[:150] if article.content else 'No content'}...")
        print()


async def demo_news_analyzer():
    """Demonstrate the NewsAnalyzer with NewsAPI integration."""
    print("\n=== NewsAnalyzer Demo ===\n")
    
    # Initialize the news analyzer
    analyzer = NewsAnalyzer()
    await analyzer.initialize()
    
    # Check if we have the NewsAPI client
    if "newsapi" in analyzer.news_clients:
        print("NewsAPI client initialized successfully")
    else:
        print("NewsAPI client not initialized - using mock client")
    
    # Collect articles
    print("\nCollecting articles...")
    articles = await analyzer.collect_articles(
        timeframe="24h",
        assets=["BTC", "ETH"]
    )
    
    print(f"Collected {len(articles)} articles")
    
    # Analyze articles
    print("\nAnalyzing articles...")
    await analyzer.analyze_articles(articles)
    
    # Get trending topics
    print("\nTrending topics:")
    topics = await analyzer.get_trending_topics()
    for topic in topics[:5]:
        print(f"- {topic['topic']}: {topic['count']} articles, sentiment: {topic['sentiment']:.2f}")
    
    # Generate market brief
    print("\nMarket brief:")
    brief = await analyzer.generate_market_brief(["BTC", "ETH"])
    for asset_data in brief["data"]:
        print(f"\n{asset_data['asset']}:")
        print(f"- Articles: {asset_data['article_count']}")
        print(f"- Sentiment: {asset_data['sentiment']:.2f}")
        print(f"- Impact: {asset_data['market_impact']}")
        print(f"- Top topics: {', '.join(asset_data['top_topics'])}")
        
        if asset_data['top_articles']:
            print(f"\nTop article: {asset_data['top_articles'][0]['title']}")
            print(f"Source: {asset_data['top_articles'][0]['source']}")
            print(f"URL: {asset_data['top_articles'][0]['url']}")
    
    # Extract events
    print("\nExtracted events:")
    events = await analyzer.extract_events(min_importance=0.4)
    for i, event in enumerate(events[:3]):
        print(f"\nEvent {i+1}: {event['title']}")
        print(f"- Importance: {event['importance']:.2f}")
        print(f"- Affected assets: {', '.join(event['affected_assets'])}")
        print(f"- Description: {event['description'][:150]}...")


async def main():
    """Run the demo."""
    # Configure logging
    configure_logging(level=logging.INFO)
    
    print("=== NewsAPI Integration Demo ===\n")
    print("This demo showcases the integration of NewsAPI with the trading agent's")
    print("news analysis system. It demonstrates fetching and analyzing news")
    print("articles related to cryptocurrency markets.\n")
    
    # Uncomment to run the NewsAPI client demo
    await demo_news_api_client()
    
    # Run the NewsAnalyzer demo
    await demo_news_analyzer()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())