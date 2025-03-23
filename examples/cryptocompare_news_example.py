"""Example script for demonstrating the CryptoCompare News API integration.

This script demonstrates how to use the CryptoCompare News API client to fetch
and analyze cryptocurrency-specific news articles.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pprint import pprint

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_agents.news.cryptocompare_news_client import CryptoCompareNewsClient, MockCryptoCompareNewsClient
from src.analysis_agents.news.news_analyzer import NewsAnalyzer, NewsArticle
from src.common.logging import get_logger, configure_logging


async def demo_cryptocompare_client():
    """Demonstrate CryptoCompare News API client functionality."""
    print("\n=== CryptoCompare News API Client Demo ===\n")
    
    # Initialize client
    # You can set the API key in environment variable CRYPTOCOMPARE_API_KEY
    # or pass it directly here
    api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
    if not api_key:
        print("No CryptoCompare API key found. Using mock client for demonstration.")
        client = MockCryptoCompareNewsClient()
    else:
        client = CryptoCompareNewsClient(api_key=api_key)
    
    # Example 1: Get latest cryptocurrency news
    print("\n--- Latest Cryptocurrency News ---\n")
    if isinstance(client, CryptoCompareNewsClient) and client.api_key:
        latest_articles = await client.get_latest_news(limit=5)
        
        for i, article in enumerate(latest_articles):
            print(f"Article {i+1}: {article.get('title')}")
            print(f"Source: {article.get('source')}")
            published_at = datetime.fromtimestamp(article.get('published_on', 0))
            print(f"Published: {published_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"URL: {article.get('url')}")
            print(f"Tags: {article.get('tags', '')}")
            print()
            
        # Example 2: Get news categories
        print("\n--- News Categories ---\n")
        categories = await client.get_categories()
        print(f"Available categories: {', '.join(list(categories.keys())[:10])}...")
        
        # Example 3: Get feeds list
        print("\n--- News Feeds ---\n")
        feeds = await client.get_feeds_list()
        for i, feed in enumerate(feeds[:5]):
            print(f"Feed {i+1}: {feed.get('name')} ({feed.get('key')})")
    else:
        print("Using mock client to demonstrate functionality...")
    
    # Example 4: Use the crypto news search helper
    print("\n--- Bitcoin News (using search helper) ---\n")
    btc_articles = await client.get_articles(timeframe="3d", assets=["BTC"])
    
    for i, article in enumerate(btc_articles[:5]):
        print(f"Article {i+1}: {article.title}")
        print(f"Source: {article.source}")
        print(f"Published: {article.published_at}")
        print(f"URL: {article.url}")
        print(f"Categories: {', '.join(article.categories)}")
        print(f"Tags: {', '.join(article.tags) if hasattr(article, 'tags') and article.tags else 'None'}")
        print(f"Content: {article.content[:150]}...")
        print()


async def demo_news_analyzer_with_cryptocompare():
    """Demonstrate the NewsAnalyzer with CryptoCompare News integration."""
    print("\n=== NewsAnalyzer with CryptoCompare Integration Demo ===\n")
    
    # Initialize the news analyzer
    analyzer = NewsAnalyzer()
    await analyzer.initialize()
    
    # Check if we have the CryptoCompare client
    if "cryptocompare" in analyzer.news_clients:
        print("CryptoCompare client initialized successfully")
    else:
        print("CryptoCompare client not initialized")
    
    # Collect articles
    print("\nCollecting articles from all sources...")
    articles = await analyzer.collect_articles(
        timeframe="24h",
        assets=["BTC", "ETH", "SOL"]
    )
    
    print(f"Collected {len(articles)} articles total")
    
    # Count articles by source
    sources = {}
    for article in articles:
        source = article.source
        sources[source] = sources.get(source, 0) + 1
    
    print("\nArticles by source:")
    for source, count in sources.items():
        print(f"- {source}: {count} articles")
    
    # Analyze articles
    print("\nAnalyzing articles...")
    await analyzer.analyze_articles(articles)
    
    # Generate market brief
    print("\nGenerating market brief for BTC...")
    brief = await analyzer.generate_market_brief(["BTC"])
    
    if brief and brief.get("data"):
        btc_data = brief["data"][0]
        print(f"\nBitcoin:")
        print(f"- Articles: {btc_data.get('article_count', 0)}")
        print(f"- Sentiment: {btc_data.get('sentiment', 0):.2f}")
        print(f"- Market impact: {btc_data.get('market_impact', 'neutral')}")
        print(f"- Top topics: {', '.join(btc_data.get('top_topics', []))}")
        
        if btc_data.get('top_articles'):
            print("\nTop Bitcoin article:")
            top_article = btc_data['top_articles'][0]
            print(f"Title: {top_article.get('title')}")
            print(f"Source: {top_article.get('source')}")
            print(f"URL: {top_article.get('url')}")
            if top_article.get('summary'):
                print(f"Summary: {top_article.get('summary')[:200]}...")
    
    # Extract events
    print("\nExtracting market events...")
    events = await analyzer.extract_events(min_importance=0.4)
    
    print(f"\nFound {len(events)} significant market events")
    for i, event in enumerate(events[:3]):
        print(f"\nEvent {i+1}: {event.get('title')}")
        print(f"- Importance: {event.get('importance', 0):.2f}")
        print(f"- Affected assets: {', '.join(event.get('affected_assets', []))}")
        print(f"- Description: {event.get('description', '')[:150]}...")


async def main():
    """Run the demo."""
    # Configure logging
    configure_logging(level=logging.INFO)
    
    print("=== CryptoCompare News API Integration Demo ===\n")
    print("This demo showcases the integration of CryptoCompare News API with")
    print("the trading agent's news analysis system. It demonstrates fetching")
    print("and analyzing cryptocurrency-specific news articles.")
    print("\nNote: For full functionality, obtain a CryptoCompare API key from")
    print("https://min-api.cryptocompare.com/ and set it as an environment variable:")
    print("export CRYPTOCOMPARE_API_KEY=your_api_key_here")
    print("\nWithout a key, the demo will use mock data.")
    
    # Run the CryptoCompare client demo
    await demo_cryptocompare_client()
    
    # Run the NewsAnalyzer demo
    await demo_news_analyzer_with_cryptocompare()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())