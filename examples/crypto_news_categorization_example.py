#!/usr/bin/env python
"""Example demonstrating the cryptocurrency news categorization system.

This script demonstrates how to use the cryptocurrency news categorization system
to categorize news articles and identify trends and narratives.
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.analysis_agents.news.news_analyzer import NewsAnalyzer, NewsArticle
from src.analysis_agents.news.crypto_news_categorizer import (
    CryptoNewsCategorizer, 
    CryptoNewsCategory, 
    CryptoNewsTopicGraph
)
from src.analysis_agents.news.cryptocompare_news_client import CryptoCompareNewsClient, MockCryptoCompareNewsClient
from src.analysis_agents.news.news_api_client import NewsAPIClient
from src.common.logging import setup_logging, get_logger


async def main():
    """Run the cryptocurrency news categorization example."""
    # Set up logging
    setup_logging(log_level="INFO")
    logger = get_logger("examples", "crypto_news_categorization")
    
    logger.info("Starting cryptocurrency news categorization example")
    
    # Initialize news analyzer
    news_analyzer = NewsAnalyzer()
    await news_analyzer.initialize()
    
    # Initialize news categorizer
    categorizer = CryptoNewsCategorizer()
    await categorizer.initialize()
    
    # Initialize topic graph
    topic_graph = CryptoNewsTopicGraph()
    
    # Collect news articles
    assets = ["BTC", "ETH", "SOL", "XRP"]
    timeframe = "48h"  # Last 48 hours
    
    logger.info(f"Collecting news articles for {assets} in the last {timeframe}")
    
    # Check if we have real API keys
    has_news_api_key = bool(os.getenv("NEWS_API_KEY", ""))
    has_cryptocompare_key = bool(os.getenv("CRYPTOCOMPARE_API_KEY", ""))
    
    if has_news_api_key:
        logger.info("Using real NewsAPI client")
        news_api_client = NewsAPIClient()
    else:
        logger.info("No NewsAPI key found - will use only CryptoCompare")
        
    if has_cryptocompare_key:
        logger.info("Using real CryptoCompare News client")
        crypto_news_client = CryptoCompareNewsClient()
    else:
        logger.info("No CryptoCompare API key found - using mock client")
        crypto_news_client = MockCryptoCompareNewsClient()
    
    # Collect articles from both sources
    all_articles = []
    
    # Get CryptoCompare articles
    crypto_articles = await crypto_news_client.get_articles(timeframe=timeframe, assets=assets)
    logger.info(f"Collected {len(crypto_articles)} articles from CryptoCompare")
    all_articles.extend(crypto_articles)
    
    # Get NewsAPI articles if available
    if has_news_api_key:
        for asset in assets:
            # Prepare search query with asset-specific terms
            if asset == "BTC":
                query = "Bitcoin OR BTC OR cryptocurrency"
            elif asset == "ETH":
                query = "Ethereum OR ETH OR cryptocurrency"
            elif asset == "SOL":
                query = "Solana OR SOL OR cryptocurrency"
            elif asset == "XRP":
                query = "Ripple OR XRP OR cryptocurrency"
            else:
                query = f"{asset} OR cryptocurrency"
            
            # Parse timeframe to days
            days = 2  # Default for 48h
            if timeframe.endswith("h"):
                hours = int(timeframe[:-1])
                days = max(1, hours // 24)
            elif timeframe.endswith("d"):
                days = int(timeframe[:-1])
                
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Fetch articles
            api_articles = await news_api_client.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                from_date=from_date,
                page_size=20
            )
            
            # Convert to NewsArticle objects
            news_articles = await news_api_client.convert_to_news_articles(api_articles)
            logger.info(f"Collected {len(news_articles)} articles about {asset} from NewsAPI")
            
            all_articles.extend(news_articles)
    
    # Analyze and categorize articles
    logger.info(f"Analyzing and categorizing {len(all_articles)} articles")
    
    categorized_articles = []
    
    for article in all_articles:
        # Calculate basic asset relevance if not already present
        if not hasattr(article, 'relevance_scores') or not article.relevance_scores:
            article.relevance_scores = calculate_asset_relevance(article, assets)
        
        # Categorize the article
        categories = await categorizer.categorize_article(article)
        
        if categories:
            # Add to topic graph
            topic_graph.add_article(article, categories)
            
            # Store categorized article
            categorized_articles.append({
                "id": article.article_id,
                "title": article.title,
                "source": article.source,
                "published_at": article.published_at.isoformat(),
                "url": article.url,
                "categories": categories,
                "relevance_scores": article.relevance_scores
            })
            
            # Log categories
            categories_str = ", ".join([f"{cat} ({score:.2f})" for cat, score in categories.items()])
            logger.info(f"Categorized article: '{article.title}' - {categories_str}")
    
    # Analyze trends and narratives
    trending_categories = topic_graph.get_trending_categories(timeframe_hours=48, limit=10)
    
    logger.info("Top trending categories:")
    for trend in trending_categories:
        logger.info(f"  {trend['category']}: {trend['score']:.2f} ({trend['article_count']} articles)")
    
    # Find narrative clusters
    narratives = topic_graph.find_narrative_clusters(timeframe_hours=48)
    
    logger.info(f"Found {len(narratives)} narrative clusters:")
    for narrative in narratives:
        logger.info(f"  {narrative['title']} - {narrative['size']} articles")
        if narrative['categories']:
            categories_str = ", ".join([cat["category"] for cat in narrative['categories']])
            logger.info(f"    Top categories: {categories_str}")
        if narrative['assets']:
            assets_str = ", ".join([ast["asset"] for ast in narrative['assets']])
            logger.info(f"    Top assets: {assets_str}")
    
    # Get asset-category associations for Bitcoin
    btc_categories = topic_graph.get_asset_category_associations("BTC")
    
    logger.info("Bitcoin category associations:")
    for category, score in sorted(btc_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {category}: {score:.2f}")
    
    # Save results to file
    results = {
        "articles": categorized_articles,
        "trending_categories": trending_categories,
        "narratives": narratives,
        "asset_categories": {
            "BTC": btc_categories
        }
    }
    
    output_dir = "examples/output/crypto_news"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/categorization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}/categorization_results.json")


def calculate_asset_relevance(article: NewsArticle, assets: List[str]) -> Dict[str, float]:
    """Calculate asset relevance scores for an article.
    
    Args:
        article: The article to analyze
        assets: List of assets to check
        
    Returns:
        Dictionary mapping asset symbols to relevance scores (0-1)
    """
    text = f"{article.title.lower()} {article.content.lower()}"
    relevance_scores = {}
    
    # Asset-specific keywords
    asset_keywords = {
        "BTC": ["bitcoin", "btc", "satoshi"],
        "ETH": ["ethereum", "eth", "vitalik", "buterin"],
        "SOL": ["solana", "sol"],
        "XRP": ["ripple", "xrp"]
    }
    
    for asset in assets:
        score = 0.0
        keywords = asset_keywords.get(asset, [asset.lower()])
        
        # Count keyword mentions
        mention_count = sum(text.count(keyword) for keyword in keywords)
        
        # Calculate base score
        if mention_count > 0:
            score = min(0.3 + (mention_count * 0.1), 1.0)
            
            # Boost score for mentions in title
            title_mentions = sum(article.title.lower().count(keyword) for keyword in keywords)
            if title_mentions > 0:
                score = min(score + 0.2, 1.0)
        
        # Store score if relevant
        if score >= 0.3:
            relevance_scores[asset] = score
    
    return relevance_scores


if __name__ == "__main__":
    asyncio.run(main())