"""
News data collector for the Early Event Detection System.

This module provides functionality for collecting data from news sources
using various news APIs and RSS feeds.
"""

import asyncio
import logging
import os
import re
import time
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
import aiohttp

from src.common.config import config
from src.common.logging import get_logger
from src.common.api_client import RetryableAPIClient, CircuitBreaker
from src.analysis_agents.news.news_api_client import NewsAPIClient
from src.analysis_agents.news.news_analyzer import NewsArticle
from src.analysis_agents.news.cryptocompare_news_client import CryptoCompareNewsClient
from src.analysis_agents.early_detection.models import SourceType, EventSource
from src.analysis_agents.early_detection.optimization import get_cost_optimizer


class NewsCollector:
    """Collector for news data from various sources."""
    
    def __init__(self):
        """Initialize the news collector."""
        self.logger = get_logger("early_detection", "news_collector")
        
        # Configuration
        self.sources = config.get("early_detection.data_collection.news.sources", 
                                 ["coindesk", "cointelegraph", "theblock", "decrypt"])
        self.keywords = config.get("early_detection.data_collection.news.keywords", 
                                  ["regulation", "sec", "fed", "central bank"])
        
        # API credentials
        self.newsapi_key = os.getenv("NEWSAPI_KEY") or config.get("apis.news_api.key", "")
        self.cryptocompare_key = os.getenv("CRYPTOCOMPARE_KEY") or config.get("apis.cryptocompare.key", "")
        
        # API clients
        self.api_client = RetryableAPIClient(
            max_retries=3,
            backoff_factor=2.0,
            logger=self.logger
        )
        
        # Specialized clients
        self.newsapi = None
        self.cryptocompare = None
        
        # Cache for last fetched article timestamps
        self.last_fetch_time = {}
        
        # RSS feed URLs
        self.rss_feeds = {
            "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "cointelegraph": "https://cointelegraph.com/rss",
            "theblock": "https://www.theblock.co/rss.xml",
            "decrypt": "https://decrypt.co/feed",
            "bitcoinmagazine": "https://bitcoinmagazine.com/feed"
        }
    
    async def initialize(self):
        """Initialize the news collector."""
        self.logger.info("Initializing news collector")
        
        # Initialize NewsAPI client if we have credentials
        if self.newsapi_key:
            try:
                self.newsapi = NewsAPIClient(api_key=self.newsapi_key)
                self.logger.info("NewsAPI client initialized")
            except Exception as e:
                self.logger.error(f"Error initializing NewsAPI client: {e}")
        else:
            self.logger.warning("Missing NewsAPI credentials")
        
        # Initialize CryptoCompare client if we have credentials
        if self.cryptocompare_key:
            try:
                self.cryptocompare = CryptoCompareNewsClient(api_key=self.cryptocompare_key)
                self.logger.info("CryptoCompare news client initialized")
            except Exception as e:
                self.logger.error(f"Error initializing CryptoCompare client: {e}")
        else:
            self.logger.warning("Missing CryptoCompare credentials")
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from news sources.
        
        Returns:
            List of collected data items
        """
        self.logger.info("Collecting data from news sources")
        
        collected_data = []
        
        try:
            # Get cost optimizer for API efficiency
            cost_optimizer = await get_cost_optimizer()
            
            # Check if we should sample news based on adaptive sampling
            if not cost_optimizer.adaptive_sampler.should_sample("news"):
                self.logger.info("Skipping news collection due to adaptive sampling")
                return []
            
            # Create collection tasks
            tasks = []
            
            # Task for NewsAPI
            if self.newsapi:
                tasks.append(
                    asyncio.create_task(
                        cost_optimizer.api_request(
                            "news",
                            self._collect_from_newsapi
                        )
                    )
                )
            
            # Task for CryptoCompare
            if self.cryptocompare:
                tasks.append(
                    asyncio.create_task(
                        cost_optimizer.api_request(
                            "news",
                            self._collect_from_cryptocompare
                        )
                    )
                )
            
            # Tasks for RSS feeds
            for source, feed_url in self.rss_feeds.items():
                if source in self.sources:
                    tasks.append(
                        asyncio.create_task(
                            cost_optimizer.api_request(
                                "news",
                                self._collect_from_rss,
                                source, feed_url
                            )
                        )
                    )
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error collecting news: {result}")
                    continue
                
                if result:
                    collected_data.extend(result)
            
            self.logger.info(f"Collected {len(collected_data)} news items")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting news data: {e}")
            return self._mock_collect()
    
    async def _collect_from_newsapi(self) -> List[Dict[str, Any]]:
        """Collect news from NewsAPI.
        
        Returns:
            List of collected news data
        """
        self.logger.debug("Collecting news from NewsAPI")
        
        collected_data = []
        
        try:
            # Check when we last fetched
            last_time = self.last_fetch_time.get("newsapi", datetime.now() - timedelta(days=1))
            from_date = last_time.strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            # Update last fetch time
            self.last_fetch_time["newsapi"] = datetime.now()
            
            # Prepare query with keywords
            query = " OR ".join([f'"{keyword}"' for keyword in self.keywords])
            
            # Get news with query
            articles = await self.newsapi.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=50,
                from_date=from_date,
                to_date=to_date
            )
            
            # Convert to internal format
            news_articles = await self.newsapi.convert_to_news_articles(articles)
            
            # Convert to collected data format
            for article in news_articles:
                # Calculate relevance score based on keyword matching
                relevance_score = self._calculate_relevance_score(article.title + " " + article.content, self.keywords)
                
                # Create source
                source = EventSource(
                    id=f"newsapi_{article.article_id}",
                    type=SourceType.NEWS,
                    name=article.source,
                    url=article.url,
                    reliability_score=0.7  # News sources are relatively reliable
                )
                
                collected_data.append({
                    "source": source,
                    "title": article.title,
                    "content": article.content,
                    "timestamp": article.published_at,
                    "metadata": {
                        "author": article.author or "Unknown",
                        "source_name": article.source,
                        "categories": article.categories,
                        "relevance_score": relevance_score,
                        "api_source": "NewsAPI"
                    }
                })
            
            self.logger.debug(f"Collected {len(collected_data)} articles from NewsAPI")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting from NewsAPI: {e}")
            return []
    
    async def _collect_from_cryptocompare(self) -> List[Dict[str, Any]]:
        """Collect news from CryptoCompare.
        
        Returns:
            List of collected news data
        """
        self.logger.debug("Collecting news from CryptoCompare")
        
        collected_data = []
        
        try:
            # Get latest news
            articles = await self.cryptocompare.get_latest_news(limit=50)
            
            # Check when we last fetched
            last_time = self.last_fetch_time.get("cryptocompare", datetime.now() - timedelta(days=1))
            
            # Filter out old articles
            articles = [a for a in articles if a.published_at > last_time]
            
            # Update last fetch time
            self.last_fetch_time["cryptocompare"] = datetime.now()
            
            # Convert to collected data format
            for article in articles:
                # Calculate relevance score based on keyword matching
                relevance_score = self._calculate_relevance_score(article.title + " " + article.content, self.keywords)
                
                # Skip if not relevant enough
                if relevance_score < 0.3:
                    continue
                
                # Create source
                source = EventSource(
                    id=f"cryptocompare_{article.article_id}",
                    type=SourceType.NEWS,
                    name=article.source,
                    url=article.url,
                    reliability_score=0.7  # News sources are relatively reliable
                )
                
                collected_data.append({
                    "source": source,
                    "title": article.title,
                    "content": article.content,
                    "timestamp": article.published_at,
                    "metadata": {
                        "author": article.author or "Unknown",
                        "source_name": article.source,
                        "categories": article.categories,
                        "tags": article.tags,
                        "relevance_score": relevance_score,
                        "api_source": "CryptoCompare"
                    }
                })
            
            self.logger.debug(f"Collected {len(collected_data)} articles from CryptoCompare")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting from CryptoCompare: {e}")
            return []
    
    async def _collect_from_rss(self, source_name: str, feed_url: str) -> List[Dict[str, Any]]:
        """Collect news from an RSS feed.
        
        Args:
            source_name: Name of the source
            feed_url: URL of the RSS feed
            
        Returns:
            List of collected news data
        """
        self.logger.debug(f"Collecting news from RSS feed: {source_name}")
        
        collected_data = []
        
        try:
            # Fetch RSS feed
            feed_data = await self._fetch_rss_feed(feed_url)
            
            if not feed_data or "entries" not in feed_data:
                self.logger.warning(f"No entries found in RSS feed: {source_name}")
                return []
            
            # Check when we last fetched
            last_time = self.last_fetch_time.get(f"rss_{source_name}", datetime.now() - timedelta(days=1))
            
            # Process entries
            for entry in feed_data.entries:
                # Parse publication date
                pub_date = None
                
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                
                # Skip if no date or too old
                if not pub_date or pub_date < last_time:
                    continue
                
                # Get title and content
                title = entry.title if hasattr(entry, "title") else ""
                
                # Try to get content
                content = ""
                if hasattr(entry, "content") and entry.content:
                    content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                elif hasattr(entry, "summary") and entry.summary:
                    content = entry.summary
                elif hasattr(entry, "description") and entry.description:
                    content = entry.description
                
                # Clean HTML from content
                content = self._clean_html(content)
                
                # Get link
                link = entry.link if hasattr(entry, "link") else ""
                
                # Calculate relevance score based on keyword matching
                relevance_score = self._calculate_relevance_score(title + " " + content, self.keywords)
                
                # Skip if not relevant enough
                if relevance_score < 0.3:
                    continue
                
                # Generate ID from link
                article_id = link.split("/")[-1] if link else f"{source_name}_{int(time.time())}"
                
                # Create source
                source = EventSource(
                    id=f"rss_{source_name}_{article_id}",
                    type=SourceType.NEWS,
                    name=source_name,
                    url=link,
                    reliability_score=0.7  # News sources are relatively reliable
                )
                
                # Get author
                author = entry.author if hasattr(entry, "author") else "Unknown"
                
                # Get categories/tags
                categories = []
                if hasattr(entry, "tags") and entry.tags:
                    categories = [tag.term for tag in entry.tags if hasattr(tag, "term")]
                elif hasattr(entry, "categories") and entry.categories:
                    categories = entry.categories
                
                collected_data.append({
                    "source": source,
                    "title": title,
                    "content": content,
                    "timestamp": pub_date,
                    "metadata": {
                        "author": author,
                        "source_name": source_name,
                        "categories": categories,
                        "relevance_score": relevance_score,
                        "api_source": "RSS"
                    }
                })
            
            # Update last fetch time
            self.last_fetch_time[f"rss_{source_name}"] = datetime.now()
            
            self.logger.debug(f"Collected {len(collected_data)} articles from {source_name} RSS")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting from RSS feed {source_name}: {e}")
            return []
    
    async def _fetch_rss_feed(self, feed_url: str) -> Any:
        """Fetch and parse an RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            Parsed feed data
        """
        try:
            # Run feedparser in a thread to avoid blocking
            feed_data = await asyncio.to_thread(
                feedparser.parse,
                feed_url
            )
            
            return feed_data
            
        except Exception as e:
            self.logger.error(f"Error fetching RSS feed {feed_url}: {e}")
            return None
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML tags from content.
        
        Args:
            html_content: HTML content to clean
            
        Returns:
            Cleaned text
        """
        # Simple regex to remove HTML tags
        clean_text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Remove excess whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def _calculate_relevance_score(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matching.
        
        Args:
            text: The text to analyze
            keywords: List of keywords to match
            
        Returns:
            Relevance score (0-1)
        """
        if not text or not keywords:
            return 0.0
        
        # Prepare text
        text = text.lower()
        
        # Count keyword matches
        matches = 0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            # Check for exact matches
            if keyword.lower() in text:
                matches += 1
        
        # Calculate base score
        if total_keywords == 0:
            return 0.0
            
        base_score = matches / total_keywords
        
        # Boost score if multiple matches
        if matches > 1:
            boost = min(0.3, 0.1 * (matches - 1))
            return min(1.0, base_score + boost)
            
        return base_score
    
    async def _mock_collect(self) -> List[Dict[str, Any]]:
        """Collect mock news data for testing.
        
        Returns:
            List of mock collected data
        """
        self.logger.info("Collecting mock news data")
        
        # Simulate a delay
        await asyncio.sleep(0.2)
        
        collected_data = []
        
        # Generate mock articles from each source
        for source in self.sources:
            # Generate 3-5 articles per source
            num_articles = 3 + (hash(source) % 3)
            
            for i in range(num_articles):
                # Create mock article
                article_id = f"mock_{source}_{i}"
                
                # Choose a keyword to focus on
                keyword = self.keywords[i % len(self.keywords)]
                
                # Create title and content based on source and keyword
                if "regulation" in keyword.lower():
                    title = f"New {['SEC', 'EU', 'global'][i % 3]} crypto regulations could {['impact', 'change', 'transform'][i % 3]} the market"
                    content = f"Regulatory bodies are planning new {['oversight', 'frameworks', 'guidelines'][i % 3]} for cryptocurrency exchanges and traders. This could lead to significant changes in how digital assets are traded and held."
                elif "fed" in keyword.lower() or "central bank" in keyword.lower():
                    title = f"Central Bank {['announces', 'considers', 'plans'][i % 3]} new {['interest rate', 'monetary policy', 'digital currency'][i % 3]}"
                    content = f"The Federal Reserve is {['raising', 'evaluating', 'implementing'][i % 3]} changes to its policy that could affect crypto markets. Analysts expect volatility in response to this development."
                elif "sec" in keyword.lower():
                    title = f"SEC {['approves', 'rejects', 'delays decision on'][i % 3]} {['Bitcoin ETF', 'crypto regulations', 'exchange licenses'][i % 3]}"
                    content = f"The Securities and Exchange Commission has {['announced', 'published', 'delayed'][i % 3]} its decision regarding cryptocurrency {['ETFs', 'regulations', 'compliance requirements'][i % 3]}. Market participants are {['optimistic', 'concerned', 'divided'][i % 3]} about the implications."
                else:
                    title = f"{['Breaking', 'Exclusive', 'Analysis'][i % 3]}: {keyword.title()} {['developments', 'news', 'updates'][i % 3]} and market impact"
                    content = f"Recent {['changes', 'announcements', 'developments'][i % 3]} related to {keyword} are causing {['excitement', 'concern', 'debate'][i % 3]} in the cryptocurrency community. Experts predict {['positive', 'negative', 'mixed'][i % 3]} effects on market prices."
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(title + " " + content, self.keywords)
                
                # Adjust publication time
                pub_time = datetime.now() - timedelta(hours=i * 3)
                
                # Create source
                source_url = f"https://{source}.com/article/{article_id}"
                source_obj = EventSource(
                    id=f"mock_{source}_{article_id}",
                    type=SourceType.NEWS,
                    name=source,
                    url=source_url,
                    reliability_score=0.7
                )
                
                collected_data.append({
                    "source": source_obj,
                    "title": title,
                    "content": content,
                    "timestamp": pub_time,
                    "metadata": {
                        "author": f"Mock Author {i}",
                        "source_name": source,
                        "categories": ["Cryptocurrency", "Regulation", "Markets"][i % 3:i % 3 + 2],
                        "relevance_score": relevance_score,
                        "api_source": "Mock"
                    }
                })
        
        self.logger.info(f"Generated {len(collected_data)} mock news items")
        return collected_data