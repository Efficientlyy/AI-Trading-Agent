"""CryptoCompare News API client.

This module provides a client for the CryptoCompare News API service, which
allows fetching cryptocurrency news articles from various sources.
"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from src.common.logging import get_logger
from src.common.config import config
from src.analysis_agents.news.news_analyzer import NewsArticle


class CryptoCompareNewsClient:
    """Client for the CryptoCompare News API.
    
    This client provides methods for fetching cryptocurrency news
    articles from the CryptoCompare News API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the CryptoCompare News API client.
        
        Args:
            api_key: API key for CryptoCompare. If not provided, it will be
                loaded from environment variables or configuration.
        """
        self.logger = get_logger("clients", "cryptocompare_news")
        
        # Set API key
        self.api_key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY") or config.get("apis.cryptocompare.key", "")
        
        if not self.api_key:
            self.logger.warning("No CryptoCompare API key provided - API requests will fail")
            
        # Set base URL
        self.base_url = "https://min-api.cryptocompare.com/data/v2/news"
        
        # Cache responses to avoid excessive API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1 hour
    
    async def get_latest_news(
        self, 
        lang: str = "EN", 
        feeds: Optional[str] = None,
        categories: Optional[str] = None,
        exclude_categories: Optional[str] = None,
        limit_timestamp: Optional[int] = None,
        sort_order: str = "latest",
        limit: int = 50,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get latest cryptocurrency news articles.
        
        Args:
            lang: Language filter (e.g., 'EN', 'PT')
            feeds: Comma-separated list of news sources
            categories: Comma-separated list of categories
            exclude_categories: Comma-separated list of categories to exclude
            limit_timestamp: Timestamp to limit results (older than)
            sort_order: Sort order ('latest' by default)
            limit: Maximum number of articles to return
            use_cache: Whether to use cached results if available
            
        Returns:
            List of news article dictionaries
        """
        if not self.api_key:
            self.logger.error("Cannot fetch news: No API key provided")
            return []
            
        # Generate cache key
        cache_key = f"latest_news_{lang}_{feeds}_{categories}_{exclude_categories}_{limit_timestamp}_{sort_order}_{limit}"
        
        # Check cache
        if use_cache and cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > datetime.now().timestamp():
            self.logger.debug("Using cached results for latest news")
            return self.cache[cache_key]
        
        # Prepare parameters
        params = {
            "lang": lang,
            "sortOrder": sort_order,
            "extraParams": "AI-Trading-Agent"
        }
        
        # Add optional parameters
        if feeds:
            params["feeds"] = feeds
        
        if categories:
            params["categories"] = categories
            
        if exclude_categories:
            params["excludeCategories"] = exclude_categories
            
        if limit_timestamp:
            params["lTs"] = limit_timestamp
        
        # Make request
        url = f"{self.base_url}/feeds"
        
        try:
            self.logger.debug("Fetching latest news", lang=lang, categories=categories)
            
            headers = {
                "authorization": f"Apikey {self.api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        json_data = await response.json()
                        
                        if json_data.get("Type") == 100:  # Success
                            articles = json_data.get("Data", [])
                            
                            # Limit the number of articles
                            articles = articles[:limit]
                            
                            # Cache results
                            if use_cache:
                                self.cache[cache_key] = articles
                                self.cache_expiry[cache_key] = (datetime.now() + timedelta(seconds=self.cache_duration)).timestamp()
                            
                            self.logger.info("Successfully fetched news articles", count=len(articles))
                            return articles
                        else:
                            error = json_data.get("Message", "Unknown error")
                            self.logger.error("API error", error=error)
                            return []
                    else:
                        self.logger.error("Failed to fetch news", 
                                       status=response.status, 
                                       reason=response.reason)
                        
                        # Check for rate limit issues
                        if response.status == 429:
                            self.logger.warning("Rate limit exceeded - backing off")
                            await asyncio.sleep(2)
                            
                        # Check for auth issues
                        elif response.status == 401:
                            self.logger.error("API key invalid or expired")
                            
                        return []
                        
        except Exception as e:
            self.logger.error("Error fetching news", error=str(e))
            return []
    
    async def get_categories(
        self, 
        lang: str = "EN",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get available news categories.
        
        Args:
            lang: Language filter (e.g., 'EN', 'PT')
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary of categories
        """
        if not self.api_key:
            self.logger.error("Cannot fetch categories: No API key provided")
            return {}
            
        # Generate cache key
        cache_key = f"categories_{lang}"
        
        # Check cache
        if use_cache and cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > datetime.now().timestamp():
            self.logger.debug("Using cached results for categories")
            return self.cache[cache_key]
        
        # Prepare parameters
        params = {
            "lang": lang,
            "extraParams": "AI-Trading-Agent"
        }
        
        # Make request
        url = f"{self.base_url}/categories"
        
        try:
            self.logger.debug("Fetching news categories", lang=lang)
            
            headers = {
                "authorization": f"Apikey {self.api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        json_data = await response.json()
                        
                        if json_data.get("Response") == "Success":
                            categories = json_data.get("Data", {})
                            
                            # Cache results
                            if use_cache:
                                self.cache[cache_key] = categories
                                self.cache_expiry[cache_key] = (datetime.now() + timedelta(seconds=self.cache_duration)).timestamp()
                            
                            self.logger.info("Successfully fetched categories", count=len(categories))
                            return categories
                        else:
                            error = json_data.get("Message", "Unknown error")
                            self.logger.error("API error", error=error)
                            return {}
                    else:
                        self.logger.error("Failed to fetch categories", 
                                       status=response.status, 
                                       reason=response.reason)
                        return {}
                        
        except Exception as e:
            self.logger.error("Error fetching categories", error=str(e))
            return {}
    
    async def get_feeds_list(
        self, 
        lang: str = "EN",
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get available news feeds.
        
        Args:
            lang: Language filter (e.g., 'EN', 'PT')
            use_cache: Whether to use cached results if available
            
        Returns:
            List of news feeds
        """
        if not self.api_key:
            self.logger.error("Cannot fetch feeds: No API key provided")
            return []
            
        # Generate cache key
        cache_key = f"feeds_list_{lang}"
        
        # Check cache
        if use_cache and cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > datetime.now().timestamp():
            self.logger.debug("Using cached results for feeds list")
            return self.cache[cache_key]
        
        # Prepare parameters
        params = {
            "lang": lang,
            "extraParams": "AI-Trading-Agent"
        }
        
        # Make request
        url = f"{self.base_url}/feedslist"
        
        try:
            self.logger.debug("Fetching news feeds", lang=lang)
            
            headers = {
                "authorization": f"Apikey {self.api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        json_data = await response.json()
                        
                        if json_data.get("Response") == "Success":
                            feeds = json_data.get("Data", [])
                            
                            # Cache results
                            if use_cache:
                                self.cache[cache_key] = feeds
                                self.cache_expiry[cache_key] = (datetime.now() + timedelta(seconds=self.cache_duration)).timestamp()
                            
                            self.logger.info("Successfully fetched feeds", count=len(feeds))
                            return feeds
                        else:
                            error = json_data.get("Message", "Unknown error")
                            self.logger.error("API error", error=error)
                            return []
                    else:
                        self.logger.error("Failed to fetch feeds", 
                                       status=response.status, 
                                       reason=response.reason)
                        return []
                        
        except Exception as e:
            self.logger.error("Error fetching feeds", error=str(e))
            return []
    
    async def convert_to_news_articles(
        self, 
        api_articles: List[Dict[str, Any]], 
        prefix: str = "cryptocompare"
    ) -> List[NewsArticle]:
        """Convert CryptoCompare news articles to internal NewsArticle objects.
        
        Args:
            api_articles: List of articles from CryptoCompare
            prefix: Prefix for article IDs
            
        Returns:
            List of NewsArticle objects
        """
        news_articles = []
        
        for i, article in enumerate(api_articles):
            try:
                # Generate a unique ID
                article_id = f"{prefix}_{article.get('id', i)}"
                
                # Parse publication date
                published_at = datetime.now()
                if article.get("published_on"):
                    try:
                        published_at = datetime.fromtimestamp(article["published_on"])
                    except (ValueError, TypeError):
                        self.logger.warning("Failed to parse publication date", date=article.get("published_on"))
                
                # Extract categories and tags
                categories = article.get("categories", "").split("|") if article.get("categories") else []
                tags = article.get("tags", "").split("|") if article.get("tags") else []
                
                # Create NewsArticle object
                news_article = NewsArticle(
                    article_id=article_id,
                    title=article.get("title", ""),
                    content=article.get("body", ""),
                    url=article.get("url", ""),
                    source=article.get("source", "CryptoCompare"),
                    published_at=published_at,
                    author=None,  # CryptoCompare doesn't provide author information
                    categories=categories,
                    tags=tags
                )
                
                news_articles.append(news_article)
                
            except Exception as e:
                self.logger.error("Error converting article", error=str(e))
                continue
        
        return news_articles
    
    async def search_crypto_news(
        self, 
        asset: str,
        categories: Optional[str] = None,
        days: int = 7,
        limit: int = 50,
        use_cache: bool = True
    ) -> List[NewsArticle]:
        """Search for cryptocurrency news for a specific asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            categories: Optional categories to filter by
            days: Number of days to search back
            limit: Maximum number of articles to return
            use_cache: Whether to use cached results if available
            
        Returns:
            List of NewsArticle objects
        """
        # Calculate the timestamp limit
        limit_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
        
        # Get news with the asset tag included
        # CryptoCompare automatically tags articles with relevant cryptocurrencies
        api_articles = await self.get_latest_news(
            lang="EN",
            categories=categories,
            limit_timestamp=limit_timestamp,
            sort_order="latest",
            limit=limit,
            use_cache=use_cache
        )
        
        # Filter articles that contain the asset in tags
        filtered_articles = []
        for article in api_articles:
            tags = article.get("tags", "").lower()
            
            # Map common asset names
            tag_map = {
                "btc": ["btc", "bitcoin"],
                "eth": ["eth", "ethereum"],
                "sol": ["sol", "solana"],
                "xrp": ["xrp", "ripple"]
            }
            
            # Get the search terms for this asset
            search_terms = tag_map.get(asset.lower(), [asset.lower()])
            
            # Check if any search term is in the tags
            if any(term in tags for term in search_terms):
                filtered_articles.append(article)
        
        # Convert to NewsArticle objects
        news_articles = await self.convert_to_news_articles(filtered_articles)
        
        self.logger.info("Fetched crypto news articles", asset=asset, count=len(news_articles))
        return news_articles
    
    async def get_articles(
        self, 
        timeframe: str = "24h",
        assets: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """Get news articles for specified assets and timeframe.
        
        This method is compatible with the interface expected by the NewsAnalyzer.
        
        Args:
            timeframe: Time period to collect articles for (e.g., "24h", "7d")
            assets: List of assets to collect news for
            
        Returns:
            List of collected NewsArticle objects
        """
        if not assets:
            assets = ["BTC", "ETH", "SOL", "XRP"]
            
        # Parse timeframe to days
        days = 1
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            days = max(1, hours // 24)
        elif timeframe.endswith("d"):
            days = int(timeframe[:-1])
            
        # Get articles for each asset
        all_articles = []
        
        for asset in assets:
            try:
                asset_articles = await self.search_crypto_news(
                    asset=asset,
                    days=days,
                    limit=50
                )
                
                all_articles.extend(asset_articles)
                
            except Exception as e:
                self.logger.error(f"Error getting articles for {asset}: {e}")
                
        return all_articles


class MockCryptoCompareNewsClient:
    """Mock CryptoCompare News client for testing and fallback."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the mock client."""
        self.logger = get_logger("clients", "mock_cryptocompare")
        
    async def get_articles(
        self, 
        timeframe: str = "24h",
        assets: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """Get mock news articles.
        
        Args:
            timeframe: Time period to fetch articles for
            assets: List of assets to fetch articles about
            
        Returns:
            List of mock NewsArticle objects
        """
        # Parse timeframe
        days = 1
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            days = max(1, hours // 24)
        elif timeframe.endswith("d"):
            days = int(timeframe[:-1])
        
        # Generate mock articles
        articles = []
        assets = assets or ["BTC", "ETH", "SOL", "XRP"]
        
        # Get current time
        now = datetime.now()
        
        # Mock news sources
        sources = ["CoinDesk", "Cointelegraph", "Decrypt", "CryptoSlate", "The Block"]
        
        # Asset-specific templates
        templates = {
            "BTC": [
                {
                    "title": "Bitcoin's Price Analysis: BTC Forms $X Pattern",
                    "content": "Bitcoin's price action has shown a specific pattern that traders are closely watching. Technical analysts suggest this could indicate a potential breakout in the coming days.",
                    "tags": ["BTC", "Bitcoin", "Technical Analysis"]
                },
                {
                    "title": "Institutional Interest in Bitcoin Grows as $X Million Enters Market",
                    "content": "New data shows institutional investors continue to accumulate Bitcoin as inflows reach significant levels. This growing interest from traditional finance may signal a maturing market.",
                    "tags": ["BTC", "Bitcoin", "Institutional"]
                },
                {
                    "title": "Bitcoin Mining Difficulty Adjusts by X% Following Hash Rate Changes",
                    "content": "The Bitcoin network has undergone its scheduled difficulty adjustment, reflecting recent changes in mining hash rate. This self-regulating mechanism helps maintain Bitcoin's security and block time consistency.",
                    "tags": ["BTC", "Bitcoin", "Mining"]
                }
            ],
            "ETH": [
                {
                    "title": "Ethereum Development Update: New EIPs Proposed for Upcoming Hard Fork",
                    "content": "Ethereum developers have proposed several new Ethereum Improvement Proposals (EIPs) for consideration in the next network upgrade. These changes aim to enhance scalability and reduce gas costs.",
                    "tags": ["ETH", "Ethereum", "Development"]
                },
                {
                    "title": "Ethereum DeFi Ecosystem Reaches $X Billion in Total Value Locked",
                    "content": "Ethereum's decentralized finance applications have collectively reached a new milestone in total value locked. This growth demonstrates the continued expansion of DeFi despite market fluctuations.",
                    "tags": ["ETH", "Ethereum", "DeFi"]
                },
                {
                    "title": "Ethereum Layer 2 Solutions See Surge in User Activity",
                    "content": "Several Ethereum Layer 2 scaling solutions are reporting significant increases in user activity and transaction volume. This adoption helps alleviate congestion on the Ethereum mainnet while maintaining security.",
                    "tags": ["ETH", "Ethereum", "Layer2", "Scaling"]
                }
            ],
            "SOL": [
                {
                    "title": "Solana Ecosystem Expands with X New Projects in Recent Weeks",
                    "content": "The Solana ecosystem continues to grow with numerous projects launching on the high-performance blockchain. Developers cite Solana's low transaction costs and high throughput as key advantages.",
                    "tags": ["SOL", "Solana", "Ecosystem"]
                },
                {
                    "title": "Solana's Validator Count Reaches New High of X",
                    "content": "Solana's network has reached a new milestone in decentralization with an increased number of validators securing the blockchain. This growth helps improve the network's resilience and security.",
                    "tags": ["SOL", "Solana", "Validators"]
                },
                {
                    "title": "Solana Foundation Announces $X Million Developer Fund",
                    "content": "The Solana Foundation has announced a new initiative to support developers building on the Solana blockchain. The fund aims to accelerate growth across various sectors including DeFi, NFTs, and gaming.",
                    "tags": ["SOL", "Solana", "Development"]
                }
            ],
            "XRP": [
                {
                    "title": "Ripple Partners with Financial Institution X for Cross-Border Payments",
                    "content": "Ripple has announced a new partnership with a major financial institution to implement its blockchain technology for cross-border payments. This collaboration aims to reduce settlement times and costs.",
                    "tags": ["XRP", "Ripple", "Partnership"]
                },
                {
                    "title": "XRP Ledger Upgrade Introduces New Features for Developers",
                    "content": "The XRP Ledger has been upgraded with new features designed to enhance its functionality and developer experience. These improvements include enhanced smart contract capabilities and DEX features.",
                    "tags": ["XRP", "XRPL", "Development"]
                },
                {
                    "title": "Ripple's Legal Case Update: Key Developments in Ongoing Proceedings",
                    "content": "Ripple's legal case has seen new developments that may impact the company's future and XRP's regulatory status. Legal experts weigh in on the potential outcomes and their market implications.",
                    "tags": ["XRP", "Ripple", "Regulation"]
                }
            ]
        }
        
        # Create mock articles for each asset
        article_id_counter = 1
        
        for asset in assets:
            asset_templates = templates.get(asset, [])
            if not asset_templates:
                continue
                
            # Generate 2-4 articles per asset
            for _ in range(min(len(asset_templates), 3)):
                template = asset_templates[_ % len(asset_templates)]
                
                # Randomize publication time within timeframe
                hours_ago = days * 24
                random_hours = int(hours_ago * ((article_id_counter % 10) / 10))
                published_at = now - timedelta(hours=random_hours)
                
                # Randomize source
                source_index = article_id_counter % len(sources)
                source = sources[source_index]
                
                # Create article ID
                article_id = f"cryptocompare_mock_{article_id_counter}"
                article_id_counter += 1
                
                # Create categories from tags
                tags = template["tags"]
                if "Technical Analysis" in tags:
                    categories = ["Trading", "Analysis"]
                elif "Development" in tags:
                    categories = ["Technology", "Development"]
                elif "Partnership" in tags:
                    categories = ["Business", "Adoption"]
                else:
                    categories = ["General", "News"]
                
                # Create mock article
                article = NewsArticle(
                    article_id=article_id,
                    title=template["title"],
                    content=template["content"],
                    url=f"https://cryptocompare.com/news/{article_id}",
                    source=source,
                    published_at=published_at,
                    author=None,
                    categories=categories,
                    tags=tags
                )
                
                articles.append(article)
        
        self.logger.info("Generated mock crypto news articles", count=len(articles))
        return articles