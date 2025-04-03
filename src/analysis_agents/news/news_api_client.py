"""NewsAPI client for fetching news articles from NewsAPI.org.

This module provides a client for the NewsAPI.org service, which allows
fetching news articles from various sources around the world.
"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging

from src.common.logging import get_logger
from src.common.config import config
from src.analysis_agents.news.news_analyzer import NewsArticle


class NewsAPIClient:
    """Client for the NewsAPI.org service.
    
    This client provides methods for fetching news articles from 
    NewsAPI.org, a comprehensive API for accessing news from thousands
    of sources worldwide.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the NewsAPI client.
        
        Args:
            api_key: API key for NewsAPI.org. If not provided, it will be
                loaded from environment variables or configuration.
        """
        self.logger = get_logger("clients", "news_api")
        
        # Set API key
        self.api_key = api_key or os.getenv("NEWS_API_KEY") or config.get("apis.news_api.key", "")
        
        if not self.api_key:
            self.logger.warning("No NewsAPI key provided - API requests will fail")
            
        # Set base URL
        self.base_url = "https://newsapi.org/v2"
        
        # Cache responses to avoid excessive API calls
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1 hour
    
    async def get_everything(
        self, 
        q: str, 
        language: str = "en", 
        sort_by: str = "publishedAt", 
        page_size: int = 20,
        page: int = 1,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        domains: Optional[str] = None,
        sources: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for news articles.
        
        Args:
            q: Search query
            language: Article language (e.g., "en", "fr", "de")
            sort_by: Sort order ("relevancy", "popularity", "publishedAt")
            page_size: Number of articles per page (max 100)
            page: Page number for pagination
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            domains: Comma-separated list of domains to restrict search to
            sources: Comma-separated list of news sources to restrict search to
            use_cache: Whether to use cached results if available
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            self.logger.error("Cannot fetch articles: No API key provided")
            return []
            
        # Generate cache key
        cache_key = f"everything_{q}_{language}_{sort_by}_{page_size}_{page}_{from_date}_{to_date}_{domains}_{sources}"
        
        # Check cache
        if use_cache and cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > datetime.now().timestamp():
            self.logger.debug("Using cached results", query=q)
            return self.cache[cache_key]
        
        # Prepare parameters
        params = {
            "q": q,
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),  # API limit
            "page": page,
            "apiKey": self.api_key
        }
        
        # Add optional parameters
        if from_date:
            params["from"] = from_date
        
        if to_date:
            params["to"] = to_date
            
        if domains:
            params["domains"] = domains
            
        if sources:
            params["sources"] = sources
        
        # Make request
        url = f"{self.base_url}/everything"
        
        try:
            self.logger.debug("Fetching news articles", query=q, page=page)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        json_data = response.json()
                        
                        if json_data.get("status") == "ok":
                            articles = json_data.get("articles", [])
                            
                            # Cache results
                            if use_cache:
                                self.cache[cache_key] = articles
                                self.cache_expiry[cache_key] = (datetime.now() + timedelta(seconds=self.cache_duration)).timestamp()
                            
                            self.logger.info("Successfully fetched articles", count=len(articles), query=q)
                            return articles
                        else:
                            error = json_data.get("message", "Unknown error")
                            self.logger.error("API error", error=error, query=q)
                            return []
                    else:
                        self.logger.error("Failed to fetch articles", 
                                        status=response.status, 
                                        reason=response.reason,
                                        query=q)
                        
                        # Check for rate limit issues
                        if response.status == 429:
                            self.logger.warning("Rate limit exceeded - backing off")
                            await asyncio.sleep(2)
                            
                        # Check for auth issues
                        elif response.status == 401:
                            self.logger.error("API key invalid or expired")
                            
                        return []
                        
        except Exception as e:
            self.logger.error("Error fetching articles", error=str(e), query=q)
            return []
    
    async def get_top_headlines(
        self, 
        country: Optional[str] = None,
        category: Optional[str] = None,
        q: Optional[str] = None,
        sources: Optional[str] = None,
        page_size: int = 20,
        page: int = 1,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get top headlines.
        
        Args:
            country: 2-letter ISO 3166-1 country code
            category: Category (business, entertainment, health, science, sports, technology)
            q: Search query
            sources: Comma-separated list of news source IDs
            page_size: Number of articles per page (max 100)
            page: Page number for pagination
            use_cache: Whether to use cached results if available
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            self.logger.error("Cannot fetch headlines: No API key provided")
            return []
            
        # Generate cache key
        cache_key = f"headlines_{country}_{category}_{q}_{sources}_{page_size}_{page}"
        
        # Check cache
        if use_cache and cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > datetime.now().timestamp():
            self.logger.debug("Using cached results", country=country, category=category)
            return self.cache[cache_key]
        
        # Prepare parameters
        params = {
            "pageSize": min(page_size, 100),  # API limit
            "page": page,
            "apiKey": self.api_key
        }
        
        # Add optional parameters
        if country:
            params["country"] = country
        
        if category:
            params["category"] = category
            
        if q:
            params["q"] = q
            
        if sources:
            params["sources"] = sources
        
        # Note: country/category and sources are mutually exclusive
        if sources and (country or category):
            self.logger.warning("Sources parameter cannot be mixed with country or category parameters - ignoring country/category")
            if "country" in params:
                del params["country"]
            if "category" in params:
                del params["category"]
        
        # Make request
        url = f"{self.base_url}/top-headlines"
        
        try:
            self.logger.debug("Fetching top headlines", country=country, category=category, query=q)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        json_data = response.json()
                        
                        if json_data.get("status") == "ok":
                            articles = json_data.get("articles", [])
                            
                            # Cache results
                            if use_cache:
                                self.cache[cache_key] = articles
                                self.cache_expiry[cache_key] = (datetime.now() + timedelta(seconds=self.cache_duration)).timestamp()
                            
                            self.logger.info("Successfully fetched headlines", count=len(articles))
                            return articles
                        else:
                            error = json_data.get("message", "Unknown error")
                            self.logger.error("API error", error=error)
                            return []
                    else:
                        self.logger.error("Failed to fetch headlines", 
                                        status=response.status, 
                                        reason=response.reason)
                        return []
                        
        except Exception as e:
            self.logger.error("Error fetching headlines", error=str(e))
            return []
    
    async def get_sources(
        self, 
        category: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get available news sources.
        
        Args:
            category: Source category
            language: Source language
            country: Source country
            use_cache: Whether to use cached results if available
            
        Returns:
            List of source dictionaries
        """
        if not self.api_key:
            self.logger.error("Cannot fetch sources: No API key provided")
            return []
            
        # Generate cache key
        cache_key = f"sources_{category}_{language}_{country}"
        
        # Check cache
        if use_cache and cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > datetime.now().timestamp():
            self.logger.debug("Using cached results for sources")
            return self.cache[cache_key]
        
        # Prepare parameters
        params = {
            "apiKey": self.api_key
        }
        
        # Add optional parameters
        if category:
            params["category"] = category
        
        if language:
            params["language"] = language
            
        if country:
            params["country"] = country
        
        # Make request
        url = f"{self.base_url}/sources"
        
        try:
            self.logger.debug("Fetching news sources")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        json_data = response.json()
                        
                        if json_data.get("status") == "ok":
                            sources = json_data.get("sources", [])
                            
                            # Cache results
                            if use_cache:
                                self.cache[cache_key] = sources
                                self.cache_expiry[cache_key] = (datetime.now() + timedelta(seconds=self.cache_duration)).timestamp()
                            
                            self.logger.info("Successfully fetched sources", count=len(sources))
                            return sources
                        else:
                            error = json_data.get("message", "Unknown error")
                            self.logger.error("API error", error=error)
                            return []
                    else:
                        self.logger.error("Failed to fetch sources", 
                                        status=response.status, 
                                        reason=response.reason)
                        return []
                        
        except Exception as e:
            self.logger.error("Error fetching sources", error=str(e))
            return []
    
    async def convert_to_news_articles(
        self, 
        api_articles: List[Dict[str, Any]], 
        prefix: str = "newsapi"
    ) -> List[NewsArticle]:
        """Convert NewsAPI articles to internal NewsArticle objects.
        
        Args:
            api_articles: List of articles from NewsAPI
            prefix: Prefix for article IDs
            
        Returns:
            List of NewsArticle objects
        """
        news_articles = []
        
        for i, article in enumerate(api_articles):
            try:
                # Generate a unique ID
                article_id = f"{prefix}_{datetime.now().strftime('%Y%m%d')}_{i}"
                
                # Parse publication date
                published_at = datetime.now()
                if article.get("publishedAt"):
                    try:
                        published_at = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        self.logger.warning("Failed to parse publication date", date=article.get("publishedAt"))
                
                # Create NewsArticle object
                news_article = NewsArticle(
                    article_id=article_id,
                    title=article.get("title", ""),
                    content=article.get("content", article.get("description", "")),
                    url=article.get("url", ""),
                    source=article.get("source", {}).get("name", "NewsAPI"),
                    published_at=published_at,
                    author=article.get("author"),
                    categories=[],  # NewsAPI doesn't provide categories
                    tags=[]  # NewsAPI doesn't provide tags
                )
                
                news_articles.append(news_article)
                
            except Exception as e:
                self.logger.error("Error converting article", error=str(e))
                continue
        
        return news_articles
    
    async def search_crypto_news(
        self, 
        asset: str,
        days: int = 7,
        page_size: int = 20,
        use_cache: bool = True
    ) -> List[NewsArticle]:
        """Search for cryptocurrency news for a specific asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            days: Number of days to search back
            page_size: Number of articles per page
            use_cache: Whether to use cached results if available
            
        Returns:
            List of NewsArticle objects
        """
        # Prepare search query
        query_terms = []
        
        # Add asset-specific terms
        if asset == "BTC":
            query_terms = ["Bitcoin", "BTC", "crypto"]
        elif asset == "ETH":
            query_terms = ["Ethereum", "ETH", "crypto"]
        elif asset == "SOL":
            query_terms = ["Solana", "SOL", "crypto"]
        elif asset == "XRP":
            query_terms = ["Ripple", "XRP", "crypto"]
        else:
            query_terms = [asset, "crypto", "cryptocurrency"]
        
        query = " OR ".join(query_terms)
        
        # Calculate date range
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Fetch articles
        api_articles = await self.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=page_size,
            from_date=from_date,
            to_date=to_date,
            use_cache=use_cache
        )
        
        # Convert to NewsArticle objects
        news_articles = await self.convert_to_news_articles(api_articles)
        
        self.logger.info("Fetched crypto news articles", asset=asset, count=len(news_articles))
        return news_articles