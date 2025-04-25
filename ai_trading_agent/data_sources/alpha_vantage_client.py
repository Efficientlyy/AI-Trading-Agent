"""
Alpha Vantage API client for fetching news sentiment data.

This module provides a client for the Alpha Vantage News Sentiment API,
which can be used to fetch sentiment data for various topics and tickers.
"""

import os
import requests
import logging
import json
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Client for the Alpha Vantage News Sentiment API."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Default fallback topics when specific queries fail
    DEFAULT_FALLBACK_TOPICS = ["blockchain", "financial_markets", "technology"]
    
    # Available topics from Alpha Vantage documentation
    AVAILABLE_TOPICS = [
        "blockchain",
        "earnings",
        "ipo",
        "mergers_and_acquisitions",
        "financial_markets",
        "economy_fiscal",
        "economy_monetary",
        "economy_macro",
        "energy_transportation",
        "finance",
        "life_sciences",
        "manufacturing",
        "real_estate",
        "retail_wholesale",
        "technology"
    ]
    
    # Premium tier settings
    TIER_SETTINGS = {
        "free": {
            "requests_per_minute": 5,
            "max_results": 50
        },
        "premium": {
            "requests_per_minute": 75,  # Typical premium tier limit
            "max_results": 1000
        },
        "enterprise": {
            "requests_per_minute": 300,  # Enterprise tier limit
            "max_results": 1000
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, tier: str = "free"):
        """
        Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key. If None, will try to load from environment variable.
            tier: API tier ('free', 'premium', or 'enterprise'). Affects rate limiting and features.
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.warning("No Alpha Vantage API key provided. API calls will fail.")
        
        # Set tier and corresponding rate limits
        self.tier = tier.lower()
        if self.tier not in self.TIER_SETTINGS:
            logger.warning(f"Unknown tier '{tier}'. Defaulting to 'free'.")
            self.tier = "free"
        
        tier_config = self.TIER_SETTINGS[self.tier]
        self.requests_per_minute = tier_config["requests_per_minute"]
        self.max_results = tier_config["max_results"]
        
        # Rate limiting parameters
        self.last_request_time = 0
        
        # Cache for API responses
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL by default
        self.use_cache = True
        
    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid API throttling."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If less than the minimum interval has passed, sleep
        min_interval = 60 / self.requests_per_minute  # seconds
        if elapsed < min_interval and self.last_request_time > 0:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate a cache key from request parameters."""
        # Sort params to ensure consistent key generation
        sorted_params = sorted(params.items())
        return json.dumps(sorted_params)
    
    def _is_cache_valid(self, cache_time: float) -> bool:
        """Check if a cached item is still valid."""
        return time.time() - cache_time < self.cache_ttl
    
    def get_news_sentiment(self, 
                          topics: Optional[List[str]] = None,
                          tickers: Optional[List[str]] = None,
                          time_from: Optional[str] = None,
                          time_to: Optional[str] = None,
                          sort: str = "RELEVANCE",
                          limit: int = 50,
                          use_cache: bool = True,
                          attempt_fallback: bool = True) -> Dict[str, Any]:
        """
        Fetch news sentiment data from Alpha Vantage.
        
        Args:
            topics: List of topics to filter news by (e.g., ["blockchain", "technology"])
            tickers: List of ticker symbols to filter news by (e.g., ["BTC", "ETH"])
            time_from: Start time for news articles (format: YYYYMMDDTHHMM)
            time_to: End time for news articles (format: YYYYMMDDTHHMM)
            sort: Sort order for results ("RELEVANCE", "LATEST", "EARLIEST")
            limit: Maximum number of results to return
            use_cache: Whether to use cached results if available
            attempt_fallback: Whether to attempt fallback queries if the primary query fails
            
        Returns:
            Dictionary containing sentiment data
        """
        # Apply tier-specific limits
        limit = min(limit, self.max_results)
        
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "sort": sort,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if topics:
            # Validate topics against available topics
            validated_topics = self._validate_topics(topics)
            if validated_topics:
                params["topics"] = ",".join(validated_topics)
            else:
                logger.warning(f"No valid topics found in {topics}. Using default topics.")
                params["topics"] = ",".join(self.DEFAULT_FALLBACK_TOPICS)
        if tickers:
            # Format tickers properly (ensure CRYPTO: prefix for crypto tickers)
            formatted_tickers = []
            for ticker in tickers:
                if ticker.upper() in ["BTC", "ETH", "XRP", "ADA", "SOL", "DOT", "DOGE", "SHIB", "AVAX", "LINK", "UNI", "AAVE", "COMP", "MKR"] and not ticker.startswith("CRYPTO:"):
                    formatted_tickers.append(f"CRYPTO:{ticker}")
                else:
                    formatted_tickers.append(ticker)
            params["tickers"] = ",".join(formatted_tickers)
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        
        # Check cache first if enabled
        cache_key = self._get_cache_key(params)
        if use_cache and self.use_cache and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry["timestamp"]):
                logger.info("Using cached sentiment data")
                return cache_entry["data"]
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        # Enforce rate limiting before making the request
        self._enforce_rate_limit()
        
        try:
            logger.info(f"Fetching news sentiment data from Alpha Vantage with params: {params}")
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            
            # Check for error messages in the response
            if "Error Message" in data:
                error_msg = data["Error Message"]
                logger.error(f"Alpha Vantage API error: {error_msg}")
                
                # Try fallback if enabled and this was a ticker-specific query
                if attempt_fallback and tickers and not topics:
                    logger.info("Attempting fallback to topic-based query")
                    return self._attempt_fallback_query(tickers, time_from, time_to, sort, limit)
                    
                return {"error": error_msg, "data": None}
            
            if "Information" in data and "feed" not in data:
                info_msg = data["Information"]
                logger.warning(f"Alpha Vantage API information: {info_msg}")
                
                # Try fallback if enabled and this looks like a rate limit or data availability issue
                if attempt_fallback and ("limit" in info_msg.lower() or "available" in info_msg.lower()):
                    logger.info("Attempting fallback due to API limitations")
                    return self._attempt_fallback_query(tickers, time_from, time_to, sort, limit)
                    
                return {"error": info_msg, "data": None}
            
            # Check if the response has actual data
            if "feed" not in data or not data.get("feed"):
                logger.warning("No feed data in response")
                
                # Try fallback if enabled and we got an empty response
                if attempt_fallback:
                    logger.info("Attempting fallback due to empty response")
                    return self._attempt_fallback_query(tickers, time_from, time_to, sort, limit)
                    
                return {"error": "No data available", "data": data}
            
            # Cache successful response
            if self.use_cache:
                self.cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": {"error": None, "data": data}
                }
            
            return {"error": None, "data": data}
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching news sentiment data: {str(e)}"
            logger.error(error_msg)
            
            # Try fallback if enabled and this was a network error
            if attempt_fallback:
                logger.info("Attempting fallback due to request exception")
                return self._attempt_fallback_query(tickers, time_from, time_to, sort, limit)
                
            return {"error": error_msg, "data": None}
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON response: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "data": None}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "data": None}
    
    def _validate_topics(self, topics: List[str]) -> List[str]:
        """Validate topics against the available topics list."""
        valid_topics = []
        for topic in topics:
            if topic in self.AVAILABLE_TOPICS:
                valid_topics.append(topic)
            else:
                # Find closest match if topic is not valid
                logger.warning(f"Topic '{topic}' is not in the list of available topics. Using closest match.")
                # Simple matching algorithm - could be improved with fuzzy matching
                for available_topic in self.AVAILABLE_TOPICS:
                    if topic.lower() in available_topic.lower() or available_topic.lower() in topic.lower():
                        logger.info(f"Using '{available_topic}' as a replacement for '{topic}'")
                        valid_topics.append(available_topic)
                        break
                else:
                    # If no match found, use the first default topic
                    logger.warning(f"No match found for '{topic}'. Using default topic.")
                    if self.DEFAULT_FALLBACK_TOPICS:
                        valid_topics.append(self.DEFAULT_FALLBACK_TOPICS[0])
        
        # If no valid topics found, use defaults
        if not valid_topics and self.DEFAULT_FALLBACK_TOPICS:
            valid_topics = self.DEFAULT_FALLBACK_TOPICS
            
        return valid_topics
    
    def _attempt_fallback_query(self, 
                             tickers: Optional[List[str]] = None,
                             time_from: Optional[str] = None,
                             time_to: Optional[str] = None,
                             sort: str = "RELEVANCE",
                             limit: int = 50) -> Dict[str, Any]:
        """Attempt a fallback query using default topics instead of specific tickers."""
        logger.info(f"Using fallback topics: {self.DEFAULT_FALLBACK_TOPICS}")
        
        # If we were using tickers, try to find related topics
        topics_to_use = self.DEFAULT_FALLBACK_TOPICS
        
        # If tickers were provided, try to map them to topics
        if tickers:
            mapped_topics = set()
            for ticker in tickers:
                # Remove CRYPTO: or FOREX: prefix if present
                clean_ticker = ticker
                for prefix in ["CRYPTO:", "FOREX:"]:  
                    if clean_ticker.startswith(prefix):
                        clean_ticker = clean_ticker[len(prefix):]
                
                # Map to topics if possible
                crypto_topics = {
                    "BTC": ["blockchain", "financial_markets"],
                    "ETH": ["blockchain", "financial_markets", "technology"],
                    "XRP": ["blockchain", "financial_markets"],
                    "ADA": ["blockchain", "technology"],
                    "SOL": ["blockchain", "technology"],
                    "DOT": ["blockchain", "technology"],
                    "DOGE": ["blockchain", "financial_markets"],
                    "SHIB": ["blockchain", "financial_markets"],
                    "AVAX": ["blockchain", "technology"],
                    "LINK": ["blockchain", "technology"],
                    "UNI": ["blockchain", "finance"],
                    "AAVE": ["blockchain", "finance"],
                    "COMP": ["blockchain", "finance"],
                    "MKR": ["blockchain", "finance"]
                }
                
                if clean_ticker in crypto_topics:
                    for topic in crypto_topics[clean_ticker]:
                        mapped_topics.add(topic)
            
            # If we found mapped topics, use them
            if mapped_topics:
                topics_to_use = list(mapped_topics)
                logger.info(f"Mapped tickers to topics: {topics_to_use}")
        
        # Validate topics
        topics_to_use = self._validate_topics(topics_to_use)
        
        # Make the fallback query (without further fallback attempts to prevent loops)
        return self.get_news_sentiment(
            topics=topics_to_use,
            tickers=None,  # Don't use tickers in fallback
            time_from=time_from,
            time_to=time_to,
            sort=sort,
            limit=limit,
            use_cache=True,
            attempt_fallback=False  # Prevent fallback loops
        )
    
    def get_sentiment_by_topic(self, topic: str, days_back: int = 7, use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch sentiment data for a specific topic.
        
        Args:
            topic: Topic to fetch sentiment for (e.g., "blockchain", "cryptocurrency")
            days_back: Number of days to look back for news
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary containing sentiment data for the topic
        """
        # Calculate time range
        time_to = datetime.now()
        time_from = time_to - timedelta(days=days_back)
        
        # Format times for Alpha Vantage API
        time_from_str = time_from.strftime("%Y%m%dT%H%M")
        time_to_str = time_to.strftime("%Y%m%dT%H%M")
        
        logger.info(f"Fetching sentiment data for topic: {topic}, days_back: {days_back}")
        
        return self.get_news_sentiment(
            topics=[topic],
            time_from=time_from_str,
            time_to=time_to_str,
            sort="LATEST",
            use_cache=use_cache,
            attempt_fallback=True
        )
    
    def get_sentiment_by_crypto(self, crypto_ticker: str, days_back: int = 7, use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch sentiment data for a specific cryptocurrency.
        
        Note: This may not work on the free tier of Alpha Vantage.
        Consider using get_sentiment_by_topic with "blockchain" or "cryptocurrency" instead.
        If this fails on the free tier, it will automatically fall back to topic-based queries.
        
        Args:
            crypto_ticker: Cryptocurrency ticker (e.g., "BTC", "ETH")
            days_back: Number of days to look back for news
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary containing sentiment data for the cryptocurrency
        """
        # Calculate time range
        time_to = datetime.now()
        time_from = time_to - timedelta(days=days_back)
        
        # Format times for Alpha Vantage API
        time_from_str = time_from.strftime("%Y%m%dT%H%M")
        time_to_str = time_to.strftime("%Y%m%dT%H%M")
        
        # Add CRYPTO: prefix for cryptocurrency tickers
        formatted_ticker = f"CRYPTO:{crypto_ticker}"
        
        logger.info(f"Fetching sentiment data for crypto: {crypto_ticker}, days_back: {days_back}")
        
        # For free tier, we'll attempt fallback to topics if ticker-specific query fails
        result = self.get_news_sentiment(
            tickers=[formatted_ticker],
            time_from=time_from_str,
            time_to=time_to_str,
            sort="LATEST",
            use_cache=use_cache,
            attempt_fallback=(self.tier == "free")  # Only attempt fallback on free tier
        )
        
        # If we're on a premium tier and the query failed, try a more targeted fallback
        if self.tier != "free" and result.get("error") and not result.get("data"):
            logger.info(f"Premium tier query for {crypto_ticker} failed, trying specific fallback")
            
            # Map common crypto tickers to relevant topics from Alpha Vantage's supported topics
            crypto_topics = {
                "BTC": ["blockchain", "financial_markets"],
                "ETH": ["blockchain", "financial_markets", "technology"],
                "XRP": ["blockchain", "financial_markets"],
                "ADA": ["blockchain", "technology"],
                "SOL": ["blockchain", "technology"],
                "DOT": ["blockchain", "technology"],
                "DOGE": ["blockchain", "financial_markets"],
                "SHIB": ["blockchain", "financial_markets"],
                "AVAX": ["blockchain", "technology"],
                "LINK": ["blockchain", "technology"],
                "UNI": ["blockchain", "finance"],
                "AAVE": ["blockchain", "finance"],
                "COMP": ["blockchain", "finance"],
                "MKR": ["blockchain", "finance"]
            }
            
            # Use specific topics for this crypto if available, otherwise use defaults
            topics_to_use = crypto_topics.get(crypto_ticker.upper(), ["cryptocurrency", "blockchain"])
            
            return self.get_news_sentiment(
                topics=topics_to_use,
                time_from=time_from_str,
                time_to=time_to_str,
                sort="LATEST",
                use_cache=use_cache,
                attempt_fallback=False  # Prevent further fallbacks
            )
        
        return result
    
    def extract_sentiment_scores(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract sentiment scores from Alpha Vantage response data.
        
        Args:
            response_data: Response data from get_news_sentiment
            
        Returns:
            List of dictionaries containing sentiment scores and metadata
        """
        if not response_data or "data" not in response_data or not response_data["data"]:
            logger.warning("No data to extract sentiment scores from")
            return []
        
        data = response_data["data"]
        if "feed" not in data:
            logger.warning("No feed data in response")
            return []
        
        sentiment_data = []
        
        for article in data["feed"]:
            # Extract basic article info
            article_data = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "time_published": article.get("time_published", ""),
                "authors": article.get("authors", []),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "overall_sentiment_score": Decimal(str(article.get("overall_sentiment_score", 0))),
                "overall_sentiment_label": article.get("overall_sentiment_label", ""),
            }
            
            # Extract ticker-specific sentiment if available
            if "ticker_sentiment" in article:
                for ticker_data in article["ticker_sentiment"]:
                    ticker = ticker_data.get("ticker", "")
                    if ticker.startswith("CRYPTO:"):
                        ticker = ticker[7:]  # Remove CRYPTO: prefix
                    
                    ticker_sentiment = {
                        **article_data,
                        "ticker": ticker,
                        "ticker_sentiment_score": Decimal(str(ticker_data.get("ticker_sentiment_score", 0))),
                        "ticker_sentiment_label": ticker_data.get("ticker_sentiment_label", ""),
                        "relevance_score": Decimal(str(ticker_data.get("relevance_score", 0))),
                    }
                    sentiment_data.append(ticker_sentiment)
            else:
                # If no ticker-specific sentiment, just use the overall sentiment
                sentiment_data.append(article_data)
        
        return sentiment_data
