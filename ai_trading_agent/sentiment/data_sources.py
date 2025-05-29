"""
Data sources for sentiment analysis.

This module provides connectors to various sources of sentiment data,
including news APIs, social media feeds, and financial report repositories.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import time
import re
from enum import Enum

from ai_trading_agent.common import logger


class DataSourceType(Enum):
    """Types of data sources for sentiment analysis."""
    NEWS_API = "news_api"
    TWITTER = "twitter"
    REDDIT = "reddit"
    FINANCIAL_REPORTS = "financial_reports"
    RSS_FEED = "rss_feed"
    MOCK = "mock"


class SentimentDataSource:
    """Base class for all sentiment data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data source.
        
        Args:
            config: Configuration dictionary
        """
        self.name = self.__class__.__name__
        self.config = config
        self.logger = logger
        
        # Common parameters
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_duration = config.get("cache_duration", 3600)  # 1 hour
        self.data_cache = {}
        self.cache_timestamps = {}
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 60)  # requests per minute
        self.request_count = 0
        self.request_reset_time = datetime.now()
        
        # Retry parameters
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)  # seconds
        
        # Common query parameters
        self.query_params = config.get("query_params", {})
        
        self.logger.info(f"Initialized {self.name}")
    
    def fetch_data(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch data from the source.
        
        Args:
            query: Search query string
            **kwargs: Additional parameters for the request
            
        Returns:
            Dictionary with the fetched data
        """
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = self._create_cache_key(query, kwargs)
            if cache_key in self.data_cache:
                cache_timestamp = self.cache_timestamps.get(cache_key, datetime.now())
                if (datetime.now() - cache_timestamp).total_seconds() < self.cache_duration:
                    self.logger.debug(f"Using cached data for query: {query}")
                    return self.data_cache[cache_key]
        
        # Implement rate limiting
        self._check_rate_limit()
        
        # Fetch data with retries
        data = None
        for retry in range(self.max_retries):
            try:
                data = self._execute_fetch(query, **kwargs)
                break
            except Exception as e:
                self.logger.warning(f"Error fetching data (retry {retry+1}/{self.max_retries}): {str(e)}")
                if retry < self.max_retries - 1:
                    time.sleep(self.retry_delay * (retry + 1))  # Exponential backoff
                else:
                    raise
        
        # Cache the result if enabled
        if self.cache_enabled and data:
            cache_key = self._create_cache_key(query, kwargs)
            self.data_cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()
        
        return data
    
    def _create_cache_key(self, query: str, params: Dict) -> str:
        """Create a unique cache key from the query and parameters."""
        sorted_params = json.dumps(params, sort_keys=True)
        return f"{query}:{sorted_params}"
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = datetime.now()
        time_diff = (current_time - self.request_reset_time).total_seconds()
        
        # Reset counter if a minute has passed
        if time_diff > 60:
            self.request_count = 0
            self.request_reset_time = current_time
        
        # Check if we've hit the rate limit
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - time_diff
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_reset_time = datetime.now()
        
        # Increment the request counter
        self.request_count += 1
    
    def _execute_fetch(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the actual data fetch. To be implemented by subclasses.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with the fetched data
        """
        raise NotImplementedError("Subclasses must implement _execute_fetch")
    
    def process_response(self, response_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process the raw response data into a structured DataFrame.
        
        Args:
            response_data: Raw response from the API
            
        Returns:
            DataFrame with processed data
        """
        raise NotImplementedError("Subclasses must implement process_response")
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_cache = {}
        self.cache_timestamps = {}
        self.logger.info(f"Cleared cache for {self.name}")


class NewsApiSource(SentimentDataSource):
    """Data source for news articles from NewsAPI."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NewsAPI source.
        
        Args:
            config: Configuration with api_key and other parameters
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.logger.warning("No API key provided for NewsAPI")
        
        self.base_url = config.get("base_url", "https://newsapi.org/v2/everything")
        self.language = config.get("language", "en")
        self.sort_by = config.get("sort_by", "publishedAt")
        self.page_size = config.get("page_size", 100)
    
    def _execute_fetch(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the fetch from NewsAPI.
        
        Args:
            query: Search query
            **kwargs: Additional parameters including:
                - days_back: Number of days to look back
                - page: Page number
                
        Returns:
            Dictionary with the API response
        """
        # Handle parameters
        days_back = kwargs.get("days_back", 7)
        page = kwargs.get("page", 1)
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        # Build request parameters
        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": self.language,
            "sortBy": self.sort_by,
            "pageSize": self.page_size,
            "page": page,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d")
        }
        
        # Add any additional parameters from query_params
        params.update(self.query_params)
        
        # Make the request
        response = requests.get(self.base_url, params=params)
        
        # Check for errors
        if response.status_code != 200:
            self.logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
            raise Exception(f"NewsAPI error: {response.status_code}")
        
        return response.json()
    
    def process_response(self, response_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process the NewsAPI response into a structured DataFrame.
        
        Args:
            response_data: Raw response from NewsAPI
            
        Returns:
            DataFrame with processed news data
        """
        if not response_data or "articles" not in response_data:
            return pd.DataFrame()
        
        articles = response_data["articles"]
        if not articles:
            return pd.DataFrame()
        
        # Extract relevant fields
        data = []
        for article in articles:
            timestamp = article.get("publishedAt")
            if timestamp:
                try:
                    # Convert ISO timestamp to datetime
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Combine title and description for sentiment analysis
            text = article.get("title", "")
            if article.get("description"):
                text += " " + article.get("description", "")
            
            # Filter out empty content
            if not text.strip():
                continue
            
            data.append({
                "timestamp": timestamp,
                "text": text,
                "source": article.get("source", {}).get("name", "unknown"),
                "url": article.get("url", ""),
                "author": article.get("author", "unknown")
            })
        
        return pd.DataFrame(data)


class RedditSource(SentimentDataSource):
    """Data source for Reddit posts and comments."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Reddit source.
        
        Args:
            config: Configuration with client_id, client_secret, and other parameters
        """
        super().__init__(config)
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.user_agent = config.get("user_agent", "ai_trading_agent/1.0")
        
        # Check if we have authentication credentials
        self.auth_enabled = self.client_id and self.client_secret
        if not self.auth_enabled:
            self.logger.warning("No authentication credentials provided for Reddit API")
        
        self.base_url = config.get("base_url", "https://www.reddit.com")
        self.subreddits = config.get("subreddits", ["CryptoCurrency", "Bitcoin", "ethereum"])
        self.sort_by = config.get("sort_by", "hot")
        self.limit = config.get("limit", 100)
        
        # Access token for authenticated requests
        self.access_token = None
        self.token_expiry = datetime.now()
    
    def _get_auth_token(self) -> str:
        """
        Get an authentication token for the Reddit API.
        
        Returns:
            Access token string
        """
        # Check if we have a valid token
        if self.access_token and datetime.now() < self.token_expiry:
            return self.access_token
        
        # Otherwise, request a new token
        auth_url = "https://www.reddit.com/api/v1/access_token"
        auth = (self.client_id, self.client_secret)
        headers = {"User-Agent": self.user_agent}
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(auth_url, auth=auth, headers=headers, data=data)
        
        if response.status_code != 200:
            self.logger.error(f"Reddit auth error: {response.status_code} - {response.text}")
            raise Exception(f"Reddit auth error: {response.status_code}")
        
        token_data = response.json()
        self.access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)  # Buffer 60 seconds
        
        return self.access_token
    
    def _execute_fetch(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the fetch from Reddit.
        
        Args:
            query: Search query or subreddit
            **kwargs: Additional parameters including:
                - subreddit: Specific subreddit to search in
                - limit: Number of posts to fetch
                - sort: Sort method (hot, new, top, etc.)
                
        Returns:
            Dictionary with the API response
        """
        # Handle parameters
        subreddit = kwargs.get("subreddit", None)
        limit = kwargs.get("limit", self.limit)
        sort = kwargs.get("sort", self.sort_by)
        
        # Determine which subreddit(s) to use
        if not subreddit:
            if query in self.subreddits:
                subreddit = query
            else:
                # Use the first subreddit in the list if none specified
                subreddit = self.subreddits[0]
        
        # Build the URL
        url = f"{self.base_url}/r/{subreddit}/{sort}.json"
        
        # Set up headers
        headers = {"User-Agent": self.user_agent}
        
        # Add auth token if available
        if self.auth_enabled:
            token = self._get_auth_token()
            headers["Authorization"] = f"Bearer {token}"
        
        # Build parameters
        params = {
            "limit": limit
        }
        
        # Add any additional parameters from query_params
        params.update(self.query_params)
        
        # Make the request
        response = requests.get(url, headers=headers, params=params)
        
        # Check for errors
        if response.status_code != 200:
            self.logger.error(f"Reddit API error: {response.status_code} - {response.text}")
            raise Exception(f"Reddit API error: {response.status_code}")
        
        return response.json()
    
    def process_response(self, response_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process the Reddit API response into a structured DataFrame.
        
        Args:
            response_data: Raw response from Reddit API
            
        Returns:
            DataFrame with processed Reddit posts
        """
        if not response_data or "data" not in response_data or "children" not in response_data["data"]:
            return pd.DataFrame()
        
        posts = response_data["data"]["children"]
        if not posts:
            return pd.DataFrame()
        
        # Extract relevant fields
        data = []
        for post in posts:
            if "data" not in post:
                continue
                
            post_data = post["data"]
            
            # Convert created timestamp to datetime
            created_utc = post_data.get("created_utc", time.time())
            timestamp = datetime.fromtimestamp(created_utc)
            
            # Combine title and selftext for sentiment analysis
            text = post_data.get("title", "")
            if post_data.get("selftext"):
                text += " " + post_data.get("selftext", "")
            
            # Filter out empty content
            if not text.strip():
                continue
            
            data.append({
                "timestamp": timestamp,
                "text": text,
                "source": f"reddit/r/{post_data.get('subreddit', 'unknown')}",
                "url": f"https://www.reddit.com{post_data.get('permalink', '')}",
                "author": post_data.get("author", "unknown"),
                "score": post_data.get("score", 0),
                "comments": post_data.get("num_comments", 0)
            })
        
        return pd.DataFrame(data)


class MockSentimentSource(SentimentDataSource):
    """Mock data source for testing and development."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock source."""
        super().__init__(config)
        
        # Sample sentiments for different assets
        self.asset_sentiments = {
            "BTC": 0.7,  # Positive
            "ETH": 0.4,  # Slightly positive
            "XRP": -0.3,  # Slightly negative
            "ADA": 0.1,  # Neutral
            "DOT": -0.6,  # Negative
        }
        
        # Sample sentiments for different topics
        self.topic_sentiments = {
            "cryptocurrency": 0.5,
            "blockchain": 0.6,
            "defi": 0.3,
            "nft": -0.2,
            "regulation": -0.4,
            "market": 0.2
        }
        
        # Sample text templates
        self.positive_templates = [
            "{asset} is showing strong bullish signals today.",
            "Investors are increasingly optimistic about {asset}.",
            "New developments in {asset} have excited the community.",
            "{asset} could see significant gains according to analysts.",
            "The future looks bright for {asset} with its latest updates."
        ]
        
        self.neutral_templates = [
            "{asset} remains stable amid market fluctuations.",
            "No significant changes for {asset} in recent trading.",
            "Analysts remain divided on the future of {asset}.",
            "{asset} continues to consolidate in its current range.",
            "Mixed signals from the market regarding {asset}."
        ]
        
        self.negative_templates = [
            "Concerns grow over {asset}'s recent performance.",
            "{asset} faces regulatory challenges that worry investors.",
            "Bearish sentiment surrounds {asset} after recent developments.",
            "Analysts predict a continued decline for {asset}.",
            "Market participants are increasingly skeptical about {asset}."
        ]
    
    def _execute_fetch(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate mock data for the query.
        
        Args:
            query: Asset symbol or topic
            **kwargs: Additional parameters including:
                - count: Number of entries to generate
                - days_back: Number of days to generate data for
                
        Returns:
            Dictionary with mock data
        """
        # Handle parameters
        count = kwargs.get("count", 20)
        days_back = kwargs.get("days_back", 7)
        
        # Determine if query is an asset or topic
        is_asset = query in self.asset_sentiments
        is_topic = query in self.topic_sentiments
        
        if not (is_asset or is_topic):
            # Default to neutral sentiment for unknown queries
            base_sentiment = 0.0
        else:
            base_sentiment = self.asset_sentiments.get(query, self.topic_sentiments.get(query, 0.0))
        
        # Generate mock data
        data = []
        for i in range(count):
            # Calculate a time in the past
            hours_back = (days_back * 24 * i) // count
            timestamp = datetime.now() - timedelta(hours=hours_back)
            
            # Add some randomness to the sentiment
            sentiment = base_sentiment + (0.2 * (0.5 - (i / count)))  # Trend over time
            sentiment += 0.2 * (0.5 - pd.np.random.random())  # Random noise
            sentiment = max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
            
            # Choose an appropriate template based on sentiment
            if sentiment > 0.3:
                templates = self.positive_templates
            elif sentiment < -0.3:
                templates = self.negative_templates
            else:
                templates = self.neutral_templates
            
            # Generate text from template
            template = pd.np.random.choice(templates)
            text = template.format(asset=query)
            
            # Add some hashtags for social media style
            if pd.np.random.random() < 0.5:
                hashtags = ["#crypto", "#trading", f"#{query}", "#investment"]
                selected_hashtags = pd.np.random.choice(hashtags, size=2, replace=False)
                text += " " + " ".join(selected_hashtags)
            
            # Add to data
            data.append({
                "timestamp": timestamp,
                "text": text,
                "sentiment": sentiment,
                "source": pd.np.random.choice(["twitter", "reddit", "news", "blog"]),
                "author": f"mock_user_{pd.np.random.randint(1, 100)}"
            })
        
        return {"data": data, "query": query, "is_asset": is_asset, "is_topic": is_topic}
    
    def process_response(self, response_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process the mock response into a structured DataFrame.
        
        Args:
            response_data: Mock response data
            
        Returns:
            DataFrame with processed mock data
        """
        if not response_data or "data" not in response_data:
            return pd.DataFrame()
        
        return pd.DataFrame(response_data["data"])


class SentimentDataManager:
    """
    Manager for multiple sentiment data sources.
    
    This class handles fetching data from various sources and combining the results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment data manager.
        
        Args:
            config: Configuration dictionary with source configurations
        """
        self.config = config
        self.logger = logger
        
        # Initialize data sources
        self.sources = {}
        source_configs = config.get("sources", {})
        
        # Check if we should use mock data
        self.use_mock = config.get("use_mock", True)
        if self.use_mock:
            self.sources[DataSourceType.MOCK.value] = MockSentimentSource(
                config.get("mock_source", {})
            )
        
        # Initialize real data sources if configured
        for source_type, source_config in source_configs.items():
            if not source_config.get("enabled", False):
                continue
                
            if source_type == DataSourceType.NEWS_API.value:
                self.sources[source_type] = NewsApiSource(source_config)
            elif source_type == DataSourceType.REDDIT.value:
                self.sources[source_type] = RedditSource(source_config)
            # Add more source types as implemented
        
        self.logger.info(f"Initialized SentimentDataManager with {len(self.sources)} sources")
        
        # Cache for combined data
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_duration = config.get("cache_duration", 3600)  # 1 hour
        self.data_cache = {}
        self.cache_timestamps = {}
    
    def fetch_sentiment_data(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Fetch sentiment data from all configured sources.
        
        Args:
            query: Search query (asset symbol or topic)
            **kwargs: Additional parameters including:
                - sources: List of specific sources to query
                - days_back: Number of days to look back
                - limit: Maximum number of results per source
                
        Returns:
            DataFrame with combined sentiment data
        """
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = f"{query}:{json.dumps(kwargs, sort_keys=True)}"
            if cache_key in self.data_cache:
                cache_timestamp = self.cache_timestamps.get(cache_key, datetime.now())
                if (datetime.now() - cache_timestamp).total_seconds() < self.cache_duration:
                    self.logger.debug(f"Using cached sentiment data for query: {query}")
                    return self.data_cache[cache_key]
        
        # Determine which sources to query
        source_names = kwargs.get("sources", None)
        if source_names:
            sources_to_query = {name: self.sources[name] for name in source_names if name in self.sources}
        else:
            sources_to_query = self.sources
        
        if not sources_to_query:
            self.logger.warning(f"No valid sources to query for: {query}")
            return pd.DataFrame()
        
        # Fetch data from each source
        all_data = []
        for source_name, source in sources_to_query.items():
            try:
                self.logger.debug(f"Fetching sentiment data from {source_name} for query: {query}")
                response = source.fetch_data(query, **kwargs)
                df = source.process_response(response)
                
                if not df.empty:
                    # Add source identifier
                    df["data_source"] = source_name
                    all_data.append(df)
            except Exception as e:
                self.logger.error(f"Error fetching data from {source_name}: {str(e)}", exc_info=True)
        
        # Combine all data
        if not all_data:
            self.logger.warning(f"No data found for query: {query}")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp (newest first)
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.sort_values("timestamp", ascending=False)
        
        # Cache the result if enabled
        if self.cache_enabled:
            cache_key = f"{query}:{json.dumps(kwargs, sort_keys=True)}"
            self.data_cache[cache_key] = combined_df
            self.cache_timestamps[cache_key] = datetime.now()
        
        return combined_df
    
    def fetch_batch_sentiment(self, queries: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch sentiment data for multiple queries.
        
        Args:
            queries: List of search queries
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping queries to DataFrames
        """
        results = {}
        for query in queries:
            results[query] = self.fetch_sentiment_data(query, **kwargs)
        return results
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_cache = {}
        self.cache_timestamps = {}
        self.logger.info("Cleared sentiment data cache")
        
        # Also clear caches in all sources
        for source in self.sources.values():
            source.clear_cache()
