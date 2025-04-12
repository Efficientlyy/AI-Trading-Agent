"""
Twitter API Collector for sentiment analysis.

This module provides functionality to collect tweets from Twitter API v2
for sentiment analysis in the AI Trading Agent.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import requests
import pandas as pd

from ...common.logging_config import setup_logging
from ..providers.base_provider import BaseSentimentProvider

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class TwitterAPICollector(BaseSentimentProvider):
    """
    Collects tweets from Twitter API v2 for sentiment analysis.
    
    Attributes:
        api_key (str): Twitter API key
        api_secret (str): Twitter API secret
        bearer_token (str): Twitter API bearer token
        access_token (str): Twitter API access token
        access_secret (str): Twitter API access token secret
        cache_dir (str): Directory to cache API responses
        cache_expiry (int): Cache expiry time in seconds
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
        access_token: Optional[str] = None,
        access_secret: Optional[str] = None,
        cache_dir: str = "cache/twitter",
        cache_expiry: int = 3600,
        max_results_per_query: int = 100,
        rate_limit_wait: int = 15,
    ):
        """
        Initialize the Twitter API collector.
        
        Args:
            api_key: Twitter API key (defaults to TWITTER_API_KEY env var)
            api_secret: Twitter API secret (defaults to TWITTER_API_SECRET env var)
            bearer_token: Twitter API bearer token (defaults to TWITTER_BEARER_TOKEN env var)
            access_token: Twitter API access token (defaults to TWITTER_ACCESS_TOKEN env var)
            access_secret: Twitter API access secret (defaults to TWITTER_ACCESS_SECRET env var)
            cache_dir: Directory to cache API responses
            cache_expiry: Cache expiry time in seconds
            max_results_per_query: Maximum results per API query
            rate_limit_wait: Time to wait when rate limited (seconds)
        """
        # Get API credentials from environment variables if not provided
        self.api_key = api_key or os.environ.get("TWITTER_API_KEY")
        self.api_secret = api_secret or os.environ.get("TWITTER_API_SECRET")
        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN")
        self.access_token = access_token or os.environ.get("TWITTER_ACCESS_TOKEN")
        self.access_secret = access_secret or os.environ.get("TWITTER_ACCESS_SECRET")
        
        # Validate credentials
        if not self.bearer_token and not (self.api_key and self.api_secret):
            logger.warning("No Twitter API credentials provided. Will use mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
        
        # Set up caching
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        os.makedirs(cache_dir, exist_ok=True)
        
        # API parameters
        self.max_results_per_query = max_results_per_query
        self.rate_limit_wait = rate_limit_wait
        
        # Initialize session
        self.session = requests.Session()
        if self.bearer_token:
            self.session.headers.update({"Authorization": f"Bearer {self.bearer_token}"})
    
    def fetch_sentiment_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch sentiment data for the specified symbols and date range.
        
        Args:
            symbols: List of ticker symbols to fetch sentiment for
            start_date: Start date for sentiment data
            end_date: End date for sentiment data
            **kwargs: Additional parameters for the API query
            
        Returns:
            DataFrame with sentiment data
        """
        if self.use_mock_data:
            return self._generate_mock_data(symbols, start_date, end_date)
        
        all_tweets = []
        
        for symbol in symbols:
            # Create search queries for the symbol (with $ and without)
            queries = [
                f"{symbol}",  # Plain symbol
                f"${symbol}",  # Cashtag
                f"{symbol} stock",  # Symbol + stock
                f"{symbol} price",  # Symbol + price
            ]
            
            for query in queries:
                tweets = self._search_tweets(
                    query=query,
                    start_time=start_date,
                    end_time=end_date,
                    **kwargs
                )
                
                # Add symbol to each tweet
                for tweet in tweets:
                    tweet["symbol"] = symbol
                
                all_tweets.extend(tweets)
        
        # Convert to DataFrame
        if not all_tweets:
            logger.warning(f"No tweets found for symbols {symbols} in date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_tweets)
        
        # Process and clean the DataFrame
        df = self._process_tweets_dataframe(df)
        
        return df
    
    def stream_sentiment_data(
        self,
        symbols: List[str],
        callback: callable,
        **kwargs
    ) -> None:
        """
        Stream sentiment data for the specified symbols.
        
        Args:
            symbols: List of ticker symbols to stream sentiment for
            callback: Callback function to process streamed data
            **kwargs: Additional parameters for the API query
        """
        if self.use_mock_data:
            logger.warning("Streaming not supported with mock data")
            return
        
        # Create rules for the stream
        rules = []
        for symbol in symbols:
            rules.append({"value": f"${symbol}", "tag": f"symbol:{symbol}"})
            rules.append({"value": f"{symbol} stock", "tag": f"symbol:{symbol}"})
        
        # Set up stream parameters
        params = {
            "tweet.fields": "created_at,public_metrics,entities",
            "expansions": "author_id",
            "user.fields": "username,name,verified",
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Start streaming
        logger.info(f"Starting Twitter stream for symbols: {symbols}")
        try:
            self._stream_tweets(rules, params, callback)
        except KeyboardInterrupt:
            logger.info("Twitter stream stopped by user")
        except Exception as e:
            logger.error(f"Error in Twitter stream: {e}")
    
    def _search_tweets(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for tweets using the Twitter API.
        
        Args:
            query: Search query
            start_time: Start time for search
            end_time: End time for search
            **kwargs: Additional parameters for the API query
            
        Returns:
            List of tweet objects
        """
        # Check cache first
        cache_key = self._get_cache_key(query, start_time, end_time, kwargs)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Using cached data for query: {query}")
            return cached_data
        
        # Format dates for Twitter API
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Set up parameters
        params = {
            "query": query,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "max_results": self.max_results_per_query,
            "tweet.fields": "created_at,public_metrics,entities",
            "expansions": "author_id",
            "user.fields": "username,name,verified",
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Make API request
        url = "https://api.twitter.com/2/tweets/search/recent"
        all_tweets = []
        next_token = None
        
        while True:
            if next_token:
                params["next_token"] = next_token
            
            try:
                response = self.session.get(url, params=params)
                
                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning(f"Rate limited. Waiting {self.rate_limit_wait} seconds.")
                    time.sleep(self.rate_limit_wait)
                    continue
                
                # Handle other errors
                if response.status_code != 200:
                    logger.error(f"Error from Twitter API: {response.status_code} - {response.text}")
                    break
                
                data = response.json()
                
                # Extract tweets
                if "data" in data:
                    all_tweets.extend(data["data"])
                
                # Check for next page
                if "meta" in data and "next_token" in data["meta"]:
                    next_token = data["meta"]["next_token"]
                else:
                    break
                
            except Exception as e:
                logger.error(f"Error fetching tweets: {e}")
                break
        
        # Cache the results
        self._save_to_cache(cache_key, all_tweets)
        
        return all_tweets
    
    def _stream_tweets(
        self,
        rules: List[Dict[str, str]],
        params: Dict[str, Any],
        callback: callable
    ) -> None:
        """
        Stream tweets using the Twitter API.
        
        Args:
            rules: List of rules for the stream
            params: Parameters for the stream
            callback: Callback function to process streamed data
        """
        # Set up the stream
        url = "https://api.twitter.com/2/tweets/search/stream"
        
        # First, delete existing rules
        rules_url = f"{url}/rules"
        response = self.session.get(rules_url)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                ids = [rule["id"] for rule in data["data"]]
                if ids:
                    payload = {"delete": {"ids": ids}}
                    self.session.post(rules_url, json=payload)
        
        # Add new rules
        payload = {"add": rules}
        response = self.session.post(rules_url, json=payload)
        if response.status_code != 201:
            logger.error(f"Error setting stream rules: {response.status_code} - {response.text}")
            return
        
        # Start the stream
        response = self.session.get(url, params=params, stream=True)
        
        for line in response.iter_lines():
            if line:
                try:
                    tweet = json.loads(line)
                    callback(tweet)
                except json.JSONDecodeError:
                    logger.error(f"Error decoding tweet: {line}")
    
    def _process_tweets_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the tweets DataFrame.
        
        Args:
            df: DataFrame with raw tweet data
            
        Returns:
            Processed DataFrame
        """
        # Convert created_at to datetime
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
        
        # Extract metrics if available
        if "public_metrics" in df.columns:
            metrics = pd.json_normalize(df["public_metrics"])
            for col in metrics.columns:
                df[f"metric_{col}"] = metrics[col]
            df = df.drop(columns=["public_metrics"])
        
        # Rename columns for consistency
        column_mapping = {
            "created_at": "timestamp",
            "id": "tweet_id",
            "text": "content",
            "author_id": "user_id",
        }
        df = df.rename(columns=column_mapping)
        
        # Add source column
        df["source"] = "twitter"
        
        # Ensure required columns exist
        required_columns = ["timestamp", "content", "symbol", "source"]
        for col in required_columns:
            if col not in df.columns:
                if col == "timestamp":
                    df[col] = datetime.now()
                else:
                    df[col] = ""
        
        return df
    
    def _get_cache_key(self, query: str, start_time: datetime, end_time: datetime, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for the query.
        
        Args:
            query: Search query
            start_time: Start time for search
            end_time: End time for search
            params: Additional parameters
            
        Returns:
            Cache key string
        """
        start_str = start_time.strftime("%Y%m%d")
        end_str = end_time.strftime("%Y%m%d")
        query_clean = query.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return f"{query_clean}_{start_str}_{end_str}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get data from cache if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is expired
        file_time = os.path.getmtime(cache_file)
        if time.time() - file_time > self.cache_expiry:
            logger.debug(f"Cache expired for key: {cache_key}")
            return None
        
        # Load cache
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: List[Dict[str, Any]]) -> None:
        """
        Save data to cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _generate_mock_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate mock tweet data for testing.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for mock data
            end_date: End date for mock data
            
        Returns:
            DataFrame with mock tweet data
        """
        logger.info("Generating mock Twitter data")
        
        # Calculate date range
        date_range = (end_date - start_date).days
        if date_range <= 0:
            date_range = 1
        
        # Generate random data
        import numpy as np
        
        all_data = []
        
        for symbol in symbols:
            # Generate 5-20 tweets per day per symbol
            for day in range(date_range):
                current_date = start_date + timedelta(days=day)
                num_tweets = np.random.randint(5, 21)
                
                for _ in range(num_tweets):
                    # Random time during the day
                    hour = np.random.randint(0, 24)
                    minute = np.random.randint(0, 60)
                    second = np.random.randint(0, 60)
                    timestamp = current_date.replace(hour=hour, minute=minute, second=second)
                    
                    # Ensure timestamp is within the specified range
                    if timestamp < start_date:
                        timestamp = start_date
                    if timestamp > end_date:
                        timestamp = end_date
                    
                    # Generate tweet text
                    sentiment = np.random.choice(["positive", "negative", "neutral"], p=[0.4, 0.3, 0.3])
                    
                    if sentiment == "positive":
                        text = np.random.choice([
                            f"${symbol} looking strong today! Bullish pattern forming.",
                            f"Just bought more ${symbol}. Great fundamentals and technical setup.",
                            f"${symbol} earnings beat expectations! Stock should rally.",
                            f"Analyst upgrade for ${symbol}. Target price increased by 15%.",
                            f"${symbol} breaking out of resistance. Time to buy!"
                        ])
                    elif sentiment == "negative":
                        text = np.random.choice([
                            f"${symbol} breaking down. Looks weak technically.",
                            f"Selling my ${symbol} position. Fundamentals deteriorating.",
                            f"${symbol} missed earnings. Expect downside tomorrow.",
                            f"Analyst downgrade for ${symbol}. Target price cut by 10%.",
                            f"${symbol} forming a bearish pattern. Time to sell."
                        ])
                    else:  # neutral
                        text = np.random.choice([
                            f"${symbol} trading sideways today. No clear direction.",
                            f"Watching ${symbol} for a potential entry point.",
                            f"${symbol} earnings coming up next week. Holding my position.",
                            f"What's everyone's take on ${symbol} at current levels?",
                            f"${symbol} volume lower than average today."
                        ])
                    
                    # Create tweet object
                    tweet = {
                        "timestamp": timestamp,
                        "content": text,
                        "symbol": symbol,
                        "source": "twitter",
                        "tweet_id": f"mock_{symbol}_{timestamp.strftime('%Y%m%d%H%M%S')}",
                        "user_id": f"user_{np.random.randint(1000, 9999)}",
                        "metric_retweet_count": np.random.randint(0, 100),
                        "metric_reply_count": np.random.randint(0, 50),
                        "metric_like_count": np.random.randint(0, 500),
                        "metric_quote_count": np.random.randint(0, 30),
                    }
                    
                    all_data.append(tweet)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        return df
