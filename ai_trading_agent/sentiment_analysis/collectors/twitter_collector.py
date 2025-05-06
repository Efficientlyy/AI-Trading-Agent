"""
Twitter API Collector for sentiment analysis.

This module provides functionality to collect tweets from Twitter API v2
for sentiment analysis in the AI Trading Agent.
"""

import os
import json
import time
import logging
import re
from typing import Dict, List, Optional, Any, Union, Callable
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
        # If using mock data, return mock data
        if self.use_mock_data:
            logger.info(f"Using mock data for query: {query}")
            symbols = [s.strip('$') for s in query.split('OR') if s.strip().startswith('$')]
            if not symbols:
                symbols = [query.split()[0]]  # Use first word as symbol if no $ symbols
            return self._generate_mock_data(symbols, start_time, end_time).to_dict('records')
        
        # Check cache first
        cache_key = self._get_cache_key(query, start_time, end_time, kwargs)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Using cached data for query: {query}")
            return cached_data
        
        # Prepare API request for Twitter API v2
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Format dates for Twitter API (ISO 8601 format)
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Prepare query parameters
        params = {
            "query": query,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "max_results": kwargs.get("max_results", self.max_results_per_query),
            "tweet.fields": kwargs.get("tweet_fields", "created_at,public_metrics,entities,context_annotations"),
            "user.fields": kwargs.get("user_fields", "username,public_metrics,verified"),
            "expansions": kwargs.get("expansions", "author_id,referenced_tweets.id"),
        }
        
        all_tweets = []
        next_token = None
        max_pages = kwargs.get("max_pages", 5)  # Limit number of pages to avoid rate limits
        page_count = 0
        
        try:
            while page_count < max_pages:
                if next_token:
                    params["next_token"] = next_token
                
                # Make API request
                logger.info(f"Making Twitter API request for query: {query} (page {page_count+1})")
                response = self.session.get(url, params=params)
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = int(response.headers.get("x-rate-limit-reset", self.rate_limit_wait))
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds.")
                    time.sleep(wait_time)
                    continue
                
                # Handle other errors
                if response.status_code != 200:
                    logger.error(f"Twitter API error: {response.status_code} - {response.text}")
                    break
                
                # Parse response
                data = response.json()
                
                # Extract tweets
                if "data" in data:
                    tweets = data["data"]
                    
                    # Process tweets to include user information
                    if "includes" in data and "users" in data["includes"]:
                        users = {user["id"]: user for user in data["includes"]["users"]}
                        
                        for tweet in tweets:
                            # Add user information to tweet
                            if "author_id" in tweet and tweet["author_id"] in users:
                                tweet["user"] = users[tweet["author_id"]]
                    
                    all_tweets.extend(tweets)
                
                # Check if there are more results
                if "meta" in data and "next_token" in data["meta"]:
                    next_token = data["meta"]["next_token"]
                    page_count += 1
                else:
                    break
            
            # Process the tweets into a standardized format
            processed_tweets = []
            for tweet in all_tweets:
                processed_tweet = {
                    "tweet_id": tweet.get("id"),
                    "content": tweet.get("text"),
                    "timestamp": datetime.strptime(tweet.get("created_at"), "%Y-%m-%dT%H:%M:%S.%fZ") if "created_at" in tweet else datetime.now(),
                    "source": "twitter",
                    "user_id": tweet.get("author_id"),
                    "username": tweet.get("user", {}).get("username") if "user" in tweet else None,
                    "metric_retweet_count": tweet.get("public_metrics", {}).get("retweet_count", 0) if "public_metrics" in tweet else 0,
                    "metric_reply_count": tweet.get("public_metrics", {}).get("reply_count", 0) if "public_metrics" in tweet else 0,
                    "metric_like_count": tweet.get("public_metrics", {}).get("like_count", 0) if "public_metrics" in tweet else 0,
                    "metric_quote_count": tweet.get("public_metrics", {}).get("quote_count", 0) if "public_metrics" in tweet else 0,
                }
                
                # Extract symbols from entities
                if "entities" in tweet and "cashtags" in tweet["entities"]:
                    processed_tweet["symbols"] = [cashtag["tag"] for cashtag in tweet["entities"]["cashtags"]]
                
                processed_tweets.append(processed_tweet)
            
            # Cache the results
            self._save_to_cache(cache_key, processed_tweets)
            
            return processed_tweets
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            # Fallback to mock data in case of error
            logger.warning("Falling back to mock data due to API error")
            symbols = [s.strip('$') for s in query.split('OR') if s.strip().startswith('$')]
            if not symbols:
                symbols = [query.split()[0]]  # Use first word as symbol if no $ symbols
            mock_data = self._generate_mock_data(symbols, start_time, end_time).to_dict('records')
            return mock_data
    
    def _stream_tweets(
        self,
        rules: List[Dict[str, str]],
        params: Dict[str, Any],
        callback: Callable[[pd.DataFrame], None]
    ) -> None:
        """
        Stream tweets using the Twitter API.
        
        Args:
            rules: List of rules for the stream
            params: Parameters for the stream
            callback: Callback function to process streamed data
        """
        # If using mock data, generate mock data periodically
        if self.use_mock_data:
            logger.info("Using mock data for streaming")
            symbols = []
            for rule in rules:
                query = rule.get("value", "")
                symbols.extend([s.strip('$') for s in query.split('OR') if s.strip().startswith('$')])
            
            if not symbols:
                symbols = ["BTC", "ETH"]  # Default symbols if none specified
            
            # Generate mock data every few seconds
            try:
                while True:
                    mock_data = self._generate_mock_data(
                        symbols,
                        datetime.now() - timedelta(minutes=5),
                        datetime.now()
                    )
                    callback(mock_data)
                    time.sleep(5)  # Sleep for 5 seconds
            except KeyboardInterrupt:
                logger.info("Mock streaming stopped")
            return
        
        # Implement Twitter API v2 filtered stream
        stream_url = "https://api.twitter.com/2/tweets/search/stream"
        rules_url = f"{stream_url}/rules"
        
        # Set up stream parameters
        stream_params = {
            "tweet.fields": params.get("tweet_fields", "created_at,public_metrics,entities,context_annotations"),
            "user.fields": params.get("user_fields", "username,public_metrics,verified"),
            "expansions": params.get("expansions", "author_id,referenced_tweets.id"),
        }
        
        # First, delete any existing rules
        try:
            response = self.session.get(rules_url)
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    existing_rules = data["data"]
                    if existing_rules:
                        ids = [rule["id"] for rule in existing_rules]
                        payload = {"delete": {"ids": ids}}
                        self.session.post(rules_url, json=payload)
            
            # Add new rules
            if rules:
                payload = {"add": rules}
                response = self.session.post(rules_url, json=payload)
                if response.status_code != 201:
                    logger.error(f"Failed to add stream rules: {response.status_code} - {response.text}")
                    # Fall back to mock streaming
                    self._stream_tweets_mock(symbols, callback)
                    return
            
            # Start the stream
            logger.info("Starting Twitter filtered stream")
            response = self.session.get(
                stream_url,
                params=stream_params,
                stream=True
            )
            
            if response.status_code != 200:
                logger.error(f"Stream connection failed: {response.status_code} - {response.text}")
                # Fall back to mock streaming
                self._stream_tweets_mock(symbols, callback)
                return
            
            # Process the stream
            buffer = ""
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    buffer += chunk.decode("utf-8")
                    if buffer.endswith("\r\n"):
                        tweets = []
                        for line in buffer.split("\r\n"):
                            if line.strip():
                                try:
                                    tweet_data = json.loads(line)
                                    if "data" in tweet_data:
                                        tweet = tweet_data["data"]
                                        
                                        # Process user information if available
                                        if "includes" in tweet_data and "users" in tweet_data["includes"]:
                                            users = {user["id"]: user for user in tweet_data["includes"]["users"]}
                                            if "author_id" in tweet and tweet["author_id"] in users:
                                                tweet["user"] = users[tweet["author_id"]]
                                        
                                        # Convert to standardized format
                                        processed_tweet = {
                                            "tweet_id": tweet.get("id"),
                                            "content": tweet.get("text"),
                                            "timestamp": datetime.strptime(tweet.get("created_at"), "%Y-%m-%dT%H:%M:%S.%fZ") if "created_at" in tweet else datetime.now(),
                                            "source": "twitter",
                                            "user_id": tweet.get("author_id"),
                                            "username": tweet.get("user", {}).get("username") if "user" in tweet else None,
                                            "metric_retweet_count": tweet.get("public_metrics", {}).get("retweet_count", 0) if "public_metrics" in tweet else 0,
                                            "metric_reply_count": tweet.get("public_metrics", {}).get("reply_count", 0) if "public_metrics" in tweet else 0,
                                            "metric_like_count": tweet.get("public_metrics", {}).get("like_count", 0) if "public_metrics" in tweet else 0,
                                            "metric_quote_count": tweet.get("public_metrics", {}).get("quote_count", 0) if "public_metrics" in tweet else 0,
                                        }
                                        
                                        # Extract symbols from entities
                                        if "entities" in tweet and "cashtags" in tweet["entities"]:
                                            processed_tweet["symbols"] = [cashtag["tag"] for cashtag in tweet["entities"]["cashtags"]]
                                        
                                        tweets.append(processed_tweet)
                                except json.JSONDecodeError:
                                    pass
                        
                        if tweets:
                            # Convert to DataFrame and pass to callback
                            df = pd.DataFrame(tweets)
                            callback(df)
                        
                        buffer = ""
        
        except Exception as e:
            logger.error(f"Error in Twitter stream: {e}")
            # Fall back to mock streaming
            self._stream_tweets_mock(symbols, callback)
        
    def _stream_tweets_mock(self, symbols: List[str], callback: Callable[[pd.DataFrame], None]) -> None:
        """
        Generate mock streaming data as a fallback.
        
        Args:
            symbols: List of symbols to generate mock data for
            callback: Callback function to process streamed data
        """
        logger.warning("Falling back to mock streaming")
        
        if not symbols:
            symbols = ["BTC", "ETH"]  # Default symbols if none specified
        
        # Generate mock data every few seconds
        try:
            while True:
                mock_data = self._generate_mock_data(
                    symbols,
                    datetime.now() - timedelta(minutes=5),
                    datetime.now()
                )
                callback(mock_data)
                time.sleep(5)  # Sleep for 5 seconds
        except KeyboardInterrupt:
            logger.info("Mock streaming stopped")
    
    def _process_tweets_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the tweets DataFrame.
        
        Args:
            df: DataFrame with raw tweet data
            
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
        
        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            if "created_at" in df.columns:
                df["timestamp"] = pd.to_datetime(df["created_at"])
            else:
                df["timestamp"] = pd.Timestamp.now()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Ensure all required columns exist
        required_columns = [
            "content", "source", "tweet_id", "user_id",
            "metric_retweet_count", "metric_reply_count", "metric_like_count", "metric_quote_count"
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Add source if not present
        if "source" not in df.columns:
            df["source"] = "twitter"
        
        # Extract symbols from content if not already present
        if "symbols" not in df.columns:
            # Extract cashtags ($BTC, $ETH, etc.)
            df["symbols"] = df["content"].apply(lambda x: re.findall(r'\$([A-Za-z0-9]+)', str(x)) if pd.notna(x) else [])
        
        # Add sentiment placeholder column (will be filled by NLP processor)
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = None
        
        # Add engagement score (weighted combination of metrics)
        if all(col in df.columns for col in ["metric_retweet_count", "metric_like_count", "metric_quote_count"]):
            df["engagement_score"] = (
                df["metric_retweet_count"].fillna(0) * 2 +  # Retweets weighted more
                df["metric_like_count"].fillna(0) * 1 +     # Likes standard weight
                df["metric_quote_count"].fillna(0) * 1.5     # Quotes weighted in between
            )
        else:
            df["engagement_score"] = 0
        
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
