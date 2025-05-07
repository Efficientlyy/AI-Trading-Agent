"""
Reddit API Collector for sentiment analysis.

This module provides functionality to collect posts and comments from Reddit API
for sentiment analysis in the AI Trading Agent.
"""

import os
import json
import time
import logging
import re
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import praw
import pandas as pd

from ...common.logging_config import setup_logging
from ..providers.base_provider import BaseSentimentProvider

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class RedditAPICollector(BaseSentimentProvider):
    """
    Collects posts and comments from Reddit API for sentiment analysis.
    
    Attributes:
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
        user_agent (str): Reddit API user agent
        username (str): Reddit username (optional)
        password (str): Reddit password (optional)
        cache_dir (str): Directory to cache API responses
        cache_expiry (int): Cache expiry time in seconds
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: str = "cache/reddit",
        cache_expiry: int = 3600,
        subreddits: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
    ):
        """
        Initialize the Reddit API collector.
        
        Args:
            client_id: Reddit API client ID (defaults to REDDIT_CLIENT_ID env var)
            client_secret: Reddit API client secret (defaults to REDDIT_CLIENT_SECRET env var)
            user_agent: Reddit API user agent (defaults to REDDIT_USER_AGENT env var)
            username: Reddit username (defaults to REDDIT_USERNAME env var)
            password: Reddit password (defaults to REDDIT_PASSWORD env var)
            cache_dir: Directory to cache API responses
            cache_expiry: Cache expiry time in seconds
            subreddits: List of subreddits to monitor
            keywords: List of keywords to search for
        """
        # Get API credentials from environment variables if not provided
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT", "AI Trading Agent Sentiment Collector")
        self.username = username or os.environ.get("REDDIT_USERNAME")
        self.password = password or os.environ.get("REDDIT_PASSWORD")
        
        # Validate credentials
        if not (self.client_id and self.client_secret):
            logger.warning("No Reddit API credentials provided. Will use mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
        
        # Set up caching
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set up subreddits and keywords
        self.subreddits = subreddits or ["wallstreetbets", "cryptocurrency", "stocks", "investing", "CryptoMarkets"]
        self.keywords = keywords or []
        
        # Initialize PRAW client if credentials are available
        if not self.use_mock_data:
            try:
                if self.username and self.password:
                    self.reddit = praw.Reddit(
                        client_id=self.client_id,
                        client_secret=self.client_secret,
                        user_agent=self.user_agent,
                        username=self.username,
                        password=self.password
                    )
                else:
                    self.reddit = praw.Reddit(
                        client_id=self.client_id,
                        client_secret=self.client_secret,
                        user_agent=self.user_agent
                    )
                
                # Set to read-only mode
                self.reddit.read_only = True
                logger.info("Reddit API client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Reddit API client: {e}")
                self.use_mock_data = True
    
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
        
        # Check cache first
        cache_key = self._get_cache_key(symbols, start_date, end_date, kwargs)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.info(f"Using cached data for symbols: {symbols}")
            return pd.DataFrame(cached_data)
        
        all_posts = []
        
        # Create search queries for each symbol
        for symbol in symbols:
            # Search for the symbol in each subreddit
            for subreddit_name in self.subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Create search queries
                    search_queries = [
                        f"{symbol}",  # Plain symbol
                        f"${symbol}",  # Symbol with $ prefix
                        f"{symbol} stock",  # Symbol + stock
                        f"{symbol} crypto",  # Symbol + crypto
                    ]
                    
                    for query in search_queries:
                        try:
                            # Search for posts
                            submissions = subreddit.search(
                                query, 
                                sort=kwargs.get("sort", "relevance"),
                                time_filter=kwargs.get("time_filter", "week"),
                                limit=kwargs.get("limit", 100)
                            )
                            
                            for submission in submissions:
                                # Check if the post is within the date range
                                post_date = datetime.fromtimestamp(submission.created_utc)
                                if start_date <= post_date <= end_date:
                                    # Extract post data
                                    post_data = {
                                        "post_id": submission.id,
                                        "title": submission.title,
                                        "content": submission.selftext,
                                        "timestamp": post_date,
                                        "source": f"reddit-{subreddit_name}",
                                        "symbol": symbol,
                                        "url": submission.url,
                                        "score": submission.score,
                                        "num_comments": submission.num_comments,
                                        "upvote_ratio": submission.upvote_ratio,
                                        "author": str(submission.author) if submission.author else "[deleted]",
                                    }
                                    
                                    all_posts.append(post_data)
                                    
                                    # Optionally fetch comments
                                    if kwargs.get("include_comments", True):
                                        submission.comments.replace_more(limit=0)  # Skip "load more comments" links
                                        for comment in submission.comments.list()[:kwargs.get("comments_limit", 20)]:
                                            comment_date = datetime.fromtimestamp(comment.created_utc)
                                            if start_date <= comment_date <= end_date:
                                                comment_data = {
                                                    "post_id": submission.id,
                                                    "comment_id": comment.id,
                                                    "content": comment.body,
                                                    "timestamp": comment_date,
                                                    "source": f"reddit-{subreddit_name}-comment",
                                                    "symbol": symbol,
                                                    "score": comment.score,
                                                    "author": str(comment.author) if comment.author else "[deleted]",
                                                }
                                                all_posts.append(comment_data)
                        
                        except Exception as e:
                            logger.error(f"Error searching for {query} in {subreddit_name}: {e}")
                
                except Exception as e:
                    logger.error(f"Error accessing subreddit {subreddit_name}: {e}")
        
        # If no data found, use mock data
        if not all_posts:
            logger.warning(f"No Reddit data found for symbols {symbols}. Using mock data.")
            df = self._generate_mock_data(symbols, start_date, end_date)
            return df
        
        # Convert to DataFrame
        df = pd.DataFrame(all_posts)
        
        # Process and clean the DataFrame
        df = self._process_reddit_dataframe(df)
        
        # Cache the results
        self._save_to_cache(cache_key, df.to_dict("records"))
        
        return df
    
    def stream_sentiment_data(
        self,
        symbols: List[str],
        callback: Callable[[pd.DataFrame], None],
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
            logger.info("Using mock data for streaming")
            
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
        
        # Set up subreddit stream
        subreddit_names = "+".join(self.subreddits)  # Combine subreddits with "+"
        
        try:
            subreddit = self.reddit.subreddit(subreddit_names)
            
            # Stream submissions
            logger.info(f"Starting Reddit stream for subreddits: {subreddit_names}")
            
            for submission in subreddit.stream.submissions():
                # Check if the submission contains any of the symbols
                submission_text = f"{submission.title} {submission.selftext}".lower()
                
                relevant_symbols = []
                for symbol in symbols:
                    # Check for symbol matches (plain symbol, $symbol, etc.)
                    if (
                        symbol.lower() in submission_text or
                        f"${symbol.lower()}" in submission_text or
                        f"{symbol.lower()} stock" in submission_text or
                        f"{symbol.lower()} crypto" in submission_text
                    ):
                        relevant_symbols.append(symbol)
                
                if relevant_symbols:
                    # Extract post data
                    posts = []
                    
                    for symbol in relevant_symbols:
                        post_data = {
                            "post_id": submission.id,
                            "title": submission.title,
                            "content": submission.selftext,
                            "timestamp": datetime.fromtimestamp(submission.created_utc),
                            "source": f"reddit-{submission.subreddit.display_name}",
                            "symbol": symbol,
                            "url": submission.url,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "upvote_ratio": submission.upvote_ratio,
                            "author": str(submission.author) if submission.author else "[deleted]",
                        }
                        
                        posts.append(post_data)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(posts)
                    
                    # Process and clean the DataFrame
                    df = self._process_reddit_dataframe(df)
                    
                    # Pass to callback
                    callback(df)
        
        except KeyboardInterrupt:
            logger.info("Reddit stream stopped by user")
        except Exception as e:
            logger.error(f"Error in Reddit stream: {e}")
            
            # Fall back to mock streaming
            logger.warning("Falling back to mock streaming")
            self._stream_mock_data(symbols, callback)
    
    def _stream_mock_data(
        self,
        symbols: List[str],
        callback: Callable[[pd.DataFrame], None]
    ) -> None:
        """
        Generate mock streaming data as a fallback.
        
        Args:
            symbols: List of symbols to generate mock data for
            callback: Callback function to process streamed data
        """
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
    
    def _process_reddit_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the Reddit DataFrame.
        
        Args:
            df: DataFrame with raw Reddit data
            
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
        
        # Ensure timestamp column exists and is datetime
        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now()
        elif not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Combine title and content for text analysis
        if "title" in df.columns and "content" in df.columns:
            df["text"] = df.apply(
                lambda row: f"{row['title']} {row['content']}" if pd.notna(row["title"]) else row["content"],
                axis=1
            )
        elif "content" in df.columns:
            df["text"] = df["content"]
        
        # Extract symbols from text if not already present
        if "symbols" not in df.columns:
            df["symbols"] = df["text"].apply(
                lambda x: re.findall(r'\$([A-Za-z0-9]+)', str(x)) if pd.notna(x) else []
            )
        
        # Add sentiment placeholder column (will be filled by NLP processor)
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = None
        
        # Calculate engagement score
        if "score" in df.columns and "num_comments" in df.columns:
            df["engagement_score"] = df["score"] + (df["num_comments"] * 2)  # Weight comments more
        elif "score" in df.columns:
            df["engagement_score"] = df["score"]
        else:
            df["engagement_score"] = 0
        
        return df
    
    def _get_cache_key(self, symbols: List[str], start_date: datetime, end_time: datetime, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for the query.
        
        Args:
            symbols: List of symbols to search for
            start_date: Start date for search
            end_time: End date for search
            params: Additional parameters
            
        Returns:
            Cache key string
        """
        symbols_str = "_".join(sorted(symbols))
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_time.strftime("%Y%m%d")
        
        # Include important parameters in the cache key
        include_comments = params.get("include_comments", True)
        time_filter = params.get("time_filter", "week")
        
        return f"reddit_{symbols_str}_{start_str}_{end_str}_{include_comments}_{time_filter}"
    
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
        Generate mock Reddit data for testing.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for mock data
            end_date: End date for mock data
            
        Returns:
            DataFrame with mock Reddit data
        """
        logger.info("Generating mock Reddit data")
        
        # Calculate date range
        date_range = (end_date - start_date).days
        if date_range <= 0:
            date_range = 1
        
        # Generate random data
        import numpy as np
        
        all_data = []
        subreddits = ["wallstreetbets", "cryptocurrency", "stocks", "investing", "CryptoMarkets"]
        
        for symbol in symbols:
            # Generate 3-10 posts per day per symbol
            for day in range(date_range):
                current_date = start_date + timedelta(days=day)
                num_posts = np.random.randint(3, 11)
                
                for _ in range(num_posts):
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
                    
                    # Random subreddit
                    subreddit = np.random.choice(subreddits)
                    
                    # Generate post sentiment
                    sentiment = np.random.choice(["positive", "negative", "neutral"], p=[0.4, 0.3, 0.3])
                    
                    if sentiment == "positive":
                        title = np.random.choice([
                            f"${symbol} is looking bullish! Technical analysis inside",
                            f"Just bought more ${symbol}. Here's why I'm bullish",
                            f"${symbol} earnings beat expectations! 🚀",
                            f"DD: Why ${symbol} is undervalued right now",
                            f"${symbol} breaking out of resistance. Technical analysis"
                        ])
                        content = np.random.choice([
                            f"I've been watching ${symbol} for a while and the technicals are looking great. MACD crossover, RSI showing strength, and volume increasing. I think we're about to see a major move up.",
                            f"${symbol} just announced a new partnership that's going to be huge for their growth. This is going to drive significant revenue in the coming quarters.",
                            f"The fundamentals for ${symbol} are stronger than ever. P/E ratio is attractive, revenue growth is accelerating, and they're expanding into new markets.",
                            f"I've analyzed ${symbol}'s chart patterns and we're seeing a classic cup and handle formation. This is typically followed by a strong upward movement.",
                            f"${symbol} just broke through a major resistance level with high volume. This is a strong buy signal according to technical analysis."
                        ])
                    elif sentiment == "negative":
                        title = np.random.choice([
                            f"Why I'm shorting ${symbol} - Red flags everywhere",
                            f"Just sold all my ${symbol} - Here's why",
                            f"${symbol} missed earnings badly. What now?",
                            f"Technical analysis shows ${symbol} is headed down",
                            f"${symbol} forming a bearish pattern. Be careful"
                        ])
                        content = np.random.choice([
                            f"I've been analyzing ${symbol}'s financials and there are serious concerns. Cash flow is negative, debt is increasing, and growth is slowing.",
                            f"${symbol} is facing increased competition that's going to hurt their margins. I expect significant downward pressure in the coming months.",
                            f"The technical indicators for ${symbol} are all bearish. Death cross on the daily chart, RSI showing weakness, and volume declining on up days.",
                            f"${symbol}'s management team has been making questionable decisions lately. I've lost confidence in their ability to execute on their strategy.",
                            f"${symbol} is overvalued by every metric. P/E ratio is through the roof, revenue growth doesn't justify the price, and the market is becoming more rational."
                        ])
                    else:  # neutral
                        title = np.random.choice([
                            f"${symbol} - What's your take?",
                            f"Considering buying ${symbol} - Need advice",
                            f"${symbol} earnings coming up - Predictions?",
                            f"Is ${symbol} a good long-term hold?",
                            f"${symbol} trading sideways - What's next?"
                        ])
                        content = np.random.choice([
                            f"I've been watching ${symbol} for a while and I'm not sure what to make of it. There are some positive signs but also some concerns. What's your analysis?",
                            f"${symbol} has been trading in a tight range for weeks now. I'm not seeing any clear signals either way. Anyone have insights on potential catalysts?",
                            f"I'm trying to decide whether to add ${symbol} to my portfolio. The fundamentals seem solid but the price action has been underwhelming. Thoughts?",
                            f"${symbol} is in an interesting position right now. On one hand, their new products look promising. On the other hand, the market conditions are challenging.",
                            f"I've been holding ${symbol} for about a year now with minimal movement. Not sure whether to keep holding or reallocate to something with more momentum."
                        ])
                    
                    # Generate post data
                    post = {
                        "post_id": f"mock_{symbol}_{timestamp.strftime('%Y%m%d%H%M%S')}",
                        "title": title,
                        "content": content,
                        "text": f"{title} {content}",
                        "timestamp": timestamp,
                        "source": f"reddit-{subreddit}",
                        "symbol": symbol,
                        "symbols": [symbol],
                        "url": f"https://reddit.com/r/{subreddit}/mock_{symbol}_{timestamp.strftime('%Y%m%d%H%M%S')}",
                        "score": np.random.randint(1, 1000) if sentiment == "positive" else np.random.randint(1, 500),
                        "num_comments": np.random.randint(0, 200),
                        "upvote_ratio": np.random.uniform(0.7, 1.0) if sentiment == "positive" else np.random.uniform(0.5, 0.9),
                        "author": f"user_{np.random.randint(1000, 9999)}",
                        "engagement_score": np.random.randint(10, 1000),
                    }
                    
                    all_data.append(post)
                    
                    # Generate 0-5 comments for each post
                    num_comments = np.random.randint(0, 6)
                    for c in range(num_comments):
                        comment_time = timestamp + timedelta(minutes=np.random.randint(1, 120))
                        
                        # Ensure comment time is within the specified range
                        if comment_time > end_date:
                            comment_time = end_date
                        
                        # Generate comment sentiment (slightly biased towards the post sentiment)
                        if sentiment == "positive":
                            comment_sentiment = np.random.choice(["positive", "negative", "neutral"], p=[0.6, 0.2, 0.2])
                        elif sentiment == "negative":
                            comment_sentiment = np.random.choice(["positive", "negative", "neutral"], p=[0.2, 0.6, 0.2])
                        else:
                            comment_sentiment = np.random.choice(["positive", "negative", "neutral"], p=[0.33, 0.33, 0.34])
                        
                        if comment_sentiment == "positive":
                            comment_text = np.random.choice([
                                f"Great analysis! I'm also bullish on ${symbol}.",
                                f"I've been accumulating ${symbol} for months. This is going to be huge.",
                                f"${symbol} is one of my best performers this year.",
                                f"Thanks for sharing! Just bought some ${symbol} based on this.",
                                f"The fundamentals for ${symbol} are so strong right now."
                            ])
                        elif comment_sentiment == "negative":
                            comment_text = np.random.choice([
                                f"I disagree. ${symbol} is overvalued at current levels.",
                                f"I sold all my ${symbol} last week. Too much risk right now.",
                                f"${symbol} has strong competition that's not being priced in.",
                                f"The chart for ${symbol} looks bearish to me. Be careful.",
                                f"I've heard rumors that ${symbol} might miss earnings."
                            ])
                        else:  # neutral
                            comment_text = np.random.choice([
                                f"Interesting take on ${symbol}. I'm still researching.",
                                f"What's your price target for ${symbol}?",
                                f"How long have you been following ${symbol}?",
                                f"Any thoughts on how ${symbol} compares to its competitors?",
                                f"I'm on the fence about ${symbol}. Need to do more DD."
                            ])
                        
                        # Create comment object
                        comment = {
                            "post_id": post["post_id"],
                            "comment_id": f"comment_{post['post_id']}_{c}",
                            "content": comment_text,
                            "text": comment_text,
                            "timestamp": comment_time,
                            "source": f"reddit-{subreddit}-comment",
                            "symbol": symbol,
                            "symbols": [symbol],
                            "score": np.random.randint(1, 100),
                            "author": f"user_{np.random.randint(1000, 9999)}",
                            "engagement_score": np.random.randint(1, 100),
                        }
                        
                        all_data.append(comment)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        return df
