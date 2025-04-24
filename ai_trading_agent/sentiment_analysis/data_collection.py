"""
Sentiment Data Collection Module.

This module provides classes and functions for collecting sentiment data from various sources
such as social media platforms, news articles, and market sentiment indicators.
"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import time

logger = logging.getLogger(__name__)

class BaseSentimentCollector(ABC):
    """
    Base class for sentiment data collectors.
    
    This abstract class defines the interface that all sentiment data collectors must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment collector with configuration.
        
        Args:
            config: Configuration dictionary with source-specific settings
        """
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initializing {self.name}")
    
    @abstractmethod
    def collect(self, symbols: List[str], start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect sentiment data for the specified symbols and time range.
        
        Args:
            symbols: List of asset symbols to collect sentiment data for
            start_date: Start date for data collection (if None, use a default like 7 days ago)
            end_date: End date for data collection (if None, use current date)
            
        Returns:
            DataFrame with sentiment data, should include at minimum:
            - timestamp: Datetime of the sentiment data point
            - symbol: Asset symbol
            - source: Source of the sentiment data
            - sentiment_score: Numerical score representing sentiment (-1 to 1)
            - volume: Volume or count of mentions/articles
            - additional source-specific fields
        """
        pass
    
    def _validate_dates(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> tuple:
        """
        Validate and set default values for date parameters.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Tuple of (start_date, end_date) with defaults applied if needed
        """
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            # Default to 7 days before end_date
            start_date = end_date - timedelta(days=7)
            
        if start_date > end_date:
            raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
            
        return start_date, end_date


class TwitterSentimentCollector(BaseSentimentCollector):
    """
    Collector for Twitter/X sentiment data.
    
    Uses the Twitter API to collect tweets mentioning specific assets and calculates
    sentiment scores based on the content.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Twitter sentiment collector.
        
        Args:
            config: Configuration dictionary with Twitter API credentials and settings
        """
        super().__init__(config)
        # Initialize Twitter API client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the Twitter API client using credentials from config."""
        # TODO: Implement Twitter API client initialization
        # This will depend on the specific Twitter API library being used
        self.client = None
        logger.info("Twitter API client initialization placeholder")
        
    def collect(self, symbols: List[str], start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect Twitter sentiment data for the specified symbols and time range.
        
        Args:
            symbols: List of asset symbols to collect sentiment data for
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with Twitter sentiment data
        """
        start_date, end_date = self._validate_dates(start_date, end_date)
        logger.info(f"Collecting Twitter sentiment data for {symbols} from {start_date} to {end_date}")
        
        # TODO: Implement actual Twitter API calls and data processing
        # For now, return a mock DataFrame with sample data
        
        # Create a sample DataFrame with mock data
        data = []
        current_date = start_date
        while current_date <= end_date:
            for symbol in symbols:
                # Generate mock data
                data.append({
                    'timestamp': current_date,
                    'symbol': symbol,
                    'source': 'twitter',
                    'sentiment_score': 0.0,  # Placeholder
                    'volume': 0,  # Placeholder
                    'positive_count': 0,  # Placeholder
                    'negative_count': 0,  # Placeholder
                    'neutral_count': 0,  # Placeholder
                })
            current_date += timedelta(hours=1)
            
        return pd.DataFrame(data)


class RedditSentimentCollector(BaseSentimentCollector):
    """
    Collector for Reddit sentiment data.
    
    Uses the Reddit API to collect posts and comments from relevant subreddits
    and calculates sentiment scores based on the content.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Reddit sentiment collector.
        
        Args:
            config: Configuration dictionary with Reddit API credentials and settings
        """
        super().__init__(config)
        # Initialize Reddit API client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the Reddit API client using PRAW library."""
        try:
            import praw
            from ..nlp_processing.sentiment_processor import SentimentProcessor
            
            # Extract configuration values with defaults
            client_id = self.config.get('client_id', '')
            client_secret = self.config.get('client_secret', '')
            user_agent = self.config.get('user_agent', 'AI Trading Agent Sentiment Collector')
            
            # Use environment variables as fallback
            if not client_id:
                client_id = os.environ.get("REDDIT_CLIENT_ID", "")
            if not client_secret:
                client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
            if not user_agent:
                user_agent = os.environ.get("REDDIT_USER_AGENT", "AI Trading Agent Sentiment Collector")
                
            # Configure subreddits and keywords
            self.subreddits = self.config.get('subreddits', ["wallstreetbets", "cryptocurrency", "stocks", "investing"])
            self.keywords = self.config.get('keywords', [])
            self.comment_limit = self.config.get('comment_limit', 10)  # Number of top comments to fetch per post
            self.post_limit = self.config.get('post_limit', 100)  # Number of posts to fetch per subreddit
            self.time_filter = self.config.get('time_filter', 'day')  # Time filter for Reddit search
            
            # Initialize PRAW Reddit client
            self.client = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                # Additional credentials if needed for script apps
                username=self.config.get('username', os.environ.get("REDDIT_USERNAME", "")),
                password=self.config.get('password', os.environ.get("REDDIT_PASSWORD", ""))
            )
            
            # Ensure read-only mode for data collection
            self.client.read_only = True
            
            # Initialize SentimentProcessor for text analysis
            self.sentiment_processor = SentimentProcessor(**self.config.get('nlp_config', {}))
            
            # Rate limiting
            self.rate_limit_wait = self.config.get('rate_limit_wait', 1.0)  # Wait time in seconds between API calls
            
            logger.info(f"Reddit API client initialized successfully with {len(self.subreddits)} subreddits")
            
        except ImportError as e:
            logger.error(f"Failed to import required libraries for Reddit API: {e}")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing Reddit API client: {e}")
            self.client = None
        
    def collect(self, symbols: List[str], start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect Reddit sentiment data for the specified symbols and time range.
        
        Args:
            symbols: List of asset symbols to collect sentiment data for
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with Reddit sentiment data
        """
        start_date, end_date = self._validate_dates(start_date, end_date)
        logger.info(f"Collecting Reddit sentiment data for {symbols} from {start_date} to {end_date}")
        
        # If client initialization failed, return mock data
        if self.client is None:
            logger.warning("Reddit API client not initialized, returning mock data")
            return self._generate_mock_data(symbols, start_date, end_date)
            
        # Check if we have actual symbols to search for
        if not symbols:
            logger.warning("No symbols provided for Reddit sentiment collection")
            return pd.DataFrame()  # Return empty DataFrame
            
        try:
            # Collect sentiment data for each symbol
            all_data = []
            
            # Create search queries based on symbols
            # Convert symbols to search terms (e.g., "BTC" -> "BTC OR Bitcoin", "ETH" -> "ETH OR Ethereum")
            search_queries = self._create_search_queries(symbols)
            
            # Collect data for each subreddit and search query
            for subreddit_name in self.subreddits:
                try:
                    subreddit = self.client.subreddit(subreddit_name)
                    
                    for symbol, search_query in search_queries.items():
                        logger.debug(f"Searching r/{subreddit_name} for '{search_query}'")
                        
                        try:
                            # Search for submissions matching the query
                            submissions = subreddit.search(
                                search_query, 
                                limit=self.post_limit, 
                                time_filter=self.time_filter,
                                sort='relevance'
                            )
                            
                            # Process each submission
                            for submission in submissions:
                                # Skip submissions outside the requested date range
                                submission_time = datetime.fromtimestamp(submission.created_utc)
                                if submission_time < start_date or submission_time > end_date:
                                    continue
                                
                                # Process submission text
                                submission_text = f"{submission.title} {submission.selftext}"
                                submission_data = {
                                    'timestamp': submission_time,
                                    'symbol': symbol,
                                    'source': 'reddit',
                                    'subreddit': subreddit_name,
                                    'content': submission_text,
                                    'url': submission.url,
                                    'score': submission.score,
                                    'num_comments': submission.num_comments,
                                    'is_comment': False,
                                    'post_id': submission.id
                                }
                                
                                # Process sentiment
                                submission_sentiment = self.sentiment_processor.process_data([{'text': submission_text}])
                                if submission_sentiment and len(submission_sentiment) > 0:
                                    submission_data['sentiment_score'] = submission_sentiment[0].get('sentiment_score', 0.0)
                                else:
                                    submission_data['sentiment_score'] = 0.0
                                
                                all_data.append(submission_data)
                                
                                # Process top comments if requested
                                if self.comment_limit > 0:
                                    submission.comments.replace_more(limit=0)  # Skip "load more comments" objects
                                    top_comments = list(submission.comments)[:self.comment_limit]
                                    
                                    for comment in top_comments:
                                        comment_time = datetime.fromtimestamp(comment.created_utc)
                                        
                                        # Skip comments outside the requested date range
                                        if comment_time < start_date or comment_time > end_date:
                                            continue
                                            
                                        # Process comment text
                                        comment_text = comment.body
                                        comment_data = {
                                            'timestamp': comment_time,
                                            'symbol': symbol,
                                            'source': 'reddit',
                                            'subreddit': subreddit_name,
                                            'content': comment_text,
                                            'url': f"https://reddit.com{comment.permalink}",
                                            'score': comment.score,
                                            'num_comments': 0,  # Comments don't have nested comments count
                                            'is_comment': True,
                                            'post_id': submission.id,
                                            'comment_id': comment.id
                                        }
                                        
                                        # Process sentiment
                                        comment_sentiment = self.sentiment_processor.process_data([{'text': comment_text}])
                                        if comment_sentiment and len(comment_sentiment) > 0:
                                            comment_data['sentiment_score'] = comment_sentiment[0].get('sentiment_score', 0.0)
                                        else:
                                            comment_data['sentiment_score'] = 0.0
                                            
                                        all_data.append(comment_data)
                                
                                # Apply rate limiting to prevent API abuse
                                time.sleep(self.rate_limit_wait)
                                
                        except Exception as e:
                            logger.error(f"Error searching r/{subreddit_name} for '{search_query}': {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error accessing subreddit r/{subreddit_name}: {e}")
                    continue
            
            # Convert to DataFrame
            if not all_data:
                logger.warning("No Reddit data collected, returning mock data")
                return self._generate_mock_data(symbols, start_date, end_date)
                
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Add volume column (count of mentions)
            volume_df = df.groupby(['symbol', 'subreddit']).size().reset_index(name='volume')
            df = pd.merge(df, volume_df, on=['symbol', 'subreddit'], how='left')
            
            # Add post_count and comment_count
            post_count_df = df[~df['is_comment']].groupby('symbol').size().reset_index(name='post_count')
            comment_count_df = df[df['is_comment']].groupby('symbol').size().reset_index(name='comment_count')
            
            # Merge counts back to main df
            df = pd.merge(df, post_count_df, on='symbol', how='left')
            df = pd.merge(df, comment_count_df, on='symbol', how='left')
            
            # Fill missing values
            df['post_count'] = df['post_count'].fillna(0).astype(int)
            df['comment_count'] = df['comment_count'].fillna(0).astype(int)
            
            logger.info(f"Collected {len(df)} Reddit sentiment data points")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting Reddit sentiment data: {e}")
            return self._generate_mock_data(symbols, start_date, end_date)
    
    def _create_search_queries(self, symbols: List[str]) -> Dict[str, str]:
        """
        Create search queries for each symbol.
        
        Args:
            symbols: List of asset symbols
            
        Returns:
            Dictionary mapping symbols to search queries
        """
        # Common name mappings for crypto and stocks
        symbol_mappings = {
            'BTC': ['Bitcoin', 'BTC'],
            'ETH': ['Ethereum', 'ETH'],
            'DOGE': ['Dogecoin', 'DOGE'],
            'AAPL': ['Apple', 'AAPL'],
            'MSFT': ['Microsoft', 'MSFT'],
            'GOOGL': ['Google', 'Alphabet', 'GOOGL'],
            'AMZN': ['Amazon', 'AMZN'],
            'TSLA': ['Tesla', 'TSLA']
            # Add more mappings as needed
        }
        
        # Create search queries
        queries = {}
        for symbol in symbols:
            # Use mapping if available, otherwise just use the symbol
            if symbol in symbol_mappings:
                terms = symbol_mappings[symbol]
                query = ' OR '.join([f'"{term}"' for term in terms])
            else:
                # For unknown symbols, just search for the symbol itself
                query = f'"{symbol}"'
                
            # Add custom keywords from config if available
            if self.keywords:
                symbol_specific_keywords = [f'{symbol} {kw}' for kw in self.keywords]
                query += ' OR ' + ' OR '.join([f'"{kw}"' for kw in symbol_specific_keywords])
                
            queries[symbol] = query
            
        return queries
            
    def _generate_mock_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate mock Reddit sentiment data for testing.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with mock Reddit sentiment data
        """
        import random
        import numpy as np
        
        # Create mock data
        data = []
        current_date = start_date
        
        # Mock subreddits
        mock_subreddits = ['wallstreetbets', 'cryptocurrency', 'stocks', 'investing']
        
        while current_date <= end_date:
            for symbol in symbols:
                # Generate mock sentiment scores - slightly biased negative or positive based on symbol
                bias = hash(symbol) % 10 / 10.0  # Deterministic bias based on symbol
                sentiment_score = (random.random() - 0.5) * 2.0 * 0.8 + bias * 0.2
                sentiment_score = max(min(sentiment_score, 1.0), -1.0)  # Clamp to [-1.0, 1.0]
                
                # Pick a random subreddit
                subreddit = random.choice(mock_subreddits)
                
                # Generate mock volume counts
                volume = int(random.expovariate(0.1)) + 1  # Exponential distribution
                post_count = max(1, int(volume * 0.3))
                comment_count = volume - post_count
                
                # Add entry
                data.append({
                    'timestamp': current_date,
                    'symbol': symbol,
                    'source': 'reddit',
                    'sentiment_score': sentiment_score,
                    'volume': volume,
                    'subreddit': subreddit,
                    'post_count': post_count,
                    'comment_count': comment_count,
                    'content': f"Mock Reddit content for {symbol}",  # Mock content
                    'url': f"https://reddit.com/r/{subreddit}/mock/{symbol}",  # Mock URL
                    'is_comment': False
                })
            
            # Advance by a random time interval (average 4 hours)
            hours_increment = random.expovariate(0.25)  # Mean of 4 hours
            current_date += timedelta(hours=hours_increment)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df


class NewsAPISentimentCollector(BaseSentimentCollector):
    """
    Collector for news article sentiment data.
    
    Uses news APIs (like NewsAPI, Alpha Vantage News, etc.) to collect news articles
    about specific assets and calculates sentiment scores based on the content.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the news API sentiment collector.
        
        Args:
            config: Configuration dictionary with news API credentials and settings
        """
        super().__init__(config)
        # Initialize news API client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the news API client using credentials from config."""
        # TODO: Implement news API client initialization
        # This will depend on the specific news API being used
        self.client = None
        logger.info("News API client initialization placeholder")
        
    def collect(self, symbols: List[str], start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect news sentiment data for the specified symbols and time range.
        
        Args:
            symbols: List of asset symbols to collect sentiment data for
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with news sentiment data
        """
        start_date, end_date = self._validate_dates(start_date, end_date)
        logger.info(f"Collecting news sentiment data for {symbols} from {start_date} to {end_date}")
        
        # TODO: Implement actual news API calls and data processing
        # For now, return a mock DataFrame with sample data
        
        # Create a sample DataFrame with mock data
        data = []
        current_date = start_date
        while current_date <= end_date:
            for symbol in symbols:
                # Generate mock data
                data.append({
                    'timestamp': current_date,
                    'symbol': symbol,
                    'source': 'news',
                    'sentiment_score': 0.0,  # Placeholder
                    'volume': 0,  # Placeholder
                    'title': '',  # Placeholder
                    'source_name': '',  # Placeholder
                    'url': '',  # Placeholder
                })
            current_date += timedelta(hours=6)  # Less frequent than social media
            
        return pd.DataFrame(data)


class FearGreedIndexCollector(BaseSentimentCollector):
    """
    Collector for market sentiment indicators like the Fear & Greed Index.
    
    Uses APIs or web scraping to collect market sentiment indicators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Fear & Greed Index collector.
        
        Args:
            config: Configuration dictionary with API credentials and settings
        """
        super().__init__(config)
        # Initialize client if needed
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the client for accessing Fear & Greed Index data."""
        # This might use web scraping or a specific API
        self.client = None
        logger.info("Fear & Greed Index client initialization placeholder")
        
    def collect(self, symbols: List[str], start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect Fear & Greed Index data for the specified time range.
        
        Note: This collector typically returns market-wide sentiment, not symbol-specific.
        
        Args:
            symbols: List of asset symbols (not used directly, but kept for interface consistency)
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with Fear & Greed Index data
        """
        start_date, end_date = self._validate_dates(start_date, end_date)
        logger.info(f"Collecting Fear & Greed Index data from {start_date} to {end_date}")
        
        # TODO: Implement actual data collection
        # For now, return a mock DataFrame with sample data
        
        # Create a sample DataFrame with mock data
        data = []
        current_date = start_date
        while current_date <= end_date:
            # Generate mock data - one entry per day
            data.append({
                'timestamp': current_date,
                'symbol': 'MARKET',  # Market-wide indicator
                'source': 'fear_greed_index',
                'sentiment_score': 0.0,  # Placeholder
                'value': 50,  # Placeholder (0-100 scale)
                'classification': 'Neutral',  # Placeholder
            })
            current_date += timedelta(days=1)  # Daily data
            
        return pd.DataFrame(data)


class SentimentCollectionService:
    """
    Service for managing and coordinating multiple sentiment data collectors.
    
    This service initializes and manages multiple sentiment collectors, and provides
    a unified interface for collecting and aggregating sentiment data from all sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment collection service.
        
        Args:
            config: Configuration dictionary with settings for all collectors
        """
        self.config = config
        self.collectors = {}
        self._initialize_collectors()
        
    def _initialize_collectors(self):
        """Initialize all configured sentiment collectors."""
        collector_configs = self.config.get('collectors', {})
        
        # Initialize Twitter collector if configured
        if 'twitter' in collector_configs:
            self.collectors['twitter'] = TwitterSentimentCollector(collector_configs['twitter'])
            
        # Initialize Reddit collector if configured
        if 'reddit' in collector_configs:
            self.collectors['reddit'] = RedditSentimentCollector(collector_configs['reddit'])
            
        # Initialize News API collector if configured
        if 'news' in collector_configs:
            self.collectors['news'] = NewsAPISentimentCollector(collector_configs['news'])
            
        # Initialize Fear & Greed Index collector if configured
        if 'fear_greed_index' in collector_configs:
            self.collectors['fear_greed_index'] = FearGreedIndexCollector(collector_configs['fear_greed_index'])
            
        logger.info(f"Initialized {len(self.collectors)} sentiment collectors: {list(self.collectors.keys())}")
        
    def collect_all(self, symbols: List[str], start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect sentiment data from all configured collectors.
        
        Args:
            symbols: List of asset symbols to collect sentiment data for
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with combined sentiment data from all collectors
        """
        all_data = []
        
        for name, collector in self.collectors.items():
            try:
                data = collector.collect(symbols, start_date, end_date)
                all_data.append(data)
                logger.info(f"Collected {len(data)} sentiment data points from {name}")
            except Exception as e:
                logger.error(f"Error collecting sentiment data from {name}: {e}")
                
        if not all_data:
            logger.warning("No sentiment data collected from any source")
            return pd.DataFrame()
            
        # Combine all data into a single DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Collected a total of {len(combined_data)} sentiment data points")
        
        return combined_data
