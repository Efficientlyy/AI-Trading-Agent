"""
Reddit sentiment data provider.

This module connects to Reddit API using PRAW to fetch posts/comments and extract sentiment data.
"""

import os
from typing import List, Dict, Any, Iterator, Optional
import praw
from .base_provider import BaseSentimentProvider
from ..nlp_processing.sentiment_processor import SentimentProcessor # Import the SentimentProcessor

class RedditSentimentProvider(BaseSentimentProvider):
    """
    Fetches and streams sentiment data from Reddit API using PRAW.
    """
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, user_agent: Optional[str] = None, **kwargs):
        """
        Initializes the RedditSentimentProvider.

        Args:
            client_id: Reddit API client ID.
            client_secret: Reddit API client secret.
            user_agent: Reddit API user agent string.
            **kwargs: Additional configuration parameters.
        """
        # TODO: Load credentials from config or environment variables if not provided
        # Prioritize explicit arguments, then environment variables with default empty string
        reddit_client_id: str = client_id if client_id is not None else os.environ.get("REDDIT_CLIENT_ID", "")
        reddit_client_secret: str = client_secret if client_secret is not None else os.environ.get("REDDIT_CLIENT_SECRET", "")
        reddit_user_agent: str = user_agent if user_agent is not None else os.environ.get("REDDIT_USER_AGENT", "AI Trading Agent Sentiment Collector")

        self.reddit = praw.Reddit(
            client_id=str(reddit_client_id),
            client_secret=str(reddit_client_secret),
            user_agent=str(reddit_user_agent),
            # Add username/password if needed for script apps, but client_id/secret is preferred
            # username=os.getenv("REDDIT_USERNAME"),
            # password=os.getenv("REDDIT_PASSWORD")
        )
        # Ensure read-only mode if only fetching data
        self.reddit.read_only = True # Set to False if posting/commenting is needed

        # TODO: Add configuration for subreddits, keywords, etc.
        self.subreddits = kwargs.get("subreddits", ["wallstreetbets", "cryptocurrency"]) # Example defaults
        self.keywords = kwargs.get("keywords", []) # Example defaults

        # Initialize the SentimentProcessor
        self.sentiment_processor = SentimentProcessor(**kwargs.get("nlp_config", {}))

    def fetch_sentiment_data(self, limit: int = 100, time_filter: str = 'day', **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch recent Reddit posts/comments and extract relevant data.

        Args:
            limit: The maximum number of items to fetch.
            time_filter: The time period to fetch from ('hour', 'day', 'week', 'month', 'year', 'all').
            **kwargs: Additional parameters for fetching (e.g., specific subreddits or keywords).

        Returns:
            List of dictionaries containing relevant data (e.g., text, timestamp, source).
        """
        sentiment_data = []
        search_query = " OR ".join(self.keywords) if self.keywords else ""

        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                # Fetch posts - can also fetch comments separately
                if search_query:
                    submissions = subreddit.search(search_query, limit=limit, time_filter=time_filter)
                else:
                    # Fetch top, hot, new, or controversial submissions
                    submissions = subreddit.hot(limit=limit) # Example: fetching hot posts

                for submission in submissions:
                    # Extract relevant data from submission
                    sentiment_data.append({
                        "source": f"reddit-{subreddit_name}",
                        "timestamp": submission.created_utc, # Unix timestamp
                        "text": submission.title + " " + submission.selftext, # Combine title and body
                        "url": submission.url,
                        "score": submission.score, # Reddit score
                        "num_comments": submission.num_comments,
                        # TODO: Optionally fetch and include comments
                    })
            except Exception as e:
                print(f"Error fetching data from r/{subreddit_name}: {e}") # TODO: Use proper logging

        # TODO: Integrate with NLP pipeline here or in a separate processing step
        # Process the raw data using the SentimentProcessor
        processed_sentiment_data = self.sentiment_processor.process_data(sentiment_data)

        return processed_sentiment_data

    def stream_sentiment_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Stream Reddit posts/comments and extract relevant data in real-time.

        Args:
            **kwargs: Additional parameters for streaming (e.g., specific subreddits or keywords).

        Yields:
            Dictionaries containing relevant data (e.g., text, timestamp, source).
        """
        # TODO: Implement proper error handling and logging
        try:
            # Stream submissions from a default subreddit (e.g., 'all')
            # This will yield new submissions as they are posted
            for submission in self.reddit.subreddit("all").stream.submissions():
                # Extract relevant data from submission
                raw_entry = {
                    "source": f"reddit-{submission.subreddit.display_name}",
                    "timestamp": submission.created_utc, # Unix timestamp
                    "text": submission.title + " " + submission.selftext, # Combine title and body
                    "url": submission.url,
                    "score": submission.score, # Reddit score
                    "num_comments": submission.num_comments,
                    # TODO: Optionally fetch and include comments
                }
                # Process the raw entry using the SentimentProcessor before yielding
                processed_entry = self.sentiment_processor.process_stream_entry(raw_entry)
                yield processed_entry
        except Exception as e:
            print(f"Error during Reddit streaming: {e}") # TODO: Use proper logging
            # Depending on the error, you might want to reconnect or stop streaming
            # For now, just print the error and the generator will stop
