"""
Twitter data collector for the Early Event Detection System.

This module provides functionality for collecting data from Twitter/X
using the Twitter API v2.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set

from src.common.config import config
from src.common.logging import get_logger
from src.common.api_client import RetryableAPIClient, CircuitBreaker
from src.analysis_agents.early_detection.models import SourceType, EventSource
from src.analysis_agents.early_detection.optimization import get_cost_optimizer


class TwitterCollector:
    """Collector for Twitter/X data."""
    
    def __init__(self):
        """Initialize the Twitter collector."""
        self.logger = get_logger("early_detection", "twitter_collector")
        
        # Configuration
        self.keywords = config.get("early_detection.data_collection.social_media.twitter.keywords", 
                                  ["crypto", "bitcoin", "ethereum"])
        self.accounts = config.get("early_detection.data_collection.social_media.twitter.accounts", 
                                  ["elonmusk", "cz_binance"])
        
        # API credentials
        self.api_key = os.getenv("TWITTER_API_KEY") or config.get("apis.twitter.api_key", "")
        self.api_secret = os.getenv("TWITTER_API_SECRET") or config.get("apis.twitter.api_secret", "")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN") or config.get("apis.twitter.access_token", "")
        self.access_secret = os.getenv("TWITTER_ACCESS_SECRET") or config.get("apis.twitter.access_secret", "")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN") or config.get("apis.twitter.bearer_token", "")
        
        # API client
        self.api_client = RetryableAPIClient(
            max_retries=3,
            backoff_factor=2.0,
            logger=self.logger
        )
        
        # Client initialization
        self.client = None
        self.use_mock = True
        
        # Cache for user IDs
        self.user_id_cache = {}
        
        # Last ID tracking for incremental loading
        self.last_tweet_ids = {
            "keyword_search": {},
            "user_tweets": {}
        }
    
    async def initialize(self):
        """Initialize the Twitter collector."""
        # Check if we have valid credentials
        if all([self.api_key, self.api_secret, self.bearer_token]):
            try:
                import tweepy
                
                # Initialize the client
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_secret,
                    wait_on_rate_limit=True
                )
                
                # Test the connection
                self.logger.info("Testing Twitter API connection")
                await self._execute_api_call(
                    lambda: self.client.get_me()
                )
                
                self.use_mock = False
                self.logger.info("Twitter API client initialized successfully")
                
                # Pre-cache user IDs for monitored accounts
                await self._cache_user_ids(self.accounts)
                
            except ImportError:
                self.logger.warning("Tweepy not installed, using mock data instead")
            except Exception as e:
                self.logger.error(f"Error initializing Twitter client: {e}, using mock data instead")
        else:
            self.logger.warning("Missing Twitter API credentials, using mock data")
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from Twitter.
        
        Returns:
            List of collected data items
        """
        if self.use_mock or self.client is None:
            return await self._mock_collect()
        
        try:
            self.logger.info("Collecting data from Twitter")
            
            collected_data = []
            
            # Get cost optimizer for API efficiency
            cost_optimizer = await get_cost_optimizer()
            
            # Check if we should sample Twitter based on adaptive sampling
            if not cost_optimizer.adaptive_sampler.should_sample("twitter"):
                self.logger.info("Skipping Twitter collection due to adaptive sampling")
                return []
            
            # Collect tweets based on keywords
            keyword_tasks = []
            for keyword in self.keywords:
                task = asyncio.create_task(
                    cost_optimizer.api_request(
                        "twitter",
                        self._search_tweets,
                        keyword
                    )
                )
                keyword_tasks.append(task)
            
            # Collect tweets from monitored accounts
            account_tasks = []
            for account in self.accounts:
                task = asyncio.create_task(
                    cost_optimizer.api_request(
                        "twitter",
                        self._get_user_tweets,
                        account
                    )
                )
                account_tasks.append(task)
            
            # Wait for all tasks to complete
            keyword_results = await asyncio.gather(*keyword_tasks, return_exceptions=True)
            account_results = await asyncio.gather(*account_tasks, return_exceptions=True)
            
            # Process keyword results
            for result in keyword_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error searching tweets: {result}")
                    continue
                
                if result:
                    collected_data.extend(result)
            
            # Process account results
            for result in account_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error getting user tweets: {result}")
                    continue
                
                if result:
                    collected_data.extend(result)
            
            self.logger.info(f"Collected {len(collected_data)} items from Twitter")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting Twitter data: {e}")
            return await self._mock_collect()
    
    async def _search_tweets(self, keyword: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for tweets matching a keyword.
        
        Args:
            keyword: The keyword to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of collected tweet data
        """
        self.logger.debug(f"Searching for tweets with keyword: {keyword}")
        
        collected_data = []
        
        try:
            # Get the last tweet ID for this keyword
            since_id = self.last_tweet_ids["keyword_search"].get(keyword)
            
            # Build the query
            query = f"{keyword} -is:retweet -is:reply lang:en"
            
            # Execute API call
            response = await self._execute_api_call(
                lambda: self.client.search_recent_tweets(
                    query=query,
                    max_results=min(max_results, 100),  # API limit is 100
                    tweet_fields=["id", "text", "created_at", "public_metrics", "entities", "author_id"],
                    user_fields=["id", "name", "username", "public_metrics"],
                    expansions=["author_id"],
                    since_id=since_id
                )
            )
            
            if not response or not response.data:
                self.logger.debug(f"No tweets found for keyword: {keyword}")
                return []
            
            # Extract tweets
            tweets = response.data
            
            # Create a dictionary of users for easy lookup
            users = {user.id: user for user in response.includes.get("users", [])} if response.includes else {}
            
            # Update the last tweet ID for incremental loading
            if tweets:
                self.last_tweet_ids["keyword_search"][keyword] = tweets[0].id
            
            # Convert to collected data format
            for tweet in tweets:
                # Get the author
                author = users.get(tweet.author_id) if hasattr(tweet, "author_id") else None
                author_username = author.username if author else "unknown"
                author_followers = author.public_metrics.get("followers_count", 0) if author and hasattr(author, "public_metrics") else 0
                
                # Get metrics
                metrics = tweet.public_metrics if hasattr(tweet, "public_metrics") else {}
                
                # Get entities
                entities = tweet.entities if hasattr(tweet, "entities") else {}
                hashtags = [tag["tag"] for tag in entities.get("hashtags", [])]
                mentions = [mention["username"] for mention in entities.get("mentions", [])]
                
                # Estimate influence based on author's followers and engagement
                retweet_count = metrics.get("retweet_count", 0)
                like_count = metrics.get("like_count", 0)
                reply_count = metrics.get("reply_count", 0)
                engagement = retweet_count + like_count + reply_count
                
                # Scale influence between 0-1 based on followers and engagement
                influence_score = min(1.0, (
                    0.7 * min(1.0, author_followers / 1000000) +  # Followers (max 1M)
                    0.3 * min(1.0, engagement / 1000)             # Engagement (max 1000)
                ))
                
                source = EventSource(
                    id=f"twitter_{tweet.id}",
                    type=SourceType.SOCIAL_MEDIA,
                    name="Twitter",
                    url=f"https://twitter.com/{author_username}/status/{tweet.id}",
                    reliability_score=0.6  # Twitter is less reliable than official sources
                )
                
                collected_data.append({
                    "source": source,
                    "content": tweet.text,
                    "timestamp": tweet.created_at,
                    "metadata": {
                        "keyword": keyword,
                        "user": author_username,
                        "followers": author_followers,
                        "retweets": retweet_count,
                        "likes": like_count,
                        "replies": reply_count,
                        "hashtags": hashtags,
                        "mentions": mentions,
                        "influence_score": influence_score
                    }
                })
            
            self.logger.debug(f"Found {len(collected_data)} tweets for keyword: {keyword}")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error searching tweets for keyword {keyword}: {e}")
            return []
    
    async def _get_user_tweets(self, username: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Get tweets from a specific user.
        
        Args:
            username: The Twitter username
            max_results: Maximum number of results to return
            
        Returns:
            List of collected tweet data
        """
        self.logger.debug(f"Getting tweets from user: {username}")
        
        collected_data = []
        
        try:
            # Get user ID (from cache or API)
            user_id = await self._get_user_id(username)
            
            if not user_id:
                self.logger.warning(f"Could not find user ID for username: {username}")
                return []
            
            # Get the last tweet ID for this user
            since_id = self.last_tweet_ids["user_tweets"].get(username)
            
            # Execute API call
            response = await self._execute_api_call(
                lambda: self.client.get_users_tweets(
                    id=user_id,
                    max_results=min(max_results, 100),  # API limit is 100
                    tweet_fields=["id", "text", "created_at", "public_metrics", "entities"],
                    exclude=["retweets", "replies"],
                    since_id=since_id
                )
            )
            
            if not response or not response.data:
                self.logger.debug(f"No tweets found for user: {username}")
                return []
            
            # Extract tweets
            tweets = response.data
            
            # Update the last tweet ID for incremental loading
            if tweets:
                self.last_tweet_ids["user_tweets"][username] = tweets[0].id
            
            # Get user details
            user_response = await self._execute_api_call(
                lambda: self.client.get_user(
                    id=user_id,
                    user_fields=["public_metrics"]
                )
            )
            
            user = user_response.data if user_response else None
            user_followers = user.public_metrics.get("followers_count", 0) if user and hasattr(user, "public_metrics") else 0
            
            # Convert to collected data format
            for tweet in tweets:
                # Get metrics
                metrics = tweet.public_metrics if hasattr(tweet, "public_metrics") else {}
                
                # Get entities
                entities = tweet.entities if hasattr(tweet, "entities") else {}
                hashtags = [tag["tag"] for tag in entities.get("hashtags", [])]
                mentions = [mention["username"] for mention in entities.get("mentions", [])]
                
                # Estimate influence based on author's followers and engagement
                retweet_count = metrics.get("retweet_count", 0)
                like_count = metrics.get("like_count", 0)
                reply_count = metrics.get("reply_count", 0)
                engagement = retweet_count + like_count + reply_count
                
                # Scale influence between 0-1 based on followers and engagement
                influence_score = min(1.0, (
                    0.7 * min(1.0, user_followers / 1000000) +  # Followers (max 1M)
                    0.3 * min(1.0, engagement / 1000)           # Engagement (max 1000)
                ))
                
                # Increase reliability for known accounts
                reliability_score = 0.7 if username in self.accounts else 0.6
                
                source = EventSource(
                    id=f"twitter_{tweet.id}",
                    type=SourceType.SOCIAL_MEDIA,
                    name="Twitter",
                    url=f"https://twitter.com/{username}/status/{tweet.id}",
                    reliability_score=reliability_score
                )
                
                collected_data.append({
                    "source": source,
                    "content": tweet.text,
                    "timestamp": tweet.created_at,
                    "metadata": {
                        "user": username,
                        "followers": user_followers,
                        "retweets": retweet_count,
                        "likes": like_count,
                        "replies": reply_count,
                        "hashtags": hashtags,
                        "mentions": mentions,
                        "influence_score": influence_score,
                        "is_monitored_account": username in self.accounts
                    }
                })
            
            self.logger.debug(f"Found {len(collected_data)} tweets from user: {username}")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error getting tweets for user {username}: {e}")
            return []
    
    async def _get_user_id(self, username: str) -> Optional[str]:
        """Get the user ID for a Twitter username.
        
        Args:
            username: The Twitter username
            
        Returns:
            The user ID if found, None otherwise
        """
        # Check cache first
        if username in self.user_id_cache:
            return self.user_id_cache[username]
        
        try:
            # Execute API call
            response = await self._execute_api_call(
                lambda: self.client.get_user(username=username)
            )
            
            if response and response.data:
                user_id = response.data.id
                
                # Cache the user ID
                self.user_id_cache[username] = user_id
                
                return user_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting user ID for {username}: {e}")
            return None
    
    async def _cache_user_ids(self, usernames: List[str]):
        """Cache user IDs for a list of usernames.
        
        Args:
            usernames: List of Twitter usernames
        """
        self.logger.debug(f"Caching user IDs for {len(usernames)} accounts")
        
        # Split into chunks to avoid rate limits
        chunk_size = 10
        for i in range(0, len(usernames), chunk_size):
            chunk = usernames[i:i+chunk_size]
            
            try:
                # Execute API call
                response = await self._execute_api_call(
                    lambda: self.client.get_users(usernames=chunk)
                )
                
                if response and response.data:
                    # Cache user IDs
                    for user in response.data:
                        self.user_id_cache[user.username.lower()] = user.id
                
                # Avoid rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error caching user IDs: {e}")
    
    async def _execute_api_call(self, call_func):
        """Execute a Twitter API call with circuit breaker pattern.
        
        Args:
            call_func: Function to call the Twitter API
            
        Returns:
            API response or None if failed
        """
        if self.use_mock:
            return None
        
        try:
            # Use circuit breaker pattern
            return await self.api_client.call_with_circuit_breaker(
                lambda: asyncio.to_thread(call_func),
                "twitter_api"
            )
        except Exception as e:
            self.logger.error(f"Error executing Twitter API call: {e}")
            return None
    
    async def _mock_collect(self) -> List[Dict[str, Any]]:
        """Collect mock Twitter data for testing.
        
        Returns:
            List of mock collected data
        """
        self.logger.info("Collecting mock Twitter data")
        
        # Simulate a delay
        await asyncio.sleep(0.2)
        
        collected_data = []
        
        # Generate mock tweets for keywords
        for keyword in self.keywords:
            for i in range(3):
                # Create mock tweet
                tweet_id = f"mock_{keyword}_{i}"
                author = "mock_user" if i % 2 == 0 else self.accounts[i % len(self.accounts)]
                
                # Vary follower counts
                followers = 10000 if author in self.accounts else 1000
                
                # Vary engagement metrics
                retweets = 50 if author in self.accounts else 5
                likes = 200 if author in self.accounts else 20
                
                # Calculate influence score
                influence_score = min(1.0, (
                    0.7 * min(1.0, followers / 1000000) +
                    0.3 * min(1.0, (retweets + likes) / 1000)
                ))
                
                # Create event source
                source = EventSource(
                    id=f"twitter_{tweet_id}",
                    type=SourceType.SOCIAL_MEDIA,
                    name="Twitter",
                    url=f"https://twitter.com/{author}/status/{tweet_id}",
                    reliability_score=0.6 if author not in self.accounts else 0.7
                )
                
                # Create mock content
                content = f"This is a mock tweet about {keyword} from {author}. #{keyword} #crypto"
                
                collected_data.append({
                    "source": source,
                    "content": content,
                    "timestamp": datetime.now() - timedelta(hours=i),
                    "metadata": {
                        "keyword": keyword,
                        "user": author,
                        "followers": followers,
                        "retweets": retweets,
                        "likes": likes,
                        "replies": i * 3,
                        "hashtags": [keyword, "crypto"],
                        "mentions": [],
                        "influence_score": influence_score,
                        "is_monitored_account": author in self.accounts
                    }
                })
        
        # Generate mock tweets for monitored accounts
        for account in self.accounts:
            for i in range(2):
                tweet_id = f"mock_{account}_{i}"
                
                # Higher metrics for monitored accounts
                followers = 100000 + (i * 10000)
                retweets = 100 + (i * 50)
                likes = 500 + (i * 100)
                
                # Calculate influence score
                influence_score = min(1.0, (
                    0.7 * min(1.0, followers / 1000000) +
                    0.3 * min(1.0, (retweets + likes) / 1000)
                ))
                
                # Create event source
                source = EventSource(
                    id=f"twitter_{tweet_id}",
                    type=SourceType.SOCIAL_MEDIA,
                    name="Twitter",
                    url=f"https://twitter.com/{account}/status/{tweet_id}",
                    reliability_score=0.7  # Higher reliability for monitored accounts
                )
                
                # Create mock content based on the account
                if "elon" in account.lower():
                    content = f"Just had a meeting about the future of cryptocurrency and AI. Exciting times ahead! #{self.keywords[i % len(self.keywords)]} #tech"
                elif "binance" in account.lower():
                    content = f"Important update on regulatory developments. Stay tuned for more information. #{self.keywords[i % len(self.keywords)]} #crypto"
                else:
                    content = f"Thoughts on the current market conditions for {self.keywords[i % len(self.keywords)]}. Thread 🧵"
                
                collected_data.append({
                    "source": source,
                    "content": content,
                    "timestamp": datetime.now() - timedelta(hours=i),
                    "metadata": {
                        "user": account,
                        "followers": followers,
                        "retweets": retweets,
                        "likes": likes,
                        "replies": i * 10,
                        "hashtags": [self.keywords[i % len(self.keywords)], "crypto"],
                        "mentions": [],
                        "influence_score": influence_score,
                        "is_monitored_account": True
                    }
                })
        
        self.logger.info(f"Generated {len(collected_data)} mock Twitter items")
        return collected_data