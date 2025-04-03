"""
Reddit data collector for the Early Event Detection System.

This module provides functionality for collecting data from Reddit
using the Reddit API via PRAW.
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


class RedditCollector:
    """Collector for Reddit data."""
    
    def __init__(self):
        """Initialize the Reddit collector."""
        self.logger = get_logger("early_detection", "reddit_collector")
        
        # Configuration
        self.subreddits = config.get("early_detection.data_collection.social_media.reddit.subreddits", 
                                    ["cryptocurrency", "bitcoin", "ethereum", "CryptoMarkets"])
        
        # API credentials
        self.client_id = os.getenv("REDDIT_CLIENT_ID") or config.get("apis.reddit.client_id", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET") or config.get("apis.reddit.client_secret", "")
        self.user_agent = os.getenv("REDDIT_USER_AGENT") or config.get("apis.reddit.user_agent", "")
        
        if not self.user_agent:
            self.user_agent = "early_event_detection_bot/1.0"
        
        # API client
        self.api_client = RetryableAPIClient(
            max_retries=3,
            backoff_factor=2.0,
            logger=self.logger
        )
        
        # Client initialization
        self.client = None
        self.use_mock = True
        
        # Cache for last seen post IDs
        self.last_post_ids = {}
    
    async def initialize(self):
        """Initialize the Reddit collector."""
        # Check if we have valid credentials
        if all([self.client_id, self.client_secret, self.user_agent]):
            try:
                import praw
                
                # Initialize the client (run in a thread to avoid blocking)
                self.client = await asyncio.to_thread(
                    praw.Reddit,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                
                # Test the connection
                self.logger.info("Testing Reddit API connection")
                subreddit = await asyncio.to_thread(
                    getattr, self.client, "subreddit", "cryptocurrency"
                )
                await asyncio.to_thread(
                    lambda: next(subreddit.hot(limit=1), None)
                )
                
                self.use_mock = False
                self.logger.info("Reddit API client initialized successfully")
                
            except ImportError:
                self.logger.warning("PRAW not installed, using mock data instead")
            except Exception as e:
                self.logger.error(f"Error initializing Reddit client: {e}, using mock data instead")
        else:
            self.logger.warning("Missing Reddit API credentials, using mock data")
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from Reddit.
        
        Returns:
            List of collected data items
        """
        if self.use_mock or self.client is None:
            return self._mock_collect()
        
        try:
            self.logger.info("Collecting data from Reddit")
            
            collected_data = []
            
            # Get cost optimizer for API efficiency
            cost_optimizer = await get_cost_optimizer()
            
            # Check if we should sample Reddit based on adaptive sampling
            if not cost_optimizer.adaptive_sampler.should_sample("reddit"):
                self.logger.info("Skipping Reddit collection due to adaptive sampling")
                return []
            
            # Collect posts from each subreddit
            subreddit_tasks = []
            for subreddit in self.subreddits:
                # Create tasks for different sorts
                hot_task = asyncio.create_task(
                    cost_optimizer.api_request(
                        "reddit",
                        self._get_subreddit_posts,
                        subreddit, "hot", 10
                    )
                )
                new_task = asyncio.create_task(
                    cost_optimizer.api_request(
                        "reddit",
                        self._get_subreddit_posts,
                        subreddit, "new", 10
                    )
                )
                top_day_task = asyncio.create_task(
                    cost_optimizer.api_request(
                        "reddit",
                        self._get_subreddit_posts,
                        subreddit, "top", 5, "day"
                    )
                )
                
                subreddit_tasks.extend([hot_task, new_task, top_day_task])
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*subreddit_tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error collecting Reddit posts: {result}")
                    continue
                
                if result:
                    collected_data.extend(result)
            
            self.logger.info(f"Collected {len(collected_data)} items from Reddit")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error collecting Reddit data: {e}")
            return self._mock_collect()
    
    async def _get_subreddit_posts(self, subreddit_name: str, sort_type: str = "hot", 
                                  limit: int = 25, time_filter: str = "all") -> List[Dict[str, Any]]:
        """Get posts from a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            sort_type: Type of sort ("hot", "new", "top", "rising")
            limit: Maximum number of posts to fetch
            time_filter: Time filter for "top" posts (hour, day, week, month, year, all)
            
        Returns:
            List of collected post data
        """
        self.logger.debug(f"Getting {sort_type} posts from r/{subreddit_name}")
        
        collected_data = []
        
        try:
            # Clean subreddit name
            subreddit_name = subreddit_name.replace("r/", "")
            
            # Get the subreddit object
            subreddit = await asyncio.to_thread(
                getattr, self.client, "subreddit", subreddit_name
            )
            
            # Cache key for tracking last seen posts
            cache_key = f"{subreddit_name}_{sort_type}_{time_filter}"
            
            # Get the appropriate listing
            if sort_type == "hot":
                listing_func = lambda: subreddit.hot(limit=limit)
            elif sort_type == "new":
                listing_func = lambda: subreddit.new(limit=limit)
            elif sort_type == "top":
                listing_func = lambda: subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_type == "rising":
                listing_func = lambda: subreddit.rising(limit=limit)
            else:
                self.logger.warning(f"Unknown sort type: {sort_type}, using hot")
                listing_func = lambda: subreddit.hot(limit=limit)
            
            # Get the posts
            posts = await asyncio.to_thread(
                lambda: list(listing_func())
            )
            
            # Get last seen post ID
            last_post_id = self.last_post_ids.get(cache_key)
            
            # Track new posts and update last seen ID
            if posts:
                # If we have a last_post_id, find its index
                if last_post_id:
                    for i, post in enumerate(posts):
                        if post.id == last_post_id:
                            # Only take posts newer than the last seen post
                            posts = posts[:i]
                            break
                
                if posts:
                    # Update the last seen post ID
                    self.last_post_ids[cache_key] = posts[0].id
            
            # Convert posts to collected data format
            for post in posts:
                # Skip posts that are just links without text
                if not hasattr(post, "selftext") or (not post.selftext and post.is_self):
                    continue
                
                # Get post content
                title = post.title
                content = post.selftext if hasattr(post, "selftext") else ""
                full_content = f"{title}\n\n{content}" if content else title
                
                # Calculate engagement and scores
                upvote_ratio = post.upvote_ratio if hasattr(post, "upvote_ratio") else 0.5
                comments = post.num_comments if hasattr(post, "num_comments") else 0
                score = post.score if hasattr(post, "score") else 0
                
                # Calculate influence score (0-1)
                influence_score = min(1.0, (
                    0.4 * min(1.0, score / 1000) +         # Score (max 1000)
                    0.3 * min(1.0, comments / 100) +       # Comments (max 100)
                    0.3 * upvote_ratio                     # Upvote ratio
                ))
                
                # Create source
                url = f"https://www.reddit.com{post.permalink}" if hasattr(post, "permalink") else f"https://www.reddit.com/r/{subreddit_name}/comments/{post.id}"
                source = EventSource(
                    id=f"reddit_{post.id}",
                    type=SourceType.SOCIAL_MEDIA,
                    name="Reddit",
                    url=url,
                    reliability_score=0.5  # Reddit is less reliable than Twitter or official sources
                )
                
                # Extract post creation time
                created_time = datetime.fromtimestamp(post.created_utc) if hasattr(post, "created_utc") else datetime.now()
                
                # Get flair
                flair = post.link_flair_text if hasattr(post, "link_flair_text") and post.link_flair_text else None
                
                collected_data.append({
                    "source": source,
                    "content": full_content,
                    "timestamp": created_time,
                    "metadata": {
                        "subreddit": subreddit_name,
                        "sort_type": sort_type,
                        "author": str(post.author) if hasattr(post, "author") and post.author else "[deleted]",
                        "score": score,
                        "upvote_ratio": upvote_ratio,
                        "comments": comments,
                        "flair": flair,
                        "is_self_post": post.is_self if hasattr(post, "is_self") else True,
                        "influence_score": influence_score
                    }
                })
            
            self.logger.debug(f"Found {len(collected_data)} posts in r/{subreddit_name} ({sort_type})")
            return collected_data
            
        except Exception as e:
            self.logger.error(f"Error getting posts from r/{subreddit_name}: {e}")
            return []
    
    async def _mock_collect(self) -> List[Dict[str, Any]]:
        """Collect mock Reddit data for testing.
        
        Returns:
            List of mock collected data
        """
        self.logger.info("Collecting mock Reddit data")
        
        # Simulate a delay
        await asyncio.sleep(0.2)
        
        collected_data = []
        
        # Generate mock posts for each subreddit
        for subreddit in self.subreddits:
            for sort_type in ["hot", "new", "top"]:
                # Generate 2-4 posts per subreddit and sort type
                num_posts = 2 + (hash(f"{subreddit}_{sort_type}") % 3)
                
                for i in range(num_posts):
                    # Create mock post
                    post_id = f"mock_{subreddit}_{sort_type}_{i}"
                    
                    # Create title and content based on subreddit
                    if "bitcoin" in subreddit.lower():
                        title = f"Thoughts on the recent BTC price {['surge', 'drop', 'volatility'][i % 3]}"
                        content = f"I've been watching the charts for the past week, and I think this {['bull run', 'correction', 'sideways action'][i % 3]} is going to continue for a while. What do you all think?"
                    elif "ethereum" in subreddit.lower():
                        title = f"ETH 2.0 {['implications', 'update', 'timeline'][i % 3]} discussion"
                        content = f"With the recent {['merge news', 'upgrade progress', 'staking changes'][i % 3]}, do you think we'll see increased adoption? How will this affect gas fees?"
                    elif "crypto" in subreddit.lower():
                        title = f"{['Breaking', 'Important', 'Interesting'][i % 3]}: New {['regulation', 'protocol', 'partnership'][i % 3]} announced"
                        content = f"Just saw that {['the SEC', 'a major exchange', 'a central bank'][i % 3]} is planning to {['regulate', 'adopt', 'investigate'][i % 3]} cryptocurrency. This could have major implications for the market."
                    else:
                        title = f"{['Analysis', 'Discussion', 'Opinion'][i % 3]}: Market trends for Q{i+1}"
                        content = f"Looking at the recent {['volatility', 'stability', 'correlation'][i % 3]} between crypto and traditional markets, I think we're seeing a {['bullish', 'bearish', 'neutral'][i % 3]} pattern emerge."
                    
                    # Combine title and content
                    full_content = f"{title}\n\n{content}"
                    
                    # Generate engagement metrics
                    upvote_ratio = 0.5 + (0.4 * (i % 2))  # 0.5 - 0.9
                    comments = 10 + (i * 20)
                    score = 50 + (i * 100)
                    
                    # Calculate influence score
                    influence_score = min(1.0, (
                        0.4 * min(1.0, score / 1000) +
                        0.3 * min(1.0, comments / 100) +
                        0.3 * upvote_ratio
                    ))
                    
                    # Create source
                    source = EventSource(
                        id=f"reddit_{post_id}",
                        type=SourceType.SOCIAL_MEDIA,
                        name="Reddit",
                        url=f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/mock_post",
                        reliability_score=0.5
                    )
                    
                    collected_data.append({
                        "source": source,
                        "content": full_content,
                        "timestamp": datetime.now() - timedelta(hours=i * 2),
                        "metadata": {
                            "subreddit": subreddit,
                            "sort_type": sort_type,
                            "author": f"mock_user_{i}",
                            "score": score,
                            "upvote_ratio": upvote_ratio,
                            "comments": comments,
                            "flair": ["Discussion", "Analysis", "News", "Comedy"][i % 4] if i % 4 != 3 else None,
                            "is_self_post": True,
                            "influence_score": influence_score
                        }
                    })
        
        self.logger.info(f"Generated {len(collected_data)} mock Reddit items")
        return collected_data