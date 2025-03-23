"""Social media sentiment analysis.

This module provides functionality for analyzing sentiment from social media
sources like Twitter, Reddit, and other social platforms.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.analysis_agents.sentiment.nlp_service import NLPService
from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import CandleData, TimeFrame


class TwitterClient:
    """Client for the Twitter/X API using Tweepy."""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_secret: str):
        """Initialize the Twitter client.
        
        Args:
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: Twitter access token
            access_secret: Twitter access secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.logger = get_logger("clients", "twitter")
        self.client = None
        self.use_mock = True  # Flag to control mock vs real API usage
        
        # Try to initialize the Twitter client
        try:
            import tweepy
            self.tweepy = tweepy
            
            # Check if we have valid credentials
            if all([self.api_key, self.api_secret, self.access_token, self.access_secret]):
                # Initialize the client
                auth = tweepy.OAuth1UserHandler(
                    self.api_key, self.api_secret, self.access_token, self.access_secret
                )
                self.client = tweepy.API(auth)
                
                # Test the connection
                self.client.verify_credentials()
                self.use_mock = False
                self.logger.info("Twitter API client initialized successfully")
            else:
                self.logger.warning("Missing Twitter API credentials, using mock data")
        except ImportError:
            self.logger.warning("Tweepy not installed, using mock data instead")
        except Exception as e:
            self.logger.error(f"Error initializing Twitter client: {e}, using mock data instead")
    
    async def search_tweets(self, query: str, count: int = 100, result_type: str = "recent") -> List[str]:
        """Search for tweets matching a query.
        
        Args:
            query: Search query
            count: Number of tweets to return
            result_type: Type of results to return (recent, popular, mixed)
            
        Returns:
            List of tweet texts
        """
        if self.use_mock or self.client is None:
            return await self._mock_search_tweets(query, count, result_type)
        
        try:
            self.logger.debug("Searching Twitter for tweets", query=query, count=count)
            
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            tweets = await loop.run_in_executor(
                None,
                lambda: self.client.search_tweets(
                    q=query,
                    count=count,
                    result_type=result_type,
                    tweet_mode="extended",
                    lang="en"
                )
            )
            
            # Extract full text from tweets
            tweet_texts = []
            for tweet in tweets:
                # Use full_text if available, otherwise fall back to text
                if hasattr(tweet, "full_text"):
                    tweet_texts.append(tweet.full_text)
                else:
                    tweet_texts.append(tweet.text)
            
            self.logger.info(f"Retrieved {len(tweet_texts)} tweets for query '{query}'")
            return tweet_texts
            
        except Exception as e:
            self.logger.error(f"Error searching tweets: {e}, falling back to mock data")
            return await self._mock_search_tweets(query, count, result_type)
    
    async def _mock_search_tweets(self, query: str, count: int = 100, result_type: str = "recent") -> List[str]:
        """Generate mock tweets for testing.
        
        Args:
            query: Search query
            count: Number of tweets to return
            result_type: Type of results to return (recent, popular, mixed)
            
        Returns:
            List of mock tweet texts
        """
        self.logger.debug("Generating mock tweets", query=query, count=count)
        
        # Simulate a delay
        await asyncio.sleep(0.1)
        
        # For testing, return between 10 and 'count' mock tweets
        actual_count = min(count, random.randint(10, count))
        
        # Extract the symbol from the query (e.g., "#BTC" -> "BTC")
        symbol = query.replace("#", "").replace("$", "").split()[0]
        
        # Create mock tweets with varying sentiment for more realistic data
        mock_tweets = []
        sentiment_types = ["bullish", "bearish", "neutral"]
        sentiment_weights = [0.4, 0.3, 0.3]  # 40% bullish, 30% bearish, 30% neutral
        
        bullish_templates = [
            f"I'm feeling bullish on {symbol}! The technical indicators look strong. #crypto #bullmarket",
            f"Just bought more {symbol}. This is going to 🚀 soon! #investing #crypto",
            f"The fundamentals for {symbol} are incredibly strong right now. Long-term holder here! 💎🙌",
            f"{symbol} looking primed for a breakout. Weekly close above resistance is very bullish!",
            f"Accumulating {symbol} at these levels. Market sentiment too bearish - contrarian play time."
        ]
        
        bearish_templates = [
            f"Not looking good for {symbol} right now. Expecting more downside. #crypto #bearmarket",
            f"Just sold my {symbol} position. Charts don't look good for the short term. #trading",
            f"The macro environment is terrible for {symbol} right now. Staying in cash.",
            f"{symbol} failed to hold support, next leg down incoming? 📉 #crypto #trading",
            f"This rally in {symbol} is just another bull trap. Don't fall for it! #bearmarket"
        ]
        
        neutral_templates = [
            f"Watching {symbol} closely. Could go either way from here. #crypto #trading",
            f"Current price action in {symbol} is indecisive. Waiting for clearer signals.",
            f"Interesting developments with {symbol}. Need more data before taking a position.",
            f"Market is unclear on {symbol} direction. Forming a plan for both scenarios.",
            f"Following {symbol} news carefully. Not ready to make a move yet."
        ]
        
        for i in range(actual_count):
            sentiment_type = random.choices(sentiment_types, weights=sentiment_weights)[0]
            
            if sentiment_type == "bullish":
                tweet = random.choice(bullish_templates)
            elif sentiment_type == "bearish":
                tweet = random.choice(bearish_templates)
            else:
                tweet = random.choice(neutral_templates)
                
            # Add some randomization to avoid duplicate tweets
            if random.random() > 0.7:
                tweet += f" {random.choice(['#hodl', '#btc', '#eth', '#altcoin', '#defi', '#nft'])}"
                
            mock_tweets.append(tweet)
        
        return mock_tweets


class RedditClient:
    """Client for the Reddit API using PRAW."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize the Reddit client.
        
        Args:
            client_id: Reddit client ID
            client_secret: Reddit client secret
            user_agent: User agent string
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.logger = get_logger("clients", "reddit")
        self.reddit = None
        self.use_mock = True  # Flag to control mock vs real API usage
        
        # Try to initialize the Reddit client
        try:
            import praw
            
            # Check if we have valid credentials
            if all([self.client_id, self.client_secret, self.user_agent]):
                # Initialize the client
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                self.use_mock = False
                self.logger.info("Reddit API client initialized successfully")
            else:
                self.logger.warning("Missing Reddit API credentials, using mock data")
        except ImportError:
            self.logger.warning("PRAW not installed, using mock data instead")
        except Exception as e:
            self.logger.error(f"Error initializing Reddit client: {e}, using mock data instead")
    
    async def get_hot_posts(self, subreddit: str, limit: int = 50, time_filter: str = "day") -> List[str]:
        """Get hot posts from a subreddit.
        
        Args:
            subreddit: Subreddit name
            limit: Number of posts to return
            time_filter: Time filter (hour, day, week, month, year, all)
            
        Returns:
            List of post texts
        """
        if self.use_mock or self.reddit is None:
            return await self._mock_get_hot_posts(subreddit, limit, time_filter)
        
        try:
            self.logger.debug("Getting Reddit posts", subreddit=subreddit, limit=limit)
            
            # Clean the subreddit name (remove r/ if present)
            subreddit_name = subreddit.replace("r/", "")
            
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Get the subreddit and fetch hot posts
            posts = await loop.run_in_executor(
                None,
                lambda: list(self.reddit.subreddit(subreddit_name).hot(limit=limit))
            )
            
            # Extract titles and selftext from posts
            post_texts = []
            for post in posts:
                # Combine title and selftext if available
                text = post.title
                if hasattr(post, "selftext") and post.selftext:
                    text += "\n" + post.selftext
                post_texts.append(text)
            
            self.logger.info(f"Retrieved {len(post_texts)} posts from r/{subreddit_name}")
            return post_texts
            
        except Exception as e:
            self.logger.error(f"Error getting Reddit posts: {e}, falling back to mock data")
            return await self._mock_get_hot_posts(subreddit, limit, time_filter)
    
    async def _mock_get_hot_posts(self, subreddit: str, limit: int = 50, time_filter: str = "day") -> List[str]:
        """Generate mock Reddit posts for testing.
        
        Args:
            subreddit: Subreddit name
            limit: Number of posts to return
            time_filter: Time filter (hour, day, week, month, year, all)
            
        Returns:
            List of mock post texts
        """
        self.logger.debug("Generating mock Reddit posts", subreddit=subreddit, limit=limit)
        
        # Simulate a delay
        await asyncio.sleep(0.1)
        
        # For testing, return between 5 and 'limit' mock posts
        actual_count = min(limit, random.randint(5, limit))
        
        # Clean the subreddit name
        subreddit_name = subreddit.replace("r/", "")
        
        # Create mock posts with varying sentiment for more realistic data
        mock_posts = []
        sentiment_types = ["bullish", "bearish", "neutral"]
        sentiment_weights = [0.4, 0.3, 0.3]  # 40% bullish, 30% bearish, 30% neutral
        
        bullish_templates = [
            f"Why I think {subreddit_name} is going to moon soon! Technical analysis inside.",
            f"Bullish case for {subreddit_name} - the fundamentals are stronger than ever",
            f"Just increased my {subreddit_name} position by 50% - here's why I'm bullish",
            f"The next bull run for {subreddit_name} is coming. Institutional adoption is increasing.",
            f"{subreddit_name} technical analysis: Major ascending triangle forming. Ready for breakout!"
        ]
        
        bearish_templates = [
            f"Bearish signals for {subreddit_name}. Technical analysis shows potential downtrend.",
            f"Why I'm reducing my {subreddit_name} exposure in the current market conditions",
            f"Red flags that everyone is missing about {subreddit_name} right now",
            f"The case for a {subreddit_name} correction in the next few weeks",
            f"Analysis: Why {subreddit_name} might see more downside before recovery"
        ]
        
        neutral_templates = [
            f"Daily Discussion Thread for {subreddit_name} - what's your strategy?",
            f"Weekly {subreddit_name} market analysis - mixed signals right now",
            f"Evaluating {subreddit_name} in the current market context - pros and cons",
            f"What's your average entry price for {subreddit_name}? Poll and discussion",
            f"New to {subreddit_name} investing - need advice on timing entry"
        ]
        
        for i in range(actual_count):
            sentiment_type = random.choices(sentiment_types, weights=sentiment_weights)[0]
            
            if sentiment_type == "bullish":
                title = random.choice(bullish_templates)
                # Add some simulated content
                content = f"\nI've been watching {subreddit_name} for months now and all indicators are pointing to a major move up. The recent consolidation is healthy, volume profile is improving, and we're testing key resistance levels. Once we break through $XX,XXX, there's very little stopping us from reaching new ATHs.\n\nNot financial advice, but I'm very bullish here."
            elif sentiment_type == "bearish":
                title = random.choice(bearish_templates)
                content = f"\nLooking at the market structure for {subreddit_name}, I'm seeing several warning signs. The RSI divergence on the daily is concerning, plus we've failed to hold key support levels multiple times. With the broader macro environment still uncertain, I think we'll see more downside before any significant recovery.\n\nI've reduced my position by 30% and set stops at $XX,XXX."
            else:
                title = random.choice(neutral_templates)
                content = f"\nThe market for {subreddit_name} is at an interesting junction. On one hand, the fundamentals look stronger than ever with development milestones being hit. On the other hand, macro pressures and regulatory concerns are creating headwinds.\n\nWhat's your take on the current situation? Are you buying, selling, or holding?"
                
            # Combine title and content
            post = title + content
            mock_posts.append(post)
        
        return mock_posts


class SocialMediaSentimentAgent(BaseSentimentAgent):
    """Analysis agent for social media sentiment.
    
    This agent processes sentiment data from social media platforms
    and publishes sentiment events with confidence scores.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the social media sentiment agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "social_media_sentiment")
        
        # Social media platforms to monitor
        self.platforms = config.get(
            f"analysis_agents.{agent_id}.platforms", 
            ["Twitter", "Reddit"]
        )
        
        # Update interval in seconds
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            300  # Default: 5 minutes
        )
        
        # API clients (will be initialized during _initialize)
        self.twitter_client = None
        self.reddit_client = None
        
        # NLP service (will be set by manager)
        self.nlp_service = None
    
    async def _initialize(self) -> None:
        """Initialize the social media sentiment agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing social media sentiment agent",
                       platforms=self.platforms)
                       
        # Initialize API clients
        try:
            # Twitter API client
            self.twitter_client = TwitterClient(
                api_key=config.get("apis.twitter.api_key", ""),
                api_secret=config.get("apis.twitter.api_secret", ""),
                access_token=config.get("apis.twitter.access_token", ""),
                access_secret=config.get("apis.twitter.access_secret", "")
            )
            
            # Reddit API client
            self.reddit_client = RedditClient(
                client_id=config.get("apis.reddit.client_id", ""),
                client_secret=config.get("apis.reddit.client_secret", ""),
                user_agent=config.get("apis.reddit.user_agent", "AI-Trading-Agent/1.0")
            )
            
            self.logger.info("Initialized social media API clients")
            
        except Exception as e:
            self.logger.error("Failed to initialize social media API clients", error=str(e))
    
    def set_nlp_service(self, nlp_service: NLPService) -> None:
        """Set the NLP service for sentiment analysis.
        
        Args:
            nlp_service: The NLP service to use
        """
        self.nlp_service = nlp_service
        self.logger.info("NLP service set for social media sentiment agent")
    
    async def _start(self) -> None:
        """Start the social media sentiment agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for social media sentiment
        self.update_task = self.create_task(
            self._update_sentiment_periodically()
        )
    
    async def _stop(self) -> None:
        """Stop the social media sentiment agent."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        await super()._stop()
    
    async def _update_sentiment_periodically(self) -> None:
        """Update social media sentiment periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    # In a real system, we would fetch actual social media data
                    # For the demo, we'll simulate it
                    await self._analyze_social_media_sentiment(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Social media sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in social media sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _analyze_social_media_sentiment(self, symbol: str) -> None:
        """Analyze social media sentiment for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "social_media", self.update_interval):
            return
        
        try:
            base_currency = symbol.split('/')[0]
            
            # Fetch Twitter data
            twitter_posts = []
            if self.twitter_client:
                twitter_posts = await self.twitter_client.search_tweets(
                    query=f"#{base_currency} OR ${base_currency}",
                    count=100,
                    result_type="recent"
                )
            
            # Fetch Reddit data
            reddit_posts = []
            if self.reddit_client:
                subreddits = [f"r/{base_currency}", "r/CryptoCurrency", "r/CryptoMarkets"]
                
                for subreddit in subreddits:
                    posts = await self.reddit_client.get_hot_posts(
                        subreddit=subreddit,
                        limit=50,
                        time_filter="day"
                    )
                    reddit_posts.extend(posts)
            
            # Combine posts
            all_posts = twitter_posts + reddit_posts
            post_count = len(all_posts)
            
            if post_count == 0:
                self.logger.warning("No social media posts found", symbol=symbol)
                return
            
            # Process posts with NLP model if available
            sentiment_scores = []
            
            if self.nlp_service:
                sentiment_scores = await self.nlp_service.analyze_sentiment(all_posts)
            else:
                # Fallback to random sentiment if NLP service is not available
                sentiment_scores = [random.uniform(0.2, 0.8) for _ in range(post_count)]
            
            # Calculate overall sentiment
            sentiment_value = sum(sentiment_scores) / len(sentiment_scores)
            
            # Calculate confidence based on post volume and agreement
            volume_factor = min(1.0, post_count / 100)
            
            # Calculate standard deviation to measure agreement
            if len(sentiment_scores) > 1:
                std_dev = (sum((s - sentiment_value) ** 2 for s in sentiment_scores) / len(sentiment_scores)) ** 0.5
                agreement_factor = 1.0 - min(1.0, std_dev * 2)  # Lower std_dev means higher agreement
            else:
                agreement_factor = 0.5  # Default if only one post
                
            confidence = volume_factor * agreement_factor
            
            # Determine direction
            if sentiment_value > 0.55:
                direction = "bullish"
            elif sentiment_value < 0.45:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Store additional metadata
            additional_data = {
                "post_count": post_count,
                "platforms": self.platforms,
                "twitter_count": len(twitter_posts),
                "reddit_count": len(reddit_posts)
            }
            
            # Update the sentiment cache
            sentiment_shift = self._update_sentiment_cache(
                symbol=symbol,
                source_type="social_media",
                sentiment_value=sentiment_value,
                direction=direction,
                confidence=confidence,
                additional_data=additional_data
            )
            
            # Publish event if significant shift or high confidence extreme reading
            is_extreme = sentiment_value > 0.8 or sentiment_value < 0.2
            
            if (sentiment_shift > self.sentiment_shift_threshold or 
                (is_extreme and confidence > self.min_confidence)):
                
                event_type = "sentiment_shift" if sentiment_shift > self.sentiment_shift_threshold else "sentiment_extreme"
                
                # Determine if extreme sentiment should be treated as contrarian
                signal_type = "sentiment"
                if is_extreme and sentiment_value > self.contrarian_threshold:
                    # Very extreme sentiment might be contrarian
                    signal_type = "contrarian"
                
                # Publish event if confidence is high enough
                if confidence >= self.min_confidence:
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        is_extreme=is_extreme,
                        signal_type=signal_type,
                        sources=self.platforms,
                        details={
                            "post_count": post_count,
                            "event_type": event_type
                        }
                    )
        
        except Exception as e:
            self.logger.error("Error analyzing social media sentiment", 
                           symbol=symbol,
                           error=str(e))
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to social media sentiment.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        if not candles or len(candles) < 10:
            return
            
        # Check if we have social media sentiment data for this symbol
        if symbol not in self.sentiment_cache or "social_media" not in self.sentiment_cache[symbol]:
            return
            
        # Get the latest social media sentiment
        sentiment_data = self.sentiment_cache[symbol]["social_media"]
        sentiment_value = sentiment_data.get("value", 0.5)
        direction = sentiment_data.get("direction", "neutral")
        confidence = sentiment_data.get("confidence", 0.0)
        
        # Get price data from candles
        closes = [candle.close for candle in candles]
        
        # Analyze sentiment divergence with price action
        if len(closes) >= 20:
            # Calculate short-term trend
            short_term_trend = "bullish" if closes[-1] > closes[-10] else "bearish"
            
            # Check for divergence between sentiment and price action
            if short_term_trend != direction and confidence > 0.7:
                is_extreme = sentiment_value > 0.8 or sentiment_value < 0.2
                
                if is_extreme:
                    # This could be a contrarian indicator
                    contrarian_confidence = confidence * 1.1  # Boost confidence slightly
                    
                    # Generate a contrarian sentiment event
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction="bullish" if direction == "bearish" else "bearish",
                        value=1 - sentiment_value,  # Inverse sentiment value
                        confidence=min(0.95, contrarian_confidence),
                        timeframe=timeframe,
                        is_extreme=True,
                        signal_type="contrarian",
                        sources=self.platforms,
                        details={
                            "price_trend": short_term_trend,
                            "event_type": "price_sentiment_divergence"
                        }
                    )
