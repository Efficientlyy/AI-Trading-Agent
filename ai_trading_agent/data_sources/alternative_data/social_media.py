"""
Social Media Sentiment Analysis for trading insights.

This module integrates with social media platforms to analyze sentiment and extract
trends relevant for trading decision-making, including:
- Brand sentiment tracking
- Product launch reception
- Corporate event reactions
- Market mood and retail investor sentiment
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set

import pandas as pd
import aiohttp
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base import AlternativeDataSource, AlternativeDataConfig
from ...health.monitoring import register_health_check, HealthStatus


logger = logging.getLogger(__name__)


class SocialMediaSentimentAnalyzer(AlternativeDataSource):
    """
    Analyzer for social media data to extract sentiment and trading insights.
    
    This class connects to various social media platforms, processes posts and comments,
    and extracts sentiment metrics relevant to trading decisions.
    """
    
    SUPPORTED_PLATFORMS = {
        "twitter": "https://api.twitter.com/2/",
        "reddit": "https://oauth.reddit.com/",
        "stocktwits": "https://api.stocktwits.com/api/2/",
        "youtube_comments": "https://www.googleapis.com/youtube/v3/"
    }
    
    def __init__(self, config: AlternativeDataConfig, platforms: List[str] = None):
        """
        Initialize the social media sentiment analyzer.
        
        Args:
            config: Configuration with API keys and settings
            platforms: List of social media platforms to monitor
        """
        self.platforms = platforms or ["twitter", "reddit", "stocktwits"]
        for platform in self.platforms:
            if platform not in self.SUPPORTED_PLATFORMS:
                raise ValueError(f"Unsupported platform: {platform}. "
                               f"Must be one of {list(self.SUPPORTED_PLATFORMS.keys())}")
        
        # Set the endpoint based on the first platform (will change dynamically)
        if not config.endpoint:
            config.endpoint = self.SUPPORTED_PLATFORMS[self.platforms[0]]
            
        super().__init__(config)
        
        # Register with health monitoring system
        register_health_check(
            component_id="social_media_sentiment",
            check_function=self.get_health_status,
            interval_seconds=300
        )
    
    def _initialize(self) -> None:
        """Initialize connections to social media platforms."""
        self.sessions = {}
        self.connected = False
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tracked_symbols = set()
        self.tracked_keywords = set()
        
        logger.info(f"Initialized social media sentiment analyzer for platforms: {self.platforms}")
    
    async def _ensure_session(self, platform: str) -> None:
        """
        Ensure an active HTTP session exists for the specified platform.
        
        Args:
            platform: The social media platform to create a session for
        """
        if platform not in self.sessions or self.sessions[platform].closed:
            headers = self._get_platform_headers(platform)
            
            self.sessions[platform] = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
    
    def _get_platform_headers(self, platform: str) -> Dict[str, str]:
        """
        Get the appropriate headers for API authentication.
        
        Args:
            platform: The social media platform
            
        Returns:
            Dictionary of HTTP headers
        """
        if platform == "twitter":
            return {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        elif platform == "reddit":
            return {
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": "AI-Trading-Agent/1.0"
            }
        elif platform == "stocktwits":
            # Stocktwits doesn't use auth in headers, uses query params
            return {
                "Content-Type": "application/json"
            }
        elif platform == "youtube_comments":
            return {
                "Content-Type": "application/json"
            }
        else:
            return {
                "Content-Type": "application/json"
            }
    
    async def _close_sessions(self) -> None:
        """Close all HTTP sessions."""
        for platform, session in self.sessions.items():
            if not session.closed:
                await session.close()
        self.sessions = {}
    
    async def fetch_data(self, query: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """
        Fetch social media data based on query parameters.
        
        Args:
            query: Dictionary containing query parameters such as:
                - symbols: List of ticker symbols to track
                - keywords: List of keywords/hashtags to search for
                - start_date: Beginning of time period
                - end_date: End of time period
                - platforms: Specific platforms to query (defaults to all configured)
            **kwargs: Additional platform-specific parameters
            
        Returns:
            DataFrame containing posts and sentiment metrics
        """
        platforms = query.get("platforms", self.platforms)
        symbols = query.get("symbols", [])
        keywords = query.get("keywords", [])
        
        # Add to tracked items
        self.tracked_symbols.update(symbols)
        self.tracked_keywords.update(keywords)
        
        # Create cache key
        cache_key = str(hash(str({
            "platforms": sorted(platforms),
            "symbols": sorted(symbols),
            "keywords": sorted(keywords),
            "start_date": query.get("start_date"),
            "end_date": query.get("end_date")
        })))
        
        # Check cache
        if cache_key in self._cache:
            logger.info(f"Returning cached results for query: {cache_key}")
            return self._cache[cache_key]
        
        # Fetch data from each platform
        all_results = []
        
        for platform in platforms:
            try:
                await self._ensure_session(platform)
                platform_results = await self._fetch_platform_data(platform, query, **kwargs)
                all_results.extend(platform_results)
            except Exception as e:
                logger.error(f"Error fetching data from {platform}: {e}")
        
        # Convert to DataFrame
        if not all_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        
        # Apply sentiment analysis
        if not df.empty and "text" in df.columns:
            # Calculate sentiment for each post
            df["sentiment_scores"] = df["text"].apply(
                lambda text: self.sentiment_analyzer.polarity_scores(text)
            )
            
            # Extract compound sentiment score
            df["sentiment_score"] = df["sentiment_scores"].apply(
                lambda scores: scores["compound"]
            )
            
            # Add sentiment classification
            df["sentiment"] = df["sentiment_score"].apply(self._classify_sentiment)
        
        # Cache results
        self._cache[cache_key] = df
        self.last_updated = datetime.now()
        
        return df
    
    async def _fetch_platform_data(
        self, 
        platform: str, 
        query: Dict[str, Any], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from a specific social media platform.
        
        Args:
            platform: Social media platform to query
            query: Query parameters
            **kwargs: Additional parameters
            
        Returns:
            List of posts/comments with metadata
        """
        # In a real implementation, this would make actual API calls
        # For now, we'll generate mock data
        
        symbols = query.get("symbols", [])
        keywords = query.get("keywords", [])
        start_date = query.get("start_date", datetime.now() - timedelta(days=7))
        end_date = query.get("end_date", datetime.now())
        
        # Convert strings to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Mock data generation
        results = []
        
        # Number of posts to generate
        num_posts = np.random.randint(50, 200)
        
        for _ in range(num_posts):
            # Generate random timestamp
            post_time = start_date + (end_date - start_date) * np.random.random()
            
            # Pick a symbol or keyword
            if symbols and np.random.random() < 0.7:  # 70% chance of symbol
                topic = np.random.choice(symbols)
                topic_type = "symbol"
            elif keywords:
                topic = np.random.choice(keywords)
                topic_type = "keyword"
            else:
                topic = "market"
                topic_type = "general"
            
            # Generate mock post text
            post_text = self._generate_mock_post(platform, topic, topic_type)
            
            # Generate engagement metrics
            likes = int(np.random.exponential(10))
            comments = int(np.random.exponential(5))
            shares = int(np.random.exponential(3))
            
            # Generate user info
            user_followers = int(np.random.exponential(500))
            user_credibility = np.random.uniform(0, 1)
            
            results.append({
                "platform": platform,
                "post_id": f"{platform}_{np.random.randint(10000, 99999)}",
                "timestamp": post_time.isoformat(),
                "text": post_text,
                "topic": topic,
                "topic_type": topic_type,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "user_followers": user_followers,
                "user_credibility": user_credibility,
                "is_verified": np.random.random() < 0.1,  # 10% chance of verified
            })
        
        return results
    
    def _generate_mock_post(self, platform: str, topic: str, topic_type: str) -> str:
        """
        Generate mock social media post text.
        
        Args:
            platform: Social media platform
            topic: The main topic (symbol or keyword)
            topic_type: Type of topic (symbol or keyword)
            
        Returns:
            Generated post text
        """
        # Templates for different sentiment types
        bullish_templates = [
            f"Really optimistic about {topic} right now! Looking strong.",
            f"{topic} showing great momentum. Expecting it to outperform.",
            f"Just bought more {topic}. The fundamentals are solid.",
            f"The latest news for {topic} is incredibly positive. Going up!",
            f"Technical indicators for {topic} all pointing up. Clear buy signal."
        ]
        
        bearish_templates = [
            f"Not feeling good about {topic}. Might be time to sell.",
            f"{topic} breaking key support levels. Looking weak.",
            f"The problems with {topic} are mounting. Staying away for now.",
            f"Disappointed by {topic}'s recent performance. Cutting losses.",
            f"The market seems to be turning against {topic}. Be cautious."
        ]
        
        neutral_templates = [
            f"Watching {topic} closely. Could go either way.",
            f"Mixed signals from {topic}. Need more data.",
            f"Anyone have insights on {topic}?",
            f"Interesting developments with {topic} today.",
            f"Holding my {topic} position for now. No clear direction."
        ]
        
        # Pick sentiment randomly with weighted distribution
        sentiment_choice = np.random.choice(
            ["bullish", "bearish", "neutral"], 
            p=[0.4, 0.3, 0.3]  # Slightly biased toward bullish
        )
        
        if sentiment_choice == "bullish":
            templates = bullish_templates
        elif sentiment_choice == "bearish":
            templates = bearish_templates
        else:
            templates = neutral_templates
        
        # Select template
        post = np.random.choice(templates)
        
        # Add platform-specific formatting
        if platform == "twitter":
            # Add hashtags
            hashtags = f"#{topic} " if topic_type == "keyword" else f"${topic} #{topic} "
            hashtags += np.random.choice([
                "#investing #markets", 
                "#trading #stocks", 
                "#finance #investing",
                "#StockMarket",
                ""
            ])
            post = f"{post} {hashtags}"
            
        elif platform == "reddit":
            # Make it more discussion-oriented
            post = f"{post}\n\nWhat do you all think? I've been following {topic} for {np.random.randint(1, 24)} months."
            
        elif platform == "stocktwits":
            # Add cashtags
            cashtag = f"${topic}" if topic_type == "symbol" else topic
            post = f"{post} {cashtag} {np.random.choice(['#bullish', '#bearish', ''])}"
        
        return post
    
    def _classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score.
        
        Args:
            compound_score: VADER sentiment compound score
            
        Returns:
            Sentiment classification
        """
        if compound_score >= 0.05:
            return "bullish"
        elif compound_score <= -0.05:
            return "bearish"
        else:
            return "neutral"
    
    def process_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process raw social media data into trading signals.
        
        Args:
            data: DataFrame with social media posts and sentiment
            
        Returns:
            Dictionary with processed signals
        """
        if data.empty:
            return {"signal": "neutral", "strength": 0, "insights": []}
        
        # Preprocess
        if "timestamp" in data.columns and isinstance(data["timestamp"].iloc[0], str):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
        
        # Group by topic (symbol or keyword)
        topic_insights = {}
        
        for topic, group in data.groupby("topic"):
            # Calculate sentiment metrics
            sentiment_counts = group["sentiment"].value_counts()
            bullish_count = sentiment_counts.get("bullish", 0)
            bearish_count = sentiment_counts.get("bearish", 0)
            neutral_count = sentiment_counts.get("neutral", 0)
            total_count = bullish_count + bearish_count + neutral_count
            
            # Skip if no data
            if total_count == 0:
                continue
            
            # Calculate sentiment ratio
            bullish_ratio = bullish_count / total_count
            bearish_ratio = bearish_count / total_count
            
            # Calculate weighted sentiment (consider engagement)
            if "likes" in group.columns and "user_followers" in group.columns:
                # Weight by engagement and user influence
                weighted_sentiment = 0
                total_weight = 0
                
                for _, post in group.iterrows():
                    # Calculate post weight based on engagement and user influence
                    engagement = post.get("likes", 0) + post.get("comments", 0) * 2 + post.get("shares", 0) * 3
                    user_weight = np.log1p(post.get("user_followers", 0)) * (2 if post.get("is_verified") else 1)
                    post_weight = engagement * user_weight
                    
                    # Add weighted sentiment
                    weighted_sentiment += post.get("sentiment_score", 0) * post_weight
                    total_weight += post_weight
                
                # Calculate final weighted sentiment
                if total_weight > 0:
                    weighted_sentiment = weighted_sentiment / total_weight
            else:
                # Simple average if engagement metrics aren't available
                weighted_sentiment = group["sentiment_score"].mean()
            
            # Calculate sentiment momentum (change over time)
            if "timestamp" in group.columns and len(group) > 5:
                # Sort by timestamp
                sorted_group = group.sort_values("timestamp")
                
                # Split into two halves
                half_point = len(sorted_group) // 2
                first_half = sorted_group.iloc[:half_point]
                second_half = sorted_group.iloc[half_point:]
                
                # Calculate sentiment for each half
                first_sentiment = first_half["sentiment_score"].mean()
                second_sentiment = second_half["sentiment_score"].mean()
                
                # Calculate momentum
                sentiment_momentum = second_sentiment - first_sentiment
            else:
                sentiment_momentum = 0
            
            # Create insight
            insight = {
                "topic": topic,
                "post_count": int(total_count),
                "bullish_ratio": float(bullish_ratio),
                "bearish_ratio": float(bearish_ratio),
                "weighted_sentiment": float(weighted_sentiment),
                "sentiment_momentum": float(sentiment_momentum),
                "interpretation": self._interpret_sentiment(
                    bullish_ratio, bearish_ratio, weighted_sentiment, sentiment_momentum
                )
            }
            
            # Store insight
            topic_insights[topic] = insight
        
        # Generate overall signal from all topics
        signals = []
        
        for topic, insight in topic_insights.items():
            # Convert sentiment to signal
            signal_strength = self._sentiment_to_signal(
                insight["weighted_sentiment"],
                insight["sentiment_momentum"]
            )
            
            signal_direction = "bullish" if signal_strength > 0 else "bearish"
            
            signals.append({
                "topic": topic,
                "direction": signal_direction,
                "strength": abs(signal_strength),
                "post_count": insight["post_count"]
            })
        
        if not signals:
            return {"signal": "neutral", "strength": 0, "insights": []}
        
        # Calculate weighted average signal
        total_posts = sum(s["post_count"] for s in signals)
        weighted_direction = sum(
            s["post_count"] * s["strength"] * (1 if s["direction"] == "bullish" else -1) 
            for s in signals
        )
        
        if weighted_direction > 0:
            signal = "bullish"
            strength = weighted_direction / total_posts
        elif weighted_direction < 0:
            signal = "bearish"
            strength = abs(weighted_direction) / total_posts
        else:
            signal = "neutral"
            strength = 0
        
        return {
            "signal": signal,
            "strength": float(strength),
            "insights": list(topic_insights.values()),
            "raw_signals": signals,
            "timestamp": datetime.now().isoformat()
        }
    
    def _interpret_sentiment(
        self, 
        bullish_ratio: float, 
        bearish_ratio: float, 
        weighted_sentiment: float,
        sentiment_momentum: float
    ) -> str:
        """
        Generate textual interpretation of sentiment metrics.
        
        Args:
            bullish_ratio: Ratio of bullish posts
            bearish_ratio: Ratio of bearish posts
            weighted_sentiment: Weighted sentiment score
            sentiment_momentum: Change in sentiment over time
            
        Returns:
            Textual interpretation
        """
        # Describe sentiment distribution
        if bullish_ratio > 0.6:
            distribution = "predominantly bullish"
        elif bearish_ratio > 0.6:
            distribution = "predominantly bearish"
        elif bullish_ratio > bearish_ratio:
            distribution = "moderately bullish"
        elif bearish_ratio > bullish_ratio:
            distribution = "moderately bearish"
        else:
            distribution = "evenly divided"
        
        # Describe momentum
        if sentiment_momentum > 0.2:
            momentum = "rapidly improving"
        elif sentiment_momentum > 0.05:
            momentum = "improving"
        elif sentiment_momentum < -0.2:
            momentum = "rapidly deteriorating"
        elif sentiment_momentum < -0.05:
            momentum = "deteriorating"
        else:
            momentum = "stable"
        
        # Generate interpretation
        interpretation = f"Social media sentiment is {distribution} with {momentum} trend."
        
        # Add trading implication
        if weighted_sentiment > 0.3 and sentiment_momentum > 0:
            interpretation += " Strong positive sentiment may indicate buying pressure."
        elif weighted_sentiment < -0.3 and sentiment_momentum < 0:
            interpretation += " Strong negative sentiment may indicate selling pressure."
        elif abs(sentiment_momentum) > 0.2:
            interpretation += " Rapid sentiment shift may precede price movement."
        
        return interpretation
    
    def _sentiment_to_signal(self, weighted_sentiment: float, momentum: float) -> float:
        """
        Convert sentiment metrics to trading signal strength.
        
        Args:
            weighted_sentiment: Weighted sentiment score
            momentum: Sentiment momentum (change)
            
        Returns:
            Signal strength (-1 to 1, negative for bearish)
        """
        # Base signal from sentiment
        base_signal = weighted_sentiment
        
        # Adjust for momentum (emphasize direction if momentum confirms)
        if (base_signal > 0 and momentum > 0) or (base_signal < 0 and momentum < 0):
            # Momentum confirms direction, strengthen signal
            momentum_factor = 0.5
        else:
            # Momentum contradicts direction, reduce impact
            momentum_factor = 0.2
        
        # Calculate final signal
        signal = base_signal + (momentum * momentum_factor)
        
        # Ensure within bounds
        return max(min(signal, 1.0), -1.0)
    
    def _is_connected(self) -> bool:
        """Check if connected to social media platforms."""
        return bool(self.sessions)
    
    async def __aenter__(self):
        """Support for async context manager."""
        for platform in self.platforms:
            await self._ensure_session(platform)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting async context."""
        await self._close_sessions()
