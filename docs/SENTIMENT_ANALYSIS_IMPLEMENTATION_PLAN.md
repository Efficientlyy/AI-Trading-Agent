# Sentiment Analysis Implementation Plan

## Overview

This document outlines the detailed implementation plan for completing the sentiment analysis integration in the AI Crypto Trading Agent. Based on the analysis of the existing codebase, the sentiment analysis system is partially implemented with a well-defined architecture but requires further development to be fully functional and integrated with the trading system.

## Current Architecture

The sentiment analysis system follows a modular design with the following components:

1. **BaseSentimentAgent**: Abstract base class providing common functionality for all sentiment agents
2. **Specialized Sentiment Agents**:
   - **SocialMediaSentimentAgent**: Analyzes sentiment from social media platforms
   - **NewsSentimentAgent**: Analyzes sentiment from news sources
   - **MarketSentimentAgent**: Analyzes sentiment from market indicators
   - **OnchainSentimentAgent**: Analyzes sentiment from blockchain data
3. **SentimentAggregator**: Combines signals from various sources
4. **SentimentAnalysisManager**: Coordinates all sentiment components

## Implementation Status

### Existing Components

The following components are already implemented but contain simulated data rather than real data sources:

1. **BaseSentimentAgent**: Fully implemented with sentiment caching, event publishing, and common utilities
2. **SocialMediaSentimentAgent**: Implemented with simulated data
3. **NewsSentimentAgent**: Implemented with simulated data
4. **MarketSentimentAgent**: Implemented with simulated data
5. **OnchainSentimentAgent**: Implemented with simulated data
6. **SentimentAggregator**: Implemented with weighted aggregation logic
7. **SentimentAnalysisManager**: Implemented with component lifecycle management

### Missing Components

The following components need to be implemented:

1. **Real Data Sources**: Replace simulated data with real API integrations
2. **NLP Models**: Implement proper sentiment analysis models
3. **Trading Strategy Integration**: Connect sentiment signals to trading decisions
4. **Backtesting Framework**: Test sentiment strategies with historical data
5. **Performance Metrics**: Measure the effectiveness of sentiment signals

## Implementation Plan

### Phase 1: Real Data Source Integration

#### 1.1 Social Media Data Integration

```python
# Implementation in src/analysis_agents/sentiment/social_media_sentiment.py

class SocialMediaSentimentAgent(BaseSentimentAgent):
    # ... existing code ...
    
    async def _initialize(self) -> None:
        await super()._initialize()
        
        if not self.enabled:
            return
            
        # Initialize API clients
        self.twitter_client = TwitterClient(
            api_key=config.get("apis.twitter.api_key"),
            api_secret=config.get("apis.twitter.api_secret"),
            access_token=config.get("apis.twitter.access_token"),
            access_secret=config.get("apis.twitter.access_secret")
        )
        
        self.reddit_client = RedditClient(
            client_id=config.get("apis.reddit.client_id"),
            client_secret=config.get("apis.reddit.client_secret"),
            user_agent=config.get("apis.reddit.user_agent")
        )
        
        self.logger.info("Initialized social media API clients")
    
    async def _analyze_social_media_sentiment(self, symbol: str) -> None:
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "social_media", self.update_interval):
            return
        
        try:
            base_currency = symbol.split('/')[0]
            
            # Fetch Twitter data
            twitter_posts = await self.twitter_client.search_tweets(
                query=f"#{base_currency} OR ${base_currency}",
                count=100,
                result_type="recent"
            )
            
            # Fetch Reddit data
            subreddits = [f"r/{base_currency}", "r/CryptoCurrency", "r/CryptoMarkets"]
            reddit_posts = []
            
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
            
            # Process posts with NLP model
            sentiment_scores = await self._analyze_text_sentiment(all_posts)
            
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
                            "twitter_count": len(twitter_posts),
                            "reddit_count": len(reddit_posts),
                            "event_type": event_type
                        }
                    )
        
        except Exception as e:
            self.logger.error("Error analyzing social media sentiment", 
                           symbol=symbol,
                           error=str(e))
    
    async def _analyze_text_sentiment(self, posts: List[str]) -> List[float]:
        """Analyze sentiment of text posts using NLP.
        
        Args:
            posts: List of text posts to analyze
            
        Returns:
            List of sentiment scores (0-1 scale)
        """
        # Use NLP model to analyze sentiment
        # This would be replaced with a proper NLP model
        
        sentiment_scores = []
        
        for post in posts:
            # Process text to extract sentiment
            # For now, use a simple lexicon-based approach
            post_lower = post.lower()
            
            # Count bullish and bearish words
            bullish_count = sum(1 for word in self.bullish_words if word in post_lower)
            bearish_count = sum(1 for word in self.bearish_words if word in post_lower)
            
            # Calculate sentiment score
            if bullish_count + bearish_count > 0:
                sentiment = bullish_count / (bullish_count + bearish_count)
            else:
                sentiment = 0.5  # Neutral if no sentiment words
                
            sentiment_scores.append(sentiment)
            
        return sentiment_scores
```

#### 1.2 News Data Integration

```python
# Implementation in src/analysis_agents/sentiment/news_sentiment.py

class NewsSentimentAgent(BaseSentimentAgent):
    # ... existing code ...
    
    async def _initialize(self) -> None:
        await super()._initialize()
        
        if not self.enabled:
            return
            
        # Initialize API clients
        self.news_api_client = NewsApiClient(
            api_key=config.get("apis.news_api.api_key")
        )
        
        self.crypto_news_client = CryptoNewsClient(
            api_key=config.get("apis.crypto_news.api_key")
        )
        
        self.logger.info("Initialized news API clients")
    
    async def _analyze_news_sentiment(self, symbol: str) -> None:
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "news", self.update_interval):
            return
        
        try:
            base_currency = symbol.split('/')[0]
            
            # Fetch news articles
            news_api_articles = await self.news_api_client.get_everything(
                q=f"{base_currency} OR cryptocurrency",
                language="en",
                sort_by="publishedAt",
                page_size=20
            )
            
            crypto_news_articles = await self.crypto_news_client.get_news(
                categories=["blockchain", "cryptocurrency", "technology"],
                keywords=[base_currency],
                limit=20
            )
            
            # Combine articles
            all_articles = news_api_articles + crypto_news_articles
            article_count = len(all_articles)
            
            if article_count == 0:
                self.logger.warning("No news articles found", symbol=symbol)
                return
            
            # Extract article texts
            article_texts = [article.get("title", "") + " " + article.get("description", "") 
                            for article in all_articles]
            
            # Process articles with NLP model
            sentiment_scores = await self._analyze_text_sentiment(article_texts)
            
            # Calculate overall sentiment
            sentiment_value = sum(sentiment_scores) / len(sentiment_scores)
            
            # Calculate confidence based on article volume and source quality
            volume_factor = min(1.0, article_count / 15)
            quality_factor = 0.9  # News sources generally have higher quality
            confidence = volume_factor * quality_factor
            
            # Determine direction
            if sentiment_value > 0.55:
                direction = "bullish"
            elif sentiment_value < 0.45:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Extract keywords
            keywords = self._extract_keywords_from_news(" ".join(article_texts))
            
            # Store additional metadata
            additional_data = {
                "article_count": article_count,
                "sources": self.sources,
                "keywords": keywords
            }
            
            # Update the sentiment cache
            sentiment_shift = self._update_sentiment_cache(
                symbol=symbol,
                source_type="news",
                sentiment_value=sentiment_value,
                direction=direction,
                confidence=confidence,
                additional_data=additional_data
            )
            
            # Publish event if significant shift
            if sentiment_shift > self.sentiment_shift_threshold:
                # Publish event if confidence is high enough
                if confidence >= self.min_confidence:
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        sources=self.sources,
                        details={
                            "article_count": article_count,
                            "keywords": keywords[:10],  # Top 10 keywords
                            "event_type": "news_sentiment_shift"
                        }
                    )
        
        except Exception as e:
            self.logger.error("Error analyzing news sentiment", 
                           symbol=symbol,
                           error=str(e))
    
    async def _analyze_text_sentiment(self, texts: List[str]) -> List[float]:
        """Analyze sentiment of text articles using NLP.
        
        Args:
            texts: List of article texts to analyze
            
        Returns:
            List of sentiment scores (0-1 scale)
        """
        # This would be replaced with a proper NLP model
        # For now, use a simple lexicon-based approach similar to social media
        
        sentiment_scores = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Count bullish and bearish words
            bullish_count = sum(1 for word in self.bullish_words if word in text_lower)
            bearish_count = sum(1 for word in self.bearish_words if word in text_lower)
            
            # Calculate sentiment score
            if bullish_count + bearish_count > 0:
                sentiment = bullish_count / (bullish_count + bearish_count)
            else:
                sentiment = 0.5  # Neutral if no sentiment words
                
            sentiment_scores.append(sentiment)
            
        return sentiment_scores
```

#### 1.3 Market Sentiment Data Integration

```python
# Implementation in src/analysis_agents/sentiment/market_sentiment.py

class MarketSentimentAgent(BaseSentimentAgent):
    # ... existing code ...
    
    async def _initialize(self) -> None:
        await super()._initialize()
        
        if not self.enabled:
            return
            
        # Initialize API clients
        self.fear_greed_client = FearGreedClient()
        self.exchange_data_client = ExchangeDataClient(
            api_key=config.get("apis.exchange_data.api_key")
        )
        
        self.logger.info("Initialized market sentiment API clients")
    
    async def _analyze_market_sentiment_indicators(self, symbol: str) -> None:
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "market_sentiment", self.update_interval):
            return
        
        try:
            base_currency = symbol.split('/')[0]
            
            # Fetch Fear & Greed Index
            fear_greed_data = await self.fear_greed_client.get_current_index()
            fear_greed = fear_greed_data.get("value", 50)
            
            # Fetch long/short ratio from exchanges
            long_short_data = await self.exchange_data_client.get_long_short_ratio(
                symbol=symbol
            )
            long_short_ratio = long_short_data.get("longShortRatio", 1.0)
            
            # Calculate overall market sentiment (0-1)
            # Fear & Greed: 0=extreme fear, 100=extreme greed
            # Convert to 0-1 scale
            fg_sentiment = fear_greed / 100.0
            
            # Long/Short ratio: <1 means more shorts, >1 means more longs
            # Convert to 0-1 scale with 0.5 at ratio=1
            if long_short_ratio < 1:
                ls_sentiment = 0.5 * long_short_ratio
            else:
                ls_sentiment = 0.5 + 0.5 * min(1.0, (long_short_ratio - 1) / 2)
            
            # Combine both indicators (equal weight)
            sentiment_value = (fg_sentiment + ls_sentiment) / 2
            
            # Determine confidence based on the agreement between indicators
            indicator_agreement = 1.0 - abs(fg_sentiment - ls_sentiment)
            confidence = 0.7 + (indicator_agreement * 0.25)  # 0.7 to 0.95 range
            
            # Determine direction
            if sentiment_value > 0.6:  # Higher threshold for market indicators
                direction = "bullish"
            elif sentiment_value < 0.4:  # Lower threshold for market indicators
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Store additional metadata
            additional_data = {
                "fear_greed_index": fear_greed,
                "long_short_ratio": long_short_ratio,
                "indicators": self.indicators
            }
            
            # Update the sentiment cache
            sentiment_shift = self._update_sentiment_cache(
                symbol=symbol,
                source_type="market_sentiment",
                sentiment_value=sentiment_value,
                direction=direction,
                confidence=confidence,
                additional_data=additional_data
            )
            
            # Check for extreme values
            is_extreme = fear_greed <= 20 or fear_greed >= 80
            
            # Publish event if significant shift or extreme values
            if sentiment_shift > self.sentiment_shift_threshold or is_extreme:
                # Determine if extreme sentiment should be treated as contrarian
                signal_type = "sentiment"
                if is_extreme:
                    # Extreme fear/greed can be contrarian
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
                        sources=self.indicators,
                        details={
                            "fear_greed_index": fear_greed,
                            "long_short_ratio": long_short_ratio,
                            "event_type": "market_sentiment_shift" if sentiment_shift > self.sentiment_shift_threshold else "extreme_market_sentiment"
                        }
                    )
        
        except Exception as e:
            self.logger.error("Error analyzing market sentiment indicators", 
                           symbol=symbol,
                           error=str(e))
```

#### 1.4 Onchain Data Integration

```python
# Implementation in src/analysis_agents/sentiment/onchain_sentiment.py

class OnchainSentimentAgent(BaseSentimentAgent):
    # ... existing code ...
    
    async def _initialize(self) -> None:
        await super()._initialize()
        
        if not self.enabled:
            return
            
        # Initialize API clients
        self.blockchain_client = BlockchainClient(
            api_key=config.get("apis.blockchain.api_key")
        )
        
        self.logger.info("Initialized blockchain API client")
    
    async def _analyze_onchain_metrics(self, symbol: str) -> None:
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "onchain", self.update_interval):
            return
            
        # Only some cryptocurrencies have meaningful on-chain metrics
        base_currency = symbol.split('/')[0]
        if base_currency not in ["BTC", "ETH"]:
            return
        
        try:
            # Fetch on-chain metrics
            
            # Large transactions count and volume
            large_tx_data = await self.blockchain_client.get_large_transactions(
                asset=base_currency,
                time_period="24h"
            )
            large_tx_count = large_tx_data.get("count", 0)
            large_tx_volume = large_tx_data.get("volume", 0)
            
            # Active addresses
            active_addr_data = await self.blockchain_client.get_active_addresses(
                asset=base_currency,
                time_period="24h"
            )
            active_addresses = active_addr_data.get("count", 0)
            active_addr_change = active_addr_data.get("change_percentage", 0)
            
            # Network hash rate (for PoW chains)
            hash_rate_data = None
            hash_rate_change = 0
            if base_currency == "BTC":
                hash_rate_data = await self.blockchain_client.get_hash_rate(
                    asset=base_currency,
                    time_period="7d"
                )
                hash_rate_change = hash_rate_data.get("change_percentage", 0) if hash_rate_data else 0
            
            # Exchange reserves
            exchange_data = await self.blockchain_client.get_exchange_reserves(
                asset=base_currency,
                time_period="7d"
            )
            exchange_reserves = exchange_data.get("reserves", 0)
            exchange_reserves_change = exchange_data.get("change_percentage", 0)
            
            # Normalize metrics to sentiment scores (0-1)
            
            # Large transactions (normalize based on historical averages)
            large_tx_normalized = min(1.0, large_tx_volume / large_tx_data.get("average_volume", large_tx_volume))
            
            # Active addresses growth (-100% to +100%, normalize to 0-1)
            active_addr_normalized = 0.5 + (active_addr_change / 200)  # Convert to 0-1 scale
            
            # Hash rate change (-100% to +100%, normalize to 0-1)
            hash_rate_normalized = 0.5 + (hash_rate_change / 200)  # Convert to 0-1 scale
            
            # Exchange reserves change (-100% to +100%, normalize to 0-1)
            # Negative means tokens leaving exchanges (bullish)
            exchange_reserves_normalized = 0.5 - (exchange_reserves_change / 200)
            
            # Combine metrics into a sentiment score
            # Each metric is weighted differently
            sentiment_metrics = {
                "large_transactions": large_tx_normalized,
                "active_addresses": active_addr_normalized,
                "hash_rate": hash_rate_normalized if hash_rate_data else 0.5,
                "exchange_reserves": exchange_reserves_normalized
            }
            
            # Calculate weighted sentiment
            metric_weights = {
                "large_transactions": 0.3,
                "active_addresses": 0.3,
                "hash_rate": 0.2 if hash_rate_data else 0,
                "exchange_reserves": 0.2
            }
            
            # Adjust weights if hash rate is not available
            if not hash_rate_data:
                total_weight = sum(metric_weights.values())
                for key in metric_weights:
                    metric_weights[key] = metric_weights[key] / total_weight
            
            sentiment_value = sum(
                sentiment_metrics[metric] * metric_weights[metric]
                for metric in sentiment_metrics
            ) / sum(metric_weights.values())
            
            # Calculate confidence based on data quality
            confidence = 0.7  # Base confidence
            
            # Determine direction
            if sentiment_value > 0.6:
                direction = "bullish"
            elif sentiment_value < 0.4:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Store additional metadata
            additional_data = {
                "large_transactions_count": large_tx_count,
                "large_transactions_volume": large_tx_volume,
                "active_addresses": active_addresses,
                "active_addresses_change": active_addr_change,
                "hash_rate_change": hash_rate_change if hash_rate_data else None,
                "exchange_reserves": exchange_reserves,
                "exchange_reserves_change": exchange_reserves_change,
                "metrics": self.metrics
            }
            
            # Update the sentiment cache
            sentiment_shift = self._update_sentiment_cache(
                symbol=symbol,
                source_type="onchain",
                sentiment_value=sentiment_value,
                direction=direction,
                confidence=confidence,
                additional_data=additional_data
            )
            
            # Publish event if significant shift or strong signal
            if sentiment_shift > self.sentiment_shift_threshold or confidence > 0.85:
                # Publish event if confidence is high enough
                if confidence >= self.min_confidence:
                    # Exchange outflows are often a strong signal
                    is_strong_signal = exchange_reserves_change < -5.0  # 5% outflow
                    
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        is_extreme=is_strong_signal,
                        sources=self.metrics,
                        details=additional_data
                    )
        
        except Exception as e:
            self.logger.error("Error analyzing onchain metrics", 
                           symbol=symbol,
                           error=str(e))
```

### Phase 2: NLP Model Implementation

#### 2.1 Create NLP Service

```python
# Implementation in src/analysis_agents/sentiment/nlp_service.py

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np
from transformers import pipeline

from src.common.config import config
from src.common.logging import get_logger

class NLPService:
    """Service for natural language processing tasks.
    
    This service provides NLP functionality for sentiment analysis,
    including text classification and entity recognition.
    """
    
    def __init__(self):
        """Initialize the NLP service."""
        self.logger = get_logger("analysis_agents", "nlp_service")
        
        # Load configuration
        self.model_name = config.get("nlp.sentiment_model", "distilbert-base-uncased-finetuned-sst-2-english")
        self.batch_size = config.get("nlp.batch_size", 16)
        
        # Initialize sentiment pipeline
        self.sentiment_pipeline = None
        
    async def initialize(self) -> None:
        """Initialize the NLP models."""
        self.logger.info("Initializing NLP service")
        
        # Load sentiment analysis model
        try:
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.sentiment_pipeline = await loop.run_in_executor(
                None, 
                lambda: pipeline("sentiment-analysis", model=self.model_name)
            )
            self.logger.info("Loaded sentiment analysis model", model=self.model_name)
        except Exception as e:
            self.logger.error("Failed to load sentiment analysis model", error=str(e))
            # Fall back to lexicon-based approach
            self.sentiment_pipeline = None
            self._load_sentiment_lexicons()
            
    def _load_sentiment_lexicons(self) -> None:
        """Load sentiment lexicons for text analysis."""
        self.logger.info("Loading sentiment lexicons as fallback")
        
        # Bullish words/phrases
        self.bullish_words = [
            "bullish", "buy", "long", "potential", "upside", "green", 
            "higher", "surge", "rally", "moon", "strong", "growth",
            "breakout", "outperform", "upgrade", "accumulate",
            "support", "bottom", "opportunity", "bullrun"
        ]
        
        # Bearish words/phrases
        self.bearish_words = [
            "bearish", "sell", "short", "downside", "red", "lower", 
            "drop", "fall", "dump", "weak", "decline", "breakdown",
            "underperform", "downgrade", "distribute", "resistance", 
            "top", "risk", "crash", "correction"
        ]
    
    async def analyze_sentiment(self, texts: List[str]) -> List[float]:
        """Analyze sentiment of text.
        
        Args:
            texts: List of text to analyze
            
        Returns:
            List of sentiment scores (0-1 scale)
        """
        if not texts:
            return []
            
        if self.sentiment_pipeline:
            return await self._analyze_with_model(texts)
        else:
            return self._analyze_with_lexicon(texts)
    
    async def _analyze_with_model(self, texts: List[str]) -> List[float]:
        """Analyze sentiment using the transformer model.
        
        Args:
            texts: List of text to analyze
            
        Returns:
            List of sentiment scores (0-1 scale)
        """
        try:
            # Process in batches to avoid memory issues
            results = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                
                # Run in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    None, 
                    lambda: self.sentiment_pipeline(batch)
                )
                
                # Convert to sentiment scores (0-1 scale)
                for result in batch_results:
                    label = result["label"]
                    score = result["score"]
                    
                    if label == "POSITIVE":
                        sentiment = score
                    elif label == "NEGATIVE":
                        sentiment = 1.0 - score
                    else:
                        sentiment = 0.5
                        
                    results.append(sentiment)
            
            return results
            
        except Exception as e:
            self.logger.error("Error in model-based sentiment analysis", error=str(e))
            # Fall back to lexicon-based approach
            return self._analyze_with_lexicon(texts)
    
    def _analyze_with_lexicon(self, texts: List[str]) -> List[float]:
        """Analyze sentiment using lexicon-based approach.
        
        Args:
            texts: List of text to analyze
            
        Returns:
            List of sentiment scores (0-1 scale)
        """
        sentiment_scores = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Count bullish and bearish words
            bullish_count = sum(1 for word in self.bullish_words if word in text_lower)
            bearish_count = sum(1 for word in self.bearish_words if word in text_lower)
            
            # Calculate sentiment score
            if bullish_count + bearish_count > 0:
                sentiment = bullish_count / (bullish_count + bearish_count)
            else:
                sentiment = 0.5  # Neutral if no sentiment words
                
            sentiment_scores.append(sentiment)
            
        return sentiment_scores
```

#### 2.2 Integrate NLP Service with Sentiment Agents

```python
# Update SentimentAnalysisManager to include NLP service

class SentimentAnalysisManager(Component):
    # ... existing code ...
    
    def __init__(self):
        """Initialize the sentiment analysis manager."""
        super().__init__("sentiment_analysis_manager")
        self.logger = get_logger("analysis_agents", "sentiment_manager")
        
        # Load configuration
        self.enabled = config.get("analysis_agents.sentiment.enabled", True)
        self.agent_configs = config.get("analysis_agents.sentiment.agents", {})
        
        # Create NLP service
        self.nlp_service = NLPService()
        
        # Create sentiment agents
        self.agents: Dict[str, BaseSentimentAgent] = {}
        self._create_agents()
    
    async def _initialize(self) -> None:
        """Initialize all sentiment analysis agents."""
        if not self.enabled:
            self.logger.info("Sentiment analysis manager is disabled")
            return
            
        self.logger.info("Initializing sentiment analysis manager")
        
        # Initialize NLP service first
        await self.nlp_service.initialize()
        
        # Initialize all agents
        init_tasks = []
        for agent_id, agent in self.agents.items():
            # Pass NLP service to agent
            agent.set_nlp_service(self.nlp_service)
            init_tasks.append(agent.initialize())
            
        # Wait for all agents to initialize
        if init_tasks:
            await asyncio.gather(*init_tasks)
```

```python
# Update BaseSentimentAgent to use NLP service

class BaseSentimentAgent(AnalysisAgent):
    # ... existing code ...
    
    def __init__(self, agent_id: str):
        """Initialize the base sentiment analysis agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", f"sentiment_{agent_id}")
        
        # NLP service (will be set by manager)
        self.nlp_service = None
        
        # ... rest of existing code ...
    
    def set_nlp_service(self, nlp_service: 'NLPService') -> None:
        """Set the NLP service for this agent.
        
        Args:
            nlp_service: The NLP service to use
        """
        self.nlp_service = nlp_service
```

### Phase 3: Trading Strategy Integration

#### 3.1 Create Sentiment Strategy

```python
# Implementation in src/strategy/enhanced_sentiment_strategy.py

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime, timedelta

from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.models.events import SentimentEvent, SignalEvent
from src.models.market_data import CandleData, TimeFrame
from src.models.signals import Signal, SignalType
from src.strategy.sentiment_strategy import SentimentStrategy

class EnhancedSentimentStrategy(SentimentStrategy):
    """Enhanced strategy that combines sentiment with technical indicators.
    
    This strategy extends the basic sentiment strategy by incorporating
    technical indicators and market regime detection to improve signal quality.
    """
    
    def __init__(self, strategy_id: str = "enhanced_sentiment"):
        """Initialize the enhanced sentiment strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", "enhanced_sentiment")
        
        # Additional configuration
        self.use_market_regime = config.get(f"strategies.{strategy_id}.use_market_regime", True)
        self.use_technical_confirmation = config.get(f"strategies.{strategy_id}.use_technical_confirmation", True)
        self.min_signal_score = config.get(f"strategies.{strategy_id}.min_signal_score", 0.7)
        
        # Technical indicators for confirmation
        self.rsi_period = config.get(f"strategies.{strategy_id}.rsi_period", 14)
        self.rsi_overbought = config.get(f"strategies.{strategy_id}.rsi_overbought", 70)
        self.rsi_oversold = config.get(f"strategies.{strategy_id}.rsi_oversold", 30)
        
        # Market regime detection
        self.regime_lookback = config.get(f"strategies.{strategy_id}.regime_lookback", 20)
        
        # Subscribe to sentiment events
        self.subscribe_to_events()
    
    def subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        event_bus.subscribe("sentiment_event", self.handle_sentiment_event)
        event_bus.subscribe("market_regime_event", self.handle_market_regime_event)
    
    async def handle_sentiment_event(self, event: SentimentEvent) -> None:
        """Handle sentiment events.
        
        Args:
            event: The sentiment event to handle
        """
        # Process the sentiment event
        symbol = event.symbol
        direction = event.sentiment_direction
        value = event.sentiment_value
        confidence = event.confidence
        details = event.details
        
        # Get the source type
        source = event.source
        source_type = source.split("_")[-1] if "_" in source else "unknown"
        
        # Store in sentiment cache
        if symbol not in self.sentiment_cache:
            self.sentiment_cache[symbol] = {}
            
        self.sentiment_cache[symbol][source_type] = {
            "direction": direction,
            "value": value,
            "confidence": confidence,
            "timestamp": datetime.utcnow(),
            "details": details
        }
        
        # Check if we have enough data to generate a signal
        await self.check_for_signal(symbol)
    
    async def handle_market_regime_event(self, event: Any) -> None:
        """Handle market regime events.
        
        Args:
            event: The market regime event to handle
        """
        # Store market regime information
        symbol = event.symbol
        regime = event.regime
        
        if symbol not in self.market_regimes:
            self.market_regimes[symbol] = {}
            
        self.market_regimes[symbol] = {
            "regime": regime,
            "timestamp": datetime.utcnow()
        }
    
    async def check_for_signal(self, symbol: str) -> None:
        """Check if we should generate a trading signal.
        
        Args:
            symbol: The trading pair symbol
        """
        # Check if we have sentiment data
        if symbol not in self.sentiment_cache:
            return
            
        # Get aggregated sentiment if available
        if "aggregator" in self.sentiment_cache[symbol]:
            sentiment_data = self.sentiment_cache[symbol]["aggregator"]
        else:
            # Need at least two sentiment sources
            if len(self.sentiment_cache[symbol]) < 2:
                return
                
            # Calculate simple average
            values = [data["value"] for data in self.sentiment_cache[symbol].values()]
            confidences = [data["confidence"] for data in self.sentiment_cache[symbol].values()]
            
            sentiment_value = sum(values) / len(values)
            confidence = sum(confidences) / len(confidences)
            
            if sentiment_value > 0.6:
                direction = "bullish"
            elif sentiment_value < 0.4:
                direction = "bearish"
            else:
                direction = "neutral"
                
            sentiment_data = {
                "direction": direction,
                "value": sentiment_value,
                "confidence": confidence
            }
        
        # Check if sentiment is strong enough
        if sentiment_data["direction"] == "neutral":
            return
            
        if sentiment_data["confidence"] < self.min_confidence:
            return
            
        # Check market regime if enabled
        if self.use_market_regime and symbol in self.market_regimes:
            regime = self.market_regimes[symbol]["regime"]
            
            # Only generate signals that align with the market regime
            if sentiment_data["direction"] == "bullish" and regime not in ["bullish", "neutral"]:
                return
                
            if sentiment_data["direction"] == "bearish" and regime not in ["bearish", "neutral"]:
                return
        
        # Calculate signal score
        signal_score = sentiment_data["value"] if sentiment_data["direction"] == "bullish" else (1 - sentiment_data["value"])
        signal_score = signal_score * sentiment_data["confidence"]
        
        # Check if score is high enough
        if signal_score < self.min_signal_score:
            return
        
        # Generate signal
        signal_type = SignalType.LONG if sentiment_data["direction"] == "bullish" else SignalType.SHORT
        
        # Create signal with metadata
        signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy=self.name,
            confidence=sentiment_data["confidence"],
            metadata={
                "sentiment_value": sentiment_data["value"],
                "signal_score": signal_score,
                "sources": list(self.sentiment_cache[symbol].keys())
            }
        )
        
        # Publish signal event
        await self.publish_signal(signal)
        
        self.logger.info("Generated sentiment-based signal", 
                       symbol=symbol,
                       signal_type=signal_type.name,
                       confidence=sentiment_data["confidence"],
                       score=signal_score)
```

#### 3.2 Create Strategy Factory

```python
# Implementation in src/strategy/factory.py

from typing import Dict, Type, Optional, Any

from src.common.config import config
from src.common.logging import get_logger
from src.strategy.base_strategy import Strategy
from src.strategy.ma_crossover import MACrossoverStrategy
from src.strategy.sentiment_strategy import SentimentStrategy
from src.strategy.enhanced_sentiment_strategy import EnhancedSentimentStrategy
from src.strategy.market_imbalance import MarketImbalanceStrategy
from src.strategy.meta_strategy import MetaStrategy

class StrategyFactory:
    """Factory for creating trading strategies.
    
    This class provides a centralized way to create strategy instances
    based on configuration.
    """
    
    # Registry of available strategies
    _strategies: Dict[str, Type[Strategy]] = {
        "ma_crossover": MACrossoverStrategy,
        "sentiment": SentimentStrategy,
        "enhanced_sentiment": EnhancedSentimentStrategy,
        "market_imbalance": MarketImbalanceStrategy,
        "meta": MetaStrategy
    }
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[Strategy]) -> None:
        """Register a new strategy class.
        
        Args:
            name: The name of the strategy
            strategy_class: The strategy class to register
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def create(cls, strategy_type: str, strategy_id: Optional[str] = None, **kwargs: Any) -> Optional[Strategy]:
        """Create a strategy instance.
        
        Args:
            strategy_type: The type of strategy to create
            strategy_id: Optional custom ID for the strategy
            **kwargs: Additional arguments to pass to the strategy constructor
            
        Returns:
            The created strategy instance, or None if the strategy type is not found
        """
        logger = get_logger("strategy", "factory")
        
        if strategy_type not in cls._strategies:
            logger.error("Unknown strategy type", strategy_type=strategy_type)
            return None
            
        strategy_class = cls._strategies[strategy_type]
        
        try:
            if strategy_id:
                strategy = strategy_class(strategy_id=strategy_id, **kwargs)
            else:
                strategy = strategy_class(**kwargs)
                
            logger.info("Created strategy", 
                      type=strategy_type, 
                      id=strategy.strategy_id)
                
            return strategy
            
        except Exception as e:
            logger.error("Failed to create strategy", 
                       type=strategy_type,
                       error=str(e))
            return None
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, Type[Strategy]]:
        """Get all available strategy types.
        
        Returns:
            Dictionary of strategy names to strategy classes
        """
        return cls._strategies.copy()
```

### Phase 4: Backtesting Framework

#### 4.1 Create Sentiment Backtester

```python
# Implementation in src/backtesting/sentiment_backtester.py

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
import numpy as np

from src.common.config import config
from src.common.logging import get_logger
from src.models.events import SentimentEvent
from src.models.market_data import CandleData, TimeFrame
from src.models.signals import Signal, SignalType
from src.strategy.sentiment_strategy import SentimentStrategy
from src.strategy.enhanced_sentiment_strategy import EnhancedSentimentStrategy
from src.backtesting.base_backtester import BaseBacktester

class SentimentBacktester(BaseBacktester):
    """Backtester for sentiment-based strategies.
    
    This class provides functionality for backtesting sentiment-based
    trading strategies using historical sentiment and market data.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the sentiment backtester.
        
        Args:
            config_path: Optional path to configuration file
        """
        super().__init__(config_path)
        self.logger = get_logger("backtesting", "sentiment_backtester")
        
        # Load sentiment data
        self.sentiment_data: Dict[str, pd.DataFrame] = {}
        
        # Strategy to test
        self.strategy = None
    
    async def initialize(self) -> None:
        """Initialize the backtester."""
        await super().initialize()
        
        # Create strategy
        strategy_type = self.config.get("strategy.type", "enhanced_sentiment")
        strategy_id = self.config.get("strategy.id", "backtest_sentiment")
        
        if strategy_type == "enhanced_sentiment":
            self.strategy = EnhancedSentimentStrategy(strategy_id=strategy_id)
        else:
            self.strategy = SentimentStrategy(strategy_id=strategy_id)
            
        # Initialize strategy
        await self.strategy.initialize()
        
        # Load sentiment data
        await self._load_sentiment_data()
    
    async def _load_sentiment_data(self) -> None:
        """Load historical sentiment data."""
        sentiment_path = self.config.get("data.sentiment_path", "data/sentiment")
        symbols = self.config.get("symbols", ["BTC/USDT"])
        
        for symbol in symbols:
            try:
                # Load sentiment data from CSV
                file_path = f"{sentiment_path}/{symbol.replace('/', '_')}_sentiment.csv"
                df = pd.read_csv(file_path, parse_dates=["timestamp"])
                
                # Store in dictionary
                self.sentiment_data[symbol] = df
                
                self.logger.info("Loaded sentiment data", 
                               symbol=symbol,
                               records=len(df))
                               
            except Exception as e:
                self.logger.error("Failed to load sentiment data", 
                               symbol=symbol,
                               error=str(e))
    
    async def run_backtest(self) -> Dict[str, Any]:
        """Run the backtest.
        
        Returns:
            Dictionary of backtest results
        """
        # Get configuration
        symbols = self.config.get("symbols", ["BTC/USDT"])
        start_date = pd.to_datetime(self.config.get("backtest.start_date", "2023-01-01"))
        end_date = pd.to_datetime(self.config.get("backtest.end_date", "2023-12-31"))
        
        # Results dictionary
        results = {
            "trades": [],
            "equity_curve": [],
            "metrics": {}
        }
        
        # Process each symbol
        for symbol in symbols:
            # Check if we have data
            if symbol not in self.market_data or symbol not in self.sentiment_data:
                self.logger.warning("Missing data for symbol", symbol=symbol)
                continue
                
            # Get market data
            candles = self.market_data[symbol]
            
            # Filter by date range
            candles = [c for c in candles if start_date <= c.timestamp <= end_date]
            
            if not candles:
                self.logger.warning("No candles in date range", symbol=symbol)
                continue
                
            # Get sentiment data
            sentiment_df = self.sentiment_data[symbol]
            
            # Filter by date range
            sentiment_df = sentiment_df[(sentiment_df["timestamp"] >= start_date) & 
                                      (sentiment_df["timestamp"] <= end_date)]
            
            if len(sentiment_df) == 0:
                self.logger.warning("No sentiment data in date range", symbol=symbol)
                continue
                
            # Run backtest for this symbol
            symbol_results = await self._backtest_symbol(symbol, candles, sentiment_df)
            
            # Add to overall results
            results["trades"].extend(symbol_results["trades"])
            
            # Merge equity curves
            if not results["equity_curve"]:
                results["equity_curve"] = symbol_results["equity_curve"]
            else:
                # Merge by timestamp
                equity_df = pd.DataFrame(results["equity_curve"])
                symbol_equity_df = pd.DataFrame(symbol_results["equity_curve"])
                
                merged_df = pd.merge(equity_df, symbol_equity_df, on="timestamp", how="outer")
                merged_df = merged_df.fillna(method="ffill")
                
                # Sum equity values
                equity_cols = [col for col in merged_df.columns if col.endswith("_equity")]
                merged_df["equity"] = merged_df[equity_cols].sum(axis=1)
                
                results["equity_curve"] = merged_df.to_dict("records")
        
        # Calculate overall metrics
        results["metrics"] = self._calculate_metrics(results["trades"], results["equity_curve"])
        
        return results
    
    async def _backtest_symbol(
        self, 
        symbol: str, 
        candles: List[CandleData], 
        sentiment_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run backtest for a single symbol.
        
        Args:
            symbol: The trading pair symbol
            candles: List of candle data
            sentiment_df: DataFrame of sentiment data
            
        Returns:
            Dictionary of backtest results for this symbol
        """
        # Reset strategy state
        await self.strategy.reset()
        
        # Sort candles by timestamp
        candles = sorted(candles, key=lambda c: c.timestamp)
        
        # Sort sentiment data by timestamp
        sentiment_df = sentiment_df.sort_values("timestamp")
        
        # Initialize results
        trades = []
        equity_curve = []
        
        # Initial equity
        equity = self.config.get("backtest.initial_equity", 10000.0)
        
        # Current position
        position = None
        
        # Process each candle
        for i, candle in enumerate(candles):
            # Current timestamp
            timestamp = candle.timestamp
            
            # Get sentiment events up to this timestamp
            current_sentiment = sentiment_df[sentiment_df["timestamp"] <= timestamp]
            
            if len(current_sentiment) == 0:
                continue
                
            # Get the latest sentiment
            latest_sentiment = current_sentiment.iloc[-1]
            
            # Create sentiment event
            event = SentimentEvent(
                source=latest_sentiment["source"],
                symbol=symbol,
                sentiment_value=latest_sentiment["value"],
                sentiment_direction=latest_sentiment["direction"],
                confidence=latest_sentiment["confidence"],
                details={}
            )
            
            # Process sentiment event
            await self.strategy.handle_sentiment_event(event)
            
            # Process market data
            await self.strategy.process_candle(candle)
            
            # Check for signals
            signals = self.strategy.get_signals(symbol)
            
            if signals:
                for signal in signals:
                    # Process signal
                    if position is None:
                        # Open new position
                        if signal.signal_type == SignalType.LONG:
                            position = {
                                "type": "long",
                                "entry_price": candle.close,
                                "entry_time": timestamp,
                                "size": equity / candle.close,
                                "equity": equity
                            }
                        elif signal.signal_type == SignalType.SHORT:
                            position = {
                                "type": "short",
                                "entry_price": candle.close,
                                "entry_time": timestamp,
                                "size": equity / candle.close,
                                "equity": equity
                            }
                    elif position["type"] == "long" and signal.signal_type == SignalType.SHORT:
                        # Close long position
                        exit_price = candle.close
                        pnl = (exit_price - position["entry_price"]) * position["size"]
                        equity += pnl
                        
                        # Record trade
                        trades.append({
                            "symbol": symbol,
                            "type": "long",
                            "entry_time": position["entry_time"],
                            "entry_price": position["entry_price"],
                            "exit_time": timestamp,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "return": pnl / position["equity"]
                        })
                        
                        # Open short position
                        position = {
                            "type": "short",
                            "entry_price": candle.close,
                            "entry_time": timestamp,
                            "size": equity / candle.close,
                            "equity": equity
                        }
                    elif position["type"] == "short" and signal.signal_type == SignalType.LONG:
                        # Close short position
                        exit_price = candle.close
                        pnl = (position["entry_price"] - exit_price) * position["size"]
                        equity += pnl
                        
                        # Record trade
                        trades.append({
                            "symbol": symbol,
                            "type": "short",
                            "entry_time": position["entry_time"],
                            "entry_price": position["entry_price"],
                            "exit_time": timestamp,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "return": pnl / position["equity"]
                        })
                        
                        # Open long position
                        position = {
                            "type": "long",
                            "entry_price": candle.close,
                            "entry_time": timestamp,
                            "size": equity / candle.close,
                            "equity": equity
                        }
            
            # Calculate current equity
            current_equity = equity
            if position:
                if position["type"] == "long":
                    unrealized_pnl = (candle.close - position["entry_price"]) * position["size"]
                else:  # short
                    unrealized_pnl = (position["entry_price"] - candle.close) * position["size"]
                    
                current_equity = equity + unrealized_pnl
            
            # Record equity curve
            equity_curve.append({
                "timestamp": timestamp,
                f"{symbol}_equity": current_equity
            })
        
        # Close any open position at the end
        if position:
            exit_price = candles[-1].close
            
            if position["type"] == "long":
                pnl = (exit_price - position["entry_price"]) * position["size"]
            else:  # short
                pnl = (position["entry_price"] - exit_price) * position["size"]
                
            equity += pnl
            
            # Record trade
            trades.append({
                "symbol": symbol,
                "type": position["type"],
                "entry_time": position["entry_time"],
                "entry_price": position["entry_price"],
                "exit_time": candles[-1].timestamp,
                "exit_price": exit_price,
                "pnl": pnl,
                "return": pnl / position["equity"]
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)
        
        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics
        }
    
    def _calculate_metrics(self, trades: List[Dict[str, Any]], equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics.
        
        Args:
            trades: List of trade records
            equity_curve: List of equity curve points
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades or not equity_curve:
            return {}
            
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        if "return" in trades_df.columns:
            avg_return = trades_df["return"].mean()
            total_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) - 1
        else:
            avg_return = 0
            total_return = 0
        
        # Calculate drawdown
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] / equity_df["peak"]) - 1
        max_drawdown = equity_df["drawdown"].min()
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_return": total_return,
            "max_drawdown": max_drawdown
        }
```

### Phase 5: Performance Metrics

#### 5.1 Create Sentiment Performance Evaluator

```python
# Implementation in src/ml/evaluation/sentiment_evaluator.py

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.common.config import config
from src.common.logging import get_logger

class SentimentPerformanceEvaluator:
    """Evaluator for sentiment analysis performance.
    
    This class provides functionality for evaluating the performance
    of sentiment analysis in predicting market movements.
    """
    
    def __init__(self):
        """Initialize the sentiment performance evaluator."""
        self.logger = get_logger("ml", "sentiment_evaluator")
    
    def evaluate_sentiment_predictions(
        self, 
        sentiment_data: pd.DataFrame, 
        price_data: pd.DataFrame, 
        prediction_horizon: int = 24,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Evaluate sentiment predictions against price movements.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            price_data: DataFrame with price data
            prediction_horizon: Hours to look ahead for price movement
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure DataFrames have datetime index
        sentiment_data = sentiment_data.copy()
        price_data = price_data.copy()
        
        if "timestamp" in sentiment_data.columns:
            sentiment_data["timestamp"] = pd.to_datetime(sentiment_data["timestamp"])
            sentiment_data = sentiment_data.set_index("timestamp")
            
        if "timestamp" in price_data.columns:
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
            price_data = price_data.set_index("timestamp")
        
        # Filter sentiment data by confidence
        high_confidence = sentiment_data[sentiment_data["confidence"] >= confidence_threshold]
        
        if len(high_confidence) == 0:
            self.logger.warning("No high-confidence sentiment data")
            return {}
        
        # Create predictions
        predictions = []
        
        for idx, row in high_confidence.iterrows():
            # Get sentiment direction
            if row["direction"] == "bullish":
                prediction = 1  # Up
            elif row["direction"] == "bearish":
                prediction = -1  # Down
            else:
                prediction = 0  # Neutral
                
            # Skip neutral predictions
            if prediction == 0:
                continue
                
            # Get future price data
            future_time = idx + timedelta(hours=prediction_horizon)
            
            # Find closest price data points
            current_price = price_data.loc[price_data.index <= idx, "close"].iloc[-1]
            
            future_prices = price_data.loc[price_data.index >= future_time, "close"]
            if len(future_prices) == 0:
                continue
                
            future_price = future_prices.iloc[0]
            
            # Calculate actual movement
            price_change = (future_price / current_price) - 1
            actual = 1 if price_change > 0 else -1
            
            # Record prediction
            predictions.append({
                "timestamp": idx,
                "sentiment_value": row["value"],
                "confidence": row["confidence"],
                "predicted": prediction,
                "actual": actual,
                "price_change": price_change,
                "correct": prediction == actual
            })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        if len(predictions_df) == 0:
            self.logger.warning("No valid predictions")
            return {}
        
        # Calculate metrics
        total_predictions = len(predictions_df)
        correct_predictions = predictions_df["correct"].sum()
        accuracy = correct_predictions / total_predictions
        
        # Calculate metrics by direction
        bullish_df = predictions_df[predictions_df["predicted"] == 1]
        bearish_df = predictions_df[predictions_df["predicted"] == -1]
        
        bullish_accuracy = bullish_df["correct"].mean() if len(bullish_df) > 0 else 0
        bearish_accuracy = bearish_df["correct"].mean() if len(bearish_df) > 0 else 0
        
        # Calculate average price change
        avg_bullish_change = bullish_df["price_change"].mean() if len(bullish_df) > 0 else 0
        avg_bearish_change = bearish_df["price_change"].mean() if len(bearish_df) > 0 else 0
        
        # Calculate metrics by confidence level
        confidence_bins = [0.7, 0.8, 0.9, 1.0]
        confidence_metrics = []
        
        for i in range(len(confidence_bins) - 1):
            lower = confidence_bins[i]
            upper = confidence_bins[i+1]
            
            bin_df = predictions_df[(predictions_df["confidence"] >= lower) & 
                                  (predictions_df["confidence"] < upper)]
            
            if len(bin_df) > 0:
                bin_accuracy = bin_df["correct"].mean()
                bin_count = len(bin_df)
            else:
                bin_accuracy = 0
                bin_count = 0
                
            confidence_metrics.append({
                "confidence_range": f"{lower:.1f}-{upper:.1f}",
                "accuracy": bin_accuracy,
                "count": bin_count
            })
        
        # Return all metrics
        return {
            "total_predictions": total_predictions,
            "accuracy": accuracy,
            "bullish_accuracy": bullish_accuracy,
            "bearish_accuracy": bearish_accuracy,
            "bullish_predictions": len(bullish_df),
            "bearish_predictions": len(bearish_df),
            "avg_bullish_change": avg_bullish_change,
            "avg_bearish_change": avg_bearish_change,
            "confidence_metrics": confidence_metrics,
            "prediction_horizon": prediction_horizon,
            "confidence_threshold": confidence_threshold
        }
    
    def evaluate_source_performance(
        self, 
        sentiment_data: pd.DataFrame, 
        price_data: pd.DataFrame,
        prediction_horizon: int = 24,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Evaluate performance of different sentiment sources.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            price_data: DataFrame with price data
            prediction_horizon: Hours to look ahead for price movement
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary of evaluation metrics by source
        """
        # Ensure DataFrames have datetime index
        sentiment_data = sentiment_data.copy()
        price_data = price_data.copy()
        
        if "timestamp" in sentiment_data.columns:
            sentiment_data["timestamp"] = pd.to_datetime(sentiment_data["timestamp"])
            sentiment_data = sentiment_data.set_index("timestamp")
            
        if "timestamp" in price_data.columns:
            price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])
            price_data = price_data.set_index("timestamp")
        
        # Get unique sources
        sources = sentiment_data["source"].unique()
        
        # Evaluate each source
        source_metrics = {}
        
        for source in sources:
            source_data = sentiment_data[sentiment_data["source"] == source]
            
            # Evaluate this source
            metrics = self.evaluate_sentiment_predictions(
                source_data,
                price_data,
                prediction_horizon,
                confidence_threshold
            )
            
            source_metrics[source] = metrics
        
        return source_metrics
```

## Configuration Updates

### sentiment_analysis.yaml

```yaml
# Configuration for sentiment analysis components

# Global settings
enabled: true
update_interval: 300  # 5 minutes

# NLP settings
nlp:
  sentiment_model: "distilbert-base-uncased-finetuned-sst-2-english"
  batch_size: 16

# API keys
apis:
  twitter:
    api_key: "${TWITTER_API_KEY}"
    api_secret: "${TWITTER_API_SECRET}"
    access_token: "${TWITTER_ACCESS_TOKEN}"
    access_secret: "${TWITTER_ACCESS_SECRET}"
  
  reddit:
    client_id: "${REDDIT_CLIENT_ID}"
    client_secret: "${REDDIT_CLIENT_SECRET}"
    user_agent: "AI-Trading-Agent/1.0"
  
  news_api:
    api_key: "${NEWS_API_KEY}"
  
  crypto_news:
    api_key: "${CRYPTO_NEWS_API_KEY}"
  
  exchange_data:
    api_key: "${EXCHANGE_DATA_API_KEY}"
  
  blockchain:
    api_key: "${BLOCKCHAIN_API_KEY}"

# Agent settings
agents:
  social_media:
    enabled: true
    update_interval: 300  # 5 minutes
    min_confidence: 0.7
    sentiment_shift_threshold: 0.15
    contrarian_threshold: 0.8
    platforms:
      - "Twitter"
      - "Reddit"
  
  news:
    enabled: true
    update_interval: 600  # 10 minutes
    min_confidence: 0.7
    sentiment_shift_threshold: 0.15
    sources:
      - "CryptoNews"
      - "CoinDesk"
      - "CoinTelegraph"
  
  market_sentiment:
    enabled: true
    update_interval: 3600  # 1 hour
    min_confidence: 0.7
    sentiment_shift_threshold: 0.15
    indicators:
      - "FearGreedIndex"
      - "LongShortRatio"
  
  onchain:
    enabled: true
    update_interval: 3600  # 1 hour
    min_confidence: 0.7
    sentiment_shift_threshold: 0.15
    metrics:
      - "LargeTransactions"
      - "ActiveAddresses"
      - "ExchangeReserves"
  
  aggregator:
    enabled: true
    update_interval: 1800  # 30 minutes
    min_confidence: 0.7
    sentiment_shift_threshold: 0.15
    source_weights:
      social_media: 0.25
      news: 0.25
      market_sentiment: 0.3
      onchain: 0.2

# Symbols to monitor
symbols:
  - "BTC/USDT"
  - "ETH/USDT"
  - "SOL/USDT"
  - "XRP/USDT"
```

### strategies.yaml

```yaml
# Configuration for trading strategies

# Enhanced sentiment strategy
enhanced_sentiment:
  enabled: true
  sentiment_threshold_bullish: 0.7
  sentiment_threshold_bearish: 0.3
  min_confidence: 0.7
  min_signal_score: 0.7
  use_market_regime: true
  use_technical_confirmation: true
  
  # Source weighting
  source_weights:
    social_media: 1.0
    news: 1.0
    market_sentiment: 1.0
    onchain: 1.0
    aggregator: 2.0  # Aggregator gets double weight
  
  # Technical indicators
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  
  # Market regime
  regime_lookback: 20
```

## Implementation Roadmap

### Phase 1: Real Data Source Integration (2 weeks)

1. Week 1:
   - Implement Twitter and Reddit API clients
   - Implement News API clients
   - Create API key management system

2. Week 2:
   - Implement Market Sentiment data sources
   - Implement Onchain data sources
   - Test data source integrations

### Phase 2: NLP Model Implementation (1 week)

1. Week 1:
   - Implement NLP Service
   - Integrate with sentiment agents
   - Test NLP model performance

### Phase 3: Trading Strategy Integration (1 week)

1. Week 1:
   - Implement Enhanced Sentiment Strategy
   - Create Strategy Factory
   - Test strategy with live data

### Phase 4: Backtesting Framework (1 week)

1. Week 1:
   - Implement Sentiment Backtester
   - Create historical sentiment data loader
   - Test backtesting framework

### Phase 5: Performance Metrics (1 week)

1. Week 1:
   - Implement Sentiment Performance Evaluator
   - Create performance dashboards
   - Test and validate metrics

## Conclusion

This implementation plan provides a comprehensive roadmap for completing the sentiment analysis integration in the AI Crypto Trading Agent. The plan builds upon the existing architecture and components, replacing simulated data with real API integrations and adding sophisticated NLP models for sentiment analysis.

The implementation is divided into five phases, each focusing on a specific aspect of the system:

1. Real Data Source Integration: Connecting to external APIs for sentiment data
2. NLP Model Implementation: Adding sophisticated text analysis capabilities
3. Trading Strategy Integration: Creating strategies that leverage sentiment signals
4. Backtesting Framework: Testing sentiment strategies with historical data
5. Performance Metrics: Measuring the effectiveness of sentiment signals

By following this plan, the sentiment analysis system will be fully integrated with the trading agent, providing valuable signals for cryptocurrency trading decisions.
