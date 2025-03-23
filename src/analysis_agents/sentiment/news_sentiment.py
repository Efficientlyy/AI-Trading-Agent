"""News sentiment analysis.

This module provides functionality for analyzing sentiment from news
sources, including crypto news sites, financial news, and press releases.
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


class NewsApiClient:
    """Client for the News API."""
    
    def __init__(self, api_key: str):
        """Initialize the News API client.
        
        Args:
            api_key: API key for the News API
        """
        self.api_key = api_key
        self.logger = get_logger("clients", "news_api")
        
        # In production, we would initialize the News API client here
        # For now, we'll use a mock implementation
    
    async def get_everything(
        self, 
        q: str, 
        language: str = "en", 
        sort_by: str = "publishedAt", 
        page_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for news articles.
        
        Args:
            q: Search query
            language: Article language
            sort_by: Sort order (relevancy, popularity, publishedAt)
            page_size: Number of articles per page
            
        Returns:
            List of article dictionaries
        """
        # In production, this would call the News API
        # For now, return mock data
        self.logger.debug("Searching News API", query=q, page_size=page_size)
        
        # Simulate a delay
        await asyncio.sleep(0.1)
        
        # For testing, return between 5 and page_size mock articles
        actual_count = min(page_size, random.randint(5, page_size))
        
        # Create mock articles
        mock_articles = []
        
        for i in range(actual_count):
            sentiment_type = random.choice(["bullish", "bearish", "neutral"])
            
            if sentiment_type == "bullish":
                title = f"Analysts predict {q} price surge in coming months"
                description = "Market analysts are predicting a significant price increase based on recent developments."
            elif sentiment_type == "bearish":
                title = f"Bearish outlook for {q} as market faces headwinds"
                description = "Investors are cautious as market indicators suggest potential downside in the near term."
            else:
                title = f"{q} market stabilizes as traders await next move"
                description = "Trading volume has decreased as market participants await clearer direction."
                
            mock_articles.append({
                "title": title,
                "description": description,
                "url": f"https://example.com/news/{i}",
                "publishedAt": datetime.utcnow().isoformat(),
                "source": {"name": "Mock News Source"}
            })
        
        return mock_articles


class CryptoNewsClient:
    """Client for crypto-specific news APIs."""
    
    def __init__(self, api_key: str):
        """Initialize the Crypto News client.
        
        Args:
            api_key: API key for the Crypto News API
        """
        self.api_key = api_key
        self.logger = get_logger("clients", "crypto_news")
        
        # In production, we would initialize the Crypto News API client here
        # For now, we'll use a mock implementation
    
    async def get_news(
        self, 
        categories: List[str] = ["blockchain", "cryptocurrency"], 
        keywords: List[str] = [], 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get cryptocurrency news articles.
        
        Args:
            categories: News categories to include
            keywords: Keywords to search for
            limit: Maximum number of articles to return
            
        Returns:
            List of article dictionaries
        """
        # In production, this would call the Crypto News API
        # For now, return mock data
        self.logger.debug("Getting crypto news", 
                       categories=categories, 
                       keywords=keywords, 
                       limit=limit)
        
        # Simulate a delay
        await asyncio.sleep(0.1)
        
        # For testing, return between 5 and limit mock articles
        actual_count = min(limit, random.randint(5, limit))
        
        # Create mock articles
        mock_articles = []
        
        for i in range(actual_count):
            sentiment_type = random.choice(["bullish", "bearish", "neutral"])
            
            if sentiment_type == "bullish":
                title = f"Bullish signals emerge for {', '.join(keywords)}" if keywords else "Crypto market shows signs of recovery"
                description = "The cryptocurrency market is showing signs of a potential bullish trend reversal."
            elif sentiment_type == "bearish":
                title = f"Bearish pressure continues for {', '.join(keywords)}" if keywords else "Crypto market faces selling pressure"
                description = "The cryptocurrency market continues to face bearish pressure amid regulatory concerns."
            else:
                title = f"{', '.join(keywords)} market consolidates" if keywords else "Crypto market in consolidation phase"
                description = "The cryptocurrency market is in a consolidation phase as traders await clearer signals."
                
            mock_articles.append({
                "title": title,
                "description": description,
                "url": f"https://example.com/crypto-news/{i}",
                "publishedAt": datetime.utcnow().isoformat(),
                "source": {"name": "Mock Crypto News Source"}
            })
        
        return mock_articles


class NewsSentimentAgent(BaseSentimentAgent):
    """Analysis agent for news sentiment.
    
    This agent processes sentiment data from news sources
    and publishes sentiment events with confidence scores.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the news sentiment agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "news_sentiment")
        
        # News sources to monitor
        self.sources = config.get(
            f"analysis_agents.{agent_id}.sources", 
            ["CryptoNews", "CoinDesk", "CoinTelegraph"]
        )
        
        # Update interval in seconds
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            600  # Default: 10 minutes
        )
        
        # API clients (will be initialized during _initialize)
        self.news_api_client = None
        self.crypto_news_client = None
        
        # NLP service (will be set by manager)
        self.nlp_service = None
    
    async def _initialize(self) -> None:
        """Initialize the news sentiment agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing news sentiment agent",
                       sources=self.sources)
                       
        # Initialize API clients
        try:
            # News API client
            self.news_api_client = NewsApiClient(
                api_key=config.get("apis.news_api.api_key", "")
            )
            
            # Crypto news API client
            self.crypto_news_client = CryptoNewsClient(
                api_key=config.get("apis.crypto_news.api_key", "")
            )
            
            self.logger.info("Initialized news API clients")
            
        except Exception as e:
            self.logger.error("Failed to initialize news API clients", error=str(e))
    
    def set_nlp_service(self, nlp_service: NLPService) -> None:
        """Set the NLP service for sentiment analysis.
        
        Args:
            nlp_service: The NLP service to use
        """
        self.nlp_service = nlp_service
        self.logger.info("NLP service set for news sentiment agent")
    
    async def _start(self) -> None:
        """Start the news sentiment agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for news sentiment
        self.update_task = self.create_task(
            self._update_sentiment_periodically()
        )
    
    async def _stop(self) -> None:
        """Stop the news sentiment agent."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        await super()._stop()
    
    async def _update_sentiment_periodically(self) -> None:
        """Update news sentiment periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    await self._analyze_news_sentiment(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("News sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in news sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _analyze_news_sentiment(self, symbol: str) -> None:
        """Analyze news sentiment for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "news", self.update_interval):
            return
        
        try:
            base_currency = symbol.split('/')[0]
            
            # Fetch news articles
            news_api_articles = []
            if self.news_api_client:
                news_api_articles = await self.news_api_client.get_everything(
                    q=f"{base_currency} OR cryptocurrency",
                    language="en",
                    sort_by="publishedAt",
                    page_size=20
                )
            
            # Fetch crypto news
            crypto_news_articles = []
            if self.crypto_news_client:
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
            
            # Process articles with NLP model if available
            sentiment_scores = []
            
            if self.nlp_service:
                sentiment_scores = await self.nlp_service.analyze_sentiment(article_texts)
            else:
                # Fallback to random sentiment if NLP service is not available
                sentiment_scores = [random.uniform(0.2, 0.8) for _ in range(article_count)]
            
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
            keywords = []
            if self.nlp_service:
                # Combine all article texts
                all_text = " ".join(article_texts)
                keywords = await self.nlp_service.extract_keywords(all_text)
            else:
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
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to news sentiment.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        # News sentiment typically doesn't do additional analysis 
        # against market data beyond what the base class provides
        pass
        
    def _extract_keywords_from_news(self, text: str) -> List[str]:
        """Extract relevant keywords from news text.
        
        Args:
            text: The news article text
            
        Returns:
            List of extracted keywords
        """
        # This is a placeholder for an NLP-based keyword extraction
        # In a real implementation, this would use a proper NLP library
        
        # Simple approach: just split text and find important words
        words = text.lower().split()
        keywords = []
        
        important_terms = [
            "launch", "partnership", "listing", "upgrade", "fork",
            "regulation", "ban", "hack", "security", "vulnerability",
            "adoption", "integration", "institutional", "whale",
            "bullish", "bearish", "rally", "crash", "correction"
        ]
        
        for word in words:
            if word in important_terms:
                keywords.append(word)
                
        return keywords