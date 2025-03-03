"""Social media sentiment analysis.

This module provides functionality for analyzing sentiment from social media
sources like Twitter, Reddit, and other social platforms.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import CandleData, TimeFrame


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
        
        # Sentiment analysis dictionaries (would be loaded from NLP model in production)
        self._load_sentiment_lexicons()
    
    def _load_sentiment_lexicons(self) -> None:
        """Load sentiment lexicons for text analysis."""
        # In a production system, this would load actual NLP models or lexicons
        # For the demo, we'll use a simplified dictionary approach
        
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
    
    async def _initialize(self) -> None:
        """Initialize the social media sentiment agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing social media sentiment agent",
                       platforms=self.platforms)
    
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
            # Simulate fetching social media posts
            post_count = random.randint(50, 200)
            
            # Simulate sentiment analysis results
            base_currency = symbol.split('/')[0]
            
            # Bias slightly based on currency (just for demo variation)
            sentiment_bias = {
                "BTC": 0.05,  # Slightly more bullish
                "ETH": 0.02,  # Slightly more bullish
                "XRP": -0.03,  # Slightly more bearish
                "SOL": 0.04,  # Slightly more bullish
            }.get(base_currency, 0)
            
            # Add some randomness with mean at neutral (0.5) plus the bias
            sentiment_value = max(0.1, min(0.9, random.normalvariate(0.5 + sentiment_bias, 0.15)))
            
            # Calculate confidence based on post volume and "agreement"
            volume_factor = min(1.0, post_count / 100)
            agreement_factor = random.uniform(0.7, 0.95)  # High agreement is more confident
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
                "platforms": self.platforms
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
