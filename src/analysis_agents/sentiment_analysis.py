"""Sentiment analysis agent.

This module provides an analysis agent that processes sentiment data from
various sources, including social media, news, and market indicators to generate
sentiment-based trading signals.
"""

import asyncio
from datetime import datetime, timedelta
import re
import random
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from src.analysis_agents.base_agent import AnalysisAgent
from src.common.config import config
from src.common.logging import get_logger
from src.models.events import SentimentEvent
from src.models.market_data import CandleData, TimeFrame


class SentimentAnalysisAgent(AnalysisAgent):
    """Analysis agent for market sentiment analysis.
    
    This agent processes sentiment data from various sources and publishes
    sentiment events with confidence scores and directional bias.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the sentiment analysis agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "sentiment_analysis")
        
        # Sentiment sources to monitor
        self.sources = config.get(
            f"analysis_agents.{agent_id}.sources", 
            {
                "social_media": {"enabled": True},
                "news": {"enabled": True},
                "market_sentiment": {"enabled": True},
                "onchain": {"enabled": True}
            }
        )
        
        # Source configurations
        self.social_media_platforms = self.sources.get("social_media", {}).get("platforms", ["Twitter", "Reddit"])
        self.news_sources = self.sources.get("news", {}).get("sources", ["CryptoNews", "CoinDesk", "CoinTelegraph"])
        self.market_indicators = self.sources.get("market_sentiment", {}).get("indicators", ["FearGreedIndex", "LongShortRatio"])
        self.onchain_metrics = self.sources.get("onchain", {}).get("metrics", ["LargeTransactions", "ActiveAddresses"])
        
        # Confidence and threshold settings
        self.min_confidence = config.get(f"analysis_agents.{agent_id}.min_confidence", 0.7)
        self.sentiment_shift_threshold = config.get(f"analysis_agents.{agent_id}.sentiment_shift_threshold", 0.15)
        self.contrarian_threshold = config.get(f"analysis_agents.{agent_id}.contrarian_threshold", 0.8)
        
        # Source update intervals (in seconds)
        self.social_media_interval = self.sources.get("social_media", {}).get("interval_seconds", 300)  # 5 minutes
        self.news_interval = self.sources.get("news", {}).get("interval_seconds", 600)  # 10 minutes
        self.market_sentiment_interval = self.sources.get("market_sentiment", {}).get("interval_seconds", 3600)  # 1 hour
        self.onchain_interval = self.sources.get("onchain", {}).get("interval_seconds", 3600)  # 1 hour
        
        # Sentiment data cache
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self.max_history_size = config.get(f"analysis_agents.{agent_id}.max_history_size", 100)
        
        # Last update times
        self.last_update: Dict[str, Dict[str, datetime]] = {
            "social_media": {},
            "news": {},
            "market_sentiment": {},
            "onchain": {}
        }
        
        # Source weights for aggregation
        self.source_weights = {
            "social_media": 0.25,
            "news": 0.25,
            "market_sentiment": 0.3,
            "onchain": 0.2
        }
        
        # Word sentiment dictionaries (would be loaded from NLP model in production)
        self._load_sentiment_lexicons()
        
        # Tasks for different update cycles
        self.update_tasks = []
    
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
        """Initialize the sentiment analysis agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing sentiment analysis agent",
                       social_media=self.social_media_platforms,
                       news=self.news_sources,
                       market_indicators=self.market_indicators,
                       onchain_metrics=self.onchain_metrics)
    
    async def _start(self) -> None:
        """Start the sentiment analysis agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update tasks for different data sources
        if self.sources.get("social_media", {}).get("enabled", False):
            task = self.create_task(self._update_social_media_sentiment_periodically())
            self.update_tasks.append(task)
            
        if self.sources.get("news", {}).get("enabled", False):
            task = self.create_task(self._update_news_sentiment_periodically())
            self.update_tasks.append(task)
            
        if self.sources.get("market_sentiment", {}).get("enabled", False):
            task = self.create_task(self._update_market_sentiment_periodically())
            self.update_tasks.append(task)
            
        if self.sources.get("onchain", {}).get("enabled", False):
            task = self.create_task(self._update_onchain_metrics_periodically())
            self.update_tasks.append(task)
    
    async def _stop(self) -> None:
        """Stop the sentiment analysis agent."""
        # Cancel all update tasks
        for task in self.update_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.update_tasks = []
        
        await super()._stop()
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        The sentiment analysis agent doesn't directly process candles,
        but we implement this method to satisfy the abstract base class.
        
        Args:
            candle: The candle data to process
        """
        # Sentiment analysis doesn't directly process individual candles
        # but we could use them for correlation with sentiment shifts
        pass
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to sentiment indicators.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        if not candles or len(candles) < 10:
            return
            
        # Check if we have any sentiment data for this symbol
        if symbol not in self.sentiment_cache:
            return
            
        # Get the latest sentiment data
        latest_sentiment = self.sentiment_cache[symbol]
        
        # Get price data from candles
        closes = [candle.close for candle in candles]
        
        # Analyze candle patterns in relation to sentiment
        if len(closes) >= 20:
            # Calculate short-term trend
            short_term_trend = "bullish" if closes[-1] > closes[-10] else "bearish"
            
            # Check if sentiment aligns with price action
            sentiment_direction = latest_sentiment.get("direction", "neutral")
            
            # If there's a discord between sentiment and price action,
            # that could be a signal of potential reversal
            if short_term_trend != sentiment_direction and latest_sentiment.get("confidence", 0) > 0.7:
                is_extreme = latest_sentiment.get("value", 0.5) > 0.8 or latest_sentiment.get("value", 0.5) < 0.2
                
                if is_extreme:
                    # This could be a contrarian indicator
                    contrarian_confidence = latest_sentiment.get("confidence", 0) * 1.1  # Boost confidence slightly
                    
                    # Generate a contrarian sentiment event
                    await self._publish_sentiment_event(
                        symbol=symbol,
                        direction="bullish" if sentiment_direction == "bearish" else "bearish",
                        value=1 - latest_sentiment.get("value", 0.5),  # Inverse sentiment value
                        confidence=min(0.95, contrarian_confidence),
                        is_extreme=True,
                        signal_type="contrarian",
                        timeframe=timeframe,
                        sources=latest_sentiment.get("sources", []),
                        details={"price_trend": short_term_trend}
                    )
    
    async def _update_social_media_sentiment_periodically(self) -> None:
        """Update social media sentiment periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    # In a real system, we would fetch actual social media data
                    # For the demo, we'll simulate it
                    await self._analyze_social_media_sentiment(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.social_media_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Social media sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in social media sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _update_news_sentiment_periodically(self) -> None:
        """Update news sentiment periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    # In a real system, we would fetch actual news data
                    # For the demo, we'll simulate it
                    await self._analyze_news_sentiment(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.news_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("News sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in news sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _update_market_sentiment_periodically(self) -> None:
        """Update market sentiment indicators periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    # In a real system, we would fetch actual market indicators
                    # For the demo, we'll simulate it
                    await self._analyze_market_sentiment_indicators(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.market_sentiment_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Market sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in market sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _update_onchain_metrics_periodically(self) -> None:
        """Update on-chain metrics periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    # Only process symbols that have on-chain data available
                    # (typically just BTC, ETH, and some major coins)
                    base_currency = symbol.split('/')[0]
                    if base_currency in ["BTC", "ETH"]:
                        # In a real system, we would fetch actual on-chain data
                        # For the demo, we'll simulate it
                        await self._analyze_onchain_metrics(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.onchain_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("On-chain metrics update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in on-chain metrics update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _analyze_social_media_sentiment(self, symbol: str) -> None:
        """Analyze social media sentiment for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # In a real implementation, we would fetch actual social media data
        # For the demo, we'll generate simulated sentiment data
        
        # Check if we need to update (respect the interval)
        now = datetime.utcnow()
        if (symbol in self.last_update["social_media"] and 
            (now - self.last_update["social_media"][symbol]).total_seconds() < self.social_media_interval):
            return
            
        self.last_update["social_media"][symbol] = now
        
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
            
            # Check if sentiment has shifted significantly from previous reading
            previous_sentiment = self.sentiment_cache.get(symbol, {}).get("social_media", {}).get("value", 0.5)
            sentiment_shift = abs(sentiment_value - previous_sentiment)
            
            # Update the cache
            if symbol not in self.sentiment_cache:
                self.sentiment_cache[symbol] = {}
            
            if "social_media" not in self.sentiment_cache[symbol]:
                self.sentiment_cache[symbol]["social_media"] = {
                    "history": [],
                    "value": sentiment_value,
                    "direction": direction,
                    "confidence": confidence,
                    "last_update": now
                }
            else:
                # Add to history
                history = self.sentiment_cache[symbol]["social_media"]["history"]
                history.append((now, sentiment_value, direction, confidence))
                
                # Limit history size
                if len(history) > self.max_history_size:
                    history = history[-self.max_history_size:]
                
                # Update current values
                self.sentiment_cache[symbol]["social_media"]["history"] = history
                self.sentiment_cache[symbol]["social_media"]["value"] = sentiment_value
                self.sentiment_cache[symbol]["social_media"]["direction"] = direction
                self.sentiment_cache[symbol]["social_media"]["confidence"] = confidence
                self.sentiment_cache[symbol]["social_media"]["last_update"] = now
            
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
                
                # Update aggregate sentiment
                await self._update_aggregate_sentiment(symbol)
                
                # Publish event if confidence is high enough
                if confidence >= self.min_confidence:
                    await self._publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        is_extreme=is_extreme,
                        signal_type=signal_type,
                        sources=["Twitter", "Reddit"],
                        details={
                            "post_count": post_count,
                            "event_type": event_type
                        }
                    )
                    
                    self.logger.info("Published social media sentiment event", 
                                    symbol=symbol,
                                    direction=direction,
                                    sentiment_value=sentiment_value,
                                    confidence=confidence,
                                    is_extreme=is_extreme)
        
        except Exception as e:
            self.logger.error("Error analyzing social media sentiment", 
                           symbol=symbol,
                           error=str(e))
    
    async def _analyze_news_sentiment(self, symbol: str) -> None:
        """Analyze news sentiment for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # In a real implementation, we would fetch actual news data
        # For the demo, we'll generate simulated sentiment data
        
        # Check if we need to update (respect the interval)
        now = datetime.utcnow()
        if (symbol in self.last_update["news"] and 
            (now - self.last_update["news"][symbol]).total_seconds() < self.news_interval):
            return
            
        self.last_update["news"][symbol] = now
        
        try:
            # Simulate fetching news articles
            article_count = random.randint(5, 30)
            
            # Simulate sentiment analysis results
            base_currency = symbol.split('/')[0]
            
            # News tends to be less volatile than social media
            sentiment_value = max(0.2, min(0.8, random.normalvariate(0.5, 0.12)))
            
            # Calculate confidence based on article volume and source quality
            volume_factor = min(1.0, article_count / 15)
            quality_factor = random.uniform(0.8, 0.98)  # News sources generally have higher quality
            confidence = volume_factor * quality_factor
            
            # Determine direction
            if sentiment_value > 0.55:
                direction = "bullish"
            elif sentiment_value < 0.45:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Check if sentiment has shifted significantly from previous reading
            previous_sentiment = self.sentiment_cache.get(symbol, {}).get("news", {}).get("value", 0.5)
            sentiment_shift = abs(sentiment_value - previous_sentiment)
            
            # Update the cache
            if symbol not in self.sentiment_cache:
                self.sentiment_cache[symbol] = {}
            
            if "news" not in self.sentiment_cache[symbol]:
                self.sentiment_cache[symbol]["news"] = {
                    "history": [],
                    "value": sentiment_value,
                    "direction": direction,
                    "confidence": confidence,
                    "last_update": now
                }
            else:
                # Add to history
                history = self.sentiment_cache[symbol]["news"]["history"]
                history.append((now, sentiment_value, direction, confidence))
                
                # Limit history size
                if len(history) > self.max_history_size:
                    history = history[-self.max_history_size:]
                
                # Update current values
                self.sentiment_cache[symbol]["news"]["history"] = history
                self.sentiment_cache[symbol]["news"]["value"] = sentiment_value
                self.sentiment_cache[symbol]["news"]["direction"] = direction
                self.sentiment_cache[symbol]["news"]["confidence"] = confidence
                self.sentiment_cache[symbol]["news"]["last_update"] = now
            
            # Publish event if significant shift
            if sentiment_shift > self.sentiment_shift_threshold:
                # Update aggregate sentiment
                await self._update_aggregate_sentiment(symbol)
                
                # Publish event if confidence is high enough
                if confidence >= self.min_confidence:
                    await self._publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        sources=self.news_sources,
                        details={
                            "article_count": article_count,
                            "event_type": "news_sentiment_shift"
                        }
                    )
                    
                    self.logger.info("Published news sentiment event", 
                                    symbol=symbol,
                                    direction=direction,
                                    sentiment_value=sentiment_value,
                                    confidence=confidence)
        
        except Exception as e:
            self.logger.error("Error analyzing news sentiment", 
                           symbol=symbol,
                           error=str(e))
    
    async def _analyze_market_sentiment_indicators(self, symbol: str) -> None:
        """Analyze market sentiment indicators for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # In a real implementation, we would fetch actual market indicators
        # For the demo, we'll generate simulated data
        
        # Check if we need to update (respect the interval)
        now = datetime.utcnow()
        if (symbol in self.last_update["market_sentiment"] and 
            (now - self.last_update["market_sentiment"][symbol]).total_seconds() < self.market_sentiment_interval):
            return
            
        self.last_update["market_sentiment"][symbol] = now
        
        try:
            # Simulate Fear & Greed Index (0-100)
            # This is typically for the whole market, but we can bias slightly per symbol
            base_currency = symbol.split('/')[0]
            fear_greed_bias = {
                "BTC": 0,       # No bias for Bitcoin
                "ETH": 2,       # Slightly higher for Ethereum
                "XRP": -3,      # Slightly lower for XRP
                "SOL": 5,       # Higher for Solana
            }.get(base_currency, 0)
            
            # Generate Fear & Greed Index (0-100)
            fear_greed = random.randint(20, 80) + fear_greed_bias
            fear_greed = max(0, min(100, fear_greed))
            
            # Simulate long/short ratio data
            long_short_ratio = max(0.3, min(3.0, random.normalvariate(1.0, 0.4)))
            
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
            
            # Check if sentiment has shifted significantly from previous reading
            previous_sentiment = self.sentiment_cache.get(symbol, {}).get("market_sentiment", {}).get("value", 0.5)
            sentiment_shift = abs(sentiment_value - previous_sentiment)
            
            # Update the cache
            if symbol not in self.sentiment_cache:
                self.sentiment_cache[symbol] = {}
            
            if "market_sentiment" not in self.sentiment_cache[symbol]:
                self.sentiment_cache[symbol]["market_sentiment"] = {
                    "history": [],
                    "value": sentiment_value,
                    "direction": direction,
                    "confidence": confidence,
                    "fear_greed": fear_greed,
                    "long_short_ratio": long_short_ratio,
                    "last_update": now
                }
            else:
                # Add to history
                history = self.sentiment_cache[symbol]["market_sentiment"]["history"]
                history.append((now, sentiment_value, direction, confidence, fear_greed, long_short_ratio))
                
                # Limit history size
                if len(history) > self.max_history_size:
                    history = history[-self.max_history_size:]
                
                # Update current values
                self.sentiment_cache[symbol]["market_sentiment"]["history"] = history
                self.sentiment_cache[symbol]["market_sentiment"]["value"] = sentiment_value
                self.sentiment_cache[symbol]["market_sentiment"]["direction"] = direction
                self.sentiment_cache[symbol]["market_sentiment"]["confidence"] = confidence
                self.sentiment_cache[symbol]["market_sentiment"]["fear_greed"] = fear_greed
                self.sentiment_cache[symbol]["market_sentiment"]["long_short_ratio"] = long_short_ratio
                self.sentiment_cache[symbol]["market_sentiment"]["last_update"] = now
            
            # Check for extreme values
            is_extreme = fear_greed <= 20 or fear_greed >= 80
            
            # Publish event if significant shift or extreme values
            if sentiment_shift > self.sentiment_shift_threshold or is_extreme:
                # Update aggregate sentiment
                await self._update_aggregate_sentiment(symbol)
                
                # Determine if extreme sentiment should be treated as contrarian
                signal_type = "sentiment"
                if is_extreme:
                    # Extreme fear/greed can be contrarian
                    signal_type = "contrarian"
                
                # Publish event if confidence is high enough
                if confidence >= self.min_confidence:
                    await self._publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        is_extreme=is_extreme,
                        signal_type=signal_type,
                        sources=["FearGreedIndex", "LongShortRatio"],
                        details={
                            "fear_greed_index": fear_greed,
                            "long_short_ratio": long_short_ratio,
                            "event_type": "market_sentiment_shift" if sentiment_shift > self.sentiment_shift_threshold else "extreme_market_sentiment"
                        }
                    )
                    
                    self.logger.info("Published market sentiment event", 
                                    symbol=symbol,
                                    direction=direction,
                                    sentiment_value=sentiment_value,
                                    confidence=confidence,
                                    fear_greed=fear_greed,
                                    long_short_ratio=long_short_ratio,
                                    is_extreme=is_extreme)
        
        except Exception as e:
            self.logger.error("Error analyzing market sentiment indicators", 
                           symbol=symbol,
                           error=str(e))
    
    async def _analyze_onchain_metrics(self, symbol: str) -> None:
        """Analyze on-chain metrics for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Only some cryptocurrencies have meaningful on-chain metrics
        base_currency = symbol.split('/')[0]
        if base_currency not in ["BTC", "ETH"]:
            return
        
        # Check if we need to update (respect the interval)
        now = datetime.utcnow()
        if (symbol in self.last_update["onchain"] and 
            (now - self.last_update["onchain"][symbol]).total_seconds() < self.onchain_interval):
            return
            
        self.last_update["onchain"][symbol] = now
        
        try:
            # Simulate on-chain metrics
            
            # Large transactions count (normalized to 0-1)
            large_tx_value = random.uniform(0.3, 0.7)
            
            # Active addresses growth rate (-1 to 1, centered at 0)
            active_addr_growth = random.uniform(-0.5, 0.5)
            
            # Network hash rate trend (-1 to 1, centered at 0)
            hash_rate_trend = random.uniform(-0.3, 0.3)
            
            # Exchange reserves growth rate (-1 to 1, centered at 0)
            # Negative means tokens leaving exchanges (bullish)
            exchange_reserves_growth = random.uniform(-0.4, 0.4)
            
            # Combine metrics into a sentiment score
            # Each metric is weighted differently
            sentiment_metrics = {
                "large_transactions": large_tx_value,
                "active_addresses": 0.5 + (active_addr_growth / 2),  # Convert to 0-1 scale
                "hash_rate": 0.5 + (hash_rate_trend / 2),  # Convert to 0-1 scale
                "exchange_reserves": 0.5 - (exchange_reserves_growth / 2)  # Negative is bullish
            }
            
            # Calculate weighted sentiment
            metric_weights = {
                "large_transactions": 0.3,
                "active_addresses": 0.3,
                "hash_rate":
