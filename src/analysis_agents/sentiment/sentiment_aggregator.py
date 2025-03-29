"""Sentiment aggregation module.

This module provides functionality for aggregating sentiment data
from various sources into a unified sentiment signal.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.analysis_agents.sentiment.nlp_service import NLPService
from src.analysis_agents.sentiment.adaptive_weights import AdaptiveSentimentWeights
from src.analysis_agents.sentiment.sentiment_validator import SentimentValidator
from src.common.config import config
from src.common.logging import get_logger
from src.common.caching import Cache
from src.common.monitoring import metrics
from src.models.market_data import CandleData, TimeFrame


class SentimentAggregator(BaseSentimentAgent):
    """Aggregates sentiment data from multiple sources.
    
    This agent combines sentiment signals from social media, news,
    market indicators, and on-chain metrics to provide a unified
    sentiment view.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the sentiment aggregator.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "sentiment_aggregator")
        
        # Source weights for aggregation
        self.source_weights = config.get(
            f"analysis_agents.{agent_id}.source_weights",
            {
                "social_media": 0.25,
                "news": 0.25,
                "market_sentiment": 0.3,
                "onchain": 0.2
            }
        )
        
        # Update interval in seconds
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            1800  # Default: 30 minutes
        )
        
        # The aggregator has its own sentiment cache
        self.aggregated_cache: Dict[str, Dict[str, Any]] = {}
        
        # Add cache for aggregated sentiment results
        self.result_cache = Cache(ttl=self.update_interval)
        
        # Create adaptive weights system if enabled
        self.use_adaptive_weights = config.get(
            f"analysis_agents.{agent_id}.use_adaptive_weights",
            False
        )
        if self.use_adaptive_weights:
            self.adaptive_weights = AdaptiveSentimentWeights()
        
        # Create sentiment validator
        self.validator = SentimentValidator()
        
        # NLP service (will be set by manager)
        self.nlp_service: Optional[NLPService] = None
    
    async def _initialize(self) -> None:
        """Initialize the sentiment aggregator."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing sentiment aggregator",
                       source_weights=self.source_weights)
        
        # Record initialization metric
        metrics.counter("sentiment_aggregator_initialized", tags={
            "agent_id": self.agent_id
        })
    
    async def _start(self) -> None:
        """Start the sentiment aggregator."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for aggregating sentiment
        self.update_task = self.create_task(
            self._aggregate_sentiment_periodically()
        )
        
        # Record start metric
        metrics.counter("sentiment_aggregator_started", tags={
            "agent_id": self.agent_id
        })
    
    async def _stop(self) -> None:
        """Stop the sentiment aggregator."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Record stop metric
        metrics.counter("sentiment_aggregator_stopped", tags={
            "agent_id": self.agent_id
        })
        
        await super()._stop()
    
    async def _aggregate_sentiment_periodically(self) -> None:
        """Aggregate sentiment from all sources periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    await self._aggregate_sentiment(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Sentiment aggregation task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in sentiment aggregation task", error=str(e))
            
            # Record error metric
            metrics.counter("sentiment_aggregator_error", tags={
                "agent_id": self.agent_id,
                "error_type": type(e).__name__
            })
            
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _aggregate_sentiment(self, symbol: str) -> None:
        """Internal method to aggregate sentiment from all sources for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Start timer for performance monitoring
        timer_end = metrics.timer("sentiment_aggregation_duration", tags={
            "agent_id": self.agent_id,
            "symbol": symbol
        })
        
        try:
            # Check if we have sentiment data for this symbol
            if symbol not in self.sentiment_cache:
                return
                
            # Get sentiment from all available sources
            source_data = {}
            sources_list = []
            
            for source_type in ["social_media", "news", "market_sentiment", "onchain"]:
                if source_type in self.sentiment_cache[symbol]:
                    source_data[source_type] = self.sentiment_cache[symbol][source_type]
                    sources_list.append(source_type)
            
            # Need at least one source to aggregate (changed from 2 for better resilience)
            if len(source_data) < 1:
                return
                
            # Get source weights (use adaptive weights if enabled)
            if self.use_adaptive_weights:
                source_weights = self.adaptive_weights.get_weights()
            else:
                source_weights = self.source_weights
                
            # Calculate weighted sentiment value using numpy for better performance
            if source_data:
                # Extract data into numpy arrays for vectorized operations
                values = np.array([data.get("value", 0.5) for data in source_data.values()])
                confidences = np.array([data.get("confidence", 0.7) for data in source_data.values()])
                
                # Get timestamps and calculate recency factors
                now = datetime.utcnow()
                timestamps = [data.get("last_update", now) for data in source_data.values()]
                age_hours = np.array([(now - ts).total_seconds() / 3600 for ts in timestamps])
                recency_factors = np.maximum(0.5, 1.0 - (age_hours / 24.0))
                
                # Get source weights
                source_types = list(source_data.keys())
                weights = np.array([source_weights.get(st, 0.25) for st in source_types])
                
                # Calculate adjusted weights
                adjusted_weights = weights * confidences * recency_factors
                total_weight = np.sum(adjusted_weights)
                
                # Calculate aggregate value
                if total_weight > 0:
                    aggregate_value = np.sum(values * adjusted_weights) / total_weight
                else:
                    aggregate_value = 0.5  # Neutral if no weights
                    
                # Determine direction
                if aggregate_value > 0.6:
                    direction = "bullish"
                elif aggregate_value < 0.4:
                    direction = "bearish"
                else:
                    direction = "neutral"
                    
                # Calculate agreement level (standard deviation of sentiment values)
                std_dev = np.std(values)
                agreement_level = 1.0 - (float(std_dev) / 0.5)  # Normalized standard deviation
                agreement_level = max(0.0, min(1.0, float(agreement_level)))
                
                # Calculate overall confidence (mean of individual confidences, boosted by agreement)
                base_confidence = float(np.mean(confidences))
                confidence = float(base_confidence * (0.7 + (0.3 * agreement_level)))  # Agreement boosts confidence
                
                # Store in aggregated cache
                if symbol not in self.aggregated_cache:
                    self.aggregated_cache[symbol] = {}
                    
                previous_aggregate = self.aggregated_cache[symbol].get("value", 0.5)
                aggregate_shift = abs(aggregate_value - previous_aggregate)
                
                self.aggregated_cache[symbol] = {
                    "value": aggregate_value,
                    "direction": direction, 
                    "confidence": confidence,
                    "agreement_level": agreement_level,
                    "source_count": len(source_data),
                    "sources": sources_list,
                    "last_update": datetime.utcnow()
                }
                
                # Record metrics
                metrics.gauge("sentiment_aggregate_value", aggregate_value, tags={
                    "symbol": symbol,
                    "direction": direction
                })
                
                metrics.gauge("sentiment_confidence", confidence, tags={
                    "symbol": symbol
                })
                
                metrics.gauge("sentiment_agreement", agreement_level, tags={
                    "symbol": symbol
                })
                
                # Publish event if significant shift or high confidence
                is_significant = aggregate_shift > self.sentiment_shift_threshold
                is_high_confidence = confidence > 0.85
                
                if is_significant or is_high_confidence:
                    details = {
                        "agreement_level": agreement_level,
                        "source_count": len(source_data),
                        "source_types": sources_list,
                        "event_type": "aggregate_sentiment_shift" if is_significant else "high_confidence_sentiment"
                    }
                    
                    # Find direction agreement
                    directions = [data.get("direction", "neutral") for data in source_data.values()]
                    bullish_count = directions.count("bullish")
                    bearish_count = directions.count("bearish")
                    neutral_count = directions.count("neutral")
                    
                    details["direction_counts"] = {
                        "bullish": bullish_count,
                        "bearish": bearish_count,
                        "neutral": neutral_count
                    }
                    
                    # Add source-specific values
                    for source_type, data in source_data.items():
                        details[f"{source_type}_value"] = data.get("value", 0.5)
                        details[f"{source_type}_confidence"] = data.get("confidence", 0.7)
                        
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=aggregate_value,
                        confidence=confidence,
                        is_extreme=abs(aggregate_value - 0.5) > 0.3,  # Extreme if far from neutral
                        signal_type="aggregate",
                        sources=sources_list,
                        details=details
                    )
                    
                    self.logger.info("Published aggregated sentiment event", 
                                   symbol=symbol,
                                   direction=direction,
                                   value=aggregate_value, 
                                   confidence=confidence,
                                   source_count=len(source_data))
                    
                    # Record event metric
                    metrics.counter("sentiment_event_published", tags={
                        "symbol": symbol,
                        "direction": direction,
                        "event_type": "aggregate"
                    })
        finally:
            # End timer and record duration
            duration = timer_end()
            self.logger.debug(f"Sentiment aggregation took {duration:.3f}s",
                           symbol=symbol)
    
    async def aggregate_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Aggregate sentiment from all sources for a symbol.
        
        This is a public method that can be called to trigger sentiment
        aggregation and retrieve the aggregated result.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            The aggregated sentiment data, or None if aggregation failed
        """
        # Check cache first
        cache_key = f"aggregate:{symbol}"
        cached_result = self.result_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Not in cache, perform aggregation
        await self._aggregate_sentiment(symbol)
        
        # Return aggregated data if available
        if symbol in self.aggregated_cache:
            # Cache the result
            self.result_cache.set(cache_key, self.aggregated_cache[symbol])
            return self.aggregated_cache[symbol]
        
        return None
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to aggregated sentiment.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        if not candles or len(candles) < 20:
            return
            
        # Check if we have aggregated sentiment data for this symbol
        if symbol not in self.aggregated_cache:
            return
            
        # Get the aggregated sentiment
        sentiment_data = self.aggregated_cache[symbol]
        sentiment_value = sentiment_data.get("value", 0.5)
        direction = sentiment_data.get("direction", "neutral")
        confidence = sentiment_data.get("confidence", 0.7)
        agreement_level = sentiment_data.get("agreement_level", 0.5)
        
        # Only perform advanced analysis if we have high confidence and agreement
        if confidence < 0.8 or agreement_level < 0.7:
            return
            
        # Get price data
        closes = np.array([candle.close for candle in candles])
        
        # Calculate recent performance using numpy for better performance
        price_change_1d = (closes[-1] / closes[-1]) - 1 if len(closes) > 1 else 0
        price_change_5d = (closes[-1] / closes[-5]) - 1 if len(closes) > 5 else 0
        price_change_20d = (closes[-1] / closes[-20]) - 1 if len(closes) > 20 else 0
        
        # Check for divergence between sentiment and price action
        # This is a complex analysis that could indicate turning points
        is_sentiment_divergence = False
        
        if direction == "bullish" and price_change_5d < -0.05:
            # Bullish sentiment despite recent price drop
            is_sentiment_divergence = True
            divergence_type = "bullish_divergence"
        elif direction == "bearish" and price_change_5d > 0.05:
            # Bearish sentiment despite recent price rise
            is_sentiment_divergence = True
            divergence_type = "bearish_divergence"
            
        if is_sentiment_divergence and confidence > 0.85:
            # Higher bar for publishing divergence signals
            await self.publish_sentiment_event(
                symbol=symbol,
                direction=direction,  # Keep original direction
                value=sentiment_value,
                confidence=confidence * 0.9,  # Slightly reduce confidence for divergence
                timeframe=timeframe,
                is_extreme=False,
                signal_type="divergence",
                sources=sentiment_data.get("sources", []),
                details={
                    "price_change_1d": price_change_1d,
                    "price_change_5d": price_change_5d,
                    "price_change_20d": price_change_20d,
                    "divergence_type": divergence_type,
                    "event_type": "sentiment_price_divergence"
                }
            )
            
            self.logger.info("Published sentiment divergence event", 
                           symbol=symbol,
                           direction=direction,
                           divergence_type=divergence_type,
                           confidence=confidence * 0.9)
            
            # Record divergence event metric
            metrics.counter("sentiment_divergence_event", tags={
                "symbol": symbol,
                "direction": direction,
                "divergence_type": divergence_type
            })
            
        # Record performance for adaptive learning if enabled
        if self.use_adaptive_weights and len(candles) > 3:
            # Get the sentiment prediction from a few candles ago
            lookback = 3  # Number of candles to look back
            
            # Get sentiment value from when the prediction was made
            prediction_time = candles[-lookback].timestamp
            
            # Find the closest sentiment update
            for source_type in ["social_media", "news", "market_sentiment", "onchain"]:
                if (symbol in self.sentiment_cache and 
                    source_type in self.sentiment_cache[symbol]):
                    source_data = self.sentiment_cache[symbol][source_type]
                    timestamp = source_data.get("last_update")
                    
                    # If this update was close to the prediction time
                    if timestamp and abs((timestamp - prediction_time).total_seconds()) < 3600:
                        # Calculate actual outcome (normalized price change)
                        price_change = (candles[-1].close / candles[-lookback].close) - 1
                        normalized_outcome = 0.5 + (price_change * 5)  # Scale to 0-1 range
                        normalized_outcome = max(0, min(1, normalized_outcome))
                        
                        # Record performance for this source
                        if self.use_adaptive_weights:
                            self.adaptive_weights.record_performance(
                                source=source_type,
                                prediction=source_data.get("value", 0.5),
                                actual_outcome=normalized_outcome,
                                timestamp=timestamp
                            )
