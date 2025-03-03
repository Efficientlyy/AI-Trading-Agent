"""Sentiment aggregation module.

This module provides functionality for aggregating sentiment data
from various sources into a unified sentiment signal.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.common.config import config
from src.common.logging import get_logger
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
    
    async def _initialize(self) -> None:
        """Initialize the sentiment aggregator."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing sentiment aggregator",
                       source_weights=self.source_weights)
    
    async def _start(self) -> None:
        """Start the sentiment aggregator."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for aggregating sentiment
        self.update_task = self.create_task(
            self._aggregate_sentiment_periodically()
        )
    
    async def _stop(self) -> None:
        """Stop the sentiment aggregator."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
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
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _aggregate_sentiment(self, symbol: str) -> None:
        """Internal method to aggregate sentiment from all sources for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
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
        
        # Need at least two sources to aggregate
        if len(source_data) < 2:
            return
            
        # Calculate weighted sentiment value
        weighted_values = []
        total_weight = 0
        
        for source_type, data in source_data.items():
            # Get the source weight
            weight = self.source_weights.get(source_type, 0.25)
            
            # Adjust weight by confidence and recency
            now = datetime.utcnow()
            last_update = data.get("last_update", now)
            age_hours = (now - last_update).total_seconds() / 3600
            recency_factor = max(0.5, 1.0 - (age_hours / 24.0))  # Decay over 24 hours
            
            confidence = data.get("confidence", 0.7)
            adjusted_weight = weight * confidence * recency_factor
            
            # Add to weighted calculation
            weighted_values.append(data.get("value", 0.5) * adjusted_weight)
            total_weight += adjusted_weight
        
        # Calculate aggregate sentiment value
        if total_weight > 0:
            aggregate_value = sum(weighted_values) / total_weight
        else:
            aggregate_value = 0.5  # Neutral if no weights
            
        # Determine direction
        if aggregate_value > 0.6:
            direction = "bullish"
        elif aggregate_value < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"
            
        # Calculate agreement level
        values = [data.get("value", 0.5) for data in source_data.values()]
        # Calculate agreement level (standard deviation of sentiment values)
        std_dev = np.std(values)
        agreement_level = 1.0 - (float(std_dev) / 0.5)  # Normalized standard deviation
        agreement_level = max(0.0, min(1.0, float(agreement_level)))
        
        # Calculate overall confidence (mean of individual confidences, boosted by agreement)
        confidences = [data.get("confidence", 0.7) for data in source_data.values()]
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
    
    async def aggregate_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Aggregate sentiment from all sources for a symbol.
        
        This is a public method that can be called to trigger sentiment
        aggregation and retrieve the aggregated result.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            The aggregated sentiment data, or None if aggregation failed
        """
        await self._aggregate_sentiment(symbol)
        
        # Return aggregated data if available
        if symbol in self.aggregated_cache:
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
        closes = [candle.close for candle in candles]
        
        # Calculate recent performance
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
