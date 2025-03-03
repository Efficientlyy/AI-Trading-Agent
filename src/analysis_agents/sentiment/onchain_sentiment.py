"""Onchain sentiment analysis.

This module provides functionality for analyzing sentiment from on-chain metrics
such as wallet activity, transaction volume, and blockchain data.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import CandleData, TimeFrame


class OnchainSentimentAgent(BaseSentimentAgent):
    """Analysis agent for onchain sentiment.
    
    This agent processes sentiment data from on-chain metrics
    and publishes sentiment events with confidence scores.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the onchain sentiment agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "onchain_sentiment")
        
        # Onchain metrics to monitor
        self.metrics = config.get(
            f"analysis_agents.{agent_id}.metrics", 
            ["LargeTransactions", "ActiveAddresses"]
        )
        
        # Update interval in seconds
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            3600  # Default: 1 hour
        )
    
    async def _initialize(self) -> None:
        """Initialize the onchain sentiment agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing onchain sentiment agent",
                       metrics=self.metrics)
    
    async def _start(self) -> None:
        """Start the onchain sentiment agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for onchain sentiment
        self.update_task = self.create_task(
            self._update_sentiment_periodically()
        )
    
    async def _stop(self) -> None:
        """Stop the onchain sentiment agent."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        await super()._stop()
    
    async def _update_sentiment_periodically(self) -> None:
        """Update onchain sentiment periodically."""
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
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Onchain sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in onchain sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _analyze_onchain_metrics(self, symbol: str) -> None:
        """Analyze onchain metrics for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "onchain", self.update_interval):
            return
            
        # Only some cryptocurrencies have meaningful on-chain metrics
        base_currency = symbol.split('/')[0]
        if base_currency not in ["BTC", "ETH"]:
            return
        
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
                "hash_rate": 0.2,
                "exchange_reserves": 0.2
            }
            
            sentiment_value = sum(
                sentiment_metrics[metric] * metric_weights[metric]
                for metric in sentiment_metrics
            ) / sum(metric_weights.values())
            
            # Calculate confidence based on data quality and coverage
            confidence_base = 0.7  # Base confidence
            confidence_adjustment = random.uniform(-0.1, 0.1)  # Random adjustment
            confidence = min(0.95, max(0.5, confidence_base + confidence_adjustment))
            
            # Determine direction
            if sentiment_value > 0.6:
                direction = "bullish"
            elif sentiment_value < 0.4:
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Store additional metadata
            additional_data = {
                "large_transactions": large_tx_value,
                "active_addresses_growth": active_addr_growth,
                "hash_rate_trend": hash_rate_trend,
                "exchange_reserves_growth": exchange_reserves_growth,
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
                    is_strong_signal = exchange_reserves_growth < -0.2
                    
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        is_extreme=is_strong_signal,
                        sources=self.metrics,
                        details={
                            "large_transactions": large_tx_value,
                            "active_addresses_growth": active_addr_growth,
                            "hash_rate_trend": hash_rate_trend,
                            "exchange_reserves_growth": exchange_reserves_growth,
                            "event_type": "onchain_sentiment_shift" if sentiment_shift > self.sentiment_shift_threshold else "strong_onchain_signal"
                        }
                    )
        
        except Exception as e:
            self.logger.error("Error analyzing onchain metrics", 
                           symbol=symbol,
                           error=str(e))
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to onchain metrics.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        if not candles or len(candles) < 10:
            return
            
        # Only some cryptocurrencies have meaningful on-chain metrics
        base_currency = symbol.split('/')[0]
        if base_currency not in ["BTC", "ETH"]:
            return
            
        # Check if we have onchain sentiment data for this symbol
        if symbol not in self.sentiment_cache or "onchain" not in self.sentiment_cache[symbol]:
            return
            
        # Get the latest onchain sentiment
        sentiment_data = self.sentiment_cache[symbol]["onchain"]
        exchange_reserves_growth = sentiment_data.get("exchange_reserves_growth", 0)
        
        # Get price data from candles
        closes = [candle.close for candle in candles]
        
        # If there's a significant exchange outflow and price is stagnant or falling,
        # that's a potential divergence and accumulation signal
        if len(closes) >= 20 and exchange_reserves_growth < -0.3:
            # Calculate short-term trend
            short_term_change = (closes[-1] / closes[-20]) - 1  # 20-period return
            
            # If price is not rising despite exchange outflows
            if short_term_change < 0.02:  # Less than 2% gain
                # This could be an accumulation signal
                await self.publish_sentiment_event(
                    symbol=symbol,
                    direction="bullish",
                    value=0.75,  # Quite bullish
                    confidence=0.85,
                    timeframe=timeframe,
                    is_extreme=False,
                    signal_type="divergence",
                    sources=self.metrics,
                    details={
                        "exchange_reserves_growth": exchange_reserves_growth,
                        "price_change": short_term_change,
                        "event_type": "accumulation_divergence"
                    }
                )
