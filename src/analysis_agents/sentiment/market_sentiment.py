"""Market sentiment analysis.

This module provides functionality for analyzing sentiment from market indicators
such as the Fear & Greed Index, Long/Short ratio, and other market sentiment metrics.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import CandleData, TimeFrame


class MarketSentimentAgent(BaseSentimentAgent):
    """Analysis agent for market sentiment indicators.
    
    This agent processes sentiment data from market indicators
    and publishes sentiment events with confidence scores.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the market sentiment agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "market_sentiment")
        
        # Market indicators to monitor
        self.indicators = config.get(
            f"analysis_agents.{agent_id}.indicators", 
            ["FearGreedIndex", "LongShortRatio"]
        )
        
        # Update interval in seconds
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            3600  # Default: 1 hour
        )
    
    async def _initialize(self) -> None:
        """Initialize the market sentiment agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing market sentiment agent",
                       indicators=self.indicators)
    
    async def _start(self) -> None:
        """Start the market sentiment agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for market sentiment
        self.update_task = self.create_task(
            self._update_sentiment_periodically()
        )
    
    async def _stop(self) -> None:
        """Stop the market sentiment agent."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        await super()._stop()
    
    async def _update_sentiment_periodically(self) -> None:
        """Update market sentiment periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    # In a real system, we would fetch actual market indicators
                    # For the demo, we'll simulate it
                    await self._analyze_market_sentiment_indicators(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Market sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in market sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _analyze_market_sentiment_indicators(self, symbol: str) -> None:
        """Analyze market sentiment indicators for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "market_sentiment", self.update_interval):
            return
        
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
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to market sentiment indicators.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        if not candles or len(candles) < 10:
            return
            
        # Check if we have market sentiment data for this symbol
        if symbol not in self.sentiment_cache or "market_sentiment" not in self.sentiment_cache[symbol]:
            return
            
        # Get the latest market sentiment
        sentiment_data = self.sentiment_cache[symbol]["market_sentiment"]
        fear_greed = sentiment_data.get("fear_greed_index", 50)
        sentiment_value = sentiment_data.get("value", 0.5)
        
        # Calculate recent volatility
        if len(candles) >= 20:
            # Calculate volatility as standard deviation of returns
            closes = [candle.close for candle in candles[-20:]]
            returns = [(closes[i] / closes[i-1]) - 1 for i in range(1, len(closes))]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
            
            # Check for extreme fear and high volatility
            if fear_greed < 25 and volatility > 0.03:  # 3% daily volatility is high
                # This is often a contrarian buy signal
                await self.publish_sentiment_event(
                    symbol=symbol,
                    direction="bullish",  # Contrarian to extreme fear
                    value=0.7,  # Moderately bullish
                    confidence=0.8,
                    timeframe=timeframe,
                    is_extreme=True,
                    signal_type="contrarian",
                    sources=self.indicators,
                    details={
                        "fear_greed_index": fear_greed,
                        "volatility": volatility,
                        "event_type": "volatility_fear_contrarian"
                    }
                )
