"""Base sentiment analysis agent.

This module provides the base class for all sentiment analysis agents,
with common functionality and interfaces.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple

from src.analysis_agents.base_agent import AnalysisAgent
from src.common.config import config
from src.common.logging import get_logger
from src.models.events import SentimentEvent
from src.models.market_data import CandleData, TimeFrame


class BaseSentimentAgent(AnalysisAgent):
    """Base class for all sentiment analysis agents.
    
    This class provides common functionality for sentiment analysis,
    including configuration loading, sentiment caching, and event
    publishing.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the base sentiment analysis agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", f"sentiment_{agent_id}")
        
        # Confidence and threshold settings
        self.min_confidence = config.get(f"analysis_agents.{agent_id}.min_confidence", 0.7)
        self.sentiment_shift_threshold = config.get(f"analysis_agents.{agent_id}.sentiment_shift_threshold", 0.15)
        self.contrarian_threshold = config.get(f"analysis_agents.{agent_id}.contrarian_threshold", 0.8)
        
        # Sentiment data cache
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self.max_history_size = config.get(f"analysis_agents.{agent_id}.max_history_size", 100)
        
        # Last update times
        self.last_update: Dict[str, Dict[str, datetime]] = {}
        
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        Sentiment analysis doesn't directly process individual candles,
        but this method is required by the AnalysisAgent base class.
        
        Args:
            candle: The candle data to process
        """
        # Sentiment analysis typically doesn't process individual candles
        pass
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to sentiment indicators.
        
        This method should be implemented by subclasses to analyze
        market data in relation to sentiment indicators.
        
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
        
        # Subclasses should implement specific analysis logic
    
    async def publish_sentiment_event(
        self,
        symbol: str,
        direction: str,
        value: float,
        confidence: float,
        timeframe: Optional[TimeFrame] = None,
        is_extreme: bool = False,
        signal_type: str = "sentiment",
        sources: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publish a sentiment event.
        
        Args:
            symbol: The trading pair symbol
            direction: Sentiment direction ("bullish", "bearish", or "neutral")
            value: Sentiment value (0.0-1.0, with 0.5 being neutral)
            confidence: Confidence score (0.0-1.0)
            timeframe: Optional timeframe associated with the sentiment
            is_extreme: Whether this is an extreme sentiment reading
            signal_type: Type of signal ("sentiment" or "contrarian")
            sources: List of data sources contributing to this sentiment
            details: Additional details about the sentiment
        """
        if sources is None:
            sources = []
            
        if details is None:
            details = {}
            
        # Create enhanced details with additional metadata
        enhanced_details = details.copy()
        
        # Add timeframe if provided
        if timeframe:
            enhanced_details["timeframe"] = timeframe.value if isinstance(timeframe, TimeFrame) else timeframe
            
        # Add tags based on properties
        tags = [f"direction:{direction}", f"source:{self.agent_id}"]
        
        if is_extreme:
            tags.append("extreme")
            
        if signal_type == "contrarian":
            tags.append("contrarian")
            
        # Add sources and tags to details
        enhanced_details["sources"] = sources
        enhanced_details["tags"] = tags
        enhanced_details["signal_type"] = signal_type
        
        # Create and publish the event
        await self.publish_event(SentimentEvent(
            source=self.name,
            symbol=symbol,
            sentiment_value=value,
            sentiment_direction=direction,
            confidence=confidence,
            details=enhanced_details
        ))
        
        self.logger.info("Published sentiment event", 
                        symbol=symbol,
                        direction=direction,
                        value=value,
                        confidence=confidence,
                        is_extreme=is_extreme,
                        signal_type=signal_type)
    
    def _should_update_sentiment(
        self, 
        symbol: str, 
        source_type: str, 
        update_interval: int
    ) -> bool:
        """Check if sentiment should be updated based on interval.
        
        Args:
            symbol: The trading pair symbol
            source_type: The type of sentiment source
            update_interval: The update interval in seconds
            
        Returns:
            True if sentiment should be updated, False otherwise
        """
        now = datetime.utcnow()
        
        # Initialize dictionaries if needed
        if source_type not in self.last_update:
            self.last_update[source_type] = {}
            
        # Check if we've updated recently
        if (symbol in self.last_update[source_type] and 
            (now - self.last_update[source_type][symbol]).total_seconds() < update_interval):
            return False
            
        # Update timestamp and return True
        self.last_update[source_type][symbol] = now
        return True
    
    def _update_sentiment_cache(
        self,
        symbol: str,
        source_type: str,
        sentiment_value: float,
        direction: str,
        confidence: float,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Update sentiment cache and return the shift in sentiment.
        
        Args:
            symbol: The trading pair symbol
            source_type: The type of sentiment source
            sentiment_value: The new sentiment value (0.0-1.0)
            direction: The sentiment direction ("bullish", "bearish", or "neutral")
            confidence: The confidence score (0.0-1.0)
            additional_data: Additional data to store in the cache
            
        Returns:
            The absolute shift in sentiment value
        """
        now = datetime.utcnow()
        
        # Initialize dictionaries if needed
        if symbol not in self.sentiment_cache:
            self.sentiment_cache[symbol] = {}
            
        # Get previous sentiment for calculating shift
        previous_sentiment = self.sentiment_cache.get(symbol, {}).get(source_type, {}).get("value", 0.5)
        sentiment_shift = abs(sentiment_value - previous_sentiment)
        
        # Update or create cache entry
        if source_type not in self.sentiment_cache[symbol]:
            self.sentiment_cache[symbol][source_type] = {
                "history": [],
                "value": sentiment_value,
                "direction": direction,
                "confidence": confidence,
                "last_update": now
            }
            
            # Add additional data if provided
            if additional_data:
                self.sentiment_cache[symbol][source_type].update(additional_data)
        else:
            # Add to history
            history = self.sentiment_cache[symbol][source_type]["history"]
            history_entry = (now, sentiment_value, direction, confidence)
            history.append(history_entry)
            
            # Limit history size
            if len(history) > self.max_history_size:
                history = history[-self.max_history_size:]
            
            # Update current values
            self.sentiment_cache[symbol][source_type]["history"] = history
            self.sentiment_cache[symbol][source_type]["value"] = sentiment_value
            self.sentiment_cache[symbol][source_type]["direction"] = direction
            self.sentiment_cache[symbol][source_type]["confidence"] = confidence
            self.sentiment_cache[symbol][source_type]["last_update"] = now
            
            # Update additional data if provided
            if additional_data:
                self.sentiment_cache[symbol][source_type].update(additional_data)
                
        return sentiment_shift
