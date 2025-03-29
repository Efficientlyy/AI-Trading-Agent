"""LLM-based sentiment analysis agent.

This module provides a sentiment analysis agent that leverages
large language models for sophisticated market sentiment analysis.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.analysis_agents.sentiment.llm_service import LLMService
from src.common.config import config
from src.common.logging import get_logger
from src.common.monitoring import metrics
from src.models.market_data import CandleData, TimeFrame
from src.models.analysis_events import SentimentEvent


class LLMSentimentAgent(BaseSentimentAgent):
    """Sentiment analysis agent using LLM models.
    
    This agent uses large language models to analyze text data from 
    various sources and extract sophisticated sentiment signals.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the LLM sentiment agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "llm_sentiment")
        
        # LLM configuration
        self.llm_service = None
        self.use_primary_model = config.get(
            f"analysis_agents.{agent_id}.use_primary_model", 
            False
        )
        self.model = config.get(
            f"analysis_agents.{agent_id}.model", 
            None  # Will use the service default
        )
        
        # Source configuration
        self.data_sources = config.get(
            f"analysis_agents.{agent_id}.data_sources", 
            ["social_media", "news"]
        )
        
        # Update interval in seconds
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            900  # Default: 15 minutes
        )
        
        # Event thresholds
        self.event_severity_threshold = config.get(
            f"analysis_agents.{agent_id}.event_severity_threshold", 
            5  # 0-10 scale
        )
        
        # Pending text for analysis
        self.pending_texts: Dict[str, List[Dict]] = {}
        
        # Market context for each symbol
        self.market_context: Dict[str, str] = {}
    
    async def _initialize(self) -> None:
        """Initialize the LLM sentiment agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing LLM sentiment agent",
                       data_sources=self.data_sources,
                       model=self.model or "default")
        
        # Initialize the LLM service
        self.llm_service = LLMService()
        await self.llm_service.initialize()
        
        # Initialize pending texts container
        self.pending_texts = {
            source: [] for source in self.data_sources
        }
        
        # Record initialization metric
        metrics.counter("llm_sentiment_agent_initialized", tags={
            "agent_id": self.agent_id
        })
    
    async def _start(self) -> None:
        """Start the LLM sentiment agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for processing pending texts
        self.update_task = self.create_task(
            self._process_pending_texts_periodically()
        )
        
        # Record start metric
        metrics.counter("llm_sentiment_agent_started", tags={
            "agent_id": self.agent_id
        })
    
    async def _stop(self) -> None:
        """Stop the LLM sentiment agent."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Close LLM service
        if self.llm_service:
            await self.llm_service.close()
        
        # Record stop metric
        metrics.counter("llm_sentiment_agent_stopped", tags={
            "agent_id": self.agent_id
        })
        
        await super()._stop()
    
    async def submit_text(self, symbol: str, text: str, source: str, metadata: Dict = None) -> None:
        """Submit text for LLM analysis.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            text: The text to analyze
            source: Source of the text (e.g., "twitter", "news", "reddit")
            metadata: Additional metadata about the text
        """
        if not self.enabled:
            return
            
        # Classify the source into broader categories
        category = self._categorize_source(source)
        
        if category not in self.data_sources:
            self.logger.debug(f"Ignoring text from unsupported source: {source}")
            return
        
        # Add to pending texts
        if category not in self.pending_texts:
            self.pending_texts[category] = []
            
        self.pending_texts[category].append({
            "symbol": symbol,
            "text": text,
            "source": source,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        })
        
        # If we have accumulated enough texts, trigger a batch process
        batch_size = 5
        if len(self.pending_texts[category]) >= batch_size:
            self.logger.debug(f"Batch size reached for {category}, scheduling processing")
            self.create_task(self._process_texts_for_source(category))
    
    def _categorize_source(self, source: str) -> str:
        """Categorize a specific source into broader categories.
        
        Args:
            source: Specific source name
            
        Returns:
            Broader category name
        """
        # Map specific sources to broader categories
        source_lower = source.lower()
        
        if source_lower in ["twitter", "x", "reddit", "telegram", "discord"]:
            return "social_media"
        elif source_lower in ["coindesk", "cointelegraph", "bloomberg", "wsj", "reuters"]:
            return "news"
        elif source_lower in ["github", "documentation", "whitepaper", "blog"]:
            return "technical"
        else:
            # Default to the original source if no mapping
            return source
    
    async def _process_pending_texts_periodically(self) -> None:
        """Process pending texts from all sources periodically."""
        try:
            while True:
                for source in self.data_sources:
                    if source in self.pending_texts and self.pending_texts[source]:
                        await self._process_texts_for_source(source)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("LLM text processing task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in LLM text processing task", error=str(e))
            
            # Record error metric
            metrics.counter("llm_sentiment_agent_error", tags={
                "agent_id": self.agent_id,
                "error_type": type(e).__name__
            })
            
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _process_texts_for_source(self, source: str) -> None:
        """Process pending texts from a specific source.
        
        Args:
            source: The text source category
        """
        if not self.llm_service:
            self.logger.error("LLM service not initialized")
            return
            
        if source not in self.pending_texts or not self.pending_texts[source]:
            return
            
        self.logger.info(f"Processing {len(self.pending_texts[source])} pending texts from {source}")
        
        # Group texts by symbol for more efficient processing
        symbol_texts: Dict[str, List[Dict]] = {}
        for text_data in self.pending_texts[source]:
            symbol = text_data["symbol"]
            if symbol not in symbol_texts:
                symbol_texts[symbol] = []
            symbol_texts[symbol].append(text_data)
        
        # Process each symbol's texts
        for symbol, texts in symbol_texts.items():
            self.logger.debug(f"Processing {len(texts)} texts for {symbol}")
            
            # Batch processing for better efficiency
            batch_size = 5
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Extract the actual text content
                text_contents = [item["text"] for item in batch]
                
                # First, run event detection to identify potential market events
                try:
                    event_results = await self.llm_service.detect_market_event(
                        text_contents, 
                        model=self.model if self.use_primary_model else None
                    )
                    
                    # Process each event result
                    for j, result in enumerate(event_results):
                        if result.get("is_market_event", False) and result.get("severity", 0) >= self.event_severity_threshold:
                            # Found a significant market event
                            await self._handle_market_event(symbol, batch[j], result)
                except Exception as e:
                    self.logger.error(f"Error in event detection: {str(e)}")
                
                # Then, run sentiment analysis on all texts
                try:
                    sentiment_results = await self.llm_service.analyze_sentiment(
                        text_contents,
                        model=self.model
                    )
                    
                    # Process each sentiment result
                    for j, result in enumerate(sentiment_results):
                        await self._handle_sentiment_result(symbol, batch[j], result)
                except Exception as e:
                    self.logger.error(f"Error in sentiment analysis: {str(e)}")
        
        # Clear processed texts
        self.pending_texts[source] = []
    
    async def _handle_market_event(self, symbol: str, text_data: Dict, event_result: Dict) -> None:
        """Handle a detected market event.
        
        Args:
            symbol: The trading pair symbol
            text_data: Original text data
            event_result: Event detection result
        """
        self.logger.info(f"Market event detected for {symbol}: {event_result.get('event_type')}",
                       severity=event_result.get("severity", 0))
        
        # Extract relevant fields
        event_type = event_result.get("event_type", "unknown")
        severity = event_result.get("severity", 0)
        credibility = event_result.get("credibility", 0.5)
        propagation = event_result.get("propagation_speed", "days")
        assets_affected = event_result.get("assets_affected", [])
        explanation = event_result.get("explanation", "")
        
        # Calculate event confidence based on credibility and other factors
        confidence = credibility * (severity / 10.0)
        
        # Determine impact direction (default to neutral)
        impact_direction = "neutral"
        
        # Assess market impact if the event is significant
        if self.llm_service and severity >= self.event_severity_threshold:
            try:
                # Get market context if available
                market_context = self.market_context.get(symbol, "")
                
                # Prepare the event description
                event_description = f"{event_type} event: {explanation}"
                
                # Get impact assessment
                impact_result = await self.llm_service.assess_market_impact(
                    event=event_description,
                    market_context=market_context,
                    model=self.model if self.use_primary_model else None
                )
                
                # Extract impact direction and adjust confidence
                if impact_result:
                    impact_direction = self._map_impact_to_sentiment(
                        impact_result.get("primary_impact_direction", "neutral")
                    )
                    
                    # Enhance confidence with impact assessment confidence
                    confidence = (confidence + impact_result.get("confidence", 0.5)) / 2
                    
                    # Add impact assessment to event details
                    event_result["impact_assessment"] = {
                        "direction": impact_result.get("primary_impact_direction"),
                        "magnitude": impact_result.get("impact_magnitude"),
                        "duration": impact_result.get("estimated_duration"),
                        "reasoning": impact_result.get("reasoning")
                    }
            except Exception as e:
                self.logger.error(f"Error assessing market impact: {str(e)}")
        
        # Update the sentiment cache with this event information
        self._update_sentiment_cache(
            symbol=symbol,
            source_type="event",
            sentiment_value=self._direction_to_value(impact_direction),
            direction=impact_direction,
            confidence=confidence,
            additional_data={
                "event_type": event_type,
                "severity": severity,
                "propagation": propagation,
                "assets_affected": assets_affected,
                "text_source": text_data.get("source"),
                "explanation": explanation
            }
        )
        
        # Publish event if significant
        if severity >= self.event_severity_threshold and confidence >= self.min_confidence:
            # Create event details
            details = {
                "event_type": event_type,
                "severity": severity,
                "credibility": credibility,
                "propagation": propagation,
                "assets_affected": assets_affected,
                "explanation": explanation,
                "text_source": text_data.get("source"),
                "original_text": text_data.get("text")[:150] + "..." if len(text_data.get("text", "")) > 150 else text_data.get("text", "")
            }
            
            # Add metadata if available
            if text_data.get("metadata"):
                details["metadata"] = text_data["metadata"]
            
            # Add impact assessment if available
            if "impact_assessment" in event_result:
                details["impact_assessment"] = event_result["impact_assessment"]
            
            # Publish the event
            await self.publish_sentiment_event(
                symbol=symbol,
                direction=impact_direction,
                value=self._direction_to_value(impact_direction),
                confidence=confidence,
                is_extreme=severity >= 8,  # Consider high severity events as extreme
                signal_type="event",
                sources=[text_data.get("source")],
                details=details
            )
            
            # Record event detection metric
            metrics.counter("market_event_detected", tags={
                "symbol": symbol,
                "event_type": event_type,
                "severity": str(severity),
                "direction": impact_direction
            })
    
    async def _handle_sentiment_result(self, symbol: str, text_data: Dict, sentiment_result: Dict) -> None:
        """Handle a sentiment analysis result.
        
        Args:
            symbol: The trading pair symbol
            text_data: Original text data
            sentiment_result: Sentiment analysis result
        """
        # Extract relevant fields
        sentiment_value = sentiment_result.get("sentiment_value", 0.5)
        direction = sentiment_result.get("direction", "neutral")
        confidence = sentiment_result.get("confidence", 0.5)
        explanation = sentiment_result.get("explanation", "")
        key_points = sentiment_result.get("key_points", [])
        
        self.logger.debug(f"Sentiment analysis for {symbol}: {direction} ({sentiment_value:.2f})",
                        confidence=confidence)
        
        # Get the source category
        source_category = self._categorize_source(text_data.get("source", "unknown"))
        
        # Update the sentiment cache
        sentiment_shift = self._update_sentiment_cache(
            symbol=symbol,
            source_type=source_category,
            sentiment_value=sentiment_value,
            direction=direction,
            confidence=confidence,
            additional_data={
                "explanation": explanation,
                "key_points": key_points,
                "text_source": text_data.get("source"),
                "timestamp": text_data.get("timestamp", datetime.utcnow()).isoformat()
            }
        )
        
        # Check if this is a significant sentiment shift or high confidence extreme reading
        is_extreme = sentiment_value > 0.8 or sentiment_value < 0.2
        is_significant_shift = sentiment_shift > self.sentiment_shift_threshold
        
        if (is_significant_shift or (is_extreme and confidence > self.min_confidence)):
            # Determine event type
            event_type = "sentiment_shift" if is_significant_shift else "extreme_sentiment"
            
            # Determine if extreme sentiment should be treated as contrarian
            signal_type = "sentiment"
            if is_extreme and sentiment_value > self.contrarian_threshold:
                # Very extreme sentiment might be contrarian
                signal_type = "contrarian"
            
            # Create event details
            details = {
                "sentiment_value": sentiment_value,
                "confidence": confidence,
                "explanation": explanation,
                "key_points": key_points,
                "text_source": text_data.get("source"),
                "source_category": source_category,
                "event_type": event_type,
                "is_extreme": is_extreme,
                "sentiment_shift": sentiment_shift
            }
            
            # Add metadata if available
            if text_data.get("metadata"):
                details["metadata"] = text_data["metadata"]
            
            # Add original text (truncated)
            details["original_text"] = text_data.get("text")[:150] + "..." if len(text_data.get("text", "")) > 150 else text_data.get("text", "")
            
            # Publish the event
            await self.publish_sentiment_event(
                symbol=symbol,
                direction=direction,
                value=sentiment_value,
                confidence=confidence,
                is_extreme=is_extreme,
                signal_type=signal_type,
                sources=[text_data.get("source")],
                details=details
            )
            
            # Record sentiment event metric
            metrics.counter("sentiment_event_detected", tags={
                "symbol": symbol,
                "direction": direction,
                "source_category": source_category,
                "is_extreme": str(is_extreme),
                "is_shift": str(is_significant_shift)
            })
    
    def _map_impact_to_sentiment(self, impact_direction: str) -> str:
        """Map impact direction to sentiment direction.
        
        Args:
            impact_direction: Impact assessment direction
            
        Returns:
            Sentiment direction
        """
        if impact_direction == "positive":
            return "bullish"
        elif impact_direction == "negative":
            return "bearish"
        else:
            return "neutral"
    
    def _direction_to_value(self, direction: str) -> float:
        """Convert direction to a sentiment value.
        
        Args:
            direction: Sentiment direction
            
        Returns:
            Sentiment value (0-1)
        """
        if direction == "bullish":
            return 0.75
        elif direction == "bearish":
            return 0.25
        else:
            return 0.5
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data to update context and check for divergences.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        if not candles or len(candles) < 10:
            return
        
        # Update market context for this symbol
        self._update_market_context(symbol, exchange, timeframe, candles)
        
        # Check for sentiment-price divergences
        if symbol in self.sentiment_cache:
            for source_type in ["social_media", "news", "event"]:
                if source_type in self.sentiment_cache[symbol]:
                    await self._check_divergence(symbol, source_type, exchange, timeframe, candles)
    
    def _update_market_context(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Update market context for a symbol.
        
        Args:
            symbol: The trading pair symbol
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        # Only update for higher timeframes to avoid too frequent updates
        if timeframe not in [TimeFrame.HOUR_1, TimeFrame.HOUR_4, TimeFrame.DAY_1]:
            return
            
        # Calculate basic market stats
        price_current = candles[-1].close
        price_24h_ago = candles[-24].close if len(candles) >= 24 else candles[0].close
        price_7d_ago = candles[-168].close if len(candles) >= 168 else candles[0].close
        
        change_24h = (price_current / price_24h_ago - 1) * 100
        change_7d = (price_current / price_7d_ago - 1) * 100
        
        # Calculate volatility
        returns = [(candles[i].close / candles[i-1].close - 1) for i in range(1, len(candles))]
        volatility = (sum([r*r for r in returns]) / len(returns)) ** 0.5 * 100  # Annualized
        
        # Create market context string
        context = f"""
Market context for {symbol} as of {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC:
- Current price: {price_current:.2f}
- 24h change: {change_24h:.2f}%
- 7d change: {change_7d:.2f}%
- Recent volatility: {volatility:.2f}%
- Exchange: {exchange}
- Timeframe: {timeframe.name}
"""
        
        # Add volume information
        avg_volume = sum([c.volume for c in candles[-24:]]) / 24 if len(candles) >= 24 else candles[-1].volume
        context += f"- 24h average volume: {avg_volume:.2f}\n"
        
        # Add trend information
        if len(candles) >= 50:
            ma50 = sum([c.close for c in candles[-50:]]) / 50
            context += f"- Price vs MA50: {((price_current / ma50) - 1) * 100:.2f}%\n"
            
        if len(candles) >= 200:
            ma200 = sum([c.close for c in candles[-200:]]) / 200
            context += f"- Price vs MA200: {((price_current / ma200) - 1) * 100:.2f}%\n"
        
        # Store the context
        self.market_context[symbol] = context
    
    async def _check_divergence(
        self, 
        symbol: str, 
        source_type: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Check for divergence between sentiment and price action.
        
        Args:
            symbol: The trading pair symbol
            source_type: The sentiment source type
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        # Get sentiment data
        sentiment_data = self.sentiment_cache[symbol][source_type]
        sentiment_value = sentiment_data.get("value", 0.5)
        direction = sentiment_data.get("direction", "neutral")
        confidence = sentiment_data.get("confidence", 0.0)
        
        # Only check for high-confidence sentiment
        if confidence < 0.8:
            return
            
        # Determine market trend
        if len(candles) >= 10:
            recent_change = (candles[-1].close / candles[-10].close) - 1
            market_direction = "bullish" if recent_change > 0.03 else "bearish" if recent_change < -0.03 else "neutral"
            
            # Check for divergence
            if direction != market_direction and direction != "neutral" and market_direction != "neutral":
                self.logger.info(f"Detected {direction} sentiment divergence from {market_direction} price action for {symbol}")
                
                # For extreme divergences, publish an event
                if abs(sentiment_value - 0.5) > 0.2 and abs(recent_change) > 0.05:
                    details = {
                        "sentiment_value": sentiment_value,
                        "sentiment_direction": direction,
                        "price_direction": market_direction,
                        "price_change": recent_change * 100,
                        "divergence_type": f"{direction}_sentiment_vs_{market_direction}_price",
                        "source_type": source_type,
                        "confidence": confidence
                    }
                    
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,  # Keep original sentiment direction
                        value=sentiment_value,
                        confidence=confidence * 0.9,  # Slightly reduce confidence due to price contradiction
                        timeframe=timeframe,
                        is_extreme=False,
                        signal_type="divergence",
                        sources=[source_type],
                        details=details
                    )
                    
                    # Record divergence metric
                    metrics.counter("sentiment_divergence_detected", tags={
                        "symbol": symbol,
                        "sentiment_direction": direction,
                        "price_direction": market_direction
                    })