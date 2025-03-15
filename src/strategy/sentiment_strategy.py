"""Sentiment-based trading strategy for the AI Crypto Trading System.

This module implements a trading strategy that makes decisions based on
sentiment analysis from various sources, including social media, news,
market indicators, and on-chain data.
"""

import asyncio
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.models.events import SentimentEvent
from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData
from src.models.signals import Signal, SignalType
from src.strategy.base_strategy import Strategy


class SentimentStrategy(Strategy):
    """Strategy that generates signals based on sentiment analysis.
    
    This strategy listens to sentiment events from various sources and generates
    trading signals when sentiment reaches significant levels or shows important shifts.
    It can be configured to focus on different sentiment sources and timeframes.
    """
    
    def __init__(self, strategy_id: str = "sentiment"):
        """Initialize the sentiment-based strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", "sentiment")
        
        # Strategy-specific configuration
        self.sentiment_threshold_bullish = config.get(f"strategies.{strategy_id}.sentiment_threshold_bullish", 0.7)
        self.sentiment_threshold_bearish = config.get(f"strategies.{strategy_id}.sentiment_threshold_bearish", 0.3)
        self.min_confidence = config.get(f"strategies.{strategy_id}.min_confidence", 0.7)
        
        # Source weighting (how much to weigh different sentiment sources)
        self.source_weights = config.get(f"strategies.{strategy_id}.source_weights", {
            "social_media": 1.0,
            "news": 1.0, 
            "market": 1.0, 
            "onchain": 1.0,
            "aggregator": 2.0,  # Aggregator gets double weight by default
        })
        
        # Contrarian settings
        self.contrarian_mode = config.get(f"strategies.{strategy_id}.contrarian_mode", False)
        self.extreme_sentiment_threshold = config.get(f"strategies.{strategy_id}.extreme_sentiment_threshold", 0.85)
        
        # Position management
        self.use_stop_loss = config.get(f"strategies.{strategy_id}.use_stop_loss", True)
        self.stop_loss_pct = config.get(f"strategies.{strategy_id}.stop_loss_pct", 0.03)  # 3%
        self.use_take_profit = config.get(f"strategies.{strategy_id}.use_take_profit", True)
        self.take_profit_pct = config.get(f"strategies.{strategy_id}.take_profit_pct", 0.06)  # 6%
        
        # Strategy state
        self.latest_candles: Dict[str, CandleData] = {}
        self.sentiment_data: Dict[str, Dict[str, Dict[str, Any]]] = {}  # symbol -> source -> data
        self.last_signal_time: Dict[str, datetime] = {}
        self.min_signal_interval = config.get(f"strategies.{strategy_id}.min_signal_interval", 3600)  # seconds
        
        # Add SentimentEvent to subscriptions if not already included
        if "SentimentEvent" not in self.event_subscriptions:
            self.event_subscriptions.append("SentimentEvent")
    
    async def _strategy_initialize(self) -> None:
        """Strategy-specific initialization."""
        self.logger.info("Initializing sentiment strategy",
                      sentiment_threshold_bullish=self.sentiment_threshold_bullish,
                      sentiment_threshold_bearish=self.sentiment_threshold_bearish,
                      contrarian_mode=self.contrarian_mode)
    
    async def _strategy_start(self) -> None:
        """Strategy-specific startup."""
        self.logger.info("Starting sentiment strategy")
        
        # Clear any existing data
        self.sentiment_data = {}
        
        # Publish status
        await self.publish_status(
            "Sentiment strategy started",
            {
                "symbols": list(self.symbols) if self.symbols else "all",
                "timeframes": [tf.value for tf in self.timeframes] if self.timeframes else "all",
                "contrarian_mode": self.contrarian_mode
            }
        )
    
    async def _strategy_stop(self) -> None:
        """Strategy-specific shutdown."""
        self.logger.info("Stopping sentiment strategy")
        
        # Publish status
        await self.publish_status("Sentiment strategy stopped")
    
    async def _handle_event(self, event) -> None:
        """Handle events the strategy is subscribed to.
        
        This overrides the base implementation to add handling for SentimentEvent.
        
        Args:
            event: The event to handle
        """
        if not self.enabled:
            return
        
        event_type = event.__class__.__name__
        
        try:
            # Handle sentiment events
            if event_type == "SentimentEvent":
                await self._handle_sentiment_event(event)
            else:
                # Use the parent class handler for other event types
                await super()._handle_event(event)
                
        except Exception as e:
            self.logger.error("Error handling event", 
                             event_type=event_type,
                             error=str(e))
    
    async def _handle_sentiment_event(self, event: SentimentEvent) -> None:
        """Handle a sentiment event.
        
        Args:
            event: The sentiment event
        """
        # Extract data from the event
        symbol = event.payload.get("symbol")
        sentiment_value = event.payload.get("sentiment_value")
        direction = event.payload.get("sentiment_direction")
        confidence = event.payload.get("confidence")
        source = event.source
        
        # Validate essential data
        if not all([symbol, sentiment_value is not None, direction, confidence]):
            return
        
        # Skip if not interested in this symbol
        if self.symbols and symbol not in self.symbols:
            return
            
        # Process the sentiment data
        await self._process_sentiment_data(
            symbol=symbol,
            source=source,
            sentiment_value=sentiment_value,
            direction=direction,
            confidence=confidence,
            details=event.payload
        )
    
    async def _process_sentiment_data(
        self,
        symbol: str,
        source: str,
        sentiment_value: float,
        direction: str,
        confidence: float,
        details: Dict[str, Any]
    ) -> None:
        """Process incoming sentiment data.
        
        Args:
            symbol: The trading pair symbol
            source: The source of the sentiment data
            sentiment_value: The sentiment value (0.0-1.0)
            direction: The sentiment direction ("bullish", "bearish", "neutral")
            confidence: The confidence level (0.0-1.0)
            details: Additional details about the sentiment
        """
        # Initialize data structures if needed
        if symbol not in self.sentiment_data:
            self.sentiment_data[symbol] = {}
            
        # Extract the source type (first part of the source string)
        source_type = source.split('_')[0]
            
        # Store the sentiment data
        self.sentiment_data[symbol][source_type] = {
            "value": sentiment_value,
            "direction": direction,
            "confidence": confidence,
            "timestamp": utc_now(),
            "details": details
        }
        
        # Analyze sentiment for signal generation
        await self._analyze_sentiment(symbol)
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        Args:
            candle: The candle data to process
        """
        # Store the latest candle for each symbol
        self.latest_candles[candle.symbol] = candle
        
        # Re-analyze sentiment when we get new price data
        if candle.symbol in self.sentiment_data:
            await self._analyze_sentiment(candle.symbol)
    
    async def _analyze_sentiment(self, symbol: str) -> None:
        """Analyze sentiment data for a symbol to potentially generate signals.
        
        Args:
            symbol: The trading pair symbol
        """
        # Skip if we don't have sentiment data for this symbol
        if symbol not in self.sentiment_data or not self.sentiment_data[symbol]:
            return
            
        # Skip if we don't have price data
        if symbol not in self.latest_candles:
            return
            
        # Check if we've recently generated a signal
        now = utc_now()
        if symbol in self.last_signal_time:
            seconds_since_last_signal = (now - self.last_signal_time[symbol]).total_seconds()
            if seconds_since_last_signal < self.min_signal_interval:
                return
        
        # Calculate weighted sentiment
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for source_type, data in self.sentiment_data[symbol].items():
            # Skip sources with low confidence
            if data["confidence"] < self.min_confidence:
                continue
                
            # Skip old data (older than 24 hours)
            if (now - data["timestamp"]).total_seconds() > 86400:
                continue
                
            # Get weight for this source
            weight = self.source_weights.get(source_type, 1.0)
            
            # Adjust weight by confidence and recency
            age_hours = (now - data["timestamp"]).total_seconds() / 3600
            recency_factor = max(0.5, 1.0 - (age_hours / 24.0))  # Decay over 24 hours
            adjusted_weight = weight * data["confidence"] * recency_factor
            
            # Convert sentiment to -1 to 1 scale for easier weighted calculation
            # (0.5 is neutral, 0 is max bearish, 1 is max bullish)
            normalized_sentiment = (data["value"] - 0.5) * 2.0
            
            # Add to weighted calculation
            weighted_sentiment += normalized_sentiment * adjusted_weight
            total_weight += adjusted_weight
        
        # Skip if we don't have enough valid sources
        if total_weight == 0:
            return
            
        # Calculate final sentiment (-1 to 1 scale)
        final_sentiment = weighted_sentiment / total_weight
        
        # Convert back to 0-1 scale
        final_sentiment_value = (final_sentiment / 2.0) + 0.5
        
        # Calculate overall confidence
        overall_confidence = total_weight / sum(self.source_weights.values())
        
        # Check if sentiment is extreme
        is_extreme = (final_sentiment_value >= self.extreme_sentiment_threshold or 
                     final_sentiment_value <= (1 - self.extreme_sentiment_threshold))
        
        # Generate signals based on sentiment
        await self._generate_sentiment_signals(
            symbol=symbol,
            sentiment_value=final_sentiment_value,
            confidence=overall_confidence,
            is_extreme=is_extreme
        )
    
    async def _generate_sentiment_signals(
        self,
        symbol: str,
        sentiment_value: float,
        confidence: float,
        is_extreme: bool
    ) -> None:
        """Generate trading signals based on sentiment analysis.
        
        Args:
            symbol: The trading pair symbol
            sentiment_value: The final sentiment value (0.0-1.0)
            confidence: The overall confidence level
            is_extreme: Whether the sentiment is extreme
        """
        # Skip if confidence is too low
        if confidence < self.min_confidence:
            return
            
        # Get current price from latest candle
        current_price = self.latest_candles[symbol].close
        
        # Determine signal direction
        signal_direction = "no_action"
        signal_type = SignalType.NO_ACTION
        direction = "neutral"
        
        # Non-contrarian mode: Follow the sentiment
        if not self.contrarian_mode:
            if sentiment_value >= self.sentiment_threshold_bullish:
                signal_direction = "long"
                signal_type = SignalType.ENTRY
                direction = "bullish"
            elif sentiment_value <= self.sentiment_threshold_bearish:
                signal_direction = "short"
                signal_type = SignalType.ENTRY
                direction = "bearish"
                
        # Contrarian mode: Go against extreme sentiment
        else:
            if is_extreme:
                if sentiment_value >= self.extreme_sentiment_threshold:
                    signal_direction = "short"
                    signal_type = SignalType.ENTRY
                    direction = "bearish"
                elif sentiment_value <= (1 - self.extreme_sentiment_threshold):
                    signal_direction = "long"
                    signal_type = SignalType.ENTRY
                    direction = "bullish"
        
        # Skip if no actionable signal
        if signal_type == SignalType.NO_ACTION:
            return
            
        # Calculate stop loss and take profit levels
        stop_loss = None
        take_profit = None
        
        if signal_direction == "long":
            if self.use_stop_loss:
                stop_loss = current_price * (1 - self.stop_loss_pct)
            if self.use_take_profit:
                take_profit = current_price * (1 + self.take_profit_pct)
        elif signal_direction == "short":
            if self.use_stop_loss:
                stop_loss = current_price * (1 + self.stop_loss_pct)
            if self.use_take_profit:
                take_profit = current_price * (1 - self.take_profit_pct)
        
        # Check if we already have an active signal for this symbol
        if symbol in self.active_signals:
            existing_signal = self.active_signals[symbol]
            
            # Skip if existing signal has same direction
            if existing_signal.direction == signal_direction:
                return
                
            # Generate exit signal for the existing position
            await self.generate_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT,
                direction=existing_signal.direction,
                price=current_price,
                confidence=confidence,
                reason=f"Sentiment shift: {existing_signal.direction} to {signal_direction}",
                metadata={
                    "sentiment_value": sentiment_value,
                    "sentiment_direction": direction,
                    "is_extreme": is_extreme,
                    "contrarian_mode": self.contrarian_mode
                }
            )
        
        # Generate the new signal
        self.last_signal_time[symbol] = utc_now()
        
        await self.generate_signal(
            symbol=symbol,
            signal_type=signal_type,
            direction=signal_direction,
            price=current_price,
            confidence=confidence,
            reason=f"Sentiment signal: {direction} {'(extreme)' if is_extreme else ''}",
            take_profit=take_profit,
            stop_loss=stop_loss,
            metadata={
                "sentiment_value": sentiment_value,
                "sentiment_direction": direction,
                "is_extreme": is_extreme,
                "contrarian_mode": self.contrarian_mode
            }
        )
