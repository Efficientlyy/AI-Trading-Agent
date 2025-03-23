"""Enhanced sentiment-based trading strategy.

This module provides an advanced strategy that combines sentiment analysis
with technical indicators and market regime detection to improve signal quality.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Set

from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.models.events import SentimentEvent, MarketRegimeEvent
from src.models.market_data import CandleData, TimeFrame
from src.models.signals import Signal, SignalType
from src.strategy.sentiment_strategy import SentimentStrategy


class EnhancedSentimentStrategy(SentimentStrategy):
    """Enhanced strategy that combines sentiment with technical indicators.
    
    This strategy extends the basic sentiment strategy by incorporating
    technical indicators and market regime detection to improve signal quality.
    """
    
    def __init__(self, strategy_id: str = "enhanced_sentiment"):
        """Initialize the enhanced sentiment strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", "enhanced_sentiment")
        
        # Additional configuration
        self.use_market_regime = config.get(f"strategies.{strategy_id}.use_market_regime", True)
        self.use_technical_confirmation = config.get(f"strategies.{strategy_id}.use_technical_confirmation", True)
        self.min_signal_score = config.get(f"strategies.{strategy_id}.min_signal_score", 0.7)
        
        # Technical indicators for confirmation
        self.rsi_period = config.get(f"strategies.{strategy_id}.rsi_period", 14)
        self.rsi_overbought = config.get(f"strategies.{strategy_id}.rsi_overbought", 70)
        self.rsi_oversold = config.get(f"strategies.{strategy_id}.rsi_oversold", 30)
        
        # Market regime detection
        self.regime_lookback = config.get(f"strategies.{strategy_id}.regime_lookback", 20)
        
        # Technical indicator values
        self.technical_indicators: Dict[str, Dict[str, Any]] = {}
        
        # Market regime data
        self.market_regimes: Dict[str, Dict[str, Any]] = {}
        
        # Add MarketRegimeEvent to subscriptions if not already included
        if "MarketRegimeEvent" not in self.event_subscriptions:
            self.event_subscriptions.append("MarketRegimeEvent")
    
    async def _strategy_initialize(self) -> None:
        """Strategy-specific initialization."""
        await super()._strategy_initialize()
        
        self.logger.info("Initializing enhanced sentiment strategy",
                      use_market_regime=self.use_market_regime,
                      use_technical_confirmation=self.use_technical_confirmation)
    
    async def _strategy_start(self) -> None:
        """Strategy-specific startup."""
        await super()._strategy_start()
        
        self.logger.info("Starting enhanced sentiment strategy")
        
        # Clear data
        self.technical_indicators = {}
        self.market_regimes = {}
    
    async def _handle_event(self, event) -> None:
        """Handle events the strategy is subscribed to.
        
        This overrides the base implementation to add handling for MarketRegimeEvent.
        
        Args:
            event: The event to handle
        """
        if not self.enabled:
            return
        
        event_type = event.__class__.__name__
        
        try:
            # Handle market regime events
            if event_type == "MarketRegimeEvent":
                await self._handle_market_regime_event(event)
            else:
                # Use the parent class handler for other event types
                await super()._handle_event(event)
                
        except Exception as e:
            self.logger.error("Error handling event", 
                             event_type=event_type,
                             error=str(e))
    
    async def _handle_market_regime_event(self, event: MarketRegimeEvent) -> None:
        """Handle a market regime event.
        
        Args:
            event: The market regime event
        """
        # Extract data from the event
        symbol = event.payload.get("symbol")
        regime = event.payload.get("regime")
        
        # Validate essential data
        if not all([symbol, regime]):
            return
        
        # Skip if not interested in this symbol
        if self.symbols and symbol not in self.symbols:
            return
            
        # Store market regime information
        self.market_regimes[symbol] = {
            "regime": regime,
            "timestamp": utc_now(),
            "details": event.payload
        }
        
        # Re-analyze sentiment with the new regime information
        if symbol in self.sentiment_data:
            await self._analyze_sentiment(symbol)
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        Args:
            candle: The candle data to process
        """
        # Call the parent implementation first
        await super().process_candle(candle)
        
        if not self.enabled or not self.use_technical_confirmation:
            return
            
        # Update technical indicators
        await self._update_technical_indicators(candle)
    
    async def _update_technical_indicators(self, candle: CandleData) -> None:
        """Update technical indicators for a symbol.
        
        Args:
            candle: The candle data to process
        """
        symbol = candle.symbol
        
        # Initialize technical indicators for this symbol if needed
        if symbol not in self.technical_indicators:
            self.technical_indicators[symbol] = {
                "candles": [],
                "rsi": None
            }
            
        # Add the candle to the list
        self.technical_indicators[symbol]["candles"].append(candle)
        
        # Keep only the most recent candles
        max_lookback = max(self.rsi_period, self.regime_lookback) + 10  # Add some buffer
        if len(self.technical_indicators[symbol]["candles"]) > max_lookback:
            self.technical_indicators[symbol]["candles"] = self.technical_indicators[symbol]["candles"][-max_lookback:]
            
        # Calculate RSI
        if len(self.technical_indicators[symbol]["candles"]) >= self.rsi_period:
            self.technical_indicators[symbol]["rsi"] = self._calculate_rsi(
                symbol, self.rsi_period)
    
    def _calculate_rsi(self, symbol: str, period: int) -> Optional[float]:
        """Calculate the RSI for a symbol.
        
        Args:
            symbol: The trading pair symbol
            period: The RSI period
            
        Returns:
            The RSI value, or None if not enough data
        """
        # Get the candles for this symbol
        candles = self.technical_indicators[symbol]["candles"]
        
        if len(candles) < period + 1:
            return None
            
        # Get the most recent 'period + 1' candles
        recent_candles = candles[-(period + 1):]
        
        # Calculate price changes
        changes = [recent_candles[i].close - recent_candles[i-1].close 
                 for i in range(1, len(recent_candles))]
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in changes]
        losses = [abs(change) if change < 0 else 0 for change in changes]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # Calculate RS and RSI
        if avg_loss == 0:
            return 100  # No losses, RSI is 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _generate_sentiment_signals(
        self,
        symbol: str,
        sentiment_value: float,
        confidence: float,
        is_extreme: bool
    ) -> None:
        """Generate trading signals based on sentiment analysis.
        
        This overrides the base implementation to add technical confirmation
        and market regime checks.
        
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
        
        # Determine base sentiment direction
        if sentiment_value >= self.sentiment_threshold_bullish:
            base_direction = "long"
            sentiment_direction = "bullish"
        elif sentiment_value <= self.sentiment_threshold_bearish:
            base_direction = "short"
            sentiment_direction = "bearish"
        else:
            return  # Neutral sentiment, no signal
            
        # Apply contrarian logic if enabled and sentiment is extreme
        if self.contrarian_mode and is_extreme:
            # Invert the direction
            base_direction = "short" if base_direction == "long" else "long"
            sentiment_direction = "bearish (contrarian)" if sentiment_direction == "bullish" else "bullish (contrarian)"
        
        # Check market regime if enabled
        if self.use_market_regime and symbol in self.market_regimes:
            regime = self.market_regimes[symbol]["regime"]
            
            # Only generate signals that align with the market regime
            if base_direction == "long" and regime == "bearish":
                self.logger.info("Skipping bullish signal due to bearish market regime",
                              symbol=symbol,
                              sentiment_value=sentiment_value)
                return
                
            if base_direction == "short" and regime == "bullish":
                self.logger.info("Skipping bearish signal due to bullish market regime",
                              symbol=symbol,
                              sentiment_value=sentiment_value)
                return
        
        # Check technical confirmation if enabled
        technical_aligned = True
        if self.use_technical_confirmation and symbol in self.technical_indicators:
            rsi = self.technical_indicators[symbol]["rsi"]
            
            if rsi is not None:
                # For long signals, check if RSI is not overbought
                if base_direction == "long" and rsi > self.rsi_overbought:
                    technical_aligned = False
                
                # For short signals, check if RSI is not oversold
                if base_direction == "short" and rsi < self.rsi_oversold:
                    technical_aligned = False
                
                if not technical_aligned:
                    self.logger.info("Skipping signal due to technical indicator disagreement",
                                  symbol=symbol,
                                  sentiment_value=sentiment_value,
                                  rsi=rsi)
                    return
        
        # Calculate signal score
        sentiment_score = sentiment_value if base_direction == "long" else (1 - sentiment_value)
        confidence_factor = confidence
        
        # Include market regime in score if enabled
        regime_factor = 1.0
        if self.use_market_regime and symbol in self.market_regimes:
            regime = self.market_regimes[symbol]["regime"]
            if regime == "bullish" and base_direction == "long":
                regime_factor = 1.2  # Boost score for aligned regime
            elif regime == "bearish" and base_direction == "short":
                regime_factor = 1.2  # Boost score for aligned regime
        
        # Include technical alignment in score
        technical_factor = 1.2 if technical_aligned else 0.8
        
        # Calculate final score
        signal_score = sentiment_score * confidence_factor * regime_factor * technical_factor
        
        # Check if score is high enough
        if signal_score < self.min_signal_score:
            self.logger.info("Skipping signal due to low signal score",
                          symbol=symbol,
                          signal_score=signal_score,
                          min_score=self.min_signal_score)
            return
        
        # Determine signal direction and type
        if base_direction == "long":
            signal_type = SignalType.ENTRY
            direction = "long"
        else:  # short
            signal_type = SignalType.ENTRY
            direction = "short"
        
        # Calculate stop loss and take profit levels
        stop_loss = None
        take_profit = None
        
        if direction == "long":
            if self.use_stop_loss:
                stop_loss = current_price * (1 - self.stop_loss_pct)
            if self.use_take_profit:
                take_profit = current_price * (1 + self.take_profit_pct)
        elif direction == "short":
            if self.use_stop_loss:
                stop_loss = current_price * (1 + self.stop_loss_pct)
            if self.use_take_profit:
                take_profit = current_price * (1 - self.take_profit_pct)
        
        # Check if we already have an active signal for this symbol
        if symbol in self.active_signals:
            existing_signal = self.active_signals[symbol]
            
            # Skip if existing signal has same direction
            if existing_signal.direction == direction:
                return
                
            # Generate exit signal for the existing position
            await self.generate_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT,
                direction=existing_signal.direction,
                price=current_price,
                confidence=confidence,
                reason=f"Sentiment shift: {existing_signal.direction} to {direction}",
                metadata={
                    "sentiment_value": sentiment_value,
                    "sentiment_direction": sentiment_direction,
                    "is_extreme": is_extreme,
                    "signal_score": signal_score,
                    "contrarian_mode": self.contrarian_mode
                }
            )
        
        # Generate the new signal
        self.last_signal_time[symbol] = utc_now()
        
        # Add technical and regime info to reason
        reason_parts = [f"Sentiment signal: {sentiment_direction}"]
        if is_extreme:
            reason_parts.append("(extreme)")
        if self.use_technical_confirmation:
            reason_parts.append(f"[RSI: {self.technical_indicators[symbol]['rsi']:.1f}]")
        if self.use_market_regime and symbol in self.market_regimes:
            reason_parts.append(f"[Regime: {self.market_regimes[symbol]['regime']}]")
        
        reason = " ".join(reason_parts)
        
        await self.generate_signal(
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            price=current_price,
            confidence=confidence,
            reason=reason,
            take_profit=take_profit,
            stop_loss=stop_loss,
            metadata={
                "sentiment_value": sentiment_value,
                "sentiment_direction": sentiment_direction,
                "is_extreme": is_extreme,
                "signal_score": signal_score,
                "contrarian_mode": self.contrarian_mode,
                "technical_aligned": technical_aligned,
                "regime": self.market_regimes.get(symbol, {}).get("regime", "unknown") if self.use_market_regime else None
            }
        )