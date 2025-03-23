"""Advanced sentiment-based trading strategy.

This module provides a sophisticated strategy that combines sentiment analysis
with market impact assessment and adaptive parameters based on market regime detection.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import numpy as np
import asyncio

from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.models.events import SentimentEvent, MarketRegimeEvent, MarketImpactEvent
from src.models.market_data import CandleData, TimeFrame, OrderBookData
from src.models.signals import Signal, SignalType
from src.strategy.enhanced_sentiment_strategy import EnhancedSentimentStrategy


class AdvancedSentimentStrategy(EnhancedSentimentStrategy):
    """Advanced strategy with market impact assessment and adaptive parameters.
    
    This strategy extends the enhanced sentiment strategy by adding:
    1. Market impact assessment to estimate how sentiment affects price
    2. Adaptive parameters based on detected market regime
    3. Multi-timeframe sentiment analysis
    4. Sentiment trend identification
    """
    
    def __init__(self, strategy_id: str = "advanced_sentiment"):
        """Initialize the advanced sentiment strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", "advanced_sentiment")
        
        # Market impact assessment
        self.use_market_impact = config.get(f"strategies.{strategy_id}.use_market_impact", True)
        self.impact_lookback_days = config.get(f"strategies.{strategy_id}.impact_lookback_days", 30)
        
        # Adaptive parameters
        self.use_adaptive_parameters = config.get(f"strategies.{strategy_id}.use_adaptive_parameters", True)
        self.adaptive_param_map = config.get(f"strategies.{strategy_id}.adaptive_param_map", {
            "bullish": {
                "sentiment_threshold_bullish": 0.65,  # Lower threshold in bullish regime
                "sentiment_threshold_bearish": 0.35,
                "stop_loss_pct": 0.04,  # Wider stop in bullish regime
                "take_profit_pct": 0.08  # Higher take profit in bullish regime
            },
            "bearish": {
                "sentiment_threshold_bullish": 0.75,  # Higher threshold in bearish regime
                "sentiment_threshold_bearish": 0.25,
                "stop_loss_pct": 0.025,  # Tighter stop in bearish regime 
                "take_profit_pct": 0.05   # Lower take profit in bearish regime
            },
            "volatile": {
                "sentiment_threshold_bullish": 0.70,
                "sentiment_threshold_bearish": 0.30,
                "stop_loss_pct": 0.05,     # Wider stop in volatile regime
                "take_profit_pct": 0.10    # Higher take profit in volatile regime
            },
            "ranging": {
                "sentiment_threshold_bullish": 0.72,
                "sentiment_threshold_bearish": 0.28,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06
            }
        })
        
        # Sentiment trend analysis
        self.use_sentiment_trend = config.get(f"strategies.{strategy_id}.use_sentiment_trend", True)
        self.trend_lookback_days = config.get(f"strategies.{strategy_id}.trend_lookback_days", 7)
        
        # Sentiment history for trend analysis
        self.sentiment_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Market impact data
        self.market_impact: Dict[str, Dict[str, Any]] = {}
        
        # Subscribe to market impact events if enabled
        if self.use_market_impact and "MarketImpactEvent" not in self.event_subscriptions:
            self.event_subscriptions.append("MarketImpactEvent")
        
        # Order book data for market impact
        self.order_book_data: Dict[str, OrderBookData] = {}

    async def _strategy_initialize(self) -> None:
        """Strategy-specific initialization."""
        await super()._strategy_initialize()
        
        self.logger.info("Initializing advanced sentiment strategy",
                      use_market_impact=self.use_market_impact,
                      use_adaptive_parameters=self.use_adaptive_parameters,
                      use_sentiment_trend=self.use_sentiment_trend)
    
    async def _strategy_start(self) -> None:
        """Strategy-specific startup."""
        await super()._strategy_start()
        
        self.logger.info("Starting advanced sentiment strategy")
        
        # Clear data
        self.sentiment_history = {}
        self.market_impact = {}
        
        # Set initial adaptive parameters if enabled
        if self.use_adaptive_parameters:
            await self._update_adaptive_parameters()
    
    async def _handle_event(self, event) -> None:
        """Handle events the strategy is subscribed to.
        
        This overrides the base implementation to add handling for MarketImpactEvent.
        
        Args:
            event: The event to handle
        """
        if not self.enabled:
            return
        
        event_type = event.__class__.__name__
        
        try:
            # Handle market impact events
            if event_type == "MarketImpactEvent":
                await self._handle_market_impact_event(event)
            else:
                # Use the parent class handler for other event types
                await super()._handle_event(event)
                
        except Exception as e:
            self.logger.error("Error handling event", 
                             event_type=event_type,
                             error=str(e))
    
    async def _handle_market_impact_event(self, event: MarketImpactEvent) -> None:
        """Handle a market impact event.
        
        Args:
            event: The market impact event
        """
        if not self.use_market_impact:
            return
            
        # Extract data from the event
        symbol = event.payload.get("symbol")
        impact_type = event.payload.get("impact_type")
        impact_value = event.payload.get("impact_value")
        confidence = event.payload.get("confidence")
        details = event.payload.get("details", {})
        
        # Validate essential data
        if not all([symbol, impact_type, impact_value is not None, confidence]):
            return
        
        # Skip if not interested in this symbol
        if self.symbols and symbol not in self.symbols:
            return
            
        # Store market impact information
        self.market_impact[symbol] = {
            "impact_type": impact_type,
            "impact_value": impact_value,
            "confidence": confidence,
            "timestamp": utc_now(),
            "details": details
        }
        
        # Re-analyze sentiment with the new market impact information
        if symbol in self.sentiment_data:
            await self._analyze_sentiment(symbol)
    
    async def _handle_market_regime_event(self, event: MarketRegimeEvent) -> None:
        """Handle a market regime event.
        
        Overrides the parent implementation to add adaptive parameters.
        
        Args:
            event: The market regime event
        """
        # Call parent implementation first
        await super()._handle_market_regime_event(event)
        
        # Update adaptive parameters based on new regime
        if self.use_adaptive_parameters:
            await self._update_adaptive_parameters()
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        Args:
            candle: The candle data to process
        """
        # Call parent implementation first
        await super().process_candle(candle)
        
        # Update sentiment history for trend analysis
        if self.use_sentiment_trend and candle.symbol in self.sentiment_data:
            await self._update_sentiment_history(candle.symbol)
    
    async def process_order_book(self, order_book: OrderBookData) -> None:
        """Process order book data.
        
        Args:
            order_book: The order book data to process
        """
        # Store the latest order book data
        self.order_book_data[order_book.symbol] = order_book
        
        # Re-analyze sentiment if we have new order book data and market impact analysis is enabled
        if self.use_market_impact and order_book.symbol in self.sentiment_data:
            await self._analyze_sentiment(order_book.symbol)
    
    async def _update_sentiment_history(self, symbol: str) -> None:
        """Update sentiment history for a symbol.
        
        Args:
            symbol: The trading pair symbol
        """
        if symbol not in self.sentiment_data or not self.sentiment_data[symbol]:
            return
            
        # Calculate current aggregated sentiment
        weighted_sentiment = 0.0
        total_weight = 0.0
        now = utc_now()
        
        for source_type, data in self.sentiment_data[symbol].items():
            # Skip sources with low confidence
            if data["confidence"] < self.min_confidence:
                continue
                
            # Skip old data (older than 24 hours)
            if (now - data["timestamp"]).total_seconds() > 86400:
                continue
                
            # Get weight for this source
            weight = self.source_weights.get(source_type, 1.0)
            
            # Adjust weight by confidence
            adjusted_weight = weight * data["confidence"]
            
            # Add to weighted calculation
            weighted_sentiment += data["value"] * adjusted_weight
            total_weight += adjusted_weight
        
        # Skip if we don't have enough valid sources
        if total_weight == 0:
            return
            
        # Calculate final sentiment
        final_sentiment_value = weighted_sentiment / total_weight
        
        # Initialize sentiment history for this symbol if needed
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []
            
        # Add current sentiment to history
        self.sentiment_history[symbol].append({
            "timestamp": now,
            "value": final_sentiment_value,
            "sources": list(self.sentiment_data[symbol].keys())
        })
        
        # Limit history to trend_lookback_days
        cutoff_time = now - timedelta(days=self.trend_lookback_days)
        self.sentiment_history[symbol] = [
            entry for entry in self.sentiment_history[symbol]
            if entry["timestamp"] > cutoff_time
        ]
    
    async def _update_adaptive_parameters(self) -> None:
        """Update strategy parameters based on current market regimes."""
        if not self.use_adaptive_parameters:
            return
            
        # Update parameters for each symbol
        for symbol in self.market_regimes:
            regime = self.market_regimes[symbol]["regime"]
            
            # Skip if we don't have adaptive parameters for this regime
            if regime not in self.adaptive_param_map:
                continue
                
            # Get the parameter map for this regime
            param_map = self.adaptive_param_map[regime]
            
            # Update each parameter
            for param, value in param_map.items():
                if hasattr(self, param):
                    old_value = getattr(self, param)
                    setattr(self, param, value)
                    self.logger.info(f"Adjusted parameter based on {regime} regime",
                                  symbol=symbol,
                                  parameter=param,
                                  old_value=old_value,
                                  new_value=value)
    
    def _analyze_sentiment_trend(self, symbol: str) -> Tuple[Optional[str], float]:
        """Analyze the trend in sentiment over time.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Tuple containing the trend direction ("rising", "falling", "flat")
            and the trend strength (0.0-1.0)
        """
        if symbol not in self.sentiment_history:
            return None, 0.0
            
        history = self.sentiment_history[symbol]
        
        # Need at least 3 data points for trend analysis
        if len(history) < 3:
            return None, 0.0
            
        # Sort by timestamp
        history = sorted(history, key=lambda x: x["timestamp"])
        
        # Extract values
        values = [entry["value"] for entry in history]
        
        # Calculate simple linear regression
        x = list(range(len(values)))
        y = values
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x_squared = sum(x_i ** 2 for x_i in x)
        sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
        
        # Calculate slope
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        except ZeroDivisionError:
            return None, 0.0
        
        # Calculate trend direction
        if slope > 0.01:
            trend_direction = "rising"
        elif slope < -0.01:
            trend_direction = "falling"
        else:
            trend_direction = "flat"
            
        # Calculate R-squared to determine trend strength
        y_mean = sum_y / n
        ss_total = sum((y_i - y_mean) ** 2 for y_i in y)
        
        if ss_total == 0:
            return trend_direction, 0.0
            
        y_pred = [x_i * slope + (sum_y - slope * sum_x) / n for x_i in x]
        ss_residual = sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(y, y_pred))
        
        r_squared = 1 - (ss_residual / ss_total)
        
        # Normalize to 0-1 scale
        trend_strength = min(1.0, max(0.0, r_squared))
        
        return trend_direction, trend_strength
    
    def _calculate_market_impact_factor(self, symbol: str) -> Optional[float]:
        """Calculate the market impact factor for a symbol.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Market impact factor (0.0-2.0, with 1.0 being neutral),
            or None if not enough data
        """
        if not self.use_market_impact:
            return None
            
        # Check if we have market impact data
        if symbol not in self.market_impact:
            return None
            
        # Check if we have order book data
        if symbol not in self.order_book_data:
            return None
            
        impact_data = self.market_impact[symbol]
        order_book = self.order_book_data[symbol]
        
        # Calculate liquidity imbalance
        bids_depth = sum(bid[1] for bid in order_book.bids[:5])
        asks_depth = sum(ask[1] for ask in order_book.asks[:5])
        
        liquidity_ratio = bids_depth / asks_depth if asks_depth > 0 else 1.0
        
        # Combine with impact data
        impact_value = impact_data["impact_value"]
        impact_confidence = impact_data["confidence"]
        
        # Calculate market impact factor (0.5 to 2.0, with 1.0 being neutral)
        if impact_value > 0:  # Positive impact
            market_impact_factor = 1.0 + (impact_value * impact_confidence * min(1.0, liquidity_ratio))
        else:  # Negative impact
            market_impact_factor = 1.0 / (1.0 + (abs(impact_value) * impact_confidence * (1.0 / max(0.01, liquidity_ratio))))
        
        # Bound between 0.5 and 2.0
        market_impact_factor = max(0.5, min(2.0, market_impact_factor))
        
        return market_impact_factor
    
    async def _analyze_sentiment(self, symbol: str) -> None:
        """Analyze sentiment data for a symbol to potentially generate signals.
        
        This overrides the parent implementation to add market impact assessment
        and sentiment trend analysis.
        
        Args:
            symbol: The trading pair symbol
        """
        # Call parent implementation first to calculate base sentiment
        await super()._analyze_sentiment(symbol)
        
        # Get the sentiment data after parent analysis
        if symbol not in self.sentiment_data or not self.sentiment_data[symbol]:
            return
            
        # Skip if we don't have price data
        if symbol not in self.latest_candles:
            return
            
        # Check if we have market impact data
        market_impact_factor = self._calculate_market_impact_factor(symbol)
        
        # Analyze sentiment trend if enabled
        sentiment_trend = None
        trend_strength = 0.0
        
        if self.use_sentiment_trend:
            sentiment_trend, trend_strength = self._analyze_sentiment_trend(symbol)
            
            # If we have a strong trend, log it
            if trend_strength > 0.7:
                self.logger.info("Detected strong sentiment trend",
                              symbol=symbol,
                              trend=sentiment_trend,
                              strength=trend_strength)
        
        # We don't need to re-generate signals here, as the parent implementation
        # will handle that. We've just enriched the data with market impact and trend
        # information that will be used when signals are generated.
    
    async def _generate_sentiment_signals(
        self,
        symbol: str,
        sentiment_value: float,
        confidence: float,
        is_extreme: bool
    ) -> None:
        """Generate trading signals based on sentiment analysis.
        
        This overrides the parent implementation to add market impact assessment
        and sentiment trend enhancement.
        
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
        
        # Get market impact factor if available
        market_impact_factor = self._calculate_market_impact_factor(symbol)
        
        # Analyze sentiment trend if enabled
        sentiment_trend = None
        trend_strength = 0.0
        
        if self.use_sentiment_trend:
            sentiment_trend, trend_strength = self._analyze_sentiment_trend(symbol)
        
        # Apply market impact to sentiment value if available
        adjusted_sentiment = sentiment_value
        if market_impact_factor is not None:
            # Adjust sentiment based on market impact (scales neutral point)
            if adjusted_sentiment > 0.5:  # Bullish sentiment
                adjusted_sentiment = 0.5 + (adjusted_sentiment - 0.5) * market_impact_factor
            else:  # Bearish sentiment
                adjusted_sentiment = 0.5 - (0.5 - adjusted_sentiment) * market_impact_factor
                
            # Log the adjustment if significant
            if abs(adjusted_sentiment - sentiment_value) > 0.05:
                self.logger.info("Adjusted sentiment based on market impact",
                              symbol=symbol,
                              original=sentiment_value,
                              adjusted=adjusted_sentiment,
                              impact_factor=market_impact_factor)
        
        # Apply sentiment trend factor if available
        if sentiment_trend and trend_strength > 0.5:
            trend_factor = 1.0 + (trend_strength * 0.2)  # Max 20% boost
            
            if sentiment_trend == "rising":
                # Boost bullish sentiment or reduce bearish sentiment
                if adjusted_sentiment > 0.5:
                    adjusted_sentiment = min(1.0, adjusted_sentiment * trend_factor)
                else:
                    adjusted_sentiment = 0.5 - (0.5 - adjusted_sentiment) / trend_factor
            elif sentiment_trend == "falling":
                # Boost bearish sentiment or reduce bullish sentiment
                if adjusted_sentiment < 0.5:
                    adjusted_sentiment = max(0.0, adjusted_sentiment / trend_factor)
                else:
                    adjusted_sentiment = 0.5 + (adjusted_sentiment - 0.5) / trend_factor
                    
            # Log the adjustment if significant
            if abs(adjusted_sentiment - sentiment_value) > 0.05:
                self.logger.info("Adjusted sentiment based on trend",
                              symbol=symbol,
                              after_impact=adjusted_sentiment,
                              final=adjusted_sentiment,
                              trend=sentiment_trend,
                              strength=trend_strength)
        
        # Determine base sentiment direction with adjusted sentiment
        if adjusted_sentiment >= self.sentiment_threshold_bullish:
            base_direction = "long"
            sentiment_direction = "bullish"
        elif adjusted_sentiment <= self.sentiment_threshold_bearish:
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
                              sentiment_value=adjusted_sentiment)
                return
                
            if base_direction == "short" and regime == "bullish":
                self.logger.info("Skipping bearish signal due to bullish market regime",
                              symbol=symbol,
                              sentiment_value=adjusted_sentiment)
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
                                  sentiment_value=adjusted_sentiment,
                                  rsi=rsi)
                    return
        
        # Calculate signal score
        sentiment_score = adjusted_sentiment if base_direction == "long" else (1 - adjusted_sentiment)
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
        
        # Include sentiment trend in score if available
        trend_factor = 1.0
        if sentiment_trend and trend_strength > 0.5:
            if (sentiment_trend == "rising" and base_direction == "long") or \
               (sentiment_trend == "falling" and base_direction == "short"):
                trend_factor = 1.0 + (trend_strength * 0.3)  # Up to 30% boost
                
        # Include market impact in score if available
        impact_factor = 1.0
        if market_impact_factor is not None:
            if base_direction == "long":
                impact_factor = market_impact_factor
            else:  # short
                impact_factor = 2.0 - market_impact_factor  # Invert for shorts
        
        # Calculate final score
        signal_score = sentiment_score * confidence_factor * regime_factor * technical_factor * trend_factor * impact_factor
        
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
                    "sentiment_value": adjusted_sentiment,
                    "original_sentiment": sentiment_value,
                    "sentiment_direction": sentiment_direction,
                    "is_extreme": is_extreme,
                    "signal_score": signal_score,
                    "contrarian_mode": self.contrarian_mode,
                    "sentiment_trend": sentiment_trend,
                    "trend_strength": trend_strength,
                    "market_impact_factor": market_impact_factor
                }
            )
        
        # Generate the new signal
        self.last_signal_time[symbol] = utc_now()
        
        # Build detailed reason string
        reason_parts = [f"Sentiment signal: {sentiment_direction}"]
        if is_extreme:
            reason_parts.append("(extreme)")
        if sentiment_trend and trend_strength > 0.5:
            reason_parts.append(f"[Trend: {sentiment_trend}, {trend_strength:.2f}]")
        if self.use_technical_confirmation:
            reason_parts.append(f"[RSI: {self.technical_indicators[symbol]['rsi']:.1f}]")
        if self.use_market_regime and symbol in self.market_regimes:
            reason_parts.append(f"[Regime: {self.market_regimes[symbol]['regime']}]")
        if market_impact_factor is not None:
            reason_parts.append(f"[Impact: {market_impact_factor:.2f}]")
        
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
                "sentiment_value": adjusted_sentiment,
                "original_sentiment": sentiment_value,
                "sentiment_direction": sentiment_direction,
                "is_extreme": is_extreme,
                "signal_score": signal_score,
                "contrarian_mode": self.contrarian_mode,
                "technical_aligned": technical_aligned,
                "regime": self.market_regimes.get(symbol, {}).get("regime", "unknown") if self.use_market_regime else None,
                "sentiment_trend": sentiment_trend,
                "trend_strength": trend_strength,
                "market_impact_factor": market_impact_factor
            }
        )