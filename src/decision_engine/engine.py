"""Decision engine for the AI Crypto Trading System.

This module provides the central decision-making component that aggregates
predictions from analysis agents and generates trading decisions.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any

from src.common.component import Component
from src.common.config import config
from src.common.datetime_utils import utc_now
from src.common.events import event_bus
from src.common.logging import get_logger
from src.decision_engine.models import (
    Prediction, AggregatedPrediction, TradingDecision,
    Direction, SignalType, PredictionSource, RiskLevel
)
from src.decision_engine.prediction_aggregator import PredictionAggregator
from src.models.events import SystemStatusEvent, SentimentEvent
from src.models.analysis_events import (
    SignalEvent, TechnicalIndicatorEvent, PatternEvent, CandleDataEvent
)
from src.models.market_data import TimeFrame
from src.models.signals import Signal, SignalType as OutputSignalType


class DecisionEngine(Component):
    """Central decision-making engine.
    
    This component aggregates predictions from analysis agents and generates
    trading decisions based on risk management rules and position sizing.
    """
    
    def __init__(self):
        """Initialize the decision engine."""
        super().__init__("decision_engine")
        self.logger = get_logger("decision_engine", "engine")
        
        # Create prediction aggregator
        self.prediction_aggregator = PredictionAggregator()
        
        # Configuration
        self.min_confidence = config.get("decision_engine.min_confidence", 0.85)
        self.min_reward_risk_ratio = config.get("decision_engine.min_reward_risk_ratio", 2.0)
        self.required_agent_types = set(config.get(
            "decision_engine.required_agent_types", 
            ["technical", "pattern"]
        ))
        
        # Risk management
        self.kelly_fraction = config.get("decision_engine.kelly_fraction", 0.4)
        self.max_per_trade_risk = config.get("decision_engine.max_per_trade_risk", 0.01)  # 1% max risk per trade
        self.max_per_symbol_exposure = config.get("decision_engine.max_per_symbol_exposure", 0.05)  # 5% per symbol
        self.max_correlated_exposure = config.get("decision_engine.max_correlated_exposure", 0.15)  # 15% correlated
        
        # Track active trading decisions
        self.active_decisions: Dict[str, TradingDecision] = {}
        
        # Performance tracking
        self.historical_win_rates: Dict[Tuple[str, Direction], float] = {}
        
        # Event mapping from analysis agents
        self.event_to_agent = {
            "TechnicalIndicatorEvent": "technical_indicators",
            "PatternEvent": "pattern_recognition",
            "SentimentEvent": "sentiment_analysis"
        }
        
        # Keep track of monitored symbols
        self.monitored_symbols: Set[str] = set()
        
        # Cached market data
        self.current_prices: Dict[str, float] = {}
    
    async def _initialize(self) -> None:
        """Initialize the decision engine."""
        self.logger.info("Initializing decision engine")
        
        # Load symbols to monitor from config
        self.monitored_symbols = set(config.get("decision_engine.monitored_symbols", []))
        
        # Load historical win rates if available
        # In a real system, this would load from a database
        
        self.logger.info("Decision engine initialized", 
                       monitored_symbols=list(self.monitored_symbols))
    
    async def _start(self) -> None:
        """Start the decision engine."""
        self.logger.info("Starting decision engine")
        
        # Register event handlers
        event_bus.subscribe("TechnicalIndicatorEvent", self._handle_technical_event)
        event_bus.subscribe("PatternEvent", self._handle_pattern_event)
        event_bus.subscribe("SentimentEvent", self._handle_sentiment_event)
        event_bus.subscribe("CandleDataEvent", self._handle_candle_event)
        
        # Start periodic tasks
        self.create_task(self._process_predictions_periodically())
        self.create_task(self._clean_expired_data_periodically())
        
        self.logger.info("Decision engine started")
        await self.publish_status("running", "Decision engine started")
    
    async def _stop(self) -> None:
        """Stop the decision engine."""
        self.logger.info("Stopping decision engine")
        
        # Unregister event handlers
        event_bus.unsubscribe("TechnicalIndicatorEvent", self._handle_technical_event)
        event_bus.unsubscribe("PatternEvent", self._handle_pattern_event)
        event_bus.unsubscribe("SentimentEvent", self._handle_sentiment_event)
        event_bus.unsubscribe("CandleDataEvent", self._handle_candle_event)
        
        self.logger.info("Decision engine stopped")
        await self.publish_status("stopped", "Decision engine stopped")
    
    async def _handle_technical_event(self, event: TechnicalIndicatorEvent) -> None:
        """Handle a technical indicator event.
        
        Args:
            event: The technical indicator event
        """
        # Convert to prediction and add to aggregator
        prediction = self._convert_technical_event_to_prediction(event)
        if prediction:
            self.prediction_aggregator.add_prediction(prediction)
    
    async def _handle_pattern_event(self, event: PatternEvent) -> None:
        """Handle a pattern event.
        
        Args:
            event: The pattern event
        """
        # Convert to prediction and add to aggregator
        prediction = self._convert_pattern_event_to_prediction(event)
        if prediction:
            self.prediction_aggregator.add_prediction(prediction)
    
    async def _handle_sentiment_event(self, event: SentimentEvent) -> None:
        """Handle a sentiment event.
        
        Args:
            event: The sentiment event
        """
        # Convert to prediction and add to aggregator
        prediction = self._convert_sentiment_event_to_prediction(event)
        if prediction:
            self.prediction_aggregator.add_prediction(prediction)
    
    async def _handle_candle_event(self, event: 'CandleDataEvent') -> None:
        """Handle a candle data event to update price data.
        
        Args:
            event: The candle data event
        """
        # Extract data from payload
        payload = event.payload
        symbol = payload.get("symbol")
        close = payload.get("close")
        
        if symbol and close is not None:
            # Update current price
            self.current_prices[symbol] = close
    
    def _convert_technical_event_to_prediction(self, event: TechnicalIndicatorEvent) -> Optional[Prediction]:
        """Convert a technical indicator event to a prediction.
        
        Args:
            event: The technical indicator event
            
        Returns:
            A prediction, or None if conversion not possible
        """
        # This is a simplified example
        # In a real system, this would analyze the indicator values to determine
        # direction, confidence, etc.
        
        # Extract data from payload
        payload = event.payload
        symbol = payload.get("symbol")
        indicator_name = payload.get("indicator_name")
        values = payload.get("values", {})
        timeframe = payload.get("timeframe")
        
        # Check if required fields are present
        if not symbol or not indicator_name or not timeframe:
            self.logger.debug("Missing required fields in technical event")
            return None
        
        # For now, just extract the last indicator value
        if not values:
            return None
        
        # Get the most recent timestamp - convert timestamp strings to datetime if needed
        timestamp_keys = []
        for ts in values.keys():
            if isinstance(ts, str):
                try:
                    timestamp_keys.append(datetime.fromisoformat(ts))
                except ValueError:
                    continue
            else:
                timestamp_keys.append(ts)
                
        if not timestamp_keys:
            return None
            
        latest_timestamp = max(timestamp_keys)
        indicator_values = values.get(str(latest_timestamp), values.get(latest_timestamp))
        
        # Determine direction and confidence based on indicator type
        direction = Direction.NEUTRAL
        confidence = 0.5
        entry_price = None
        stop_loss = None
        take_profit = None
        signal_type = SignalType.NO_ACTION
        rationale = ""
        
        if indicator_name == "RSI":
            # RSI indicator logic
            rsi_value = float(values) if isinstance(values, (int, float)) else 0.0
            
            if rsi_value < 30:  # Oversold
                direction = Direction.BULLISH
                confidence = (30 - rsi_value) / 30 * 0.5 + 0.3  # Confidence increases as RSI decreases
                signal_type = SignalType.ENTRY
                rationale = f"RSI oversold at {rsi_value:.2f}"
            elif rsi_value > 70:  # Overbought
                direction = Direction.BEARISH
                confidence = (rsi_value - 70) / 30 * 0.5 + 0.3  # Confidence increases as RSI increases
                signal_type = SignalType.ENTRY
                rationale = f"RSI overbought at {rsi_value:.2f}"
        
        elif indicator_name == "MACD":
            # MACD indicator logic
            if isinstance(indicator_values, dict) and "macd" in indicator_values and "signal" in indicator_values:
                macd = indicator_values["macd"]
                signal = indicator_values["signal"]
                histogram = indicator_values.get("histogram", macd - signal)
                
                # MACD crossing above signal line = bullish
                if macd > signal and histogram > 0:
                    direction = Direction.BULLISH
                    confidence = min(0.8, abs(histogram) * 10)
                    signal_type = SignalType.ENTRY
                    rationale = f"MACD ({macd:.4f}) crossed above signal ({signal:.4f})"
                # MACD crossing below signal line = bearish
                elif macd < signal and histogram < 0:
                    direction = Direction.BEARISH
                    confidence = min(0.8, abs(histogram) * 10)
                    signal_type = SignalType.ENTRY
                    rationale = f"MACD ({macd:.4f}) crossed below signal ({signal:.4f})"
        
        # If no clear signal or low confidence, return None
        if direction == Direction.NEUTRAL or confidence < self.min_confidence or signal_type == SignalType.NO_ACTION:
            return None
        
        # Create prediction
        return Prediction(
            id=f"tech_{uuid.uuid4().hex[:8]}",
            source=PredictionSource.TECHNICAL,
            agent_id=self.event_to_agent.get("TechnicalIndicatorEvent", "technical"),
            symbol=symbol,
            timestamp=event.timestamp,
            direction=direction,
            confidence=confidence,
            timeframe=str(timeframe),  # Ensure string type
            signal_type=signal_type,
            entry_price=self.current_prices.get(symbol, None),  # Use None as fallback
            stop_loss=stop_loss,
            take_profit=take_profit,
            rationale=rationale
        )
    
    def _convert_pattern_event_to_prediction(self, event: PatternEvent) -> Optional[Prediction]:
        """Convert a pattern event to a prediction.
        
        Args:
            event: The pattern event
            
        Returns:
            A prediction, or None if conversion not possible
        """
        # Extract data from payload
        payload = event.payload
        pattern_name = payload.get("pattern_name", "")
        symbol = payload.get("symbol")
        timeframe = payload.get("timeframe")
        confidence = payload.get("confidence", 0.0)
        target_price = payload.get("target_price")
        invalidation_price = payload.get("invalidation_price")
        
        # Determine direction based on pattern name
        direction = Direction.NEUTRAL
        signal_type = SignalType.NO_ACTION
        
        # Bullish patterns
        bullish_patterns = [
            "InverseHeadAndShoulders", "DoubleBottom", "BullishEngulfing", 
            "Hammer", "BullishFlag", "BullishPennant", "BullishTriangle"
        ]
        
        # Bearish patterns
        bearish_patterns = [
            "HeadAndShoulders", "DoubleTop", "BearishEngulfing", 
            "ShootingStar", "BearishFlag", "BearishPennant", "BearishTriangle"
        ]
        
        if any(bull_pattern in pattern_name for bull_pattern in bullish_patterns):
            direction = Direction.BULLISH
            signal_type = SignalType.ENTRY
        elif any(bear_pattern in pattern_name for bear_pattern in bearish_patterns):
            direction = Direction.BEARISH
            signal_type = SignalType.ENTRY
        
        # If no clear direction or confidence too low, return None
        if direction == Direction.NEUTRAL or confidence < self.min_confidence:
            return None
        
        # Check if required fields are present
        if not symbol or not timeframe:
            self.logger.debug("Missing required fields in pattern event")
            return None
            
        # Create prediction
        return Prediction(
            id=f"pat_{uuid.uuid4().hex[:8]}",
            source=PredictionSource.PATTERN,
            agent_id=self.event_to_agent.get("PatternEvent", "pattern"),
            symbol=symbol,
            timestamp=event.timestamp,
            direction=direction,
            confidence=confidence,
            timeframe=str(timeframe),  # Ensure string type
            signal_type=signal_type,
            entry_price=self.current_prices.get(symbol, None),  # Use None as fallback
            stop_loss=invalidation_price,
            take_profit=target_price,
            rationale=f"Detected {pattern_name} pattern with {confidence:.2f} confidence"
        )
    
    def _convert_sentiment_event_to_prediction(self, event: SentimentEvent) -> Optional[Prediction]:
        """Convert a sentiment event to a prediction.
        
        Args:
            event: The sentiment event
            
        Returns:
            A prediction, or None if conversion not possible
        """
        # Extract details from payload
        details = event.payload
        symbol = details.get("symbol", event.source)
        confidence = details.get("confidence", 0.5)
        
        # Determine direction
        direction = Direction.NEUTRAL
        if details.get("sentiment_direction") == "bullish":
            direction = Direction.BULLISH
        elif details.get("sentiment_direction") == "bearish":
            direction = Direction.BEARISH
        
        # Determine signal type
        signal_type = SignalType.NO_ACTION
        if confidence >= self.min_confidence:
            signal_type = SignalType.ENTRY
        
        # If sentiment is extreme, it might be contrarian
        is_extreme = details.get("is_extreme", False)
        if is_extreme and details.get("signal_type") == "contrarian":
            # Reverse the direction for contrarian signals
            if direction == Direction.BULLISH:
                direction = Direction.BEARISH
            elif direction == Direction.BEARISH:
                direction = Direction.BULLISH
        
        # If no clear direction or low confidence, return None
        if direction == Direction.NEUTRAL or confidence < self.min_confidence / 1.2:  # Lower threshold for sentiment
            return None
        
        # Create prediction
        return Prediction(
            id=f"sent_{uuid.uuid4().hex[:8]}",
            source=PredictionSource.SENTIMENT,
            agent_id=self.event_to_agent.get("SentimentEvent", "sentiment"),
            symbol=symbol,
            timestamp=event.timestamp,
            direction=direction,
            confidence=confidence,
            timeframe="4h",  # Default timeframe for sentiment
            signal_type=signal_type,
            entry_price=self.current_prices.get(symbol),
            rationale=f"Market sentiment is {details.get('sentiment_direction')} with {confidence:.2f} confidence"
        )
    
    async def _process_predictions_periodically(self) -> None:
        """Process predictions periodically to generate trading decisions."""
        try:
            while True:
                # Process each monitored symbol
                for symbol in self.monitored_symbols:
                    await self._process_symbol_predictions(symbol)
                
                # Process any symbols that have predictions but aren't explicitly monitored
                for symbol in list(self.prediction_aggregator.active_predictions.keys()):
                    if symbol not in self.monitored_symbols:
                        await self._process_symbol_predictions(symbol)
                
                # Sleep for a short interval
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except asyncio.CancelledError:
            self.logger.debug("Prediction processing task cancelled")
            raise
        except Exception as e:
            self.logger.exception("Error in prediction processing task", error=str(e))
            await asyncio.sleep(60)  # Sleep longer on error
    
    async def _clean_expired_data_periodically(self) -> None:
        """Clean expired predictions and decisions periodically."""
        try:
            while True:
                # Clear expired predictions
                self.prediction_aggregator.clear_expired_predictions()
                
                # Clear expired decisions
                now = utc_now()
                for decision_id in list(self.active_decisions.keys()):
                    decision = self.active_decisions[decision_id]
                    if decision.entry_valid_until and now > decision.entry_valid_until:
                        del self.active_decisions[decision_id]
                
                # Sleep for a longer interval
                await asyncio.sleep(300)  # Run every 5 minutes
                
        except asyncio.CancelledError:
            self.logger.debug("Data cleaning task cancelled")
            raise
        except Exception as e:
            self.logger.exception("Error in data cleaning task", error=str(e))
            await asyncio.sleep(300)  # Sleep on error
    
    async def _process_symbol_predictions(self, symbol: str) -> None:
        """Process predictions for a symbol to generate trading decisions.
        
        Args:
            symbol: The trading pair symbol
        """
        # Skip if no price data available
        if symbol not in self.current_prices:
            return
            
        # Aggregate predictions
        aggregated = self.prediction_aggregator.aggregate_predictions(symbol)
        if not aggregated:
            return
            
        # Check if this is a high-confidence aggregate prediction
        if not aggregated.is_high_confidence(self.min_confidence):
            self.logger.debug("Low confidence aggregated prediction", 
                           confidence=aggregated.confidence,
                           threshold=self.min_confidence)
            return
            
        # Check reward-risk ratio if targets are available
        reward_risk = aggregated.get_reward_risk_ratio()
        if reward_risk is not None and reward_risk < self.min_reward_risk_ratio:
            self.logger.debug("Insufficient reward-risk ratio", 
                           ratio=reward_risk,
                           threshold=self.min_reward_risk_ratio)
            return
            
        # Check if required agent types are represented
        prediction_sources = set(p.source.value for p in aggregated.predictions)
        if not self.required_agent_types.issubset(prediction_sources):
            missing = self.required_agent_types - prediction_sources
            self.logger.debug("Missing required agent types", 
                           missing=list(missing))
            return
            
        # Generate trading decision
        decision = self._generate_trading_decision(aggregated)
        if not decision:
            return
            
        # Store the decision
        self.active_decisions[decision.id] = decision
        
        # Publish trading signal
        await self._publish_trading_signal(decision)
        
        self.logger.info("Generated trading decision", 
                       decision_id=decision.id,
                       symbol=symbol,
                       direction=decision.direction.value,
                       confidence=decision.confidence,
                       position_size=decision.position_size)
    
    def _generate_trading_decision(self, prediction: AggregatedPrediction) -> Optional[TradingDecision]:
        """Generate a trading decision from an aggregated prediction.
        
        Args:
            prediction: The aggregated prediction
            
        Returns:
            A trading decision, or None if not possible
        """
        # Check if entry is possible
        if prediction.entry_price is None:
            prediction.entry_price = self.current_prices.get(prediction.symbol)
            
        if prediction.entry_price is None:
            self.logger.warning("No entry price available", symbol=prediction.symbol)
            return None
            
        # Calculate position size using Kelly Criterion
        position_size = self._calculate_position_size(prediction)
        if position_size <= 0:
            self.logger.debug("Position size too small", position_size=position_size)
            return None
            
        # Determine risk level
        risk_level = self._determine_risk_level(prediction)
        
        # Set expiration for entry
        entry_valid_minutes = 60  # Default 1 hour validity
        if prediction.timeframe == "1m":
            entry_valid_minutes = 5
        elif prediction.timeframe == "5m":
            entry_valid_minutes = 15
        elif prediction.timeframe == "15m":
            entry_valid_minutes = 30
        elif prediction.timeframe == "1h":
            entry_valid_minutes = 60
        elif prediction.timeframe == "4h":
            entry_valid_minutes = 240
        elif prediction.timeframe == "1d":
            entry_valid_minutes = 1440
        
        entry_valid_until = utc_now() + timedelta(minutes=entry_valid_minutes)
        
        # Create trading decision
        return TradingDecision(
            id=f"dec_{uuid.uuid4().hex[:8]}",
            symbol=prediction.symbol,
            timestamp=utc_now(),
            decision_type=prediction.signal_type,
            direction=prediction.direction,
            confidence=prediction.confidence,
            risk_level=risk_level,
            position_size=position_size,
            entry_price=prediction.entry_price,
            entry_valid_until=entry_valid_until,
            stop_loss=prediction.stop_loss,
            source_predictions=[p.id for p in prediction.predictions],
            aggregated_prediction=prediction.id,
            rationale=prediction.rationale,
            tags=["auto_generated"] + prediction.tags
        )
    
    def _calculate_position_size(self, prediction: AggregatedPrediction) -> float:
        """Calculate position size using Kelly Criterion.
        
        Args:
            prediction: The aggregated prediction
            
        Returns:
            Position size as a percentage of available capital
        """
        # Get the win rate for this symbol and direction
        symbol_direction_key = (prediction.symbol, prediction.direction)
        win_rate = self.historical_win_rates.get(symbol_direction_key, 0.6)  # Default 60% win rate
        
        # Apply confidence adjustment to win rate
        adjusted_win_rate = win_rate * (prediction.confidence / 0.9)  # Scale by confidence
        
        # Calculate reward-risk ratio
        reward_risk = prediction.get_reward_risk_ratio()
        if reward_risk is None:
            reward_risk = 2.0  # Default 2:1 reward:risk
        
        # Apply Kelly formula with fractional Kelly to be more conservative
        kelly_percentage = (adjusted_win_rate - ((1 - adjusted_win_rate) / reward_risk)) * self.kelly_fraction
        
        # Apply risk limits
        position_size = min(kelly_percentage, self.max_per_trade_risk)
        
        # Convert negative results to zero
        return max(0, position_size)
    
    def _determine_risk_level(self, prediction: AggregatedPrediction) -> RiskLevel:
        """Determine risk level based on prediction characteristics.
        
        Args:
            prediction: The aggregated prediction
            
        Returns:
            Risk level
        """
        # Calculate a risk score based on multiple factors
        risk_score = 0.0
        
        # Confidence factor (higher confidence = lower risk)
        confidence_risk = 1.0 - prediction.confidence
        risk_score += confidence_risk * 0.4
        
        # Market volatility factor
        # (This is a placeholder - in a real system, you'd incorporate volatility metrics)
        volatility_risk = 0.5  # Moderate volatility assumed
        risk_score += volatility_risk * 0.3
        
        # Trend alignment factor
        trend_risk = 0.3  # Placeholder value
        risk_score += trend_risk * 0.2
        
        # Liquidity factor
        liquidity_risk = 0.2  # Placeholder value
        risk_score += liquidity_risk * 0.1
        
        # Map risk score to risk level
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    async def _publish_trading_signal(self, decision: TradingDecision) -> None:
        """Publish a trading signal from a decision.
        
        Args:
            decision: The trading decision to publish
        """
        # Map decision type to signal type
        signal_type = OutputSignalType.ENTRY
        if decision.decision_type == SignalType.EXIT:
            signal_type = OutputSignalType.EXIT
        elif decision.decision_type == SignalType.ADJUST:
            signal_type = OutputSignalType.ADJUST
        
        # Map direction to string
        direction = "long" if decision.direction == Direction.BULLISH else "short"
        
        # Get timeframe from decision source predictions if available
        timeframe_value = "1h"  # Default timeframe
        
        # Create signal
        signal = Signal(
            source=self.name,
            symbol=decision.symbol,
            signal_type=signal_type,
            direction=direction,
            timeframe=TimeFrame(timeframe_value),
            price=decision.entry_price or self.current_prices.get(decision.symbol, 0.0),
            confidence=decision.confidence,
            reason=decision.rationale,
            take_profit=decision.take_profit_levels[0][0] if decision.take_profit_levels else None,
            stop_loss=decision.stop_loss,
            expiration=decision.entry_valid_until,
            metadata={
                "decision_id": decision.id,
                "risk_level": decision.risk_level.value,
                "position_size": decision.position_size,
                "aggregated_prediction_id": decision.aggregated_prediction
            },
            timestamp=decision.timestamp
        )
        
        # Publish signal event
        await self.publish_event(SignalEvent(
            source=self.name,
            symbol=decision.symbol,
            signal_type=signal_type.value,
            direction=direction,
            timeframe=TimeFrame(timeframe_value),
            price=signal.price,
            confidence=signal.confidence,
            strategy_id=self.name,  # Use decision engine name as strategy ID
            reason=signal.reason,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            expiration=signal.expiration,
            metadata=signal.metadata,
            timestamp=signal.timestamp
        ))
