"""Base strategy for the AI Crypto Trading System.

This module defines the base class for all trading strategies in the system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.models.events import (
    CandleDataEvent, ErrorEvent, OrderBookEvent, PatternEvent,
    SignalEvent, SystemStatusEvent, TechnicalIndicatorEvent,
    TradeDataEvent
)
from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData
from src.models.signals import Signal, SignalType


class Strategy(Component, ABC):
    """Base class for all trading strategies.
    
    A strategy is responsible for generating trading signals based on
    market data and analysis results.
    """
    
    def __init__(self, strategy_id: str):
        """Initialize the strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(f"strategy_{strategy_id}")
        self.strategy_id = strategy_id
        self.logger = get_logger("strategy", strategy_id)
        
        # Configuration
        self.enabled = config.get(f"strategies.{strategy_id}.enabled", True)
        self.symbols: Set[str] = set(config.get(f"strategies.{strategy_id}.symbols", []))
        self.exchanges: Set[str] = set(config.get(f"strategies.{strategy_id}.exchanges", []))
        self.timeframes: Set[TimeFrame] = set()
        
        # Parse timeframes from config
        timeframe_strs = config.get(f"strategies.{strategy_id}.timeframes", [])
        for tf_str in timeframe_strs:
            try:
                self.timeframes.add(TimeFrame(tf_str))
            except ValueError:
                self.logger.error("Invalid timeframe in configuration", timeframe=tf_str)
        
        # Strategy state
        self.active_signals: Dict[str, Signal] = {}  # symbol -> active signal
        self.signal_history: List[Signal] = []
        self.last_evaluation_time: Dict[Tuple[str, str, TimeFrame], datetime] = {}
        
        # Define event subscriptions
        self.event_subscriptions = config.get(
            f"strategies.{strategy_id}.subscriptions", 
            ["CandleDataEvent", "TechnicalIndicatorEvent", "PatternEvent"]
        )
    
    async def _initialize(self) -> None:
        """Initialize the strategy."""
        if not self.enabled:
            self.logger.info("Strategy is disabled")
            return
        
        self.logger.info("Initializing strategy", 
                        symbols=list(self.symbols),
                        exchanges=list(self.exchanges),
                        timeframes=[tf.value for tf in self.timeframes])
        
        # Additional initialization
        self._strategy_initialize()
    
    async def _start(self) -> None:
        """Start the strategy."""
        if not self.enabled:
            return
        
        self.logger.info("Starting strategy")
        
        # Subscribe to events
        for event_type in self.event_subscriptions:
            event_bus.subscribe(event_type, self._handle_event)
        
        # Additional startup
        self._strategy_start()
    
    async def _stop(self) -> None:
        """Stop the strategy."""
        if not self.enabled:
            return
        
        self.logger.info("Stopping strategy")
        
        # Unsubscribe from events
        for event_type in self.event_subscriptions:
            event_bus.unsubscribe(event_type, self._handle_event)
        
        # Additional shutdown
        self._strategy_stop()
    
    async def _handle_event(self, event) -> None:
        """Handle an event.
        
        This method is called for all events the strategy is subscribed to.
        It dispatches the event to the appropriate handler based on the event type.
        
        Args:
            event: The event to handle
        """
        if not self.enabled:
            return
        
        event_type = event.__class__.__name__
        
        try:
            if event_type == "CandleDataEvent":
                await self._handle_candle_event(event)
            elif event_type == "TradeDataEvent":
                await self._handle_trade_event(event)
            elif event_type == "OrderBookEvent":
                await self._handle_orderbook_event(event)
            elif event_type == "TechnicalIndicatorEvent":
                await self._handle_indicator_event(event)
            elif event_type == "PatternEvent":
                await self._handle_pattern_event(event)
            # Add handlers for other event types as needed
            
        except Exception as e:
            self.logger.error("Error handling event", 
                              event_type=event_type,
                              error=str(e))
    
    async def _handle_candle_event(self, event: CandleDataEvent) -> None:
        """Handle a candle data event.
        
        Args:
            event: The candle data event
        """
        # Check if this candle is relevant to this strategy
        candle = event.candle
        if (not self.symbols or candle.symbol in self.symbols) and \
           (not self.exchanges or candle.exchange in self.exchanges) and \
           (not self.timeframes or candle.timeframe in self.timeframes):
            
            await self.process_candle(candle)
    
    async def _handle_trade_event(self, event: TradeDataEvent) -> None:
        """Handle a trade data event.
        
        Args:
            event: The trade data event
        """
        # Check if this trade is relevant to this strategy
        trade = event.trade
        if (not self.symbols or trade.symbol in self.symbols) and \
           (not self.exchanges or trade.exchange in self.exchanges):
            
            await self.process_trade(trade)
    
    async def _handle_orderbook_event(self, event: OrderBookEvent) -> None:
        """Handle an order book event.
        
        Args:
            event: The order book event
        """
        # Check if this order book is relevant to this strategy
        orderbook = event.orderbook
        if (not self.symbols or orderbook.symbol in self.symbols) and \
           (not self.exchanges or orderbook.exchange in self.exchanges):
            
            await self.process_orderbook(orderbook)
    
    async def _handle_indicator_event(self, event: TechnicalIndicatorEvent) -> None:
        """Handle a technical indicator event.
        
        Args:
            event: The technical indicator event
        """
        # Check if this indicator is relevant to this strategy
        if (not self.symbols or event.symbol in self.symbols) and \
           (not self.timeframes or event.timeframe in self.timeframes):
            
            await self.process_indicator(
                event.symbol, 
                event.timeframe, 
                event.indicator_name, 
                event.values
            )
    
    async def _handle_pattern_event(self, event: PatternEvent) -> None:
        """Handle a pattern event.
        
        Args:
            event: The pattern event
        """
        # Check if this pattern is relevant to this strategy
        if (not self.symbols or event.symbol in self.symbols) and \
           (not self.timeframes or event.timeframe in self.timeframes):
            
            await self.process_pattern(
                event.symbol,
                event.timeframe,
                event.pattern_name,
                event.confidence,
                event.target_price,
                event.invalidation_price
            )
    
    async def publish_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        direction: str,
        timeframe: TimeFrame,
        price: float,
        confidence: float,
        reason: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        expiration: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Publish a trading signal.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            signal_type: The type of signal (entry, exit, etc.)
            direction: The direction of the signal (long, short)
            timeframe: The timeframe of the signal
            price: The price at which the signal was generated
            confidence: The confidence score for the signal (0.0 to 1.0)
            reason: The reason for the signal
            take_profit: Optional price target for take profit
            stop_loss: Optional price level for stop loss
            expiration: Optional expiration time for the signal
            metadata: Optional additional data for the signal
        """
        # Create a signal object
        signal = Signal(
            source=self.name,
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            timeframe=timeframe,
            price=price,
            confidence=confidence,
            reason=reason,
            take_profit=take_profit,
            stop_loss=stop_loss,
            expiration=expiration,
            metadata=metadata or {},
            timestamp=utc_now()
        )
        
        # If this is an entry signal, store it in active signals
        if signal_type == SignalType.ENTRY:
            self.active_signals[symbol] = signal
        
        # If this is an exit signal, remove any active signal for this symbol
        elif signal_type == SignalType.EXIT and symbol in self.active_signals:
            del self.active_signals[symbol]
        
        # Add to signal history
        self.signal_history.append(signal)
        
        # Trim history if getting too large
        max_history = config.get(f"strategies.{self.strategy_id}.max_signal_history", 1000)
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
        
        # Publish the signal event
        await self.publish_event(SignalEvent(
            source=self.name,
            signal=signal
        ))
        
        self.logger.info("Published trading signal", 
                        symbol=symbol,
                        signal_type=signal_type.value,
                        direction=direction,
                        price=price,
                        confidence=confidence,
                        reason=reason)
    
    async def publish_error(
        self, 
        error_type: str, 
        error_message: str, 
        error_details: Optional[Dict] = None
    ) -> None:
        """Publish an error event.
        
        Args:
            error_type: The type of error
            error_message: The error message
            error_details: Optional details about the error
        """
        await self.publish_event(ErrorEvent(
            source=self.name,
            error_type=error_type,
            error_message=error_message,
            error_details=error_details or {}
        ))
    
    async def publish_status(
        self, 
        message: str, 
        details: Optional[Dict] = None
    ) -> None:
        """Publish a status event.
        
        Args:
            message: The status message
            details: Optional details about the status
        """
        await self.publish_event(SystemStatusEvent(
            source=self.name,
            status="info",
            message=message,
            details=details or {}
        ))
    
    @abstractmethod
    async def _strategy_initialize(self) -> None:
        """Strategy-specific initialization.
        
        This method should be implemented by subclasses to perform
        strategy-specific initialization.
        """
        pass
    
    @abstractmethod
    async def _strategy_start(self) -> None:
        """Strategy-specific startup.
        
        This method should be implemented by subclasses to perform
        strategy-specific startup tasks.
        """
        pass
    
    @abstractmethod
    async def _strategy_stop(self) -> None:
        """Strategy-specific shutdown.
        
        This method should be implemented by subclasses to perform
        strategy-specific shutdown tasks.
        """
        pass
    
    @abstractmethod
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        Args:
            candle: The candle data to process
        """
        pass
    
    @abstractmethod
    async def process_trade(self, trade: TradeData) -> None:
        """Process a new trade data event.
        
        Args:
            trade: The trade data to process
        """
        pass
    
    @abstractmethod
    async def process_orderbook(self, orderbook: OrderBookData) -> None:
        """Process a new order book event.
        
        Args:
            orderbook: The order book data to process
        """
        pass
    
    @abstractmethod
    async def process_indicator(
        self,
        symbol: str,
        timeframe: TimeFrame,
        indicator_name: str,
        values: Dict
    ) -> None:
        """Process a technical indicator update.
        
        Args:
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            indicator_name: The name of the indicator
            values: The indicator values
        """
        pass
    
    @abstractmethod
    async def process_pattern(
        self,
        symbol: str,
        timeframe: TimeFrame,
        pattern_name: str,
        confidence: float,
        target_price: Optional[float],
        invalidation_price: Optional[float]
    ) -> None:
        """Process a pattern detection.
        
        Args:
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            pattern_name: The name of the pattern
            confidence: The confidence score for the pattern
            target_price: Optional price target for the pattern
            invalidation_price: Optional price level that would invalidate the pattern
        """
        pass 