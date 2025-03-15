"""
Event models for analysis agents.

This module contains event classes for various analysis agents to publish their
results to the event bus for consumption by the decision engine and other components.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from src.common.events import Event, EventPriority
from src.models.market_data import TimeFrame


class TechnicalIndicatorEvent(Event):
    """Event for technical indicator updates."""
    
    def __init__(
        self,
        source: str,
        symbol: str,
        indicator_name: str,
        values: Dict[datetime, Any],
        timeframe: TimeFrame,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Initialize a technical indicator event.
        
        Args:
            source: The source of the indicator data
            symbol: The trading pair symbol
            indicator_name: Name of the indicator (e.g., "RSI", "MACD")
            values: Dictionary of indicator values by timestamp
            timeframe: The timeframe of the indicator
            metadata: Optional additional metadata
            timestamp: Optional custom timestamp (defaults to now)
        """
        # Create payload for Event
        payload = {
            "symbol": symbol,
            "indicator_name": indicator_name,
            "values": values,
            "timeframe": timeframe.value if isinstance(timeframe, TimeFrame) else timeframe
        }
        
        # Add metadata to payload if provided
        if metadata:
            payload["metadata"] = metadata
            
        # Initialize Event
        super().__init__(
            event_type="TechnicalIndicatorEvent",
            source=source,
            payload=payload
        )
        
        if timestamp:
            self.timestamp = timestamp


class PatternEvent(Event):
    """Event for chart pattern detections."""
    
    def __init__(
        self,
        source: str,
        symbol: str,
        pattern_name: str,
        timeframe: TimeFrame,
        confidence: float,
        target_price: Optional[float] = None,
        invalidation_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Initialize a pattern event.
        
        Args:
            source: The source of the pattern detection
            symbol: The trading pair symbol
            pattern_name: Name of the detected pattern
            timeframe: The timeframe of the pattern
            confidence: Confidence level (0.0-1.0)
            target_price: Target price if pattern completes
            invalidation_price: Price level where pattern is invalidated
            metadata: Optional additional metadata
            timestamp: Optional custom timestamp (defaults to now)
        """
        # Create payload for Event
        payload = {
            "symbol": symbol,
            "pattern_name": pattern_name,
            "timeframe": timeframe.value if isinstance(timeframe, TimeFrame) else timeframe,
            "confidence": confidence
        }
        
        # Add optional fields
        if target_price is not None:
            payload["target_price"] = target_price
        if invalidation_price is not None:
            payload["invalidation_price"] = invalidation_price
        
        # Add metadata to payload if provided
        if metadata:
            payload["metadata"] = metadata
            
        # Initialize Event
        super().__init__(
            event_type="PatternEvent",
            source=source,
            payload=payload
        )
        
        if timestamp:
            self.timestamp = timestamp


class CandleDataEvent(Event):
    """Event for candle data updates."""
    
    def __init__(
        self,
        source: str,
        symbol: str,
        timeframe: TimeFrame,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        candle_timestamp: datetime,
        event_timestamp: Optional[datetime] = None
    ):
        """Initialize a candle data event.
        
        Args:
            source: The source of the candle data
            symbol: The trading pair symbol
            timeframe: The timeframe of the candle
            open: Opening price
            high: Highest price
            low: Lowest price
            close: Closing price
            volume: Trading volume
            candle_timestamp: Timestamp of the candle
            event_timestamp: Optional custom event timestamp (defaults to now)
        """
        # Create payload for Event
        payload = {
            "symbol": symbol,
            "timeframe": timeframe.value if isinstance(timeframe, TimeFrame) else timeframe,
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "candle_timestamp": candle_timestamp.isoformat() if isinstance(candle_timestamp, datetime) else candle_timestamp
        }
            
        # Initialize Event
        super().__init__(
            event_type="CandleDataEvent",
            source=source,
            payload=payload
        )
        
        if event_timestamp:
            self.timestamp = event_timestamp


class SignalEvent(Event):
    """Event for trading signals."""
    
    def __init__(
        self,
        source: str,
        symbol: str,
        signal_type: str,
        direction: str,
        price: float,
        timeframe: TimeFrame,
        confidence: float,
        strategy_id: Optional[str] = None,
        reason: Optional[str] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        expiration: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Initialize a signal event.
        
        Args:
            source: The source of the signal
            symbol: The trading pair symbol
            signal_type: Type of signal (entry, exit, adjust)
            direction: Direction of signal (long, short)
            price: Price level for the signal
            timeframe: The timeframe of the signal
            confidence: Confidence level (0.0-1.0)
            strategy_id: Optional strategy identifier
            reason: Optional reason for the signal
            take_profit: Optional take profit price
            stop_loss: Optional stop loss price
            expiration: Optional expiration time for the signal
            metadata: Optional additional metadata
            timestamp: Optional custom timestamp (defaults to now)
        """
        # Create payload for Event
        payload = {
            "symbol": symbol,
            "signal_type": signal_type,
            "direction": direction,
            "price": price,
            "timeframe": timeframe.value if isinstance(timeframe, TimeFrame) else timeframe,
            "confidence": confidence
        }
        
        # Add optional fields
        if strategy_id is not None:
            payload["strategy_id"] = strategy_id
        if reason is not None:
            payload["reason"] = reason
        if take_profit is not None:
            payload["take_profit"] = take_profit
        if stop_loss is not None:
            payload["stop_loss"] = stop_loss
        if expiration is not None:
            payload["expiration"] = expiration.isoformat() if isinstance(expiration, datetime) else expiration
        
        # Add metadata to payload if provided
        if metadata:
            payload["metadata"] = metadata
            
        # Initialize Event
        super().__init__(
            event_type="SignalEvent",
            source=source,
            payload=payload
        )
        
        if timestamp:
            self.timestamp = timestamp
