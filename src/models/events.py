"""
Event models for the system-wide event bus.

This module contains the classes representing different types of events
that can be published to the event bus for inter-component communication.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.common.events import Event as BaseEvent, EventPriority
from src.models.order import Order


class EventType(Enum):
    """Enum for event types in the system."""
    SYSTEM_STATUS = "system_status"
    ORDER = "order"
    PERFORMANCE = "performance"
    POSITION = "position"
    DATA = "data"
    SENTIMENT = "sentiment"


class Event:
    """Base class for all events in the system."""
    
    def __init__(self, event_type: str):
        """Initialize an event.
        
        Args:
            event_type: The type of the event
        """
        self.event_type = event_type
        self.timestamp = datetime.now()


class SystemStatusEvent(Event):
    """Event for system status updates from components."""
    
    def __init__(self, component_id: str, status: str, message: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize a system status event.
        
        Args:
            component_id: The ID of the component reporting status
            status: The status of the component ("ok", "warning", "error", etc.)
            message: Optional message describing the status
            details: Optional details about the status
        """
        super().__init__(EventType.SYSTEM_STATUS.value)
        self.component_id = component_id
        self.status = status
        self.message = message
        self.details = details or {}


class OrderEvent(Event):
    """Event for order-related notifications."""
    
    def __init__(self, order_id: str, order_event_type: str, order: Optional[Order] = None,
                 details: Optional[Dict[str, Any]] = None):
        """Initialize an order event.
        
        Args:
            order_id: The ID of the order
            order_event_type: The type of order event ("SUBMITTED", "FILLED", etc.)
            order: Optional Order object
            details: Optional additional details
        """
        super().__init__(EventType.ORDER.value)
        self.order_id = order_id
        self.order_event_type = order_event_type
        self.order = order
        self.details = details or {}


class PerformanceEvent(Event):
    """Event for reporting performance metrics."""
    
    def __init__(self, metric_type: str, metrics: Dict[str, Any], 
                 strategy_id: Optional[str] = None, portfolio_id: Optional[str] = None):
        """Initialize a performance event.
        
        Args:
            metric_type: The type of metrics being reported
            metrics: The performance metrics
            strategy_id: Optional strategy ID for strategy-specific metrics
            portfolio_id: Optional portfolio ID for portfolio-specific metrics
        """
        super().__init__(EventType.PERFORMANCE.value)
        self.metric_type = metric_type
        self.metrics = metrics
        self.strategy_id = strategy_id
        self.portfolio_id = portfolio_id
        

class PositionEvent(Event):
    """Event for position-related notifications."""
    
    def __init__(self, position_id: str, position_event_type: str, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize a position event.
        
        Args:
            position_id: The ID of the position
            position_event_type: The type of position event ("OPENED", "CLOSED", etc.)
            details: Optional additional details
        """
        super().__init__(EventType.POSITION.value)
        self.position_id = position_id
        self.position_event_type = position_event_type
        self.details = details or {}
        

class DataEvent(Event):
    """Event for data-related notifications."""
    
    def __init__(self, source: str, data_type: str, symbol: Optional[str] = None, 
                 data: Any = None):
        """Initialize a data event.
        
        Args:
            source: The source of the data
            data_type: The type of data
            symbol: Optional symbol for the data
            data: Optional data payload
        """
        super().__init__(EventType.DATA.value)
        self.source = source
        self.data_type = data_type
        self.symbol = symbol
        self.data = data


class SentimentEvent(BaseEvent):
    """Event for sentiment analysis updates."""
    
    def __init__(self, source: str, symbol: str, sentiment_value: float,
                 sentiment_direction: str, confidence: float,
                 details: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime] = None):
        """Initialize a sentiment event.
        
        Args:
            source: The source of the sentiment data
            symbol: The trading pair symbol
            sentiment_value: The numerical sentiment value (0.0-1.0)
            sentiment_direction: The direction of sentiment ("bullish", "bearish")
            confidence: The confidence level of the sentiment prediction (0.0-1.0)
            details: Optional additional details about the sentiment
            timestamp: Optional custom timestamp (defaults to now)
        """
        # Create payload for BaseEvent
        payload = {
            "symbol": symbol,
            "sentiment_value": sentiment_value,
            "sentiment_direction": sentiment_direction,
            "confidence": confidence,
        }
        
        # Add details to payload if provided
        if details:
            payload.update(details)
            
        # Initialize BaseEvent
        super().__init__(
            event_type=EventType.SENTIMENT.value,
            source=source,
            payload=payload
        )
