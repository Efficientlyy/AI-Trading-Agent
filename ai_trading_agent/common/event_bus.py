"""
Event Bus System

This module provides a centralized event bus for system-wide communication
between components of the trading system.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
import threading
import queue
import time
import uuid
from datetime import datetime

from .utils import get_logger

# Setup logging
logger = get_logger(__name__)

class Event:
    """
    Event class for communication between system components.
    """
    
    def __init__(self, event_type: str, data: Any = None, source: str = None):
        """
        Initialize a new event.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Source of the event
        """
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"Event({self.event_type}, source={self.source}, timestamp={self.timestamp})"

class EventBus:
    """
    Event bus for decoupled communication between system components.
    """
    
    def __init__(self, async_processing: bool = True):
        """
        Initialize the event bus.
        
        Args:
            async_processing: Whether to process events asynchronously
        """
        self.subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self.event_queue = queue.Queue()
        self.async_processing = async_processing
        self.running = False
        self.processor_thread = None
        
        # Start the event processor if async processing is enabled
        if self.async_processing:
            self.start()
    
    def start(self):
        """Start the event processor."""
        if self.running:
            logger.warning("Event bus already running")
            return
        
        self.running = True
        self.processor_thread = threading.Thread(
            target=self._process_events,
            daemon=True
        )
        self.processor_thread.start()
        logger.info("Started event bus processor")
    
    def stop(self):
        """Stop the event processor."""
        if not self.running:
            logger.warning("Event bus not running")
            return
        
        self.running = False
        logger.info("Stopping event bus processor")
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> str:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
            
        Returns:
            Subscription ID
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        subscription_id = f"{event_type}_{len(self.subscribers[event_type])}"
        
        logger.debug(f"Subscribed to {event_type} events with ID {subscription_id}")
        
        return subscription_id
    
    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]):
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from {event_type} events")
    
    def publish(self, event_type: str, data: Any = None, source: str = None):
        """
        Publish an event.
        
        Args:
            event_type: Type of event to publish
            data: Event data
            source: Source of the event
        """
        event = Event(event_type, data, source)
        
        if self.async_processing:
            # Put the event in the queue for async processing
            self.event_queue.put(event)
        else:
            # Process the event synchronously
            self._deliver_event(event)
        
        logger.debug(f"Published event: {event}")
    
    def _process_events(self):
        """Process events from the queue."""
        while self.running:
            try:
                # Get event from queue with timeout to allow for clean shutdown
                try:
                    event = self.event_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Deliver the event to subscribers
                self._deliver_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
    
    def _deliver_event(self, event: Event):
        """
        Deliver an event to subscribers.
        
        Args:
            event: Event to deliver
        """
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {str(e)}")

class EventLogger:
    """
    Event logger for debugging and auditing purposes.
    """
    
    def __init__(self, event_bus: EventBus, log_file: Optional[str] = None):
        """
        Initialize the event logger.
        
        Args:
            event_bus: Event bus to log events from
            log_file: Optional file to log events to
        """
        self.event_bus = event_bus
        self.log_file = log_file
        self.event_count = 0
        
        # Subscribe to all events with a wildcard
        self.event_bus.subscribe("*", self.log_event)
        
        logger.info("Event logger initialized")
    
    def log_event(self, event: Event):
        """
        Log an event.
        
        Args:
            event: Event to log
        """
        self.event_count += 1
        
        # Log to console
        logger.info(f"EVENT: {event}")
        
        # Log to file if specified
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"{event.timestamp.isoformat()} - {event.event_type} - {event.source} - {event.data}\n")
            except Exception as e:
                logger.error(f"Error logging event to file: {str(e)}")

def get_event_bus() -> EventBus:
    """
    Get or create the global event bus instance.
    
    Returns:
        Global event bus instance
    """
    # Use a global instance
    if not hasattr(get_event_bus, "_instance"):
        get_event_bus._instance = EventBus()
    
    return get_event_bus._instance
