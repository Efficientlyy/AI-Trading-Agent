#!/usr/bin/env python
"""
Event Bus for System Overseer

This module provides an event bus for system-wide event publishing and subscription,
with support for event history and filtering.
"""

import os
import sys
import json
import time
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Type
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_overseer.event_bus")


class Event:
    """Event class for system events."""
    
    def __init__(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        timestamp: Optional[int] = None,
        event_id: Optional[str] = None
    ):
        """Initialize event.
        
        Args:
            event_type: Event type
            source: Event source
            data: Event data
            timestamp: Event timestamp (milliseconds since epoch)
            event_id: Event identifier
        """
        self.event_type = event_type
        self.source = source
        self.data = data
        self.timestamp = timestamp or int(time.time() * 1000)
        self.event_id = event_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary.
        
        Returns:
            dict: Event as dictionary
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary.
        
        Args:
            data: Event data
            
        Returns:
            Event: Event instance
        """
        return cls(
            event_type=data["event_type"],
            source=data["source"],
            data=data["data"],
            timestamp=data["timestamp"],
            event_id=data["event_id"]
        )


class EventFilter:
    """Event filter for filtering events."""
    
    def __init__(
        self,
        event_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        data_filters: Optional[Dict[str, Any]] = None
    ):
        """Initialize event filter.
        
        Args:
            event_types: Event types to filter
            sources: Event sources to filter
            start_time: Start time (milliseconds since epoch)
            end_time: End time (milliseconds since epoch)
            data_filters: Data filters
        """
        self.event_types = event_types
        self.sources = sources
        self.start_time = start_time
        self.end_time = end_time
        self.data_filters = data_filters or {}
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter.
        
        Args:
            event: Event to check
            
        Returns:
            bool: True if event matches filter
        """
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Check source
        if self.sources and event.source not in self.sources:
            return False
        
        # Check time range
        if self.start_time and event.timestamp < self.start_time:
            return False
        
        if self.end_time and event.timestamp > self.end_time:
            return False
        
        # Check data filters
        for key, value in self.data_filters.items():
            if key not in event.data or event.data[key] != value:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary.
        
        Returns:
            dict: Filter as dictionary
        """
        return {
            "event_types": self.event_types,
            "sources": self.sources,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "data_filters": self.data_filters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventFilter':
        """Create filter from dictionary.
        
        Args:
            data: Filter data
            
        Returns:
            EventFilter: Filter instance
        """
        return cls(
            event_types=data.get("event_types"),
            sources=data.get("sources"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            data_filters=data.get("data_filters")
        )


class EventHistory:
    """Event history for storing and retrieving events."""
    
    def __init__(self, max_events: int = 1000):
        """Initialize event history.
        
        Args:
            max_events: Maximum number of events to store
        """
        self.events = []
        self.max_events = max_events
        self.lock = threading.RLock()
    
    def add_event(self, event: Event) -> bool:
        """Add event to history.
        
        Args:
            event: Event to add
            
        Returns:
            bool: True if event added successfully
        """
        with self.lock:
            self.events.append(event)
            
            # Limit history size
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
            
            return True
    
    def get_events(
        self,
        filter: Optional[EventFilter] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get events from history.
        
        Args:
            filter: Event filter
            limit: Maximum number of events to return
            
        Returns:
            list: Filtered events
        """
        with self.lock:
            # Apply filter
            if filter:
                events = [e for e in self.events if filter.matches(e)]
            else:
                events = self.events.copy()
            
            # Apply limit
            if limit:
                events = events[-limit:]
            
            return events
    
    def clear(self) -> bool:
        """Clear event history.
        
        Returns:
            bool: True if history cleared successfully
        """
        with self.lock:
            self.events = []
            return True
    
    def save_to_file(self, file_path: str) -> bool:
        """Save event history to file.
        
        Args:
            file_path: File path
            
        Returns:
            bool: True if history saved successfully
        """
        try:
            with self.lock:
                # Convert events to dictionaries
                events_data = [e.to_dict() for e in self.events]
                
                # Save to file
                with open(file_path, "w") as f:
                    json.dump(events_data, f, indent=2)
                
                return True
                
        except Exception as e:
            logger.error(f"Error saving event history: {e}")
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """Load event history from file.
        
        Args:
            file_path: File path
            
        Returns:
            bool: True if history loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Event history file not found: {file_path}")
                return False
            
            with open(file_path, "r") as f:
                events_data = json.load(f)
            
            with self.lock:
                # Convert dictionaries to events
                self.events = [Event.from_dict(e) for e in events_data]
                
                # Limit history size
                if len(self.events) > self.max_events:
                    self.events = self.events[-self.max_events:]
                
                return True
                
        except Exception as e:
            logger.error(f"Error loading event history: {e}")
            return False


class EventBus:
    """Event bus for system-wide event publishing and subscription."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize event bus.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self.subscribers = {}
        self.history = EventHistory(max_events=max_history)
        self.lock = threading.RLock()
        self.running = False
        self.event_queue = []
        self.event_thread = None
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> bool:
        """Subscribe to events.
        
        Args:
            event_type: Event type to subscribe to (use "*" for all events)
            callback: Callback function
            
        Returns:
            bool: True if subscription successful
        """
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            if callback in self.subscribers[event_type]:
                logger.warning(f"Callback already subscribed to event type: {event_type}")
                return False
            
            self.subscribers[event_type].append(callback)
            logger.info(f"Subscribed to event type: {event_type}")
            return True
    
    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> bool:
        """Unsubscribe from events.
        
        Args:
            event_type: Event type to unsubscribe from
            callback: Callback function
            
        Returns:
            bool: True if unsubscription successful
        """
        with self.lock:
            if event_type not in self.subscribers:
                logger.warning(f"No subscribers for event type: {event_type}")
                return False
            
            if callback not in self.subscribers[event_type]:
                logger.warning(f"Callback not subscribed to event type: {event_type}")
                return False
            
            self.subscribers[event_type].remove(callback)
            logger.info(f"Unsubscribed from event type: {event_type}")
            return True
    
    def publish(self, event_type: str, event_data=None) -> bool:
        """Publish event.
        
        Args:
            event_type: Event type
            event_data: Event data
            
        Returns:
            bool: True if publish successful
        """
        # Create event
        if isinstance(event_type, Event):
            event = event_type
        else:
            event = Event(
                event_type=event_type,
                source="system",
                data=event_data or {}
            )
        
        with self.lock:
            # Add to queue
            self.event_queue.append(event)
            
            # Add to history
            self.history.add_event(event)
            
            # Notify subscribers
            callbacks = []
            
            # Event type subscribers
            if event.event_type in self.subscribers:
                callbacks.extend(self.subscribers[event.event_type])
            
            # Wildcard subscribers
            if "*" in self.subscribers:
                callbacks.extend(self.subscribers["*"])
            
            # Call callbacks outside lock
            
        # Call callbacks
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
        
        logger.debug(f"Event published: {event.event_type}")
        return True
    
    def process_events(self) -> bool:
        """Process events in queue.
        
        Returns:
            bool: True if processing successful
        """
        logger.info("Starting event processing")
        
        self.running = True
        self.event_thread = threading.Thread(target=self._process_events_thread)
        self.event_thread.daemon = True
        self.event_thread.start()
        
        return True
    
    def _process_events_thread(self):
        """Process events in background thread."""
        logger.info("Event processing thread started")
        
        while self.running:
            events = []
            
            # Get events from queue
            with self.lock:
                if self.event_queue:
                    events = self.event_queue.copy()
                    self.event_queue = []
            
            # Process events
            for event in events:
                logger.debug(f"Processing event: {event.event_type}")
                # Event processing logic here
            
            # Sleep to avoid CPU hogging
            time.sleep(0.1)
        
        logger.info("Event processing thread stopped")
    
    def stop_processing(self) -> bool:
        """Stop event processing.
        
        Returns:
            bool: True if stop successful
        """
        logger.info("Stopping event processing")
        
        self.running = False
        
        if self.event_thread:
            self.event_thread.join(timeout=5)
            if self.event_thread.is_alive():
                logger.warning("Event processing thread did not stop gracefully")
        
        return True
    
    def get_history(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Get event history.
        
        Args:
            event_type: Filter by event type
            source: Filter by source
            start_time: Filter by start time (milliseconds since epoch)
            end_time: Filter by end time (milliseconds since epoch)
            limit: Limit number of events
            
        Returns:
            list: Filtered events
        """
        # Create filter
        filter = None
        if event_type or source or start_time or end_time:
            filter = EventFilter(
                event_types=[event_type] if event_type else None,
                sources=[source] if source else None,
                start_time=start_time,
                end_time=end_time
            )
        
        # Get events
        return self.history.get_events(filter=filter, limit=limit)
    
    def clear_history(self) -> bool:
        """Clear event history.
        
        Returns:
            bool: True if clear successful
        """
        return self.history.clear()
    
    def save_history(self, file_path: str) -> bool:
        """Save event history to file.
        
        Args:
            file_path: File path
            
        Returns:
            bool: True if save successful
        """
        return self.history.save_to_file(file_path)
    
    def load_history(self, file_path: str) -> bool:
        """Load event history from file.
        
        Args:
            file_path: File path
            
        Returns:
            bool: True if load successful
        """
        return self.history.load_from_file(file_path)
