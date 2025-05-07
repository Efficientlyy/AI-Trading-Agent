"""
Event emitter module for the AI Trading Agent.

This module provides an event emitter system for publishing and subscribing to events.
"""

import asyncio
from typing import Dict, Any, List, Callable, Coroutine, Optional, Set
from ..common import logger


class EventEmitter:
    """
    Event emitter for publishing and subscribing to events.
    
    This class provides a simple event system that allows components to:
    1. Subscribe to specific event types
    2. Publish events to all subscribers
    3. Handle both synchronous and asynchronous event handlers
    """
    
    def __init__(self):
        """Initialize the event emitter."""
        self._subscribers: Dict[str, List[Callable]] = {}
        self._async_subscribers: Dict[str, List[Callable]] = {}
        self._active_topics: Set[str] = set()
        
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to an event type with a handler function.
        
        Args:
            event_type: The type of event to subscribe to
            handler: The function to call when the event is emitted
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            self._active_topics.add(event_type)
            logger.debug(f"Subscribed to event: {event_type}")
    
    def subscribe_async(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to an event type with an async handler function.
        
        Args:
            event_type: The type of event to subscribe to
            handler: The async function to call when the event is emitted
        """
        if event_type not in self._async_subscribers:
            self._async_subscribers[event_type] = []
        
        if handler not in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].append(handler)
            self._active_topics.add(event_type)
            logger.debug(f"Subscribed to async event: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler function to remove
        """
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed from event: {event_type}")
            
            # Clean up empty subscriber lists
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]
                
                # Check if there are no async subscribers either
                if event_type not in self._async_subscribers:
                    self._active_topics.discard(event_type)
    
    def unsubscribe_async(self, event_type: str, handler: Callable) -> None:
        """
        Unsubscribe from an async event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            handler: The async handler function to remove
        """
        if event_type in self._async_subscribers and handler in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed from async event: {event_type}")
            
            # Clean up empty subscriber lists
            if not self._async_subscribers[event_type]:
                del self._async_subscribers[event_type]
                
                # Check if there are no sync subscribers either
                if event_type not in self._subscribers:
                    self._active_topics.discard(event_type)
    
    def emit(self, event_type: str, data: Any = None) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event_type: The type of event to emit
            data: The data to pass to the event handlers
        """
        # Skip if no subscribers for this event type
        if event_type not in self._active_topics:
            return
            
        # Call synchronous handlers
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}", exc_info=True)
        
        # Create tasks for asynchronous handlers
        if event_type in self._async_subscribers:
            for handler in self._async_subscribers[event_type]:
                try:
                    # Try to get the current event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        # If no event loop is available, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Create a task for the handler
                    loop.create_task(handler(data))
                except Exception as e:
                    logger.error(f"Error creating task for async event handler for {event_type}: {e}", exc_info=True)
    
    async def emit_async(self, event_type: str, data: Any = None) -> None:
        """
        Emit an event to all subscribers asynchronously.
        
        Args:
            event_type: The type of event to emit
            data: The data to pass to the event handlers
        """
        # Skip if no subscribers for this event type
        if event_type not in self._active_topics:
            return
            
        # Call synchronous handlers
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}", exc_info=True)
        
        # Call asynchronous handlers
        if event_type in self._async_subscribers:
            for handler in self._async_subscribers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Error in async event handler for {event_type}: {e}", exc_info=True)
    
    def has_subscribers(self, event_type: str) -> bool:
        """
        Check if an event type has any subscribers.
        
        Args:
            event_type: The type of event to check
            
        Returns:
            bool: True if the event has subscribers, False otherwise
        """
        return event_type in self._active_topics


# Create a global event emitter instance
global_event_emitter = EventEmitter()
