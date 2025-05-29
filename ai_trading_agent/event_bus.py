"""
Event Bus module for the AI Trading Agent.

This module provides a central event bus for components to communicate through
pub/sub patterns, allowing for loose coupling between components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Callable, Awaitable, Optional, Set

# Configure logger
logger = logging.getLogger(__name__)

class EventBus:
    """
    Event bus for the AI Trading Agent system.
    
    Implements a publish-subscribe pattern to allow components to communicate
    without direct dependencies. Events are processed asynchronously.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one event bus exists."""
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the event bus if not already initialized."""
        if self._initialized:
            return
            
        self._handlers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
        self._topic_metadata: Dict[str, Dict[str, Any]] = {}
        self._topic_history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history_per_topic = 100
        self._initialized = True
        logger.info("EventBus initialized")
    
    async def publish(self, topic: str, data: Dict[str, Any], retain: bool = False) -> None:
        """
        Publish an event to a topic.
        
        Args:
            topic: The topic to publish to
            data: The event data
            retain: Whether to retain this event for future subscribers
        """
        logger.debug(f"Publishing to topic {topic}: {data}")
        
        # Store in history if requested
        if retain:
            if topic not in self._topic_history:
                self._topic_history[topic] = []
            
            self._topic_history[topic].append(data)
            
            # Trim history if it gets too long
            if len(self._topic_history[topic]) > self._max_history_per_topic:
                self._topic_history[topic] = self._topic_history[topic][-self._max_history_per_topic:]
        
        # If no handlers, just return
        if topic not in self._handlers:
            logger.debug(f"No handlers for topic {topic}")
            return
            
        # Notify all handlers
        tasks = []
        for handler in self._handlers[topic]:
            tasks.append(asyncio.create_task(self._call_handler(handler, data)))
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_handler(self, handler: Callable, data: Dict[str, Any]) -> None:
        """
        Call a handler safely, catching exceptions.
        
        Args:
            handler: The handler function
            data: The event data
        """
        try:
            await handler(data)
        except Exception as e:
            logger.error(f"Error in event handler: {e}")
    
    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to
            handler: The handler function
        """
        if topic not in self._handlers:
            self._handlers[topic] = []
        
        self._handlers[topic].append(handler)
        logger.debug(f"Subscribed handler to topic {topic}")
        
        # Send retained messages to new subscribers
        if topic in self._topic_history and self._topic_history[topic]:
            for data in self._topic_history[topic]:
                asyncio.create_task(self._call_handler(handler, data))
    
    def unsubscribe(self, topic: str, handler: Callable) -> None:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: The topic to unsubscribe from
            handler: The handler function
        """
        if topic in self._handlers and handler in self._handlers[topic]:
            self._handlers[topic].remove(handler)
            logger.debug(f"Unsubscribed handler from topic {topic}")
            
            # Clean up empty handler lists
            if not self._handlers[topic]:
                del self._handlers[topic]
    
    def get_topics(self) -> Set[str]:
        """
        Get all active topics.
        
        Returns:
            Set of active topics
        """
        return set(self._handlers.keys())
    
    def clear_topic_history(self, topic: Optional[str] = None) -> None:
        """
        Clear the history for a topic or all topics.
        
        Args:
            topic: The topic to clear, or None to clear all
        """
        if topic is None:
            self._topic_history.clear()
            logger.debug("Cleared all topic history")
        elif topic in self._topic_history:
            del self._topic_history[topic]
            logger.debug(f"Cleared history for topic {topic}")
    
    def set_topic_metadata(self, topic: str, metadata: Dict[str, Any]) -> None:
        """
        Set metadata for a topic.
        
        Args:
            topic: The topic to set metadata for
            metadata: The metadata
        """
        self._topic_metadata[topic] = metadata
        logger.debug(f"Set metadata for topic {topic}: {metadata}")
    
    def get_topic_metadata(self, topic: str) -> Dict[str, Any]:
        """
        Get metadata for a topic.
        
        Args:
            topic: The topic to get metadata for
            
        Returns:
            Topic metadata
        """
        return self._topic_metadata.get(topic, {})
    
    def reset(self) -> None:
        """Reset the event bus, clearing all subscriptions and history."""
        self._handlers.clear()
        self._topic_metadata.clear()
        self._topic_history.clear()
        logger.info("EventBus reset")

# Create a singleton instance
event_bus = EventBus()
