"""
Event Bus

This module provides a class for event distribution between system components
using a publish-subscribe pattern. It allows components to publish events and
subscribe to events from other components.
"""

import json
import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Callable, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("event_bus")

class EventBus:
    """
    Provides publish-subscribe messaging between system components.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        # Subscribers by topic
        self.subscribers: Dict[str, Dict[str, Callable]] = {}
        
        # Event history for replay
        self.event_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history_per_topic = 100
        
        # Statistics
        self.stats = {
            "events_published": 0,
            "events_delivered": 0,
            "subscribers_total": 0,
            "start_time": time.time()
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Event bus initialized")
        
    def subscribe(self, topic: str, callback: Callable, subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to
            callback: The callback function to call when an event is published
            subscriber_id: Optional subscriber ID, generated if not provided
            
        Returns:
            The subscriber ID
        """
        with self.lock:
            # Generate subscriber ID if not provided
            if subscriber_id is None:
                subscriber_id = str(uuid.uuid4())
                
            # Create topic if it doesn't exist
            if topic not in self.subscribers:
                self.subscribers[topic] = {}
                
            # Add subscriber
            self.subscribers[topic][subscriber_id] = callback
            
            # Update statistics
            self.stats["subscribers_total"] += 1
            
            logger.debug(f"Subscriber {subscriber_id} subscribed to topic {topic}")
            
            return subscriber_id
            
    def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: The topic to unsubscribe from
            subscriber_id: The subscriber ID
            
        Returns:
            True if the subscriber was unsubscribed, False otherwise
        """
        with self.lock:
            # Check if topic exists
            if topic not in self.subscribers:
                return False
                
            # Check if subscriber exists
            if subscriber_id not in self.subscribers[topic]:
                return False
                
            # Remove subscriber
            del self.subscribers[topic][subscriber_id]
            
            # Remove topic if empty
            if not self.subscribers[topic]:
                del self.subscribers[topic]
                
            logger.debug(f"Subscriber {subscriber_id} unsubscribed from topic {topic}")
            
            return True
            
    def publish(self, topic: str, data: Any, retain: bool = False) -> int:
        """
        Publish an event to a topic.
        
        Args:
            topic: The topic to publish to
            data: The event data
            retain: Whether to retain the event for future subscribers
            
        Returns:
            The number of subscribers the event was delivered to
        """
        with self.lock:
            # Create event
            event = {
                "topic": topic,
                "data": data,
                "timestamp": time.time()
            }
            
            # Add to history if retain is True
            if retain:
                if topic not in self.event_history:
                    self.event_history[topic] = []
                    
                self.event_history[topic].append(event)
                
                # Limit history size
                if len(self.event_history[topic]) > self.max_history_per_topic:
                    self.event_history[topic].pop(0)
                    
            # Update statistics
            self.stats["events_published"] += 1
            
            # Check if topic has subscribers
            if topic not in self.subscribers:
                return 0
                
            # Deliver event to subscribers
            delivered_count = 0
            
            for subscriber_id, callback in list(self.subscribers[topic].items()):
                try:
                    # Call subscriber callback
                    callback(event)
                    delivered_count += 1
                except Exception as e:
                    logger.error(f"Error delivering event to subscriber {subscriber_id}: {e}")
                    
            # Update statistics
            self.stats["events_delivered"] += delivered_count
            
            logger.debug(f"Published event to topic {topic}, delivered to {delivered_count} subscribers")
            
            return delivered_count
            
    def get_retained_events(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get retained events for a topic.
        
        Args:
            topic: The topic to get events for
            
        Returns:
            List of retained events
        """
        with self.lock:
            return self.event_history.get(topic, []).copy()
            
    def subscribe_with_history(self, topic: str, callback: Callable, subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to a topic and receive retained events.
        
        Args:
            topic: The topic to subscribe to
            callback: The callback function to call when an event is published
            subscriber_id: Optional subscriber ID, generated if not provided
            
        Returns:
            The subscriber ID
        """
        # Subscribe to topic
        subscriber_id = self.subscribe(topic, callback, subscriber_id)
        
        # Get retained events
        retained_events = self.get_retained_events(topic)
        
        # Deliver retained events
        for event in retained_events:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error delivering retained event to subscriber {subscriber_id}: {e}")
                
        return subscriber_id
        
    def clear_history(self, topic: Optional[str] = None) -> int:
        """
        Clear retained events.
        
        Args:
            topic: The topic to clear events for, or None to clear all topics
            
        Returns:
            The number of events cleared
        """
        with self.lock:
            if topic is None:
                # Clear all topics
                event_count = sum(len(events) for events in self.event_history.values())
                self.event_history.clear()
                return event_count
            elif topic in self.event_history:
                # Clear specific topic
                event_count = len(self.event_history[topic])
                del self.event_history[topic]
                return event_count
            else:
                return 0
                
    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.
        
        Returns:
            Dict containing event bus statistics
        """
        with self.lock:
            uptime = time.time() - self.stats["start_time"]
            
            # Count current subscribers
            current_subscribers = sum(len(subscribers) for subscribers in self.subscribers.values())
            
            # Count retained events
            retained_events = sum(len(events) for events in self.event_history.values())
            
            return {
                "events_published": self.stats["events_published"],
                "events_delivered": self.stats["events_delivered"],
                "subscribers_total": self.stats["subscribers_total"],
                "subscribers_current": current_subscribers,
                "topics": len(self.subscribers),
                "retained_events": retained_events,
                "uptime_seconds": uptime,
                "uptime_formatted": self._format_uptime(uptime)
            }
            
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in seconds to a human-readable string"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0 or days > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        
        return " ".join(parts)