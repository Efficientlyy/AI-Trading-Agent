#!/usr/bin/env python
"""
EventBus Module for Modular System Overseer

This module implements a central event distribution system with support for
plugins to publish and subscribe to events. It provides topic-based filtering,
event prioritization, and asynchronous event processing.
"""

import os
import json
import time
import logging
import threading
import queue
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("event_bus")

class Event:
    """Event object for the event bus"""
    
    def __init__(
        self,
        event_type: str,
        data: Dict[str, Any],
        publisher_id: str = None,
        timestamp: float = None,
        event_id: str = None,
        priority: int = 0,
        metadata: Dict[str, Any] = None
    ):
        """Initialize event
        
        Args:
            event_type: Type of event
            data: Event data
            publisher_id: ID of publisher (optional)
            timestamp: Event timestamp (defaults to current time)
            event_id: Unique event ID (defaults to generated UUID)
            priority: Event priority (0 = normal, negative = lower, positive = higher)
            metadata: Additional metadata
        """
        self.event_type = event_type
        self.data = data
        self.publisher_id = publisher_id
        self.timestamp = timestamp or time.time()
        self.event_id = event_id or str(uuid.uuid4())
        self.priority = priority
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary
        
        Returns:
            dict: Dictionary representation of event
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "publisher_id": self.publisher_id,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'Event':
        """Create event from dictionary
        
        Args:
            event_dict: Dictionary representation of event
            
        Returns:
            Event: Event object
        """
        return cls(
            event_type=event_dict["event_type"],
            data=event_dict["data"],
            publisher_id=event_dict.get("publisher_id"),
            timestamp=event_dict.get("timestamp"),
            event_id=event_dict.get("event_id"),
            priority=event_dict.get("priority", 0),
            metadata=event_dict.get("metadata", {})
        )


class EventSubscription:
    """Subscription to events"""
    
    def __init__(
        self,
        subscriber_id: str,
        event_type: str,
        callback: Callable[[Event], None],
        filter_func: Callable[[Event], bool] = None,
        priority: int = 0,
        max_events_per_second: float = None,
        subscription_id: str = None
    ):
        """Initialize subscription
        
        Args:
            subscriber_id: ID of subscriber
            event_type: Type of event to subscribe to (can include wildcards)
            callback: Function to call when event is received
            filter_func: Function to filter events (optional)
            priority: Subscription priority (0 = normal, negative = lower, positive = higher)
            max_events_per_second: Maximum events per second (optional)
            subscription_id: Unique subscription ID (defaults to generated UUID)
        """
        self.subscriber_id = subscriber_id
        self.event_type = event_type
        self.callback = callback
        self.filter_func = filter_func
        self.priority = priority
        self.max_events_per_second = max_events_per_second
        self.subscription_id = subscription_id or str(uuid.uuid4())
        
        # Rate limiting
        self.last_event_time = 0
        self.event_count = 0
        self.rate_limit_window = 1.0  # 1 second window for rate limiting
    
    def matches_event_type(self, event_type: str) -> bool:
        """Check if subscription matches event type
        
        Args:
            event_type: Event type to check
            
        Returns:
            bool: True if subscription matches event type
        """
        # Exact match
        if self.event_type == event_type:
            return True
        
        # Wildcard match
        if '*' in self.event_type:
            pattern = self.event_type.replace('*', '.*')
            import re
            return bool(re.match(f"^{pattern}$", event_type))
        
        # Hierarchical match (e.g. "system.log" matches "system.*")
        if '.' in self.event_type and self.event_type.endswith('.*'):
            prefix = self.event_type[:-1]  # Remove the '*'
            return event_type.startswith(prefix)
        
        return False
    
    def should_deliver(self, event: Event) -> bool:
        """Check if event should be delivered to subscriber
        
        Args:
            event: Event to check
            
        Returns:
            bool: True if event should be delivered
        """
        # Check if event type matches
        if not self.matches_event_type(event.event_type):
            return False
        
        # Apply custom filter if provided
        if self.filter_func and not self.filter_func(event):
            return False
        
        # Apply rate limiting if configured
        if self.max_events_per_second:
            current_time = time.time()
            
            # Reset counter if window has passed
            if current_time - self.last_event_time > self.rate_limit_window:
                self.event_count = 0
                self.last_event_time = current_time
            
            # Increment counter
            self.event_count += 1
            
            # Check if rate limit exceeded
            if self.event_count > self.max_events_per_second * self.rate_limit_window:
                return False
        
        return True
    
    def deliver(self, event: Event) -> bool:
        """Deliver event to subscriber
        
        Args:
            event: Event to deliver
            
        Returns:
            bool: True if event was delivered successfully
        """
        try:
            self.callback(event)
            return True
        except Exception as e:
            logger.error(f"Error delivering event to subscriber {self.subscriber_id}: {e}")
            return False


class EventTypeSchema:
    """Schema for event types"""
    
    def __init__(
        self,
        event_type: str,
        schema: Dict[str, Any] = None,
        description: str = None,
        required_fields: List[str] = None,
        example: Dict[str, Any] = None
    ):
        """Initialize event type schema
        
        Args:
            event_type: Type of event
            schema: JSON schema for event data
            description: Description of event type
            required_fields: List of required fields
            example: Example event data
        """
        self.event_type = event_type
        self.schema = schema or {}
        self.description = description
        self.required_fields = required_fields or []
        self.example = example or {}
    
    def validate(self, event_data: Dict[str, Any]) -> bool:
        """Validate event data against schema
        
        Args:
            event_data: Event data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check required fields
        for field in self.required_fields:
            if field not in event_data:
                logger.error(f"Missing required field '{field}' in event data")
                return False
        
        # Validate against JSON schema if available
        if self.schema:
            try:
                import jsonschema
                jsonschema.validate(instance=event_data, schema=self.schema)
            except ImportError:
                logger.warning("jsonschema package not available, skipping schema validation")
            except jsonschema.exceptions.ValidationError as e:
                logger.error(f"Event data validation failed: {e}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary
        
        Returns:
            dict: Dictionary representation of schema
        """
        return {
            "event_type": self.event_type,
            "schema": self.schema,
            "description": self.description,
            "required_fields": self.required_fields,
            "example": self.example
        }


class EventHistory:
    """History of events"""
    
    def __init__(
        self,
        max_size: int = 1000,
        storage_dir: str = None
    ):
        """Initialize event history
        
        Args:
            max_size: Maximum number of events to keep in memory
            storage_dir: Directory for persistent storage (optional)
        """
        self.max_size = max_size
        self.storage_dir = storage_dir
        
        # Create storage directory if specified
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
        
        # In-memory event storage
        self.events = {}  # {event_type: [event]}
        self.event_count = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def add_event(self, event: Event) -> bool:
        """Add event to history
        
        Args:
            event: Event to add
            
        Returns:
            bool: True if event was added
        """
        with self.lock:
            # Initialize event type list if needed
            if event.event_type not in self.events:
                self.events[event.event_type] = []
            
            # Add event to list
            self.events[event.event_type].append(event)
            self.event_count += 1
            
            # Trim oldest events if needed
            if self.event_count > self.max_size:
                # Find event type with most events
                max_type = max(self.events.items(), key=lambda x: len(x[1]))[0]
                
                # Remove oldest event of that type
                self.events[max_type].pop(0)
                self.event_count -= 1
            
            # Persist event if storage directory specified
            if self.storage_dir:
                self._persist_event(event)
            
            return True
    
    def get_events(
        self,
        event_type: str = None,
        start_time: float = None,
        end_time: float = None,
        limit: int = None,
        publisher_id: str = None,
        filter_func: Callable[[Event], bool] = None
    ) -> List[Event]:
        """Get events from history
        
        Args:
            event_type: Type of events to get (optional)
            start_time: Start time for events (optional)
            end_time: End time for events (optional)
            limit: Maximum number of events to return (optional)
            publisher_id: Filter by publisher ID (optional)
            filter_func: Custom filter function (optional)
            
        Returns:
            list: List of events
        """
        with self.lock:
            # Get events of specified type or all events
            if event_type:
                events = self.events.get(event_type, [])
            else:
                events = [e for event_list in self.events.values() for e in event_list]
            
            # Apply filters
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
            
            if publisher_id:
                events = [e for e in events if e.publisher_id == publisher_id]
            
            if filter_func:
                events = [e for e in events if filter_func(e)]
            
            # Sort by timestamp (newest first)
            events = sorted(events, key=lambda e: e.timestamp, reverse=True)
            
            # Apply limit
            if limit:
                events = events[:limit]
            
            return events
    
    def clear(self, event_type: str = None) -> int:
        """Clear event history
        
        Args:
            event_type: Type of events to clear (optional, None for all)
            
        Returns:
            int: Number of events cleared
        """
        with self.lock:
            if event_type:
                # Clear specific event type
                count = len(self.events.get(event_type, []))
                self.event_count -= count
                self.events[event_type] = []
                return count
            else:
                # Clear all events
                count = self.event_count
                self.events = {}
                self.event_count = 0
                return count
    
    def _persist_event(self, event: Event) -> bool:
        """Persist event to storage
        
        Args:
            event: Event to persist
            
        Returns:
            bool: True if event was persisted
        """
        try:
            # Create event type directory if needed
            event_dir = os.path.join(self.storage_dir, event.event_type.replace('.', '_'))
            os.makedirs(event_dir, exist_ok=True)
            
            # Create filename based on timestamp and event ID
            filename = f"{int(event.timestamp)}_{event.event_id}.json"
            filepath = os.path.join(event_dir, filename)
            
            # Write event to file
            with open(filepath, 'w') as f:
                json.dump(event.to_dict(), f)
            
            return True
        except Exception as e:
            logger.error(f"Error persisting event: {e}")
            return False
    
    def load_persisted_events(
        self,
        event_type: str = None,
        start_time: float = None,
        end_time: float = None,
        limit: int = None
    ) -> List[Event]:
        """Load persisted events from storage
        
        Args:
            event_type: Type of events to load (optional)
            start_time: Start time for events (optional)
            end_time: End time for events (optional)
            limit: Maximum number of events to return (optional)
            
        Returns:
            list: List of events
        """
        if not self.storage_dir:
            return []
        
        events = []
        
        try:
            # Get event type directories
            if event_type:
                event_dirs = [os.path.join(self.storage_dir, event_type.replace('.', '_'))]
            else:
                event_dirs = [os.path.join(self.storage_dir, d) for d in os.listdir(self.storage_dir)
                             if os.path.isdir(os.path.join(self.storage_dir, d))]
            
            # Process each directory
            for event_dir in event_dirs:
                if not os.path.exists(event_dir):
                    continue
                
                # Get event files
                files = [f for f in os.listdir(event_dir) if f.endswith('.json')]
                
                # Apply time filters if specified
                if start_time or end_time:
                    filtered_files = []
                    for f in files:
                        try:
                            file_time = float(f.split('_')[0])
                            if start_time and file_time < start_time:
                                continue
                            if end_time and file_time > end_time:
                                continue
                            filtered_files.append(f)
                        except (ValueError, IndexError):
                            continue
                    files = filtered_files
                
                # Sort files by timestamp (newest first)
                files.sort(reverse=True)
                
                # Apply limit if specified
                if limit and len(events) + len(files) > limit:
                    files = files[:limit - len(events)]
                
                # Load events from files
                for f in files:
                    try:
                        with open(os.path.join(event_dir, f), 'r') as file:
                            event_dict = json.load(file)
                            events.append(Event.from_dict(event_dict))
                    except Exception as e:
                        logger.error(f"Error loading event from {f}: {e}")
                
                # Check if limit reached
                if limit and len(events) >= limit:
                    break
            
            return events
        
        except Exception as e:
            logger.error(f"Error loading persisted events: {e}")
            return []


class EventProcessor:
    """Base class for event processors"""
    
    def __init__(
        self,
        processor_id: str,
        event_types: List[str],
        description: str = None
    ):
        """Initialize event processor
        
        Args:
            processor_id: ID of processor
            event_types: Types of events to process
            description: Description of processor
        """
        self.processor_id = processor_id
        self.event_types = event_types
        self.description = description
    
    def process(self, event: Event) -> Optional[Event]:
        """Process event
        
        Args:
            event: Event to process
            
        Returns:
            Event: New event to publish (optional)
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    def get_info(self) -> Dict[str, Any]:
        """Get processor information
        
        Returns:
            dict: Processor information
        """
        return {
            "processor_id": self.processor_id,
            "event_types": self.event_types,
            "description": self.description
        }


class EventBus:
    """Central event distribution system with plugin support"""
    
    def __init__(
        self,
        config_registry=None,
        async_delivery: bool = True,
        worker_threads: int = 2,
        max_queue_size: int = 1000,
        history_size: int = 1000,
        history_storage_dir: str = None
    ):
        """Initialize event bus
        
        Args:
            config_registry: Configuration registry (optional)
            async_delivery: Whether to deliver events asynchronously
            worker_threads: Number of worker threads for async delivery
            max_queue_size: Maximum size of event queue
            history_size: Maximum number of events to keep in history
            history_storage_dir: Directory for persistent event storage
        """
        self.config_registry = config_registry
        self.async_delivery = async_delivery
        self.worker_threads = worker_threads
        self.max_queue_size = max_queue_size
        
        # Event type schemas
        self.event_schemas = {}  # {event_type: EventTypeSchema}
        
        # Subscriptions
        self.subscriptions = {}  # {subscriber_id: [EventSubscription]}
        self.subscriptions_by_type = {}  # {event_type: [EventSubscription]}
        
        # Event processors
        self.processors = {}  # {processor_id: EventProcessor}
        
        # Event history
        self.history = EventHistory(
            max_size=history_size,
            storage_dir=history_storage_dir
        )
        
        # Async delivery
        self.event_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.workers = []
        self.running = False
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize from config if provided
        if self.config_registry:
            self._init_from_config()
        
        logger.info("EventBus initialized")
    
    def _init_from_config(self):
        """Initialize from configuration registry"""
        if not self.config_registry:
            return
        
        try:
            # Get event bus configuration
            async_delivery = self.config_registry.get_parameter(
                "event_bus", "async_delivery", self.async_delivery)
            worker_threads = self.config_registry.get_parameter(
                "event_bus", "worker_threads", self.worker_threads)
            max_queue_size = self.config_registry.get_parameter(
                "event_bus", "max_queue_size", self.max_queue_size)
            
            # Update settings
            self.async_delivery = async_delivery
            self.worker_threads = worker_threads
            self.max_queue_size = max_queue_size
            
            logger.info("EventBus initialized from configuration")
        except Exception as e:
            logger.error(f"Error initializing EventBus from configuration: {e}")
    
    def start(self):
        """Start event bus"""
        with self.lock:
            if self.running:
                logger.warning("EventBus already running")
                return
            
            self.running = True
            
            # Start worker threads for async delivery
            if self.async_delivery:
                for i in range(self.worker_threads):
                    worker = threading.Thread(
                        target=self._worker_thread,
                        name=f"EventBus-Worker-{i}",
                        daemon=True
                    )
                    worker.start()
                    self.workers.append(worker)
                
                logger.info(f"EventBus started with {self.worker_threads} worker threads")
            else:
                logger.info("EventBus started in synchronous mode")
    
    def stop(self):
        """Stop event bus"""
        with self.lock:
            if not self.running:
                logger.warning("EventBus not running")
                return
            
            self.running = False
            
            # Stop worker threads
            if self.async_delivery:
                # Add sentinel values to queue to signal workers to stop
                for _ in range(len(self.workers)):
                    try:
                        self.event_queue.put((0, None), block=False)
                    except queue.Full:
                        pass
                
                # Wait for workers to finish
                for worker in self.workers:
                    worker.join(timeout=1.0)
                
                self.workers = []
            
            logger.info("EventBus stopped")
    
    def _worker_thread(self):
        """Worker thread for async event delivery"""
        while self.running:
            try:
                # Get event from queue
                priority, event = self.event_queue.get(timeout=0.1)
                
                # Check for sentinel value
                if event is None:
                    break
                
                # Deliver event to subscribers
                self._deliver_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
            
            except queue.Empty:
                # Queue is empty, continue
                continue
            
            except Exception as e:
                logger.error(f"Error in EventBus worker thread: {e}")
    
    def register_event_type(
        self,
        event_type: str,
        schema: Dict[str, Any] = None,
        description: str = None,
        required_fields: List[str] = None,
        example: Dict[str, Any] = None
    ) -> bool:
        """Register event type schema
        
        Args:
            event_type: Type of event
            schema: JSON schema for event data
            description: Description of event type
            required_fields: List of required fields
            example: Example event data
            
        Returns:
            bool: True if event type was registered
        """
        with self.lock:
            # Check if event type already registered
            if event_type in self.event_schemas:
                logger.warning(f"Event type {event_type} already registered")
                return False
            
            # Create event type schema
            schema_obj = EventTypeSchema(
                event_type=event_type,
                schema=schema,
                description=description,
                required_fields=required_fields,
                example=example
            )
            
            # Store schema
            self.event_schemas[event_type] = schema_obj
            
            logger.info(f"Event type {event_type} registered")
            return True
    
    def subscribe(
        self,
        subscriber_id: str,
        event_type: str,
        callback: Callable[[Event], None],
        filter_func: Callable[[Event], bool] = None,
        priority: int = 0,
        max_events_per_second: float = None
    ) -> str:
        """Subscribe to events
        
        Args:
            subscriber_id: ID of subscriber
            event_type: Type of event to subscribe to (can include wildcards)
            callback: Function to call when event is received
            filter_func: Function to filter events (optional)
            priority: Subscription priority (0 = normal, negative = lower, positive = higher)
            max_events_per_second: Maximum events per second (optional)
            
        Returns:
            str: Subscription ID
        """
        with self.lock:
            # Create subscription
            subscription = EventSubscription(
                subscriber_id=subscriber_id,
                event_type=event_type,
                callback=callback,
                filter_func=filter_func,
                priority=priority,
                max_events_per_second=max_events_per_second
            )
            
            # Initialize subscriber list if needed
            if subscriber_id not in self.subscriptions:
                self.subscriptions[subscriber_id] = []
            
            # Add subscription to subscriber list
            self.subscriptions[subscriber_id].append(subscription)
            
            # Initialize event type list if needed
            if event_type not in self.subscriptions_by_type:
                self.subscriptions_by_type[event_type] = []
            
            # Add subscription to event type list
            self.subscriptions_by_type[event_type].append(subscription)
            
            logger.info(f"Subscriber {subscriber_id} subscribed to {event_type}")
            return subscription.subscription_id
    
    def unsubscribe(
        self,
        subscription_id: str = None,
        subscriber_id: str = None,
        event_type: str = None
    ) -> int:
        """Unsubscribe from events
        
        Args:
            subscription_id: Specific subscription ID to unsubscribe (optional)
            subscriber_id: Unsubscribe all subscriptions for this subscriber (optional)
            event_type: Unsubscribe all subscriptions for this event type (optional)
            
        Returns:
            int: Number of subscriptions removed
        """
        with self.lock:
            count = 0
            
            # Unsubscribe specific subscription
            if subscription_id:
                for sid, subscriptions in self.subscriptions.items():
                    for i, sub in enumerate(subscriptions):
                        if sub.subscription_id == subscription_id:
                            # Remove from subscriber list
                            self.subscriptions[sid].pop(i)
                            
                            # Remove from event type list
                            if sub.event_type in self.subscriptions_by_type:
                                for j, type_sub in enumerate(self.subscriptions_by_type[sub.event_type]):
                                    if type_sub.subscription_id == subscription_id:
                                        self.subscriptions_by_type[sub.event_type].pop(j)
                                        break
                            
                            count = 1
                            break
                    
                    if count > 0:
                        break
            
            # Unsubscribe all subscriptions for subscriber
            elif subscriber_id:
                if subscriber_id in self.subscriptions:
                    count = len(self.subscriptions[subscriber_id])
                    
                    # Remove from event type lists
                    for sub in self.subscriptions[subscriber_id]:
                        if sub.event_type in self.subscriptions_by_type:
                            self.subscriptions_by_type[sub.event_type] = [
                                s for s in self.subscriptions_by_type[sub.event_type]
                                if s.subscriber_id != subscriber_id
                            ]
                    
                    # Remove from subscriber list
                    del self.subscriptions[subscriber_id]
            
            # Unsubscribe all subscriptions for event type
            elif event_type:
                if event_type in self.subscriptions_by_type:
                    count = len(self.subscriptions_by_type[event_type])
                    
                    # Remove from subscriber lists
                    for sub in self.subscriptions_by_type[event_type]:
                        if sub.subscriber_id in self.subscriptions:
                            self.subscriptions[sub.subscriber_id] = [
                                s for s in self.subscriptions[sub.subscriber_id]
                                if s.event_type != event_type
                            ]
                    
                    # Remove from event type list
                    del self.subscriptions_by_type[event_type]
            
            if count > 0:
                logger.info(f"Removed {count} subscriptions")
            
            return count
    
    def register_processor(
        self,
        processor: EventProcessor
    ) -> bool:
        """Register event processor
        
        Args:
            processor: Event processor to register
            
        Returns:
            bool: True if processor was registered
        """
        with self.lock:
            # Check if processor already registered
            if processor.processor_id in self.processors:
                logger.warning(f"Processor {processor.processor_id} already registered")
                return False
            
            # Store processor
            self.processors[processor.processor_id] = processor
            
            # Subscribe to event types
            for event_type in processor.event_types:
                self.subscribe(
                    subscriber_id=f"processor:{processor.processor_id}",
                    event_type=event_type,
                    callback=self._process_event,
                    filter_func=lambda e, pid=processor.processor_id: self._filter_processor_event(e, pid)
                )
            
            logger.info(f"Processor {processor.processor_id} registered for {processor.event_types}")
            return True
    
    def unregister_processor(
        self,
        processor_id: str
    ) -> bool:
        """Unregister event processor
        
        Args:
            processor_id: ID of processor to unregister
            
        Returns:
            bool: True if processor was unregistered
        """
        with self.lock:
            # Check if processor exists
            if processor_id not in self.processors:
                logger.warning(f"Processor {processor_id} not registered")
                return False
            
            # Unsubscribe from events
            self.unsubscribe(subscriber_id=f"processor:{processor_id}")
            
            # Remove processor
            del self.processors[processor_id]
            
            logger.info(f"Processor {processor_id} unregistered")
            return True
    
    def _filter_processor_event(self, event: Event, processor_id: str) -> bool:
        """Filter event for processor
        
        Args:
            event: Event to filter
            processor_id: ID of processor
            
        Returns:
            bool: True if event should be processed
        """
        # Prevent processors from processing their own events
        if event.publisher_id == f"processor:{processor_id}":
            return False
        
        return True
    
    def _process_event(self, event: Event):
        """Process event with registered processor
        
        Args:
            event: Event to process
        """
        # Extract processor ID from subscriber ID
        processor_id = event.metadata.get("processor_id")
        if not processor_id and ":" in event.metadata.get("subscriber_id", ""):
            processor_id = event.metadata.get("subscriber_id").split(":", 1)[1]
        
        if not processor_id or processor_id not in self.processors:
            return
        
        try:
            # Process event
            processor = self.processors[processor_id]
            result = processor.process(event)
            
            # Publish result if returned
            if result:
                self.publish(
                    event_type=result.event_type,
                    data=result.data,
                    publisher_id=f"processor:{processor_id}",
                    priority=result.priority,
                    metadata=result.metadata
                )
        
        except Exception as e:
            logger.error(f"Error processing event with processor {processor_id}: {e}")
    
    def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        publisher_id: str = None,
        priority: int = 0,
        metadata: Dict[str, Any] = None,
        validate: bool = True
    ) -> Optional[str]:
        """Publish event
        
        Args:
            event_type: Type of event
            data: Event data
            publisher_id: ID of publisher (optional)
            priority: Event priority (0 = normal, negative = lower, positive = higher)
            metadata: Additional metadata
            validate: Whether to validate event data against schema
            
        Returns:
            str: Event ID if published, None if validation failed
        """
        # Validate event data if schema exists and validation requested
        if validate and event_type in self.event_schemas:
            schema = self.event_schemas[event_type]
            if not schema.validate(data):
                logger.warning(f"Event data validation failed for {event_type}")
                return None
        
        # Create event
        event = Event(
            event_type=event_type,
            data=data,
            publisher_id=publisher_id,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Add to history
        self.history.add_event(event)
        
        # Deliver event
        if self.async_delivery:
            try:
                # Add to queue with priority (negated so higher priority comes first)
                self.event_queue.put((-priority, event), block=False)
            except queue.Full:
                logger.warning("Event queue full, dropping event")
                return event.event_id
        else:
            # Deliver synchronously
            self._deliver_event(event)
        
        return event.event_id
    
    def _deliver_event(self, event: Event):
        """Deliver event to subscribers
        
        Args:
            event: Event to deliver
        """
        # Find matching subscriptions
        matching_subs = []
        
        with self.lock:
            # Check all subscription types
            for event_type, subs in self.subscriptions_by_type.items():
                for sub in subs:
                    if sub.should_deliver(event):
                        matching_subs.append(sub)
        
        # Sort by priority (higher first)
        matching_subs.sort(key=lambda s: s.priority, reverse=True)
        
        # Deliver to subscribers
        for sub in matching_subs:
            try:
                # Add metadata about subscription
                event.metadata["subscriber_id"] = sub.subscriber_id
                
                # Deliver event
                sub.deliver(event)
            
            except Exception as e:
                logger.error(f"Error delivering event to subscriber {sub.subscriber_id}: {e}")
    
    def get_event_history(
        self,
        event_type: str = None,
        start_time: float = None,
        end_time: float = None,
        limit: int = None,
        publisher_id: str = None,
        include_persisted: bool = False
    ) -> List[Event]:
        """Get event history
        
        Args:
            event_type: Type of events to get (optional)
            start_time: Start time for events (optional)
            end_time: End time for events (optional)
            limit: Maximum number of events to return (optional)
            publisher_id: Filter by publisher ID (optional)
            include_persisted: Whether to include persisted events
            
        Returns:
            list: List of events
        """
        # Get in-memory events
        events = self.history.get_events(
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            publisher_id=publisher_id
        )
        
        # Include persisted events if requested
        if include_persisted:
            # Determine limit for persisted events
            persisted_limit = None
            if limit:
                persisted_limit = limit - len(events)
                if persisted_limit <= 0:
                    return events
            
            # Get persisted events
            persisted_events = self.history.load_persisted_events(
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
                limit=persisted_limit
            )
            
            # Filter by publisher ID if specified
            if publisher_id:
                persisted_events = [e for e in persisted_events if e.publisher_id == publisher_id]
            
            # Combine events
            events.extend(persisted_events)
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda e: e.timestamp, reverse=True)
            
            # Apply limit
            if limit:
                events = events[:limit]
        
        return events
    
    def clear_history(self, event_type: str = None) -> int:
        """Clear event history
        
        Args:
            event_type: Type of events to clear (optional, None for all)
            
        Returns:
            int: Number of events cleared
        """
        return self.history.clear(event_type)
    
    def get_event_types(self) -> List[str]:
        """Get registered event types
        
        Returns:
            list: List of event types
        """
        return list(self.event_schemas.keys())
    
    def get_event_schema(self, event_type: str) -> Optional[EventTypeSchema]:
        """Get event type schema
        
        Args:
            event_type: Type of event
            
        Returns:
            EventTypeSchema: Event type schema or None if not found
        """
        return self.event_schemas.get(event_type)
    
    def get_subscribers(self, event_type: str = None) -> List[Dict[str, Any]]:
        """Get subscribers
        
        Args:
            event_type: Filter by event type (optional)
            
        Returns:
            list: List of subscriber information
        """
        with self.lock:
            result = []
            
            if event_type:
                # Get subscribers for specific event type
                if event_type in self.subscriptions_by_type:
                    for sub in self.subscriptions_by_type[event_type]:
                        result.append({
                            "subscriber_id": sub.subscriber_id,
                            "subscription_id": sub.subscription_id,
                            "event_type": sub.event_type,
                            "priority": sub.priority,
                            "max_events_per_second": sub.max_events_per_second
                        })
            else:
                # Get all subscribers
                for subscriber_id, subs in self.subscriptions.items():
                    for sub in subs:
                        result.append({
                            "subscriber_id": subscriber_id,
                            "subscription_id": sub.subscription_id,
                            "event_type": sub.event_type,
                            "priority": sub.priority,
                            "max_events_per_second": sub.max_events_per_second
                        })
            
            return result
    
    def get_processors(self) -> List[Dict[str, Any]]:
        """Get registered processors
        
        Returns:
            list: List of processor information
        """
        with self.lock:
            return [p.get_info() for p in self.processors.values()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics
        
        Returns:
            dict: Event bus statistics
        """
        with self.lock:
            return {
                "event_types": len(self.event_schemas),
                "subscribers": len(self.subscriptions),
                "subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
                "processors": len(self.processors),
                "events_in_memory": self.history.event_count,
                "queue_size": self.event_queue.qsize() if self.async_delivery else 0,
                "worker_threads": len(self.workers),
                "running": self.running
            }


# Example usage
if __name__ == "__main__":
    # Create event bus
    event_bus = EventBus(
        async_delivery=True,
        worker_threads=2,
        history_size=1000
    )
    
    # Register event types
    event_bus.register_event_type(
        event_type="system.log",
        description="System log event",
        required_fields=["level", "message"]
    )
    
    event_bus.register_event_type(
        event_type="market.price",
        description="Market price update",
        required_fields=["symbol", "price"]
    )
    
    # Define event handlers
    def log_handler(event):
        print(f"Log: [{event.data['level']}] {event.data['message']}")
    
    def price_handler(event):
        print(f"Price: {event.data['symbol']} = {event.data['price']}")
    
    # Subscribe to events
    event_bus.subscribe(
        subscriber_id="logger",
        event_type="system.log",
        callback=log_handler
    )
    
    event_bus.subscribe(
        subscriber_id="price_tracker",
        event_type="market.price",
        callback=price_handler
    )
    
    # Start event bus
    event_bus.start()
    
    # Publish events
    event_bus.publish(
        event_type="system.log",
        data={
            "level": "INFO",
            "message": "System started"
        }
    )
    
    event_bus.publish(
        event_type="market.price",
        data={
            "symbol": "BTC/USDC",
            "price": 50000.0
        }
    )
    
    # Wait for events to be processed
    time.sleep(1)
    
    # Stop event bus
    event_bus.stop()
