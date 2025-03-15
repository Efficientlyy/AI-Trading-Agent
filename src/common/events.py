"""Event system for inter-component communication.

This module provides a centralized event bus for publishing and subscribing to events
across different components of the system.
"""

import asyncio
import inspect
import json
import uuid
from asyncio import Queue
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, cast

import structlog
from pydantic import BaseModel, Field

from src.common.config import config
from src.common.datetime_utils import utc_now
from src.common.logging import get_logger

# Configure logger
logger = get_logger("system", "event_bus")


class EventPriority(Enum):
    """Priority levels for events."""

    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


class Event(BaseModel):
    """Base class for all system events."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(...)
    timestamp: datetime = Field(default_factory=utc_now)
    priority: EventPriority = Field(default=EventPriority.NORMAL)
    source: str = Field(...)
    payload: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class EventBus:
    """
    Central event bus for the system.
    
    Handles event publishing, subscription, and distribution to listeners.
    """

    _instance = None
    _initialized = False
    
    # Event queue
    _queue: "Queue[Event]" = None
    
    # Event subscribers by event type
    _subscribers: Dict[str, List[Callable]] = {}
    
    # Event types that can be subscribed to
    _registered_event_types: Set[str] = set()

    def __new__(cls, *args, **kwargs):
        """Implement the singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the event bus."""
        if not self._initialized:
            self._queue = asyncio.Queue(
                maxsize=config.get("system.event_bus.buffer_size", 1000)
            )
            self._subscribers = {}
            self._registered_event_types = set()
            self._processing_task = None
            self._running = False
            self._initialized = True

    def register_event_type(self, event_type: str) -> None:
        """
        Register an event type that can be published and subscribed to.
        
        Args:
            event_type: The type of event to register
        """
        self._registered_event_types.add(event_type)
        logger.debug("Registered event type", event_type=event_type)

    def register_event_class(self, event_class: Type[Event]) -> None:
        """
        Register an event class and its event type.
        
        Args:
            event_class: The event class to register
        """
        event_type = event_class.__name__
        self._registered_event_types.add(event_type)
        logger.debug("Registered event class", event_class=event_type)

    async def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: The event to publish
            
        Raises:
            ValueError: If the event type is not registered
        """
        if event.event_type not in self._registered_event_types:
            raise ValueError(f"Event type '{event.event_type}' is not registered")
        
        await self._queue.put(event)
        logger.debug(
            "Published event",
            event_id=event.event_id,
            event_type=event.event_type,
            priority=event.priority.name,
        )

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: The type of event to subscribe to
            callback: The callback function to call when an event is received
            
        Raises:
            ValueError: If the event type is not registered
        """
        if event_type not in self._registered_event_types:
            raise ValueError(f"Event type '{event_type}' is not registered")
        
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        logger.debug(
            "Subscribed to event",
            event_type=event_type,
            callback=callback.__name__,
        )

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: The callback function to unsubscribe
            
        Raises:
            ValueError: If the event type is not registered or the callback is not subscribed
        """
        if event_type not in self._registered_event_types:
            raise ValueError(f"Event type '{event_type}' is not registered")
        
        if event_type not in self._subscribers:
            raise ValueError(f"No subscribers for event type '{event_type}'")
        
        if callback not in self._subscribers[event_type]:
            raise ValueError(f"Callback not subscribed to event type '{event_type}'")
        
        self._subscribers[event_type].remove(callback)
        logger.debug(
            "Unsubscribed from event",
            event_type=event_type,
            callback=callback.__name__,
        )

    async def start(self) -> None:
        """Start processing events from the queue."""
        if self._running:
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop processing events from the queue."""
        if not self._running:
            return
        
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
        
        logger.info("Event bus stopped")

    async def _process_events(self) -> None:
        """Process events from the queue and dispatch them to subscribers."""
        batch_size = config.get("system.event_bus.batch_size", 100)
        worker_count = config.get("system.event_bus.worker_count", 4)
        retry_count = config.get("system.event_bus.retry_count", 3)
        retry_delay_ms = config.get("system.event_bus.retry_delay_ms", 500)
        
        while self._running:
            try:
                # Process events in batches if available
                events = []
                for _ in range(batch_size):
                    try:
                        event = self._queue.get_nowait()
                        events.append(event)
                    except asyncio.QueueEmpty:
                        break
                
                if not events:
                    # If no events in the queue, wait for one
                    event = await self._queue.get()
                    events.append(event)
                
                # Process all collected events
                await asyncio.gather(
                    *[self._dispatch_event(event) for event in events]
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error processing events", error=str(e))
                await asyncio.sleep(1)  # Prevent busy-waiting on repeated errors

    async def _dispatch_event(self, event: Event) -> None:
        """
        Dispatch an event to its subscribers.
        
        Args:
            event: The event to dispatch
        """
        event_type = event.event_type
        
        if event_type not in self._subscribers or not self._subscribers[event_type]:
            # Mark as done even if no subscribers
            self._queue.task_done()
            return
        
        subscribers = self._subscribers[event_type]
        
        try:
            # Run all subscriber callbacks
            await asyncio.gather(
                *[self._call_subscriber(subscriber, event) for subscriber in subscribers]
            )
        except Exception as e:
            logger.exception(
                "Error dispatching event",
                event_id=event.event_id,
                event_type=event.event_type,
                error=str(e),
            )
        finally:
            # Mark as done after all subscribers have been called
            self._queue.task_done()

    async def _call_subscriber(self, subscriber: Callable, event: Event) -> None:
        """
        Call a subscriber with an event.
        
        Args:
            subscriber: The subscriber to call
            event: The event to pass to the subscriber
        """
        retry_count = config.get("system.event_bus.retry_count", 3)
        retry_delay_ms = config.get("system.event_bus.retry_delay_ms", 500)
        
        for attempt in range(retry_count + 1):
            try:
                # Check if the subscriber is a coroutine function
                if inspect.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
                return
            except Exception as e:
                if attempt < retry_count:
                    # Log and retry
                    logger.warning(
                        "Error calling subscriber, retrying",
                        subscriber=subscriber.__name__,
                        event_id=event.event_id,
                        attempt=attempt + 1,
                        max_attempts=retry_count + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(retry_delay_ms / 1000)
                else:
                    # Log the final error
                    logger.exception(
                        "Error calling subscriber, max retries exceeded",
                        subscriber=subscriber.__name__,
                        event_id=event.event_id,
                        error=str(e),
                    )


# Create a singleton instance
event_bus = EventBus() 