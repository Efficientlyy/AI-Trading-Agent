"""Tests for the event system."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.common.events import Event, EventBus, EventPriority


class TestEvent(Event):
    """Test event class for testing."""
    
    event_type: str = "TestEvent"
    source: str = "test"


@pytest.fixture
def event_bus():
    """Fixture to provide a clean event bus instance for each test."""
    # Create a fresh event bus instance (reset the singleton)
    EventBus._instance = None
    EventBus._initialized = False
    EventBus._subscribers = {}
    EventBus._registered_event_types = set()
    bus = EventBus()
    
    # Register the test event type
    bus.register_event_type("TestEvent")
    
    # Start the event bus
    asyncio.create_task(bus.start())
    
    yield bus
    
    # Stop the event bus
    asyncio.create_task(bus.stop())


@pytest.mark.asyncio
async def test_singleton_pattern():
    """Test that EventBus implements the singleton pattern."""
    # Create a fresh event bus instance (reset the singleton)
    EventBus._instance = None
    EventBus._initialized = False
    
    bus1 = EventBus()
    bus2 = EventBus()
    
    # Both instances should be the same object
    assert bus1 is bus2


@pytest.mark.asyncio
async def test_register_event_type(event_bus):
    """Test registering event types."""
    event_bus.register_event_type("AnotherEvent")
    
    # Check that the event type was registered
    assert "AnotherEvent" in event_bus._registered_event_types


@pytest.mark.asyncio
async def test_register_event_class(event_bus):
    """Test registering event classes."""
    class AnotherTestEvent(Event):
        event_type: str = "AnotherTestEvent"
        source: str = "test"
    
    event_bus.register_event_class(AnotherTestEvent)
    
    # Check that the event type was registered
    assert "AnotherTestEvent" in event_bus._registered_event_types


@pytest.mark.asyncio
async def test_subscribe_and_publish(event_bus):
    """Test subscribing to events and publishing them."""
    # Create a mock callback
    mock_callback = MagicMock()
    
    # Subscribe to the test event
    event_bus.subscribe("TestEvent", mock_callback)
    
    # Create a test event
    event = TestEvent(source="test", payload={"test": "data"})
    
    # Publish the event
    await event_bus.publish(event)
    
    # Wait a bit for the event to be processed
    await asyncio.sleep(0.1)
    
    # Check that the callback was called with the event
    mock_callback.assert_called_once()
    assert mock_callback.call_args[0][0].event_id == event.event_id


@pytest.mark.asyncio
async def test_unsubscribe(event_bus):
    """Test unsubscribing from events."""
    # Create a mock callback
    mock_callback = MagicMock()
    
    # Subscribe to the test event
    event_bus.subscribe("TestEvent", mock_callback)
    
    # Unsubscribe from the test event
    event_bus.unsubscribe("TestEvent", mock_callback)
    
    # Create a test event
    event = TestEvent(source="test", payload={"test": "data"})
    
    # Publish the event
    await event_bus.publish(event)
    
    # Wait a bit for the event to be processed
    await asyncio.sleep(0.1)
    
    # Check that the callback was not called
    mock_callback.assert_not_called()


@pytest.mark.asyncio
async def test_publish_unregistered_event(event_bus):
    """Test publishing an unregistered event."""
    class UnregisteredEvent(Event):
        event_type: str = "UnregisteredEvent"
        source: str = "test"
    
    # Create an unregistered event
    event = UnregisteredEvent(source="test", payload={"test": "data"})
    
    # Publishing the event should raise a ValueError
    with pytest.raises(ValueError):
        await event_bus.publish(event)


@pytest.mark.asyncio
async def test_async_subscriber(event_bus):
    """Test async subscriber callbacks."""
    # Create a mock for tracking calls
    mock_tracker = MagicMock()
    
    # Create an async callback
    async def async_callback(event):
        await asyncio.sleep(0.1)  # Simulate async processing
        mock_tracker()
    
    # Subscribe to the test event
    event_bus.subscribe("TestEvent", async_callback)
    
    # Create a test event
    event = TestEvent(source="test", payload={"test": "data"})
    
    # Publish the event
    await event_bus.publish(event)
    
    # Wait for the async callback to complete
    await asyncio.sleep(0.2)
    
    # Check that the callback was called
    mock_tracker.assert_called_once()


@pytest.mark.asyncio
async def test_event_priority(event_bus):
    """Test event priorities."""
    # Create a test event with different priorities
    low_priority = TestEvent(
        source="test",
        priority=EventPriority.LOW,
        payload={"priority": "low"}
    )
    high_priority = TestEvent(
        source="test",
        priority=EventPriority.HIGH,
        payload={"priority": "high"}
    )
    
    # Both should be publishable
    await event_bus.publish(low_priority)
    await event_bus.publish(high_priority)
    
    # Wait for events to be processed
    await asyncio.sleep(0.1)
    
    # No assertion here, just checking that they can be published 