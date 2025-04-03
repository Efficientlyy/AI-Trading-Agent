"""Tests for the base component class."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.common.component import Component
from src.common.events import Event, event_bus
from src.models.events import ErrorEvent, SystemStatusEvent


class TestComponent(Component):
    """Test component for testing."""
    
    def __init__(self, name: str = "TestComponent"):
        """Initialize the test component."""
        super().__init__(name)
        self.initialize_called = False
        self.start_called = False
        self.stop_called = False
    
    async def _initialize(self) -> None:
        """Initialize the test component."""
        self.initialize_called = True
    
    async def _start(self) -> None:
        """Start the test component."""
        self.start_called = True
    
    async def _stop(self) -> None:
        """Stop the test component."""
        self.stop_called = True


@pytest.fixture
def component():
    """Fixture to provide a test component."""
    return TestComponent()


@pytest.mark.asyncio
async def test_initialization(component):
    """Test component initialization."""
    component.initialize()
    
    assert component.initialized
    assert component.initialize_called
    assert not component.running
    assert not component.start_called
    assert not component.stop_called


@pytest.mark.asyncio
async def test_start(component):
    """Test component start."""
    component.start()
    
    assert component.initialized
    assert component.initialize_called
    assert component.running
    assert component.start_called
    assert not component.stop_called


@pytest.mark.asyncio
async def test_stop(component):
    """Test component stop."""
    component.start()
    component.stop()
    
    assert component.initialized
    assert component.initialize_called
    assert not component.running
    assert component.start_called
    assert component.stop_called


@pytest.mark.asyncio
async def test_get_config(component):
    """Test getting component configuration."""
    # Mock the config.get method
    with patch("src.common.component.config") as mock_config:
        mock_config.get.return_value = "test_value"
        
        # Get a configuration value
        value = component.get_config("test_key")
        
        # Check that the config.get method was called with the correct key
        mock_config.get.assert_called_once_with("testcomponent.test_key", None)
        
        # Check that the correct value was returned
        assert value == "test_value"


@pytest.mark.asyncio
async def test_publish_event(component):
    """Test publishing an event."""
    # Initialize the event bus (register event types)
    event_bus.register_event_type("TestEvent")
    
    # Create a mock subscriber
    mock_subscriber = MagicMock()
    
    # Subscribe to the test event
    event_bus.subscribe("TestEvent", mock_subscriber)
    
    # Start the event bus
    event_bus.start()
    
    try:
        # Create a test event
        event = Event(event_type="TestEvent", source="test")
        
        # Publish the event
        await component.publish_event(event)
        
        # Wait for the event to be processed
        await asyncio.sleep(0.1)
        
        # Check that the subscriber was called with the event
        mock_subscriber.assert_called_once()
        assert mock_subscriber.call_args[0][0].event_id == event.event_id
    finally:
        # Stop the event bus
        event_bus.stop()
        
        # Unsubscribe from the test event
        event_bus.unsubscribe("TestEvent", mock_subscriber)


@pytest.mark.asyncio
async def test_publish_error(component):
    """Test publishing an error event."""
    # Initialize the event bus (register event types)
    event_bus.register_event_type("ErrorEvent")
    
    # Create a mock subscriber
    mock_subscriber = MagicMock()
    
    # Subscribe to the error event
    event_bus.subscribe("ErrorEvent", mock_subscriber)
    
    # Start the event bus
    event_bus.start()
    
    try:
        # Publish an error event
        await component.publish_error("TestError", "Test error message")
        
        # Wait for the event to be processed
        await asyncio.sleep(0.1)
        
        # Check that the subscriber was called
        mock_subscriber.assert_called_once()
        
        # Check the error event details
        error_event = mock_subscriber.call_args[0][0]
        assert isinstance(error_event, ErrorEvent)
        assert error_event.source == "TestComponent"
        assert error_event.error_type == "TestError"
        assert error_event.error_message == "Test error message"
    finally:
        # Stop the event bus
        event_bus.stop()
        
        # Unsubscribe from the error event
        event_bus.unsubscribe("ErrorEvent", mock_subscriber)


@pytest.mark.asyncio
async def test_publish_status(component):
    """Test publishing a status event."""
    # Initialize the event bus (register event types)
    event_bus.register_event_type("SystemStatusEvent")
    
    # Create a mock subscriber
    mock_subscriber = MagicMock()
    
    # Subscribe to the status event
    event_bus.subscribe("SystemStatusEvent", mock_subscriber)
    
    # Start the event bus
    event_bus.start()
    
    try:
        # Publish a status event
        await component.publish_status("test_status", "Test status message")
        
        # Wait for the event to be processed
        await asyncio.sleep(0.1)
        
        # Check that the subscriber was called
        mock_subscriber.assert_called_once()
        
        # Check the status event details
        status_event = mock_subscriber.call_args[0][0]
        assert isinstance(status_event, SystemStatusEvent)
        assert status_event.source == "TestComponent"
        assert status_event.status == "test_status"
        assert status_event.message == "Test status message"
    finally:
        # Stop the event bus
        event_bus.stop()
        
        # Unsubscribe from the status event
        event_bus.unsubscribe("SystemStatusEvent", mock_subscriber)


@pytest.mark.asyncio
async def test_create_task(component):
    """Test creating a component task."""
    # Mock coroutine
    async def mock_coro():
        await asyncio.sleep(0.1)
        return "test_result"
    
    # Create a task
    task = component.create_task(mock_coro())
    
    # Check that the task was added to the component's tasks
    assert task in component.tasks
    
    # Wait for the task to complete
    result = await task
    
    # Check that the task was removed from the component's tasks
    assert task not in component.tasks
    
    # Check the task result
    assert result == "test_result" 