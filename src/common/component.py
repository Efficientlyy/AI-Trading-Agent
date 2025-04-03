"""Base component class for all system components.

This module provides a common base class that all components should inherit from,
ensuring consistent initialization, configuration, lifecycle management, and cleanup.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

import structlog

from src.common.config import config
from src.common.events import Event, event_bus
from src.common.logging import get_logger
from src.models.events import ErrorEvent, SystemStatusEvent


class Component(ABC):
    """Base class for all system components."""
    
    def __init__(self, name: str, depends_on: Optional[List[str]] = None):
        """
        Initialize the component.
        
        Args:
            name: Component name
            depends_on: List of component names this component depends on
        """
        self.name = name
        self.depends_on = depends_on or []
        self.logger = get_logger(name)
        self.config_prefix = name.lower().replace(" ", "_")
        self.running = False
        self.initialized = False
        self.tasks: List[asyncio.Task] = []
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value for this component.
        
        Args:
            key: Configuration key (without component prefix)
            default: Default value if key not found
            
        Returns:
            The configuration value
        """
        full_key = f"{self.config_prefix}.{key}"
        return config.get(full_key, default)
    
    async def initialize(self) -> None:
        """
        Initialize the component. This is called before start.
        
        Can be overridden by subclasses for additional initialization.
        """
        if self.initialized:
            return
        
        self.logger.info("Initializing component")
        
        # Register event handlers
        self._register_event_handlers()
        
        # Call subclass initialization
        self._initialize()
        
        self.initialized = True
        self.logger.info("Component initialized")
        
        # Publish system status event
        await self.publish_status("initialized", "Component initialized")
    
    @abstractmethod
    async def _initialize(self) -> None:
        """
        Initialize the component. To be implemented by subclasses.
        """
        pass
    
    async def start(self) -> None:
        """
        Start the component.
        
        This starts the component's main processing and creates any necessary tasks.
        """
        if self.running:
            return
        
        if not self.initialized:
            self.initialize()
        
        self.logger.info("Starting component")
        
        # Call subclass start
        self._start()
        
        self.running = True
        self.logger.info("Component started")
        
        # Publish system status event
        await self.publish_status("running", "Component running")
    
    @abstractmethod
    async def _start(self) -> None:
        """
        Start the component. To be implemented by subclasses.
        """
        pass
    
    async def stop(self) -> None:
        """
        Stop the component.
        
        This stops all tasks and releases resources.
        """
        if not self.running:
            return
        
        self.logger.info("Stopping component")
        
        # Cancel all running tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks = []
        
        # Call subclass stop
        self._stop()
        
        self.running = False
        self.logger.info("Component stopped")
        
        # Publish system status event
        await self.publish_status("stopped", "Component stopped")
    
    @abstractmethod
    async def _stop(self) -> None:
        """
        Stop the component. To be implemented by subclasses.
        """
        pass
    
    def _register_event_handlers(self) -> None:
        """
        Register event handlers for this component.
        
        Can be overridden by subclasses to register additional event handlers.
        """
        pass
    
    def create_task(self, coro) -> asyncio.Task:
        """
        Create a component task that will be automatically managed.
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The created task
        """
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        
        # Set up task cleanup
        task.add_done_callback(lambda t: self.tasks.remove(t) if t in self.tasks else None)
        
        return task
    
    async def publish_event(self, event: Event) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: The event to publish
        """
        try:
            await event_bus.publish(event)
        except Exception as e:
            self.logger.exception("Error publishing event", event_type=event.event_type, error=str(e))
    
    async def publish_error(self, error_type: str, error_message: str, 
                            error_details: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish an error event.
        
        Args:
            error_type: Type of error
            error_message: Error message
            error_details: Additional error details
        """
        event = ErrorEvent(
            source=self.name,
            error_type=error_type,
            error_message=error_message,
            error_details=error_details or {},
        )
        await self.publish_event(event)
    
    async def publish_status(self, status: str, message: str, 
                             details: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish a system status event.
        
        Args:
            status: Component status
            message: Status message
            details: Additional status details
        """
        event = SystemStatusEvent(
            source=self.name,
            status=status,
            message=message,
            details=details or {},
        )
        await self.publish_event(event) 