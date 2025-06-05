#!/usr/bin/env python
"""
Integration Tests for System Overseer

This module contains integration tests for the System Overseer components,
including the module registry, config registry, event bus, plugin manager,
and conversational interface.
"""

import os
import sys
import json
import time
import logging
import unittest
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Type
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import System Overseer components
from system_overseer.module_registry import ModuleRegistry
from system_overseer.config_registry import ConfigRegistry
from system_overseer.event_bus import EventBus, Event, EventHistory, EventFilter
from system_overseer.core import SystemCore
from system_overseer.plugin_manager import PluginManager, PluginInterface
from system_overseer.llm_client import LLMClient, LLMMessage, LLMResponse
from system_overseer.dialogue_manager import DialogueManager
from system_overseer.personality_system import PersonalitySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_overseer.tests.integration")


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, provider_id: str = "mock", name: str = "Mock Provider"):
        """Initialize mock provider.
        
        Args:
            provider_id: Provider identifier
            name: Provider name
        """
        self.provider_id = provider_id
        self.name = name  # Added name attribute to fix test failures
        self.calls = []
        # Add models attribute to fix test failures
        self.models = ["gpt-3.5-turbo", "gpt-4"]
    
    def generate_response(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        options: Dict[str, Any] = None
    ) -> LLMResponse:
        """Generate response.
        
        Args:
            messages: Input messages
            max_tokens: Maximum tokens
            temperature: Temperature
            options: Additional options
            
        Returns:
            LLMResponse: Response
        """
        # Record call
        self.calls.append({
            "messages": [m.to_dict() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "options": options
        })
        
        # Create response
        system_message = next((m for m in messages if m.role == "system"), None)
        user_message = next((m for m in messages if m.role == "user"), None)
        
        # Generate simple response
        content = f"This is a mock response to: {user_message.content if user_message else 'No user message'}"
        
        # Create message
        message = LLMMessage(role="assistant", content=content)
        
        # Create response using the correct constructor signature
        return LLMResponse(
            message=message,
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in messages),
                "completion_tokens": len(content.split()),
                "total_tokens": sum(len(m.content.split()) for m in messages) + len(content.split())
            }
        )
    
    # Add get_completion method to fix test failures with **kwargs to handle any arguments
    def get_completion(
        self,
        messages: List[LLMMessage],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs  # Accept any additional kwargs
    ) -> LLMResponse:
        """Get completion from LLM.
        
        This is an alias for generate_response to maintain compatibility with LLMClient.
        
        Args:
            messages: Input messages
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Temperature
            **kwargs: Additional keyword arguments
            
        Returns:
            LLMResponse: Response
        """
        # Extract options from kwargs if needed
        options = kwargs.get('options', {})
        
        return self.generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            options=options
        )


class MockPlugin(PluginInterface):
    """Mock plugin for testing."""
    
    def __init__(
        self,
        plugin_id: str,
        name: str,
        description: str,
        version: str = "1.0.0",
        dependencies: List[str] = None
    ):
        """Initialize mock plugin.
        
        Args:
            plugin_id: Plugin identifier
            name: Plugin name
            description: Plugin description
            version: Plugin version
            dependencies: Plugin dependencies
        """
        self._plugin_id = plugin_id
        self._name = name
        self._description = description
        self._version = version
        self._dependencies = dependencies or []
        self.initialized = False
        self.running = False
        self.context = None
    
    @property
    def plugin_id(self) -> str:
        """Get plugin identifier.
        
        Returns:
            str: Plugin identifier
        """
        return self._plugin_id
    
    @property
    def name(self) -> str:
        """Get plugin name.
        
        Returns:
            str: Plugin name
        """
        return self._name
    
    @property
    def description(self) -> str:
        """Get plugin description.
        
        Returns:
            str: Plugin description
        """
        return self._description
    
    @property
    def version(self) -> str:
        """Get plugin version.
        
        Returns:
            str: Plugin version
        """
        return self._version
    
    @property
    def dependencies(self) -> List[str]:
        """Get plugin dependencies.
        
        Returns:
            list: List of plugin identifiers
        """
        return self._dependencies
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize plugin.
        
        Args:
            context: Plugin context
            
        Returns:
            bool: True if initialization successful
        """
        self.initialized = True
        self.context = context
        return True
    
    def start(self) -> bool:
        """Start plugin.
        
        Returns:
            bool: True if start successful
        """
        if not self.initialized:
            return False
        
        self.running = True
        return True
    
    def stop(self) -> bool:
        """Stop plugin.
        
        Returns:
            bool: True if stop successful
        """
        self.running = False
        return True


class TestModuleRegistry(unittest.TestCase):
    """Test module registry."""
    
    def setUp(self):
        """Set up test case."""
        self.registry = ModuleRegistry()
    
    def test_register_module(self):
        """Test registering a module."""
        # Register module
        module = {"id": "test_module", "name": "Test Module"}
        self.assertTrue(self.registry.register_module("test_module", module))
        
        # Check if module is registered
        self.assertTrue(self.registry.has_module("test_module"))
        self.assertEqual(self.registry.get_module("test_module"), module)
    
    def test_register_service(self):
        """Test registering a service."""
        # Register service
        service = {"id": "test_service", "name": "Test Service"}
        self.assertTrue(self.registry.register_service("test_service", service))
        
        # Check if service is registered
        self.assertTrue(self.registry.has_service("test_service"))
        self.assertEqual(self.registry.get_service("test_service"), service)
    
    def test_get_all_modules(self):
        """Test getting all modules."""
        # Register modules
        module1 = {"id": "module1", "name": "Module 1"}
        module2 = {"id": "module2", "name": "Module 2"}
        self.registry.register_module("module1", module1)
        self.registry.register_module("module2", module2)
        
        # Get all modules
        modules = self.registry.get_all_modules()
        self.assertEqual(len(modules), 2)
        self.assertIn("module1", modules)
        self.assertIn("module2", modules)
    
    def test_get_all_services(self):
        """Test getting all services."""
        # Register services
        service1 = {"id": "service1", "name": "Service 1"}
        service2 = {"id": "service2", "name": "Service 2"}
        self.registry.register_service("service1", service1)
        self.registry.register_service("service2", service2)
        
        # Get all services
        services = self.registry.get_all_services()
        self.assertEqual(len(services), 2)
        self.assertIn("service1", services)
        self.assertIn("service2", services)


class TestConfigRegistry(unittest.TestCase):
    """Test configuration registry."""
    
    def setUp(self):
        """Set up test case."""
        self.config_dir = os.path.join(os.path.dirname(__file__), "test_data", "config")
        os.makedirs(self.config_dir, exist_ok=True)
        self.registry = ConfigRegistry(config_dir=self.config_dir)
    
    def tearDown(self):
        """Tear down test case."""
        # Clean up config files
        for filename in os.listdir(self.config_dir):
            os.remove(os.path.join(self.config_dir, filename))
    
    def test_register_parameter(self):
        """Test registering a parameter."""
        # Register parameter
        self.assertTrue(self.registry.register_parameter(
            module_id="test_module",
            param_id="test_param",
            default_value="test_value",
            description="Test parameter"
        ))
        
        # Check if parameter is registered
        self.assertTrue(self.registry.has_parameter("test_module", "test_param"))
        
        # Get parameter
        param = self.registry.get_parameter_info("test_module", "test_param")
        self.assertEqual(param["default_value"], "test_value")
        self.assertEqual(param["description"], "Test parameter")
    
    def test_set_get_parameter(self):
        """Test setting and getting a parameter."""
        # Register parameter
        self.registry.register_parameter(
            module_id="test_module",
            param_id="test_param",
            default_value="default_value",
            description="Test parameter"
        )
        
        # Set parameter
        self.assertTrue(self.registry.set_parameter(
            module_id="test_module",
            param_id="test_param",
            value="new_value",
            user_id="test_user"
        ))
        
        # Get parameter
        value = self.registry.get_parameter("test_module", "test_param")
        self.assertEqual(value, "new_value")
    
    def test_get_module_parameters(self):
        """Test getting all parameters for a module."""
        # Register parameters
        self.registry.register_parameter(
            module_id="test_module",
            param_id="param1",
            default_value="value1",
            description="Parameter 1"
        )
        self.registry.register_parameter(
            module_id="test_module",
            param_id="param2",
            default_value="value2",
            description="Parameter 2"
        )
        
        # Get module parameters
        params = self.registry.get_module_parameters("test_module")
        self.assertEqual(len(params), 2)
        self.assertIn("param1", params)
        self.assertIn("param2", params)
    
    def test_save_load_config(self):
        """Test saving and loading configuration."""
        # Register and set parameters
        self.registry.register_parameter(
            module_id="test_module",
            param_id="param1",
            default_value="default1",
            description="Parameter 1"
        )
        self.registry.set_parameter(
            module_id="test_module",
            param_id="param1",
            value="value1",
            user_id="test_user"
        )
        
        # Save config
        self.assertTrue(self.registry.save_config())
        
        # Create new registry
        new_registry = ConfigRegistry(config_dir=self.config_dir)
        
        # Check if parameter is loaded
        self.assertTrue(new_registry.has_parameter("test_module", "param1"))
        self.assertEqual(new_registry.get_parameter("test_module", "param1"), "value1")


class TestEventBus(unittest.TestCase):
    """Test event bus."""
    
    def setUp(self):
        """Set up test case."""
        self.event_bus = EventBus()
        self.events_received = []
    
    def event_handler(self, event: Event):
        """Event handler.
        
        Args:
            event: Event
        """
        self.events_received.append(event)
    
    def test_subscribe_publish(self):
        """Test subscribing to and publishing events."""
        # Subscribe to event
        self.event_bus.subscribe("test_event", self.event_handler)
        
        # Publish event
        event = Event(
            event_type="test_event",
            source="test_source",
            data={"message": "Hello, world!"}
        )
        self.event_bus.publish(event)
        
        # Check if event was received
        self.assertEqual(len(self.events_received), 1)
        self.assertEqual(self.events_received[0].event_type, "test_event")
        self.assertEqual(self.events_received[0].source, "test_source")
        self.assertEqual(self.events_received[0].data["message"], "Hello, world!")
    
    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        # Subscribe to event
        self.event_bus.subscribe("test_event", self.event_handler)
        
        # Unsubscribe from event
        self.event_bus.unsubscribe("test_event", self.event_handler)
        
        # Publish event
        event = Event(
            event_type="test_event",
            source="test_source",
            data={"message": "Hello, world!"}
        )
        self.event_bus.publish(event)
        
        # Check if event was not received
        self.assertEqual(len(self.events_received), 0)
    
    def test_wildcard_subscription(self):
        """Test wildcard subscription."""
        # Subscribe to all events
        self.event_bus.subscribe("*", self.event_handler)
        
        # Publish events
        event1 = Event(
            event_type="event1",
            source="test_source",
            data={"message": "Event 1"}
        )
        event2 = Event(
            event_type="event2",
            source="test_source",
            data={"message": "Event 2"}
        )
        self.event_bus.publish(event1)
        self.event_bus.publish(event2)
        
        # Check if events were received
        self.assertEqual(len(self.events_received), 2)
        self.assertEqual(self.events_received[0].event_type, "event1")
        self.assertEqual(self.events_received[1].event_type, "event2")


class TestPluginManager(unittest.TestCase):
    """Test plugin manager."""
    
    def setUp(self):
        """Set up test case."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "test_data", "plugins")
        os.makedirs(self.data_dir, exist_ok=True)
        self.plugin_manager = PluginManager(data_dir=self.data_dir)
    
    def tearDown(self):
        """Tear down test case."""
        # Clean up data files
        if os.path.exists(os.path.join(self.data_dir, "metadata.json")):
            os.remove(os.path.join(self.data_dir, "metadata.json"))
    
    def test_register_plugin(self):
        """Test registering a plugin."""
        # Create plugin
        plugin = MockPlugin(
            plugin_id="test_plugin",
            name="Test Plugin",
            description="Test plugin for testing"
        )
        
        # Register plugin
        self.assertTrue(self.plugin_manager.register_plugin(plugin))
        
        # Check if plugin is registered
        self.assertIn("test_plugin", self.plugin_manager.get_plugins())
        self.assertEqual(self.plugin_manager.get_plugin("test_plugin"), plugin)
    
    def test_initialize_start_stop_plugin(self):
        """Test initializing, starting, and stopping a plugin."""
        # Create plugin
        plugin = MockPlugin(
            plugin_id="test_plugin",
            name="Test Plugin",
            description="Test plugin for testing"
        )
        
        # Register plugin
        self.plugin_manager.register_plugin(plugin)
        
        # Initialize plugin
        context = {"test_key": "test_value"}
        self.assertTrue(self.plugin_manager.initialize_plugin("test_plugin", context))
        self.assertTrue(plugin.initialized)
        self.assertEqual(plugin.context, context)
        
        # Start plugin
        self.assertTrue(self.plugin_manager.start_plugin("test_plugin"))
        self.assertTrue(plugin.running)
        
        # Stop plugin
        self.assertTrue(self.plugin_manager.stop_plugin("test_plugin"))
        self.assertFalse(plugin.running)
    
    def test_plugin_dependencies(self):
        """Test plugin dependencies."""
        # Create plugins
        plugin1 = MockPlugin(
            plugin_id="plugin1",
            name="Plugin 1",
            description="Plugin 1"
        )
        plugin2 = MockPlugin(
            plugin_id="plugin2",
            name="Plugin 2",
            description="Plugin 2",
            dependencies=["plugin1"]
        )
        plugin3 = MockPlugin(
            plugin_id="plugin3",
            name="Plugin 3",
            description="Plugin 3",
            dependencies=["plugin2"]
        )
        
        # Register plugins
        self.plugin_manager.register_plugin(plugin1)
        self.plugin_manager.register_plugin(plugin2)
        self.plugin_manager.register_plugin(plugin3)
        
        # Sort plugins by dependencies
        sorted_plugins = self.plugin_manager._sort_plugins_by_dependencies()
        
        # Check if plugins are sorted correctly
        self.assertEqual(sorted_plugins[0], "plugin1")
        self.assertEqual(sorted_plugins[1], "plugin2")
        self.assertEqual(sorted_plugins[2], "plugin3")


class TestDialogueManager(unittest.TestCase):
    """Test dialogue manager."""
    
    def setUp(self):
        """Set up test case."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "test_data", "dialogue")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create LLM client with mock provider
        self.llm_client = LLMClient()
        self.mock_provider = MockLLMProvider()
        self.llm_client.register_provider(self.mock_provider, is_default=True)
        
        # Create dialogue manager
        self.dialogue_manager = DialogueManager(
            llm_client=self.llm_client,
            data_dir=self.data_dir
        )
    
    def tearDown(self):
        """Tear down test case."""
        # Clean up data files
        for filename in os.listdir(self.data_dir):
            os.remove(os.path.join(self.data_dir, filename))
    
    def test_process_user_message(self):
        """Test processing a user message."""
        # Process message
        response, context = self.dialogue_manager.process_user_message(
            user_id="test_user",
            message_text="Hello, world!"
        )
        
        # Check response
        self.assertTrue(hasattr(response, 'message'))
        self.assertIn("Hello, world!", response.message.content)
        
        # Check context
        self.assertEqual(context.user_id, "test_user")
        self.assertEqual(len(context.messages), 3)  # system, user, assistant
        self.assertEqual(context.messages[1].role, "user")
        self.assertEqual(context.messages[1].content, "Hello, world!")
    
    def test_context_persistence(self):
        """Test context persistence."""
        # Process first message
        self.dialogue_manager.process_user_message(
            user_id="test_user",
            message_text="Hello, world!"
        )
        
        # Process second message
        response, context = self.dialogue_manager.process_user_message(
            user_id="test_user",
            message_text="How are you?"
        )
        
        # Check context
        self.assertEqual(context.user_id, "test_user")
        self.assertEqual(len(context.messages), 5)  # system, user, assistant, user, assistant
        self.assertEqual(context.messages[3].role, "user")
        self.assertEqual(context.messages[3].content, "How are you?")


class TestSystemIntegration(unittest.TestCase):
    """Test system integration."""
    
    def setUp(self):
        """Set up test case."""
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_data", "integration")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create system components
        self.module_registry = ModuleRegistry()
        self.config_registry = ConfigRegistry(
            config_dir=os.path.join(self.test_dir, "config")
        )
        self.event_bus = EventBus()
        
        # Create system core
        self.system_core = SystemCore(
            config_registry=self.config_registry,
            event_bus=self.event_bus,
            data_dir=os.path.join(self.test_dir, "system")
        )
        
        # Create plugin manager
        self.plugin_manager = PluginManager(
            data_dir=os.path.join(self.test_dir, "plugins")
        )
        
        # Create LLM client with mock provider
        self.llm_client = LLMClient()
        self.mock_provider = MockLLMProvider()
        self.llm_client.register_provider(self.mock_provider, is_default=True)
        
        # Create dialogue manager
        self.dialogue_manager = DialogueManager(
            llm_client=self.llm_client,
            data_dir=os.path.join(self.test_dir, "dialogue")
        )
        
        # Create personality system
        self.personality_system = PersonalitySystem(
            data_dir=os.path.join(self.test_dir, "personality")
        )
        
        # Register components with module registry
        self.module_registry.register_service("config_registry", self.config_registry)
        self.module_registry.register_service("event_bus", self.event_bus)
        self.module_registry.register_service("plugin_manager", self.plugin_manager)
        self.module_registry.register_service("llm_client", self.llm_client)
        self.module_registry.register_service("dialogue_manager", self.dialogue_manager)
        self.module_registry.register_service("personality_system", self.personality_system)
        
        # Initialize system core
        self.system_core.initialize()
    
    def tearDown(self):
        """Tear down test case."""
        # Clean up test directory
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
    
    def test_event_propagation(self):
        """Test event propagation through the system."""
        # Create event handler
        events_received = []
        def event_handler(event: Event):
            events_received.append(event)
        
        # Subscribe to events
        self.event_bus.subscribe("test_event", event_handler)
        
        # Publish event
        self.system_core.publish_event(
            event_type="test_event",
            source="test_source",
            data={"message": "Test event"}
        )
        
        # Check if event was received
        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0].event_type, "test_event")
        self.assertEqual(events_received[0].data["message"], "Test event")
    
    def test_plugin_integration(self):
        """Test plugin integration with the system."""
        # Create plugin
        plugin = MockPlugin(
            plugin_id="test_plugin",
            name="Test Plugin",
            description="Test plugin for integration testing"
        )
        
        # Register plugin
        self.plugin_manager.register_plugin(plugin)
        
        # Initialize plugin
        context = {
            "system_core": self.system_core,
            "event_bus": self.event_bus
        }
        self.plugin_manager.initialize_plugin("test_plugin", context)
        
        # Start plugin
        self.plugin_manager.start_plugin("test_plugin")
        
        # Check if plugin is running
        self.assertTrue(plugin.running)
        
        # Check if plugin received context
        self.assertEqual(plugin.context["system_core"], self.system_core)
        self.assertEqual(plugin.context["event_bus"], self.event_bus)
    
    def test_config_parameter_integration(self):
        """Test configuration parameter integration."""
        # Register parameter
        self.config_registry.register_parameter(
            module_id="test_module",
            param_id="test_param",
            default_value="default_value",
            description="Test parameter"
        )
        
        # Set parameter
        self.config_registry.set_parameter(
            module_id="test_module",
            param_id="test_param",
            value="new_value",
            user_id="test_user"
        )
        
        # Get parameter through system core
        value = self.system_core.get_parameter("test_module", "test_param")
        self.assertEqual(value, "new_value")
    
    def test_dialogue_system_integration(self):
        """Test dialogue system integration."""
        # Process message
        response, context = self.dialogue_manager.process_user_message(
            user_id="test_user",
            message_text="What is the system status?"
        )
        
        # Check response
        self.assertTrue(hasattr(response, 'message'))
        self.assertIn("system status", response.message.content.lower())
        
        # Check if LLM provider was called
        self.assertEqual(len(self.mock_provider.calls), 1)
        self.assertIn("What is the system status?", self.mock_provider.calls[0]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
