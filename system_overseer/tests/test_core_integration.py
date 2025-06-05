#!/usr/bin/env python
"""
Integration tests for System Overseer core components.

This module tests the integration between ModuleRegistry, ConfigRegistry, and EventBus
to ensure they work together seamlessly as the foundation of the System Overseer.
"""

import os
import sys
import unittest
import time
import threading
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from module_registry import ModuleRegistry, BaseModule, BaseService, BasePlugin
from config_registry import ConfigRegistry, ConfigChangeEvent
from event_bus import EventBus, Event, EventHistory


class TestModule(BaseModule):
    """Test module for integration testing."""
    
    def __init__(self, module_id="test_module"):
        super().__init__(
            module_id=module_id,
            name="Test Module",
            version="1.0.0",
            description="Test module for integration testing",
            dependencies=[]
        )
        self.initialized = False
        self.started = False
        self.events_received = []
        self.config_changes = []
    
    def initialize(self, registry):
        """Initialize module with registry reference."""
        result = super().initialize(registry)
        self.initialized = True
        return result
    
    def start(self):
        """Start module operation."""
        result = super().start()
        self.started = True
        return result
    
    def stop(self):
        """Stop module operation."""
        result = super().stop()
        self.started = False
        return result
    
    def handle_event(self, event):
        """Handle event from event bus."""
        self.events_received.append(event)
    
    def handle_config_change(self, event):
        """Handle configuration change event."""
        self.config_changes.append(event)


class TestService(BaseService):
    """Test service for integration testing."""
    
    def __init__(self, module_id="test_service"):
        super().__init__(
            module_id=module_id,
            name="Test Service",
            version="1.0.0",
            description="Test service for integration testing",
            service_type="test",
            dependencies=["test_module"]
        )
        self.api_calls = 0
    
    def get_api(self):
        """Get service API object."""
        return self
    
    def test_api_method(self, param=None):
        """Test API method."""
        self.api_calls += 1
        return f"API call {self.api_calls} with param: {param}"


class TestPlugin(BasePlugin):
    """Test plugin for integration testing."""
    
    def __init__(self, module_id="test_plugin"):
        super().__init__(
            module_id=module_id,
            name="Test Plugin",
            version="1.0.0",
            description="Test plugin for integration testing",
            plugin_type="test",
            capabilities=["test"],
            dependencies=["test_service"]
        )
        self.configured = False
    
    def configure(self, config):
        """Configure plugin with settings."""
        result = super().configure(config)
        self.configured = True
        return result


class CoreComponentsIntegrationTest(unittest.TestCase):
    """Integration tests for core System Overseer components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create module registry
        self.module_registry = ModuleRegistry()
        
        # Create config registry
        self.config_file = os.path.join(self.temp_dir.name, "config.json")
        self.config_registry = ConfigRegistry(config_file=self.config_file)
        
        # Create event bus
        self.event_file = os.path.join(self.temp_dir.name, "events.json")
        self.event_bus = EventBus(
            persistence_file=self.event_file,
            worker_count=1
        )
        
        # Create event history
        self.event_history = EventHistory(
            event_bus=self.event_bus,
            max_size=100,
            index_fields=["test_field"]
        )
        
        # Create test modules
        self.test_module = TestModule()
        self.test_service = TestService()
        self.test_plugin = TestPlugin()
        
        # Start event bus
        self.event_bus.start()
    
    def tearDown(self):
        """Clean up test environment."""
        # Stop event bus
        self.event_bus.stop()
        
        # Close event history
        self.event_history.close()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_module_registration_and_lifecycle(self):
        """Test module registration and lifecycle management."""
        # Register modules
        self.assertTrue(self.module_registry.register_module(self.test_module))
        self.assertTrue(self.module_registry.register_module(self.test_service))
        self.assertTrue(self.module_registry.register_module(self.test_plugin))
        
        # Check registration
        self.assertEqual(len(self.module_registry.get_modules()), 3)
        self.assertEqual(len(self.module_registry.get_services()), 1)
        self.assertEqual(len(self.module_registry.get_plugins()), 1)
        
        # Initialize modules
        self.assertTrue(self.module_registry.initialize_all())
        self.assertTrue(self.test_module.initialized)
        self.assertTrue(self.test_service.initialized)
        self.assertTrue(self.test_plugin.initialized)
        
        # Start modules
        self.assertTrue(self.module_registry.start_all())
        self.assertTrue(self.test_module.started)
        self.assertTrue(self.test_service.started)
        self.assertTrue(self.test_plugin.started)
        
        # Stop modules
        self.assertTrue(self.module_registry.stop_all())
        self.assertFalse(self.test_module.started)
        self.assertFalse(self.test_service.started)
        self.assertFalse(self.test_plugin.started)
    
    def test_config_registration_and_changes(self):
        """Test configuration parameter registration and changes."""
        # Register modules
        self.module_registry.register_module(self.test_module)
        
        # Register parameters
        self.assertTrue(self.config_registry.register_parameter(
            module_id="test_module",
            param_id="test_param",
            default_value="default",
            description="Test parameter"
        ))
        
        self.assertTrue(self.config_registry.register_parameter(
            module_id="test_module",
            param_id="int_param",
            default_value=42,
            param_type=int,
            min_value=0,
            max_value=100
        ))
        
        # Check parameter values
        self.assertEqual(
            self.config_registry.get_parameter("test_module", "test_param"),
            "default"
        )
        self.assertEqual(
            self.config_registry.get_parameter("test_module", "int_param"),
            42
        )
        
        # Add config change listener
        self.config_registry.add_listener(
            self.test_module.handle_config_change,
            module_id="test_module"
        )
        
        # Change parameter values
        self.assertTrue(self.config_registry.set_parameter(
            module_id="test_module",
            param_id="test_param",
            value="new_value"
        ))
        
        self.assertTrue(self.config_registry.set_parameter(
            module_id="test_module",
            param_id="int_param",
            value=50
        ))
        
        # Check updated values
        self.assertEqual(
            self.config_registry.get_parameter("test_module", "test_param"),
            "new_value"
        )
        self.assertEqual(
            self.config_registry.get_parameter("test_module", "int_param"),
            50
        )
        
        # Check change events
        self.assertEqual(len(self.test_module.config_changes), 2)
        self.assertEqual(self.test_module.config_changes[0].param_id, "test_param")
        self.assertEqual(self.test_module.config_changes[0].old_value, "default")
        self.assertEqual(self.test_module.config_changes[0].new_value, "new_value")
        
        # Save and load configuration
        self.assertTrue(self.config_registry.save())
        
        # Create new registry and load
        new_registry = ConfigRegistry(config_file=self.config_file)
        self.assertTrue(new_registry.load())
        
        # Check loaded values
        self.assertEqual(
            new_registry.get_parameter("test_module", "test_param"),
            "new_value"
        )
        self.assertEqual(
            new_registry.get_parameter("test_module", "int_param"),
            50
        )
    
    def test_event_publishing_and_handling(self):
        """Test event publishing and handling."""
        # Register modules
        self.module_registry.register_module(self.test_module)
        
        # Register event handler
        handler_id = self.event_bus.register_handler(
            callback=self.test_module.handle_event,
            event_types=["test.event"],
            name="TestHandler"
        )
        
        # Publish event
        event = self.event_bus.publish_event(
            event_type="test.event",
            source="test_source",
            data={"test_field": "test_value"},
            sync=True  # Process synchronously for testing
        )
        
        # Check event handling
        self.assertEqual(len(self.test_module.events_received), 1)
        self.assertEqual(self.test_module.events_received[0].event_id, event.event_id)
        self.assertEqual(self.test_module.events_received[0].event_type, "test.event")
        self.assertEqual(self.test_module.events_received[0].source, "test_source")
        self.assertEqual(self.test_module.events_received[0].data["test_field"], "test_value")
        
        # Check event history
        events = self.event_history.query(event_types=["test.event"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, event.event_id)
        
        # Query by field
        events = self.event_history.query(data_filters={"test_field": "test_value"})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, event.event_id)
        
        # Unregister handler
        self.assertTrue(self.event_bus.unregister_handler(handler_id))
        
        # Publish another event
        self.event_bus.publish_event(
            event_type="test.event",
            source="test_source",
            data={"test_field": "another_value"},
            sync=True
        )
        
        # Check that event was not handled
        self.assertEqual(len(self.test_module.events_received), 1)
        
        # Check event history
        events = self.event_history.query(event_types=["test.event"])
        self.assertEqual(len(events), 2)
    
    def test_service_discovery_and_api_access(self):
        """Test service discovery and API access."""
        # Register modules
        self.module_registry.register_module(self.test_module)
        self.module_registry.register_module(self.test_service)
        
        # Initialize modules
        self.module_registry.initialize_all()
        self.module_registry.start_all()
        
        # Get service
        service = self.module_registry.get_service("test")
        self.assertIsNotNone(service)
        self.assertEqual(service.module_id, "test_service")
        
        # Access API
        api = service.get_api()
        self.assertIsNotNone(api)
        
        # Call API method
        result = api.test_api_method("test")
        self.assertEqual(result, "API call 1 with param: test")
        
        # Call again
        result = api.test_api_method("another")
        self.assertEqual(result, "API call 2 with param: another")
    
    def test_plugin_configuration_and_capabilities(self):
        """Test plugin configuration and capabilities."""
        # Register modules
        self.module_registry.register_module(self.test_module)
        self.module_registry.register_module(self.test_service)
        self.module_registry.register_module(self.test_plugin)
        
        # Initialize modules
        self.module_registry.initialize_all()
        
        # Get plugin
        plugin = self.module_registry.get_plugin("test_plugin")
        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.module_id, "test_plugin")
        
        # Check capabilities
        self.assertIn("test", plugin.capabilities)
        
        # Configure plugin
        self.assertTrue(plugin.configure({"setting": "value"}))
        self.assertTrue(plugin.configured)
        
        # Get plugins by type
        plugins = self.module_registry.get_plugins("test")
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0].module_id, "test_plugin")
    
    def test_full_integration(self):
        """Test full integration of all components."""
        # Register modules
        self.module_registry.register_module(self.test_module)
        self.module_registry.register_module(self.test_service)
        self.module_registry.register_module(self.test_plugin)
        
        # Register parameters
        self.config_registry.register_parameter(
            module_id="test_module",
            param_id="event_enabled",
            default_value=True,
            param_type=bool,
            description="Enable event publishing"
        )
        
        self.config_registry.register_parameter(
            module_id="test_plugin",
            param_id="plugin_setting",
            default_value="default",
            param_type=str,
            description="Plugin setting"
        )
        
        # Add config change listener
        self.config_registry.add_listener(
            self.test_module.handle_config_change,
            module_id="test_module"
        )
        
        # Register event handler
        self.event_bus.register_handler(
            callback=self.test_module.handle_event,
            event_types=["config.changed", "plugin.action"],
            name="TestHandler"
        )
        
        # Initialize and start modules
        self.module_registry.initialize_all()
        self.module_registry.start_all()
        
        # Configure plugin
        plugin = self.module_registry.get_plugin("test_plugin")
        plugin.configure({
            "plugin_setting": self.config_registry.get_parameter(
                "test_plugin", "plugin_setting"
            )
        })
        
        # Change configuration
        self.config_registry.set_parameter(
            module_id="test_plugin",
            param_id="plugin_setting",
            value="new_value"
        )
        
        # Publish event from plugin
        if self.config_registry.get_parameter("test_module", "event_enabled"):
            self.event_bus.publish_event(
                event_type="plugin.action",
                source="test_plugin",
                data={
                    "action": "test",
                    "setting": self.config_registry.get_parameter(
                        "test_plugin", "plugin_setting"
                    )
                },
                sync=True
            )
        
        # Check event handling
        self.assertGreaterEqual(len(self.test_module.events_received), 1)
        plugin_events = [e for e in self.test_module.events_received 
                        if e.event_type == "plugin.action"]
        self.assertEqual(len(plugin_events), 1)
        self.assertEqual(plugin_events[0].data["setting"], "new_value")
        
        # Stop modules
        self.module_registry.stop_all()
        
        # Persist state
        self.config_registry.save()
        self.event_bus.persist_events()
        
        # Check persistence files
        self.assertTrue(os.path.exists(self.config_file))
        self.assertTrue(os.path.exists(self.event_file))


if __name__ == "__main__":
    unittest.main()
