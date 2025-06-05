#!/usr/bin/env python
"""
Configuration Registry for System Overseer.

This module provides the ConfigRegistry class for managing configuration.
"""

import os
import sys
import json
import logging
import threading
from typing import Dict, Any, List, Optional

logger = logging.getLogger("system_overseer.config_registry")

class ConfigChangeEvent:
    """Configuration Change Event."""
    
    def __init__(self, key: str, old_value: Any, new_value: Any):
        """Initialize Configuration Change Event.
        
        Args:
            key: Configuration key
            old_value: Old value
            new_value: New value
        """
        self.key = key
        self.old_value = old_value
        self.new_value = new_value

class ParameterDefinition:
    """Parameter Definition."""
    
    def __init__(self, key: str, default_value: Any, description: str = "", 
                 param_type: str = "string", options: List[Any] = None, 
                 min_value: Any = None, max_value: Any = None):
        """Initialize Parameter Definition.
        
        Args:
            key: Parameter key
            default_value: Default value
            description: Parameter description
            param_type: Parameter type (string, number, boolean, array, object)
            options: List of valid options (for enum types)
            min_value: Minimum value (for number types)
            max_value: Maximum value (for number types)
        """
        self.key = key
        self.default_value = default_value
        self.description = description
        self.param_type = param_type
        self.options = options
        self.min_value = min_value
        self.max_value = max_value

class ConfigRegistry:
    """Configuration Registry for System Overseer."""
    
    def __init__(self, event_bus=None, config_dir: str = "./config"):
        """Initialize Configuration Registry.
        
        Args:
            event_bus: Event bus instance
            config_dir: Configuration directory
        """
        self.event_bus = event_bus
        self.config_dir = config_dir
        self.config = {}
        self.parameter_definitions = {}
        self.lock = threading.RLock()
        
        # Create config directory
        os.makedirs(config_dir, exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        logger.info("ConfigRegistry initialized")
    
    def _load_config(self):
        """Load configuration from file."""
        config_file = os.path.join(self.config_dir, "config.json")
        
        try:
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"Configuration file not found: {config_file}")
                # Create default configuration
                self.config = {
                    "system": {
                        "name": "Trading System Overseer",
                        "version": "1.0.0"
                    },
                    "trading": {
                        "default_pairs": ["BTCUSDC", "ETHUSDC", "SOLUSDC"],
                        "risk_level": "moderate"
                    },
                    "notifications": {
                        "level": "all"
                    },
                    "plugins": []
                }
                # Save default configuration
                self._save_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _save_config(self):
        """Save configuration to file."""
        config_file = os.path.join(self.config_dir, "config.json")
        
        try:
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_config(self, key: str, default=None):
        """Get configuration value.
        
        Args:
            key: Configuration key (dot notation)
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        with self.lock:
            # Split key into parts
            parts = key.split(".")
            
            # Navigate through config
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
    
    def set_config(self, key: str, value):
        """Set configuration value.
        
        Args:
            key: Configuration key (dot notation)
            value: Configuration value
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            try:
                # Split key into parts
                parts = key.split(".")
                
                # Navigate through config
                config = self.config
                for i, part in enumerate(parts[:-1]):
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                
                # Get old value
                old_value = config.get(parts[-1])
                
                # Set value
                config[parts[-1]] = value
                
                # Save configuration
                self._save_config()
                
                # Publish event
                if self.event_bus:
                    event = ConfigChangeEvent(key, old_value, value)
                    self.event_bus.publish("config.change", event)
                
                return True
            except Exception as e:
                logger.error(f"Error setting configuration {key}: {e}")
                return False
    
    def register_parameter(self, definition: ParameterDefinition):
        """Register parameter definition.
        
        Args:
            definition: Parameter definition
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            try:
                self.parameter_definitions[definition.key] = definition
                
                # Set default value if not already set
                if self.get_config(definition.key) is None:
                    self.set_config(definition.key, definition.default_value)
                
                return True
            except Exception as e:
                logger.error(f"Error registering parameter {definition.key}: {e}")
                return False
    
    def get_parameter_definition(self, key: str):
        """Get parameter definition.
        
        Args:
            key: Parameter key
            
        Returns:
            ParameterDefinition: Parameter definition or None if not found
        """
        with self.lock:
            return self.parameter_definitions.get(key)
    
    def get_parameter_definitions(self):
        """Get all parameter definitions.
        
        Returns:
            dict: Dictionary of parameter definitions
        """
        with self.lock:
            return self.parameter_definitions.copy()
    
    def validate_value(self, key: str, value):
        """Validate value against parameter definition.
        
        Args:
            key: Parameter key
            value: Value to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        with self.lock:
            # Get parameter definition
            definition = self.get_parameter_definition(key)
            if not definition:
                return True
            
            # Validate type
            if definition.param_type == "string":
                if not isinstance(value, str):
                    return False
            elif definition.param_type == "number":
                if not isinstance(value, (int, float)):
                    return False
                if definition.min_value is not None and value < definition.min_value:
                    return False
                if definition.max_value is not None and value > definition.max_value:
                    return False
            elif definition.param_type == "boolean":
                if not isinstance(value, bool):
                    return False
            elif definition.param_type == "array":
                if not isinstance(value, list):
                    return False
            elif definition.param_type == "object":
                if not isinstance(value, dict):
                    return False
            
            # Validate options
            if definition.options and value not in definition.options:
                return False
            
            return True
