#!/usr/bin/env python
"""
System Core for System Overseer.

This module provides the SystemCore class for managing system components.
"""

import os
import sys
import json
import logging
import threading
from typing import Dict, Any, List, Optional

logger = logging.getLogger("system_overseer.core")

class SystemCore:
    """System Core for System Overseer."""
    
    def __init__(self, config_registry=None, event_bus=None, data_dir: str = "./data/system"):
        """Initialize System Core.
        
        Args:
            config_registry: Configuration registry instance
            event_bus: Event bus instance
            data_dir: Data directory for system data
        """
        self.config_registry = config_registry
        self.event_bus = event_bus
        self.data_dir = data_dir
        self.services = {}
        self.lock = threading.RLock()
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Register core services
        if config_registry:
            self.register_service("config_registry", config_registry)
        
        if event_bus:
            self.register_service("event_bus", event_bus)
        
        logger.info("SystemCore initialized")
    
    def initialize(self):
        """Initialize system.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("System initialized")
        return True
    
    def register_service(self, service_id: str, service):
        """Register service.
        
        Args:
            service_id: Service ID
            service: Service instance
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            self.services[service_id] = service
            return True
    
    def get_service(self, service_id: str):
        """Get service by ID.
        
        Args:
            service_id: Service ID
            
        Returns:
            object: Service instance or None if not found
        """
        with self.lock:
            return self.services.get(service_id)
    
    def get_services(self):
        """Get all services.
        
        Returns:
            dict: Dictionary of service instances
        """
        with self.lock:
            return self.services.copy()
    
    def get_config(self, key: str, default=None):
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        if self.config_registry:
            return self.config_registry.get_config(key, default)
        else:
            return default
    
    def set_config(self, key: str, value):
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.config_registry:
            return self.config_registry.set_config(key, value)
        else:
            return False
    
    def publish_event(self, event_type: str, event_data=None):
        """Publish event.
        
        Args:
            event_type: Event type
            event_data: Event data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.event_bus:
            return self.event_bus.publish(event_type, event_data)
        else:
            return False
    
    def subscribe_to_event(self, event_type: str, callback):
        """Subscribe to event.
        
        Args:
            event_type: Event type
            callback: Callback function
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.event_bus:
            return self.event_bus.subscribe(event_type, callback)
        else:
            return False
