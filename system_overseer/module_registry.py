#!/usr/bin/env python
"""
Module Registry for System Overseer

This module provides a registry for system modules and services,
enabling dependency injection and service location.
"""

import os
import sys
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_overseer.module_registry")


class ModuleRegistry:
    """Registry for system modules and services."""
    
    def __init__(self):
        """Initialize module registry."""
        self.modules = {}
        self.services = {}
        self.lock = threading.RLock()
    
    def register_module(self, module_id: str, module: Any) -> bool:
        """Register a module.
        
        Args:
            module_id: Module identifier
            module: Module instance
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            if module_id in self.modules:
                logger.warning(f"Module already registered: {module_id}")
                return False
            
            self.modules[module_id] = module
            logger.info(f"Module registered: {module_id}")
            return True
    
    def register_service(self, service_id: str, service: Any) -> bool:
        """Register a service.
        
        Args:
            service_id: Service identifier
            service: Service instance
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            if service_id in self.services:
                logger.warning(f"Service already registered: {service_id}")
                return False
            
            self.services[service_id] = service
            logger.info(f"Service registered: {service_id}")
            return True
    
    def get_module(self, module_id: str) -> Any:
        """Get a module.
        
        Args:
            module_id: Module identifier
            
        Returns:
            Any: Module instance
        """
        with self.lock:
            if module_id not in self.modules:
                logger.warning(f"Module not found: {module_id}")
                return None
            
            return self.modules[module_id]
    
    def get_service(self, service_id: str) -> Any:
        """Get a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Any: Service instance
        """
        with self.lock:
            if service_id not in self.services:
                logger.warning(f"Service not found: {service_id}")
                return None
            
            return self.services[service_id]
    
    def has_module(self, module_id: str) -> bool:
        """Check if module exists.
        
        Args:
            module_id: Module identifier
            
        Returns:
            bool: True if module exists
        """
        with self.lock:
            return module_id in self.modules
    
    def has_service(self, service_id: str) -> bool:
        """Check if service exists.
        
        Args:
            service_id: Service identifier
            
        Returns:
            bool: True if service exists
        """
        with self.lock:
            return service_id in self.services
    
    def get_all_modules(self) -> Dict[str, Any]:
        """Get all modules.
        
        Returns:
            dict: All modules
        """
        with self.lock:
            return self.modules.copy()
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all services.
        
        Returns:
            dict: All services
        """
        with self.lock:
            return self.services.copy()
    
    def unregister_module(self, module_id: str) -> bool:
        """Unregister a module.
        
        Args:
            module_id: Module identifier
            
        Returns:
            bool: True if unregistration successful
        """
        with self.lock:
            if module_id not in self.modules:
                logger.warning(f"Module not found: {module_id}")
                return False
            
            del self.modules[module_id]
            logger.info(f"Module unregistered: {module_id}")
            return True
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            bool: True if unregistration successful
        """
        with self.lock:
            if service_id not in self.services:
                logger.warning(f"Service not found: {service_id}")
                return False
            
            del self.services[service_id]
            logger.info(f"Service unregistered: {service_id}")
            return True
    
    def clear(self) -> bool:
        """Clear registry.
        
        Returns:
            bool: True if clear successful
        """
        with self.lock:
            self.modules = {}
            self.services = {}
            logger.info("Registry cleared")
            return True
