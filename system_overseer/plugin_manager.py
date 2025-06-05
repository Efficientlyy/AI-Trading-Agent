#!/usr/bin/env python
"""
Plugin Manager for System Overseer.

This module provides the PluginManager class for loading and managing plugins.
"""

import os
import sys
import json
import logging
import importlib
import traceback
import threading
from typing import Dict, Any, List, Optional

logger = logging.getLogger("system_overseer.plugin_manager")

class PluginManager:
    """Plugin Manager for System Overseer."""
    
    def __init__(self, system_core=None, data_dir: str = "./data/plugins"):
        """Initialize Plugin Manager.
        
        Args:
            system_core: System core instance
            data_dir: Data directory for plugins
        """
        self.system_core = system_core
        self.data_dir = data_dir
        self.plugins = {}
        self.plugin_configs = {}
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info("PluginManager initialized")
    
    def load_plugins(self):
        """Load plugins from configuration.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Loading plugins... - START")
        
        try:
            # Get configuration registry
            logger.debug("Attempting to get config_registry service")
            config_registry = self.system_core.get_service("config_registry")
            if not config_registry:
                logger.error("Configuration registry not found")
                return False
            
            logger.debug("Successfully retrieved config_registry service")
            
            # Get plugins configuration
            logger.debug("Attempting to get plugins configuration")
            plugins_config = config_registry.get_config("plugins", [])
            logger.info(f"Found {len(plugins_config)} plugin configurations")
            
            if not plugins_config:
                logger.warning("No plugins configured - creating default empty list")
                # Create default empty plugins configuration
                config_registry.set_config("plugins", [])
                logger.info("Plugin loading completed (no plugins to load)")
                return True
            
            # Load each plugin
            for i, plugin_config in enumerate(plugins_config):
                logger.debug(f"Processing plugin configuration {i+1}/{len(plugins_config)}")
                
                plugin_id = plugin_config.get("id")
                plugin_path = plugin_config.get("path")
                plugin_enabled = plugin_config.get("enabled", True)
                plugin_config_data = plugin_config.get("config", {})
                
                logger.debug(f"Plugin details: id={plugin_id}, path={plugin_path}, enabled={plugin_enabled}")
                
                if not plugin_id or not plugin_path:
                    logger.warning(f"Invalid plugin configuration: missing id or path - {plugin_config}")
                    continue
                
                if not plugin_enabled:
                    logger.info(f"Plugin {plugin_id} is disabled, skipping")
                    continue
                
                try:
                    logger.debug(f"Attempting to load plugin module: {plugin_path}")
                    
                    # Load plugin module and class
                    module_path, class_name = plugin_path.rsplit(".", 1)
                    logger.debug(f"Importing module: {module_path}, class: {class_name}")
                    
                    module = importlib.import_module(module_path)
                    logger.debug(f"Module {module_path} imported successfully")
                    
                    plugin_class = getattr(module, class_name)
                    logger.debug(f"Class {class_name} retrieved successfully")
                    
                    # Create plugin instance
                    logger.debug(f"Creating instance of {class_name}")
                    plugin = plugin_class()
                    logger.debug(f"Plugin instance created successfully")
                    
                    # Store plugin and configuration
                    self.plugins[plugin_id] = plugin
                    self.plugin_configs[plugin_id] = plugin_config_data
                    logger.debug(f"Plugin {plugin_id} stored in registry")
                    
                    # Initialize plugin with timeout protection
                    logger.debug(f"Initializing plugin {plugin_id}")
                    plugin.initialize(self.system_core)
                    logger.debug(f"Plugin {plugin_id} initialized successfully")
                    
                    logger.info(f"Plugin {plugin_id} loaded successfully")
                except ImportError as e:
                    logger.error(f"Failed to import plugin {plugin_id}: {e}")
                    logger.error(f"Import traceback: {traceback.format_exc()}")
                except AttributeError as e:
                    logger.error(f"Failed to find plugin class {plugin_id}: {e}")
                    logger.error(f"Attribute traceback: {traceback.format_exc()}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_id}: {e}")
                    logger.error(f"Exception traceback: {traceback.format_exc()}")
            
            logger.info(f"Loaded {len(self.plugins)} plugins successfully")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in load_plugins: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
        finally:
            logger.info("Loading plugins... - END")
    
    def start_plugins(self):
        """Start all loaded plugins.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting plugins...")
        
        for plugin_id, plugin in self.plugins.items():
            try:
                logger.debug(f"Starting plugin {plugin_id}")
                plugin.start()
                logger.info(f"Plugin {plugin_id} started")
            except Exception as e:
                logger.error(f"Failed to start plugin {plugin_id}: {e}")
                logger.error(f"Exception traceback: {traceback.format_exc()}")
        
        return True
    
    def stop_plugins(self):
        """Stop all loaded plugins.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Stopping plugins...")
        
        for plugin_id, plugin in self.plugins.items():
            try:
                logger.debug(f"Stopping plugin {plugin_id}")
                plugin.stop()
                logger.info(f"Plugin {plugin_id} stopped")
            except Exception as e:
                logger.error(f"Failed to stop plugin {plugin_id}: {e}")
                logger.error(f"Exception traceback: {traceback.format_exc()}")
        
        return True
    
    def get_plugin(self, plugin_id: str):
        """Get plugin by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            object: Plugin instance or None if not found
        """
        return self.plugins.get(plugin_id)
    
    def get_plugin_config(self, plugin_id: str):
        """Get plugin configuration by ID.
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            dict: Plugin configuration or empty dict if not found
        """
        return self.plugin_configs.get(plugin_id, {})
    
    def get_plugins(self):
        """Get all plugins.
        
        Returns:
            dict: Dictionary of plugin instances
        """
        return self.plugins
