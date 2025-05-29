"""
Configuration handler for data source selection (mock vs real)

This module provides configuration management for toggling between mock and real data sources
in the trading system. It includes event handling to propagate changes throughout the system.
"""

import os
import json
import logging
from typing import Dict, Any, List, Callable, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class DataSourceConfig:
    """
    Configuration manager for data source selection.
    
    Handles the toggling between mock and real data sources and propagates
    configuration changes to registered listeners.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data source configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default location.
        """
        self._lock = threading.RLock()
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._default_config = {
            "use_mock_data": True,
            "mock_data_settings": {
                "volatility": 0.015,
                "trend_strength": 0.3,
                "seed": 42,
                "generate_regimes": True
            },
            "real_data_settings": {
                "primary_source": "alpha_vantage",
                "fallback_source": "yahoo_finance",
                "cache_timeout_minutes": 15,
                "api_retry_attempts": 3
            }
        }
        
        if config_path is None:
            # Use default location in config directory
            base_dir = Path(__file__).parent.parent.parent
            self._config_path = os.path.join(base_dir, 'config', 'data_source_config.json')
        else:
            self._config_path = config_path
            
        # Load or create configuration
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default if it doesn't exist.
        
        Returns:
            Dict containing the configuration.
        """
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded data source configuration from {self._config_path}")
                    return config
            else:
                # Create default configuration file
                os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
                with open(self._config_path, 'w') as f:
                    json.dump(self._default_config, f, indent=4)
                logger.info(f"Created default data source configuration at {self._config_path}")
                return self._default_config.copy()
        except Exception as e:
            logger.error(f"Error loading data source configuration: {e}")
            return self._default_config.copy()
    
    def save_config(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self._lock:
                with open(self._config_path, 'w') as f:
                    json.dump(self._config, f, indent=4)
                logger.info(f"Saved data source configuration to {self._config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data source configuration: {e}")
            return False
    
    def register_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a listener to be notified of configuration changes.
        
        Args:
            listener: Callback function that receives the updated configuration.
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)
    
    def unregister_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister a previously registered listener.
        
        Args:
            listener: The listener to remove.
        """
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
    
    def _notify_listeners(self) -> None:
        """Notify all listeners of configuration changes."""
        config_copy = self.get_config()
        for listener in self._listeners[:]:  # Create a copy to avoid modification during iteration
            try:
                listener(config_copy)
            except Exception as e:
                logger.error(f"Error notifying listener of configuration change: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get a copy of the current configuration.
        
        Returns:
            Dict containing the current configuration.
        """
        with self._lock:
            return self._config.copy()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates to apply.
        """
        with self._lock:
            # Deep update for nested dictionaries
            self._deep_update(self._config, updates)
            self.save_config()
        
        # Notify listeners outside the lock to avoid deadlocks
        self._notify_listeners()
        
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            target: Target dictionary to update.
            source: Source dictionary with updates.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    @property
    def use_mock_data(self) -> bool:
        """
        Check if mock data should be used.
        
        Returns:
            True if mock data should be used, False for real data.
        """
        with self._lock:
            return self._config.get("use_mock_data", True)
    
    @use_mock_data.setter
    def use_mock_data(self, value: bool) -> None:
        """
        Set whether to use mock data.
        
        Args:
            value: True to use mock data, False for real data.
        """
        with self._lock:
            if self._config.get("use_mock_data") != value:
                self._config["use_mock_data"] = value
                self.save_config()
        
        # Notify listeners outside the lock
        self._notify_listeners()
        logger.info(f"Data source switched to {'mock' if value else 'real'} data")
    
    def get_mock_data_settings(self) -> Dict[str, Any]:
        """
        Get mock data generation settings.
        
        Returns:
            Dict containing mock data settings.
        """
        with self._lock:
            return self._config.get("mock_data_settings", {}).copy()
    
    def get_real_data_settings(self) -> Dict[str, Any]:
        """
        Get real data source settings.
        
        Returns:
            Dict containing real data settings.
        """
        with self._lock:
            return self._config.get("real_data_settings", {}).copy()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        with self._lock:
            self._config = self._default_config.copy()
            self.save_config()
        
        # Notify listeners outside the lock
        self._notify_listeners()
        logger.info("Data source configuration reset to defaults")


# Singleton instance for global access
_instance = None

def get_data_source_config() -> DataSourceConfig:
    """
    Get the global data source configuration instance.
    
    Returns:
        DataSourceConfig: The global configuration instance.
    """
    global _instance
    if _instance is None:
        _instance = DataSourceConfig()
    return _instance
