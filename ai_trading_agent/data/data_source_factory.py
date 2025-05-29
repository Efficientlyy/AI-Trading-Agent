"""
Data Source Factory - Handles creation and switching between data sources.

This module provides a factory pattern implementation for creating and managing
different data sources (mock vs real) for the trading system.
"""

import logging
from typing import Dict, Any, List, Optional, Type

from ..config.data_source_config import get_data_source_config, DataSourceConfig
from .mock_data_generator import MockDataGenerator
from .market_data_provider import MarketDataProvider
from ..common.utils import get_logger

class DataSourceFactory:
    """
    Factory for creating and managing data sources.
    
    This class handles the dynamic switching between mock and real data sources
    based on configuration settings.
    """
    
    def __init__(self):
        """Initialize the data source factory."""
        self.logger = get_logger("DataSourceFactory")
        self.config = get_data_source_config()
        
        # Create instances of data providers
        self.mock_data_generator = None
        self.real_data_provider = None
        
        # Register as a listener for configuration changes
        self.config.register_listener(self._handle_config_change)
        
        self.logger.info("DataSourceFactory initialized")
    
    def _handle_config_change(self, updated_config: Dict[str, Any]) -> None:
        """
        Handle configuration changes.
        
        Args:
            updated_config: The updated configuration
        """
        self.logger.info("Data source configuration changed")
        
        # Reset providers to ensure they pick up new settings
        if not updated_config.get("use_mock_data", True):
            # Only reinitialize real data provider if using real data
            self.real_data_provider = None
    
    def get_data_provider(self) -> Any:
        """
        Get the appropriate data provider based on current configuration.
        
        Returns:
            Either a MockDataGenerator or a MarketDataProvider
        """
        if self.config.use_mock_data:
            return self._get_mock_data_generator()
        else:
            return self._get_real_data_provider()
    
    def _get_mock_data_generator(self) -> MockDataGenerator:
        """
        Get or create a MockDataGenerator instance.
        
        Returns:
            MockDataGenerator instance
        """
        if self.mock_data_generator is None:
            # Initialize with settings from config
            mock_settings = self.config.get_mock_data_settings()
            self.mock_data_generator = MockDataGenerator(seed=mock_settings.get("seed", 42))
            self.logger.info("Created new MockDataGenerator")
        
        return self.mock_data_generator
    
    def _get_real_data_provider(self) -> MarketDataProvider:
        """
        Get or create a MarketDataProvider instance.
        
        Returns:
            MarketDataProvider instance
        """
        if self.real_data_provider is None:
            # Initialize with settings from config
            real_settings = self.config.get_real_data_settings()
            self.real_data_provider = MarketDataProvider(
                primary_source=real_settings.get("primary_source", "alpha_vantage"),
                fallback_source=real_settings.get("fallback_source", "yahoo_finance"),
                cache_timeout_minutes=real_settings.get("cache_timeout_minutes", 15)
            )
            self.logger.info(f"Created new MarketDataProvider with primary source: {real_settings.get('primary_source')}")
        
        return self.real_data_provider
    
    def toggle_data_source(self) -> bool:
        """
        Toggle between mock and real data sources.
        
        Returns:
            bool: New state (True for mock, False for real)
        """
        new_state = not self.config.use_mock_data
        self.config.use_mock_data = new_state
        self.logger.info(f"Toggled data source to {'mock' if new_state else 'real'}")
        return new_state


# Create a singleton instance
_instance = None

def get_data_source_factory() -> DataSourceFactory:
    """
    Get the global data source factory instance.
    
    Returns:
        DataSourceFactory: The global factory instance
    """
    global _instance
    if _instance is None:
        _instance = DataSourceFactory()
    return _instance
