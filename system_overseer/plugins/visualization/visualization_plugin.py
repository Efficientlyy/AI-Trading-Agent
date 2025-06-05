#!/usr/bin/env python
"""
Visualization Plugin

This module provides the VisualizationPlugin class for the System Overseer.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional

from .data_providers.base import DataProvider
from .data_providers.mexc import MexcDataProvider
from .chart_manager import ChartManager

logger = logging.getLogger("system_overseer.plugins.visualization")

class VisualizationPlugin:
    """Visualization Plugin for System Overseer."""
    
    def __init__(self):
        """Initialize Visualization Plugin."""
        self.system_core = None
        self.config = {}
        self.data_providers = {}
        self.chart_manager = None
        self.initialized = False
        self.running = False
        
        logger.info("VisualizationPlugin created")
    
    def initialize(self, system_core):
        """Initialize plugin with system core.
        
        Args:
            system_core: System core instance
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        logger.info("Initializing VisualizationPlugin")
        
        try:
            self.system_core = system_core
            
            # Get configuration
            config_registry = system_core.get_service("config_registry")
            if not config_registry:
                logger.error("Configuration registry not found")
                return False
            
            # Get visualization configuration
            self.config = config_registry.get_config("visualization", {})
            if not self.config:
                logger.warning("No visualization configuration found, using defaults")
                self.config = {
                    "default_pairs": ["BTCUSDC", "ETHUSDC", "SOLUSDC"],
                    "default_timeframe": "1h",
                    "chart_types": ["candlestick", "line"],
                    "indicators": ["sma", "ema"],
                    "auto_refresh": True,
                    "refresh_interval": 60,
                    "data_provider": "mexc"
                }
                config_registry.set_config("visualization", self.config)
            
            # Initialize chart manager
            data_dir = os.path.join(system_core.data_dir, "visualization")
            self.chart_manager = ChartManager(data_dir)
            
            # Initialize data providers
            self._initialize_data_providers()
            
            # Register commands with Telegram integration
            self._register_telegram_commands()
            
            self.initialized = True
            logger.info("VisualizationPlugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing VisualizationPlugin: {e}")
            return False
    
    def _initialize_data_providers(self):
        """Initialize data providers."""
        logger.info("Initializing data providers")
        
        # Initialize MEXC data provider
        mexc_provider = MexcDataProvider()
        mexc_config = self.config.get("mexc", {})
        
        if mexc_provider.initialize(mexc_config):
            self.data_providers["mexc"] = mexc_provider
            logger.info("MEXC data provider initialized")
        else:
            logger.error("Failed to initialize MEXC data provider")
    
    def _register_telegram_commands(self):
        """Register commands with Telegram integration."""
        try:
            telegram_integration = self.system_core.get_service("telegram_integration")
            if not telegram_integration:
                logger.warning("Telegram integration not found, skipping command registration")
                return
            
            # Register chart command
            telegram_integration.register_command(
                "chart",
                self._handle_chart_command,
                "Get a chart for a trading pair. Usage: /chart <symbol> <type> <interval> [indicators]"
            )
            
            logger.info("Telegram commands registered")
        except Exception as e:
            logger.error(f"Error registering Telegram commands: {e}")
    
    def _handle_chart_command(self, args):
        """Handle chart command from Telegram.
        
        Args:
            args: Command arguments
            
        Returns:
            tuple: (success, message, image_path)
        """
        try:
            # Parse arguments
            if len(args) < 3:
                return (False, "Usage: /chart <symbol> <type> <interval> [indicators]", None)
            
            symbol = args[0].upper()
            chart_type = args[1].lower()
            interval = args[2].lower()
            indicators = args[3:] if len(args) > 3 else []
            
            # Get chart
            chart_data = self.get_chart(symbol, chart_type, interval, indicators)
            if not chart_data:
                return (False, f"Failed to generate {chart_type} chart for {symbol}", None)
            
            # Save chart image
            image_path = self.chart_manager.save_chart_image(
                chart_data, symbol, chart_type, interval
            )
            
            if not image_path:
                return (False, "Failed to save chart image", None)
            
            return (True, f"{symbol} {interval} {chart_type} chart", image_path)
            
        except Exception as e:
            logger.error(f"Error handling chart command: {e}")
            return (False, f"Error: {str(e)}", None)
    
    def start(self):
        """Start plugin operation."""
        if not self.initialized:
            logger.error("Cannot start VisualizationPlugin: not initialized")
            return False
        
        logger.info("Starting VisualizationPlugin")
        self.running = True
        
        # TODO: Start background tasks if needed
        
        return True
    
    def stop(self):
        """Stop plugin operation."""
        logger.info("Stopping VisualizationPlugin")
        self.running = False
        
        # TODO: Stop background tasks if needed
        
        return True
    
    def get_chart(self, symbol, chart_type, interval, indicators=None):
        """Get chart for specified parameters.
        
        Args:
            symbol: Trading pair symbol
            chart_type: Type of chart (candlestick, line, volume)
            interval: Time interval
            indicators: List of indicators to include
            
        Returns:
            bytes: PNG image data or None if generation failed
        """
        if not self.initialized or not self.running:
            logger.error("Cannot get chart: plugin not initialized or not running")
            return None
        
        try:
            # Validate chart type
            valid_chart_types = ["candlestick", "line", "volume"]
            if chart_type not in valid_chart_types:
                logger.error(f"Unsupported chart type: {chart_type}")
                return None
                
            # Validate interval
            provider_name = self.config.get("data_provider", "mexc")
            provider = self.data_providers.get(provider_name)
            
            if not provider:
                logger.error(f"Data provider {provider_name} not found")
                return None
                
            # Check if interval is valid
            valid_intervals = provider.get_available_intervals()
            if interval not in valid_intervals:
                logger.error(f"Invalid interval: {interval}. Valid intervals are: {valid_intervals}")
                return None
            
            # Get klines data
            df = provider.get_klines(symbol, interval, 100)
            if df is None or df.empty:
                logger.error(f"Failed to get klines data for {symbol}")
                return None
            
            # Generate chart based on type
            if chart_type == "candlestick":
                return self.chart_manager.create_candlestick_chart(df, symbol, interval, indicators)
            elif chart_type == "line":
                return self.chart_manager.create_line_chart(df, symbol, interval, indicators)
            elif chart_type == "volume":
                return self.chart_manager.create_volume_chart(df, symbol, interval)
            else:
                # This should never happen due to validation above, but kept for safety
                logger.error(f"Unsupported chart type: {chart_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None
    
    def get_available_charts(self):
        """Get list of available chart configurations.
        
        Returns:
            dict: Dictionary with available chart options
        """
        if not self.initialized:
            logger.error("Cannot get available charts: plugin not initialized")
            return {}
        
        try:
            result = {
                "symbols": [],
                "chart_types": ["candlestick", "line", "volume"],
                "intervals": [],
                "indicators": ["sma", "ema", "rsi", "macd", "bollinger"]
            }
            
            # Get available symbols from data provider
            provider_name = self.config.get("data_provider", "mexc")
            provider = self.data_providers.get(provider_name)
            
            if provider:
                result["symbols"] = provider.get_available_symbols()
                result["intervals"] = provider.get_available_intervals()
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting available charts: {e}")
            return {}
