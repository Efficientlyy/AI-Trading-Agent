#!/usr/bin/env python
"""
Plugin Configuration

This module provides the configuration for the Visualization Plugin.
"""

# Default configuration for the Visualization Plugin
DEFAULT_CONFIG = {
    "default_pairs": ["BTCUSDC", "ETHUSDC", "SOLUSDC"],  # Prioritizing USDC pairs as requested by user
    "default_timeframe": "15m",  # Changed from 1h to 15m since 1h has API compatibility issues
    "chart_types": ["candlestick", "line", "volume"],
    "indicators": ["sma", "ema", "rsi", "macd", "bollinger"],
    "auto_refresh": True,
    "refresh_interval": 60,
    "data_provider": "mexc",
    "mexc": {
        # MEXC-specific configuration
        # API credentials will be loaded from environment variables
    }
}
