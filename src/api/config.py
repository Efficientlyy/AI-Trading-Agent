"""Configuration settings for the Market Regime Detection API."""

import os
import logging
from pathlib import Path

# API settings
API_TITLE = "Market Regime Detection API"
API_DESCRIPTION = "API for detecting market regimes and backtesting trading strategies"
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FILE = "api.log"

# Directory settings
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
VISUALIZATION_DIR = BASE_DIR / "static" / "visualizations"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Default parameters
DEFAULT_LOOKBACK_WINDOW = 60
DEFAULT_METHODS = ["volatility", "momentum"]
DEFAULT_STRATEGY = "momentum"

# Available methods and strategies
AVAILABLE_METHODS = ["volatility", "momentum", "hmm", "trend", "clustering"]
AVAILABLE_STRATEGIES = ["momentum", "mean_reversion", "volatility", "regime_based", "custom"]

# Performance settings
MAX_WORKERS = 4
REQUEST_TIMEOUT = 60  # seconds 