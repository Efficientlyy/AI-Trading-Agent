"""
MEXC Exchange Configuration

This module provides configuration settings for connecting to the MEXC exchange
for both data acquisition and trading. It includes settings for the V3 API which
will remain supported after the V2 API deprecation in August 2025.
"""

import os
import json
from typing import Dict, Any, List
from pathlib import Path

# Try to load API keys from api_keys.json
def _load_api_keys():
    try:
        api_keys_path = Path(__file__).parent.parent.parent / 'api_keys.json'
        if api_keys_path.exists():
            with open(api_keys_path, 'r') as f:
                keys = json.load(f)
                if 'mexc' in keys:
                    return keys['mexc'].get('api_key', ''), keys['mexc'].get('api_secret', '')
        return None, None
    except Exception as e:
        print(f"Warning: Could not load API keys from api_keys.json: {e}")
        return None, None

# Get API keys from environment variables or api_keys.json
API_KEY = os.environ.get('MEXC_API_KEY', '')
API_SECRET = os.environ.get('MEXC_API_SECRET', '')

# If not in environment variables, try to load from api_keys.json
if not API_KEY or not API_SECRET:
    json_api_key, json_api_secret = _load_api_keys()
    if json_api_key and json_api_secret:
        API_KEY = json_api_key
        API_SECRET = json_api_secret

# MEXC connection configuration for V3 API
MEXC_CONFIG: Dict[str, Any] = {
    'exchange_id': 'mexc',
    'default_pair': 'BTC/USDC',
    'API_KEY': API_KEY,
    'API_SECRET': API_SECRET,
    'options': {
        'defaultType': 'spot',  # Use spot trading
        'adjustForTimeDifference': True,
        'recvWindow': 5000,  # Window for API requests (ms)
    },
    'api_version': 'v3',  # Using V3 API (current version)
    'rate_limits': {
        'max_requests_per_second': 20,  # Conservative estimate
        'max_orders_per_second': 5,
    }
}

# MEXC trading pairs of interest
TRADING_PAIRS: List[str] = [
    'BTC/USDC',  # Primary focus
    'ETH/USDC',
    'BNB/USDC',
    'SOL/USDC',
    'XRP/USDC',
]

# MEXC timeframes for chart data
SUPPORTED_TIMEFRAMES: Dict[str, str] = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',
}

# MEXC V3 API endpoints
API_ENDPOINTS = {
    # REST API endpoints
    'spot_v3': 'https://api.mexc.com/api/v3',
    'contract': 'https://contract.mexc.com',
    
    # WebSocket endpoints
    'public_ws': 'wss://stream.mexc.com/ws',  # Public streams
    'private_ws': 'wss://stream.mexc.com/ws',  # Private streams (user data)
    
    # The older WebSocket service will be replaced by August 4, 2025
    'legacy_ws': 'wss://wbs.mexc.com/ws',  # Deprecated in 2025
}

# Trading fees on MEXC (for reference)
TRADING_FEES = {
    'maker': 0.002,  # 0.2%
    'taker': 0.002,  # 0.2%
}

# Default trade execution parameters
DEFAULT_TRADE_PARAMS = {
    'timeout': 30000,  # Timeout for trade execution in ms
    'limit_price_buffer': 0.005,  # 0.5% buffer for limit orders
    'use_limit_orders': True,  # Use limit orders by default
}
