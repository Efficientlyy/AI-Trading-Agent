#!/usr/bin/env python
"""
MEXC API Fix for Trading-Agent System

This module provides fixes for the MEXC API integration, specifically
addressing signature generation issues and symbol format compatibility.
"""

import os
import sys
import json
import hmac
import hashlib
import logging
import requests
import urllib.parse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mexc_api_fix.log")
    ]
)

logger = logging.getLogger("mexc_api_fix")

class FixedMexcClient:
    """Fixed MEXC client with correct signature generation"""
    
    def __init__(self, api_key=None, api_secret=None, env_path=None):
        """Initialize MEXC client
        
        Args:
            api_key: API key (optional, will load from env if not provided)
            api_secret: API secret (optional, will load from env if not provided)
            env_path: Path to .env file (optional)
        """
        # Load credentials from environment if not provided
        if api_key is None or api_secret is None:
            env_vars = self._load_environment_variables(env_path)
            api_key = api_key or env_vars.get('MEXC_API_KEY')
            api_secret = api_secret or env_vars.get('MEXC_SECRET_KEY')
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.mexc.com"
        
        # Verify credentials
        if not self.api_key or not self.api_secret:
            logger.warning("API key or secret not provided, some functions will be unavailable")
        else:
            logger.info("MEXC client initialized with API credentials")
    
    def _load_environment_variables(self, env_path=None):
        """Load environment variables from .env file
        
        Args:
            env_path: Path to .env file
            
        Returns:
            dict: Environment variables
        """
        env_vars = {}
        try:
            if env_path is None:
                env_path = '.env-secure/.env'
            
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            env_vars[key] = value
        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")
        
        return env_vars
    
    def _generate_signature(self, params: Dict) -> str:
        """Generate signature for authenticated requests using the correct method
        
        Args:
            params: Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        """
        # IMPORTANT: Do NOT sort the parameters - MEXC API requires the exact order
        # Convert params to query string using urllib.parse.urlencode
        query_string = urllib.parse.urlencode(params)
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None, auth: bool = False) -> Dict:
        """Make request to MEXC API
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            auth: Whether to use authentication
            
        Returns:
            dict: Response data
        """
        # Initialize params if None
        if params is None:
            params = {}
        
        # Add timestamp for authenticated requests
        if auth:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            
            # Generate signature
            signature = self._generate_signature(params)
            params['signature'] = signature
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Make request
        try:
            headers = {}
            if auth:
                headers['X-MEXC-APIKEY'] = self.api_key
            
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers, timeout=10)
            else:
                raise Exception(f"Unsupported method: {method}")
            
            # Check if response is valid
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": response.text}
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
    
    def get_server_time(self) -> Dict:
        """Get server time
        
        Returns:
            dict: Server time
        """
        return self._request('GET', '/api/v3/time')
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information
        
        Returns:
            dict: Exchange information
        """
        return self._request('GET', '/api/v3/exchangeInfo')
    
    def get_account(self) -> Dict:
        """Get account information
        
        Returns:
            dict: Account information
        """
        return self._request('GET', '/api/v3/account', auth=True)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List:
        """Get klines (candlestick data)
        
        Args:
            symbol: Trading symbol
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of klines
            
        Returns:
            list: Klines
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = self._request('GET', '/api/v3/klines', params)
        
        if 'error' in response:
            return []
        
        return response

def test_fixed_client():
    """Test the fixed MEXC client"""
    logger.info("Testing fixed MEXC client...")
    
    # Initialize client
    client = FixedMexcClient()
    
    # Test server time
    logger.info("Testing server time...")
    server_time = client.get_server_time()
    logger.info(f"Server time: {server_time}")
    
    # Test account info
    logger.info("Testing account info...")
    account = client.get_account()
    logger.info(f"Account info: {json.dumps(account, indent=2)[:200]}...")
    
    # Test exchange info
    logger.info("Testing exchange info...")
    exchange_info = client.get_exchange_info()
    
    # Extract symbols
    symbols = exchange_info.get('symbols', [])
    
    # Check for specific symbols
    target_symbols = ['BTCUSDC', 'ETHUSDC', 'SOLUSDC', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    found_symbols = []
    
    for symbol in symbols:
        if symbol.get('symbol') in target_symbols:
            found_symbols.append({
                'symbol': symbol.get('symbol'),
                'baseAsset': symbol.get('baseAsset'),
                'quoteAsset': symbol.get('quoteAsset'),
                'status': symbol.get('status')
            })
    
    logger.info(f"Found target symbols: {json.dumps(found_symbols, indent=2)}")
    
    # Test klines for each symbol
    for symbol_info in found_symbols:
        symbol = symbol_info['symbol']
        logger.info(f"Testing klines for {symbol}...")
        klines = client.get_klines(symbol, '5m', 10)
        
        if klines:
            logger.info(f"Got {len(klines)} klines for {symbol}")
            logger.info(f"First kline: {klines[0]}")
        else:
            logger.error(f"Failed to get klines for {symbol}")
    
    logger.info("Fixed MEXC client test completed")

def update_optimized_mexc_client():
    """Update the optimized_mexc_client.py file with the fixed signature generation"""
    try:
        # Read the original file
        with open('optimized_mexc_client.py', 'r') as f:
            content = f.read()
        
        # Make a backup
        with open('optimized_mexc_client.py.bak', 'w') as f:
            f.write(content)
        
        # Replace the signature generation method
        old_signature_method = """    def _generate_signature(self, params: Dict) -> str:
        \"\"\"Generate signature for authenticated requests
        
        Args:
            params: Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        \"\"\"
        # Convert params to query string
        query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature"""
        
        new_signature_method = """    def _generate_signature(self, params: Dict) -> str:
        \"\"\"Generate signature for authenticated requests
        
        Args:
            params: Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        \"\"\"
        # IMPORTANT: Do NOT sort the parameters - MEXC API requires the exact order
        # Convert params to query string using urllib.parse.urlencode
        query_string = urllib.parse.urlencode(params)
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature"""
        
        # Update the content
        updated_content = content.replace(old_signature_method, new_signature_method)
        
        # Add urllib.parse import if not present
        if 'import urllib.parse' not in updated_content:
            updated_content = updated_content.replace(
                'import requests',
                'import requests\nimport urllib.parse'
            )
        
        # Write the updated content
        with open('optimized_mexc_client.py', 'w') as f:
            f.write(updated_content)
        
        logger.info("Successfully updated optimized_mexc_client.py with fixed signature generation")
        return True
    except Exception as e:
        logger.error(f"Error updating optimized_mexc_client.py: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the fixed client
    test_fixed_client()
    
    # Update the optimized_mexc_client.py file
    update_optimized_mexc_client()
