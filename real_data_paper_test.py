#!/usr/bin/env python
"""
Real Data Paper Trading Test

This script implements a test of the AI Trading Agent system using:
1. Real cryptocurrency market data (BTC, ETH)
2. Paper trading for execution
3. Mocked sentiment analysis
4. All other components using real data and algorithms

The test runs for 30 minutes and logs all activities and decisions.
"""

import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
import threading
import queue
import json
import requests
import math
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path to help with imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the MockLLMOversight class for decision review
try:
    from mock_llm_oversight import MockLLMOversight
    logger.info("Using imported MockLLMOversight class")
except ImportError:
    logger.warning("Could not import MockLLMOversight, will use internal definition")
    pass

# Try importing actual system components
try:
    from ai_trading_agent.common.health_monitoring.core_definitions import HealthStatus, AlertSeverity
    from ai_trading_agent.market_regime import MarketRegimeType, VolatilityRegimeType
    from ai_trading_agent.market_regime.regime_classifier import MarketRegimeClassifier
    from ai_trading_agent.market_data import MarketDataProvider
    actual_components_available = True
    logger.info("Using actual system components")
except ImportError:
    logger.warning("Using mock components as actual components could not be imported")
    actual_components_available = False
    
    # Define mock enums for testing if needed
    class HealthStatus(Enum):
        HEALTHY = auto()
        DEGRADED = auto()
        CRITICAL = auto()
        FAILED = auto()
    
    class AlertSeverity(Enum):
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
    
    class MarketRegimeType(Enum):
        UNKNOWN = "unknown"
        BULL = "bull"
        BEAR = "bear"
        SIDEWAYS = "sideways"
        VOLATILE = "volatile"
        VOLATILE_BULL = "volatile_bull"
        VOLATILE_BEAR = "volatile_bear"
        RECOVERY = "recovery"
        BREAKDOWN = "breakdown"
        TRENDING = "trending"
    
    class VolatilityRegimeType(Enum):
        UNKNOWN = "unknown"
        VERY_LOW = "very_low"
        LOW = "low"
        MODERATE = "moderate"
        HIGH = "high"
        VERY_HIGH = "very_high"
        EXTREME = "extreme"
        CRISIS = "crisis"

# Constants for the test
TEST_DURATION_SEC = 30 * 60  # 30 minutes
CYCLE_INTERVAL_SEC = 10      # How often to check for updates
ASSETS = ["BTC/USD", "ETH/USD"]
INITIAL_CASH = 100000        # Starting cash for paper trading
MAX_TEST_CYCLES = int(TEST_DURATION_SEC / CYCLE_INTERVAL_SEC)

#=============================================================================
# Real-time Cryptocurrency Data Provider
#=============================================================================

class CryptoDataProvider:
    """Real-time cryptocurrency data provider with multiple API fallbacks."""
    
    def __init__(self, api_key=None):
        """Initialize the crypto data provider.
        
        Args:
            api_key: Optional API key for authenticated data access
        """
        self.subscribed_symbols = set()
        self.data_queue = queue.Queue(maxsize=1000)
        self.latest_data = {}
        self.historical_data = {}
        self.health_status = HealthStatus.HEALTHY
        self.failure_mode = False
        self.data_thread = None
        self.running = False
        self.update_interval = 10.0  # seconds
        
        # API configuration and statistics
        self.api_key = api_key
        self.use_authenticated_api = api_key is not None
        self.provider_name = "CryptoCompare" if self.use_authenticated_api else "CoinGecko"
        
        # API provider statistics
        self.api_stats = {
            'CryptoCompare': {'success': 0, 'failure': 0, 'last_success': None},
            'CoinGecko': {'success': 0, 'failure': 0, 'last_success': None},
            'Binance': {'success': 0, 'failure': 0, 'last_success': None},
            'CoinAPI': {'success': 0, 'failure': 0, 'last_success': None},
            'CoinMarketCap': {'success': 0, 'failure': 0, 'last_success': None},
            'Yahoo Finance': {'success': 0, 'failure': 0, 'last_success': None},
            'Alternative.me': {'success': 0, 'failure': 0, 'last_success': None},
            'Simulation': {'success': 0, 'failure': 0, 'last_success': None}
        }
        
        # Cache to avoid hammering the API
        self.last_update_time = {}
        self.price_cache = {}
        self.data_source = {}  # Tracks which provider supplied each price
        
        # Simulated prices for testing purposes only
        # These are NOT meant to reflect actual market prices
        self.simulation_prices = {
            "BTC/USD": 50000,  # Simulated price - NOT actual market price
            "ETH/USD": 3000    # Simulated price - NOT actual market price
        }
        
        # Configure logger
        self.log_fetch_details = True  # Set to False to reduce log verbosity
        
        logger.info(f"Initialized Crypto Data Provider with {'authenticated' if self.use_authenticated_api else 'unauthenticated'} API access")
        logger.info(f"Primary provider: {self.provider_name}, Fallbacks: Binance, CoinGecko, Alternative.me, Yahoo Finance")
        logger.info(f"API key present: {self.api_key is not None}, CoinAPI key present: {hasattr(self, 'coinapi_key')}")
        
    def get_api_stats(self):
        """Return statistics about API provider usage."""
        total_requests = sum([p['success'] + p['failure'] for p in self.api_stats.values()])
        if total_requests == 0:
            return "No API requests made yet"
            
        stats = [f"API Provider Statistics (Total Requests: {total_requests}):\n"]
        for provider, data in self.api_stats.items():
            success_rate = 0 if (data['success'] + data['failure']) == 0 else \
                          (data['success'] / (data['success'] + data['failure']) * 100)
            last_success = data['last_success'].strftime("%H:%M:%S") if data['last_success'] else "Never"
            stats.append(f"  {provider}: {data['success']} success, {data['failure']} failure, "
                       f"{success_rate:.1f}% success rate, Last success: {last_success}")
            
        return "\n".join(stats)
        
    def _get_symbol_id(self, symbol):
        """Convert trading symbol to provider-specific ID."""
        if symbol == "BTC/USD" or symbol == "BTC":
            return "bitcoin" if self.provider_name == "CoinGecko" else "BTC"
        elif symbol == "ETH/USD" or symbol == "ETH":
            return "ethereum" if self.provider_name == "CoinGecko" else "ETH"
        return symbol.split('/')[0].lower()
    
    def start(self):
        """Start the data provider."""
        if self.running:
            logger.warning("Crypto data provider already running")
            return False
        
        self.running = True
        self.data_thread = threading.Thread(target=self._data_fetch_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        logger.info("Started crypto data provider")
        return True
    
    def stop(self):
        """Stop the data provider."""
        if not self.running:
            logger.warning("Crypto data provider not running")
            return False
        
        self.running = False
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=3.0)
        
        logger.info("Stopped crypto data provider")
        return True
    
    def subscribe(self, symbol):
        """Subscribe to data updates for a symbol."""
        # Convert to standard format if needed
        if symbol == "BTC":
            symbol = "BTC/USD"
        elif symbol == "ETH":
            symbol = "ETH/USD"
        
        self.subscribed_symbols.add(symbol)
        logger.info(f"Subscribed to {symbol}")
        return True
    
    def get_latest_data(self, timeout=0.1):
        """Get the latest data from the queue."""
        try:
            return self.data_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_current_price(self, symbol):
        """Get the current price for a symbol."""
        # Convert to standard format if needed
        if symbol == "BTC":
            symbol = "BTC/USD"
        elif symbol == "ETH":
            symbol = "ETH/USD"
            
        # If we have the price in cache and it's recent (within 30 seconds), return it
        current_time = time.time()
        cache_expiry = 30  # seconds before cache expires
        
        if (symbol in self.price_cache and 
            self.price_cache[symbol] is not None and 
            symbol in self.last_update_time and 
            current_time - self.last_update_time.get(symbol, 0) < cache_expiry):
            
            data_source = self.data_source.get(symbol, "Unknown")
            logger.debug(f"Returning cached price for {symbol}: ${self.price_cache[symbol]:.2f} (from {data_source})")
            return self.price_cache[symbol]
        
        # Try to fetch the price from API
        price = self._fetch_real_time_price(symbol)
        
        # If API call fails, use simulated prices with clear labeling
        if price is None:
            # Record simulation usage in stats
            self.api_stats['Simulation']['success'] += 1
            self.api_stats['Simulation']['last_success'] = datetime.now()
            
            # Generate simulated price with realistic variation
            base_price = self.simulation_prices.get(symbol, 100)  # Default $100 for unknown assets
            variation_factor = 0.01  # 1% variation
            variation = base_price * variation_factor * (random.random() * 2 - 1)
            price = base_price + variation
            
            # Mark this as simulated data
            self.data_source[symbol] = 'Simulation'
            
            # Log the simulation usage
            logger.warning(f"[SIMULATED] Using synthetic price for {symbol}: ${price:.2f} (not actual market data)")
            
            # Print API stats at DEBUG level
            if logger.level <= logging.DEBUG:
                logger.debug(self.get_api_stats())
        
        # Cache the price and update timestamp
        if price is not None:
            self.price_cache[symbol] = price
            self.last_update_time[symbol] = current_time
        
        return price
        
    def _fetch_real_time_price(self, symbol, validate=True):
        """Fetch real-time price from API providers with robust fallback and retry logic.
        
        Args:
            symbol: The trading symbol (e.g., "BTC/USD")
            validate: Whether to validate the price against other sources
            
        Returns:
            tuple: (price, source, confidence) or (None, None, 0) if all providers fail
        """
        # List of API providers to try in order
        providers = []
        
        # Add authenticated provider first if available
        if self.use_authenticated_api:
            providers.append({
                'name': 'CryptoCompare',
                'fetch_func': self._fetch_from_cryptocompare,
                'requires_auth': True,
                'priority': 1  # Highest priority (authenticated)
            })
        
        # Always add free providers as fallbacks with priorities
        providers.extend([
            {
                'name': 'Binance',  # Binance is highly reliable
                'fetch_func': self._fetch_from_binance,
                'requires_auth': False,
                'priority': 2
            },
            {
                'name': 'CoinMarketCap',  # Most accurate but may have rate limits
                'fetch_func': self._fetch_from_coinmarketcap,
                'requires_auth': False,
                'priority': 3
            },
            {
                'name': 'CoinGecko',
                'fetch_func': self._fetch_from_coingecko,
                'requires_auth': False,
                'priority': 4
            },
            {
                'name': 'CoinAPI',
                'fetch_func': self._fetch_from_coinapi,
                'requires_auth': True if hasattr(self, 'coinapi_key') else False,
                'priority': 5
            }
        ])
        
        # Remove providers that require auth if we don't have keys
        providers = [p for p in providers if not (p['requires_auth'] and not self.api_key)]
        
        # Sort by priority
        providers = sorted(providers, key=lambda p: p['priority'])
        
        # For price validation, we'll collect all successful prices
        all_prices = []
        successful_providers = []
        
        # Try each provider with retries
        max_retries = 3
        errors = []
        primary_price = None
        primary_source = None
        confidence = 0.0
        
        # First pass: Try to get prices from multiple sources if validation is requested
        validation_sources = 1 if validate else 0
        
        for provider in providers:
            provider_name = provider['name']
            fetch_func = provider['fetch_func']
            
            for attempt in range(1, max_retries + 1):
                try:
                    if self.log_fetch_details:
                        logger.debug(f"Attempting to fetch {symbol} price from {provider_name} (attempt {attempt}/{max_retries})")
                    
                    price = fetch_func(symbol)
                    
                    if price is not None:
                        # Record success statistics
                        self.api_stats[provider_name]['success'] += 1
                        self.api_stats[provider_name]['last_success'] = datetime.now()
                        
                        # Add to our collection of prices
                        all_prices.append(price)
                        successful_providers.append(provider_name)
                        
                        # If this is our first successful price, use it as primary
                        if primary_price is None:
                            primary_price = price
                            primary_source = provider_name
                            logger.info(f"[{provider_name}] Primary price for {symbol}: ${price:.2f}")
                        else:
                            # This is a validation source
                            logger.info(f"[{provider_name}] Validation price for {symbol}: ${price:.2f} (diff: {((price/primary_price)-1)*100:.2f}%)")
                        
                        # If we're validating, try to get multiple sources
                        if validate and len(all_prices) <= validation_sources:
                            # Continue to next provider for validation
                            break
                        else:
                            # We have enough prices, exit the loop
                            break
                    else:
                        # Record failure for this attempt
                        self.api_stats[provider_name]['failure'] += 1
                        if self.log_fetch_details:
                            logger.debug(f"[{provider_name}] Failed to fetch price for {symbol} (no data returned)")
                    
                except Exception as e:
                    # Record failure for this attempt
                    self.api_stats[provider_name]['failure'] += 1
                    
                    error_msg = f"[{provider_name}] Error fetching {symbol} price: {str(e)}"
                    errors.append(error_msg)
                    
                    if self.log_fetch_details or attempt == max_retries:
                        logger.warning(error_msg)
                    
                    # Add exponential backoff between retries
                    if attempt < max_retries:
                        backoff_time = 0.1 * (2 ** (attempt - 1))
                        time.sleep(backoff_time)
            
            # Exit provider loop if we have enough prices
            if (not validate and primary_price is not None) or \
               (validate and len(all_prices) > validation_sources):
                break
        
        # If we got at least one price
        if primary_price is not None:
            # Validate the price if requested and we have multiple sources
            if validate and len(all_prices) > 1:
                # Calculate confidence based on agreement between sources
                confidence = self._validate_price(symbol, all_prices, successful_providers)
                
                # Store data source for this price with confidence
                self.data_source[symbol] = f"{primary_source} ({confidence:.0%} confidence)"
                
                if confidence < 0.5:
                    logger.warning(f"Low confidence ({confidence:.0%}) in {symbol} price: ${primary_price:.2f} from {primary_source}")
                else:
                    logger.info(f"Validated {symbol} price: ${primary_price:.2f} from {primary_source} with {confidence:.0%} confidence")
            else:
                # Single source, medium confidence
                confidence = 0.7
                self.data_source[symbol] = primary_source
            
            return primary_price
        
        # If we get here, all providers failed
        logger.error(f"All API providers failed to fetch price for {symbol}. Tried: {', '.join([p['name'] for p in providers])}")
        
        # Return detailed error information for debugging at DEBUG level
        if logger.level <= logging.DEBUG and errors:
            logger.debug(f"Detailed fetch errors for {symbol}:\n" + "\n".join(errors))
            
        return None
        
    def _validate_price(self, symbol, prices, sources):
        """Validate a price against multiple sources and return confidence score."""
        if not prices or len(prices) <= 1:
            return 0.7  # Medium confidence for single source
            
        # Calculate mean and standard deviation
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_dev = math.sqrt(variance) if variance > 0 else 0
        
        # Calculate coefficient of variation (CV)
        cv = std_dev / mean_price if mean_price > 0 else 1.0
        
        # Map CV to confidence (lower CV = higher confidence)
        # CV < 0.01 (1%) is excellent agreement
        # CV > 0.05 (5%) indicates significant disagreement
        if cv < 0.01:
            confidence = 0.95  # Very high confidence
        elif cv < 0.02:
            confidence = 0.85  # High confidence
        elif cv < 0.05:
            confidence = 0.7   # Medium confidence
        else:
            confidence = 0.5   # Low confidence
            
        # Log details at debug level
        logger.debug(f"Price validation for {symbol}: Mean=${mean_price:.2f}, StdDev=${std_dev:.2f}, CV={cv:.4f}, Confidence={confidence:.0%}")
        logger.debug(f"Prices: {', '.join([f'{s}=${p:.2f}' for s, p in zip(sources, prices)])}")
        
        return confidence
    
    def _fetch_from_cryptocompare(self, symbol):
        """Fetch price from CryptoCompare API."""
        symbol_id = self._get_symbol_id(symbol)
        url = "https://min-api.cryptocompare.com/data/price"
        params = {
            'fsym': symbol_id,
            'tsyms': 'USD',
            'api_key': self.api_key
        }
        
        response = requests.get(url, params=params, timeout=5.0)
        
        if response.status_code == 200:
            data = response.json()
            if 'USD' in data:
                return float(data['USD'])
        elif response.status_code == 429:
            # Rate limit hit
            logger.warning("CryptoCompare rate limit exceeded")
        
        return None
    
    def _fetch_from_coingecko(self, symbol):
        """Fetch price from CoinGecko API with reliability improvements."""
        # Map symbol to CoinGecko ID format
        if symbol == "BTC/USD" or symbol == "BTC":
            symbol_id = "bitcoin"
        elif symbol == "ETH/USD" or symbol == "ETH":
            symbol_id = "ethereum"
        else:
            # Try to use the generic ID mapping function
            symbol_id = self._get_symbol_id(symbol)
        
        # Try multiple CoinGecko API endpoints for reliability
        endpoints = [
            # Simple price endpoint (fastest)
            {
                'url': "https://api.coingecko.com/api/v3/simple/price",
                'params': {
                    'ids': symbol_id,
                    'vs_currencies': 'usd',
                },
                'price_path': [symbol_id, 'usd']
            },
            # Coins markets endpoint (more reliable but slower)
            {
                'url': f"https://api.coingecko.com/api/v3/coins/markets",
                'params': {
                    'vs_currency': 'usd',
                    'ids': symbol_id,
                    'per_page': 1,
                    'page': 1
                },
                'price_path': [0, 'current_price']
            },
            # Single coin endpoint (most reliable but slowest)
            {
                'url': f"https://api.coingecko.com/api/v3/coins/{symbol_id}",
                'params': {
                    'localization': 'false',
                    'tickers': 'false',
                    'market_data': 'true',
                    'community_data': 'false',
                    'developer_data': 'false'
                },
                'price_path': ['market_data', 'current_price', 'usd']
            }
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint['url'], params=endpoint['params'], timeout=8.0)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract price using the specified path
                    try:
                        price_data = data
                        for key in endpoint['price_path']:
                            price_data = price_data[key]
                        
                        price = float(price_data)
                        logger.debug(f"CoinGecko success using {endpoint['url'].split('/')[-1]}: {symbol_id} = ${price}")
                        return price
                    except (KeyError, IndexError, TypeError) as e:
                        logger.debug(f"Failed to parse CoinGecko response: {e}, Response: {data}")
                        continue
                        
                elif response.status_code == 429:
                    # Rate limit hit
                    logger.warning(f"CoinGecko rate limit exceeded on {endpoint['url'].split('/')[-1]}")
                    # Add longer delay before the next endpoint
                    time.sleep(1.0)
                else:
                    logger.warning(f"CoinGecko returned status {response.status_code} on {endpoint['url'].split('/')[-1]}")
                    
            except Exception as e:
                logger.warning(f"CoinGecko API error on {endpoint['url']}: {e}")
                continue
        
        return None
        
    def _fetch_from_coinmarketcap(self, symbol):
        """Fetch price from CoinMarketCap API (public, no API key needed)."""
        # Convert symbol to CMC format
        if symbol == "BTC/USD" or symbol == "BTC":
            cmc_symbol = "BTC"
            cmc_id = 1  # Bitcoin's CMC ID
        elif symbol == "ETH/USD" or symbol == "ETH":
            cmc_symbol = "ETH"
            cmc_id = 1027  # Ethereum's CMC ID
        else:
            # Extract the base symbol
            try:
                cmc_symbol = symbol.split('/')[0]
                # We don't know the ID for other symbols, so use symbol-based endpoint
            except:
                return None
        
        # Try multiple approaches
        try:
            # First approach: Use the quotes/latest endpoint with ID
            if symbol in ["BTC/USD", "BTC", "ETH/USD", "ETH"]:
                # We know the exact ID for these major coins
                url = f"https://web-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
                params = {
                    'id': cmc_id,
                    'convert': 'USD'
                }
                
                response = requests.get(url, params=params, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and str(cmc_id) in data['data']:
                        coin_data = data['data'][str(cmc_id)]
                        if 'quote' in coin_data and 'USD' in coin_data['quote']:
                            price = float(coin_data['quote']['USD']['price'])
                            logger.debug(f"Using CoinMarketCap API (ID): {cmc_symbol} = ${price}")
                            return price
            
            # Second approach: Use public widget data (more reliable, no rate limits)
            widget_url = "https://3rdparty-apis.coinmarketcap.com/v1/cryptocurrency/widget"
            params = {
                'symbols': cmc_symbol,
                'convert': 'USD'
            }
            
            response = requests.get(widget_url, params=params, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and cmc_symbol in data['data']:
                    coin_data = data['data'][cmc_symbol]
                    if 'quote' in coin_data and 'USD' in coin_data['quote']:
                        price = float(coin_data['quote']['USD']['price'])
                        logger.debug(f"Using CoinMarketCap widget API: {cmc_symbol} = ${price}")
                        return price
                        
        except Exception as e:
            logger.warning(f"CoinMarketCap API error: {e}")
        
        return None
    
    def _fetch_from_binance(self, symbol):
        """Fetch price from Binance public API."""
        # Convert symbol to Binance format
        if symbol == "BTC/USD" or symbol == "BTC":
            binance_symbol = "BTCUSDT"
        elif symbol == "ETH/USD" or symbol == "ETH":
            binance_symbol = "ETHUSDT"
        else:
            # Try to convert other symbols
            try:
                binance_symbol = symbol.replace("/", "").replace("USD", "USDT")
            except:
                return None
        
        # Try multiple Binance API endpoints for redundancy
        endpoints = [
            # Ticker price endpoint
            {
                'url': "https://api.binance.com/api/v3/ticker/price",
                'params': {'symbol': binance_symbol},
                'price_key': 'price'
            },
            # 24hr ticker endpoint (more data but more reliable)
            {
                'url': "https://api.binance.com/api/v3/ticker/24hr",
                'params': {'symbol': binance_symbol},
                'price_key': 'lastPrice'
            },
            # Average price endpoint
            {
                'url': "https://api.binance.com/api/v3/avgPrice",
                'params': {'symbol': binance_symbol},
                'price_key': 'price'
            }
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint['url'], params=endpoint['params'], timeout=5.0)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.debug(f"Binance response for {binance_symbol}: {data}")
                    
                    if endpoint['price_key'] in data:
                        price = float(data[endpoint['price_key']])
                        # Log the exact endpoint that worked
                        logger.debug(f"Using Binance {endpoint['url'].split('/')[-1]} endpoint: {binance_symbol} = ${price}")
                        return price
            except Exception as e:
                logger.warning(f"Binance API error on {endpoint['url']}: {e}")
                continue
        
        # Try the binance.us API as a fallback for USD pairs
        if symbol.endswith("/USD"):
            try:
                us_url = "https://api.binance.us/api/v3/ticker/price"
                response = requests.get(us_url, params={'symbol': binance_symbol}, timeout=5.0)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'price' in data:
                        price = float(data['price'])
                        logger.debug(f"Using Binance.US API: {binance_symbol} = ${price}")
                        return price
            except Exception as e:
                logger.warning(f"Binance.US API error: {e}")
        
        return None
    
    def _fetch_from_coinapi(self, symbol):
        """Fetch price from CoinAPI (as another fallback)."""
        # Convert symbol to CoinAPI format (BTC/USD stays as is)
        
        # CoinAPI needs a special header for authentication
        headers = {}
        if hasattr(self, 'coinapi_key'):
            headers['X-CoinAPI-Key'] = self.coinapi_key
        
        url = f"https://rest.coinapi.io/v1/exchangerate/{symbol.split('/')[0]}/USD"
        
        try:
            response = requests.get(url, headers=headers, timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                if 'rate' in data:
                    return float(data['rate'])
            elif response.status_code == 429:
                logger.warning("CoinAPI rate limit exceeded")
        except Exception as e:
            logger.warning(f"CoinAPI error: {e}")
        
        return None
    
    def get_historical_data(self, symbol, days=30):
        """Get historical data for a symbol with multiple provider fallbacks."""
        # List of API providers to try in order
        providers = []
        
        # Use the same provider priority as for real-time data
        if self.use_authenticated_api:
            providers.append({
                'name': 'CryptoCompare',
                'fetch_func': self._fetch_history_from_cryptocompare,
                'requires_auth': True
            })
        
        # Always add free providers as fallbacks
        providers.extend([
            {
                'name': 'CoinGecko',
                'fetch_func': self._fetch_history_from_coingecko,
                'requires_auth': False
            },
            {
                'name': 'Alternative.me',
                'fetch_func': self._fetch_history_from_alternative,
                'requires_auth': False
            },
            {
                'name': 'Yahoo Finance',
                'fetch_func': self._fetch_history_from_yahoo,
                'requires_auth': False
            }
        ])
        
        # Remove providers that require auth if we don't have keys
        providers = [p for p in providers if not (p['requires_auth'] and not self.api_key)]
        
        # Try each provider with retries
        max_retries = 2
        errors = []
        
        for provider in providers:
            provider_name = provider['name']
            fetch_func = provider['fetch_func']
            
            for attempt in range(1, max_retries + 1):
                try:
                    logger.debug(f"Attempting to fetch historical data for {symbol} from {provider_name} (attempt {attempt}/{max_retries})")
                    df = fetch_func(symbol, days)
                    
                    if df is not None and not df.empty:
                        logger.info(f"Successfully fetched historical data for {symbol} from {provider_name} ({len(df)} data points)")
                        return df
                    
                except Exception as e:
                    error_msg = f"Error fetching historical data for {symbol} from {provider_name}: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    
                    # Add exponential backoff between retries
                    if attempt < max_retries:
                        backoff_time = 0.2 * (2 ** (attempt - 1))
                        time.sleep(backoff_time)
        
        # If all APIs fail, generate synthetic data
        logger.warning(f"All API providers failed to fetch historical data for {symbol}, generating synthetic data. Errors: {errors}")
        return self._generate_synthetic_historical_data(symbol, days)
    
    def _fetch_history_from_cryptocompare(self, symbol, days):
        """Fetch historical data from CryptoCompare API."""
        symbol_id = self._get_symbol_id(symbol)
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            'fsym': symbol_id,
            'tsym': 'USD',
            'limit': days,
            'api_key': self.api_key
        }
        
        response = requests.get(url, params=params, timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            if 'Data' in data and 'Data' in data['Data']:
                history_data = data['Data']['Data']
                dates = [datetime.fromtimestamp(item['time']) for item in history_data]
                prices = [item['close'] for item in history_data]
                
                df = pd.DataFrame({
                    'date': dates,
                    'price': prices
                })
                df.set_index('date', inplace=True)
                return df
        
        return None
    
    def _fetch_history_from_coingecko(self, symbol, days):
        """Fetch historical data from CoinGecko API."""
        symbol_id = self._get_symbol_id(symbol)
        url = f"https://api.coingecko.com/api/v3/coins/{symbol_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params, timeout=10.0)
        
        if response.status_code == 200:
            data = response.json()
            prices = data.get('prices', [])
            if prices:
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)
                df.drop('timestamp', axis=1, inplace=True)
                return df
        
        return None
    
    def _fetch_history_from_alternative(self, symbol, days):
        """Fetch historical data from Alternative.me API (free for BTC/ETH)."""
        # This API is mainly for BTC
        if not (symbol.startswith("BTC") or symbol.startswith("ETH")):
            return None
            
        # Alternative.me uses 'bitcoin' and 'ethereum'
        symbol_name = "bitcoin" if symbol.startswith("BTC") else "ethereum"
        
        url = f"https://api.alternative.me/v2/historical_price/{symbol_name}/"
        params = {
            'limit': days,
            'end': int(time.time())  # current timestamp
        }
        
        try:
            response = requests.get(url, params=params, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    # Process their data format
                    history_data = data['data']
                    dates = [datetime.fromtimestamp(int(item['time'])) for item in history_data]
                    prices = [float(item['price']) for item in history_data]
                    
                    df = pd.DataFrame({
                        'date': dates,
                        'price': prices
                    })
                    df.set_index('date', inplace=True)
                    return df
        except Exception:
            pass
        
        return None
    
    def _fetch_history_from_yahoo(self, symbol, days):
        """Fetch historical data from Yahoo Finance (as last resort)."""
        try:
            import yfinance as yf
        except ImportError:
            # If yfinance is not installed, skip this provider
            logger.warning("yfinance not installed, skipping Yahoo Finance data source")
            return None
        
        # Convert symbol to Yahoo format
        yahoo_symbols = {
            "BTC/USD": "BTC-USD",
            "ETH/USD": "ETH-USD"
        }
        
        yahoo_symbol = yahoo_symbols.get(symbol)
        if not yahoo_symbol:
            return None
            
        try:
            # Get data from n days ago until today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(yahoo_symbol, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                # Use closing prices
                df = pd.DataFrame({
                    'price': data['Close']
                })
                return df
        except Exception as e:
            logger.warning(f"Yahoo Finance API error: {e}")
        
        return None
    
    def _generate_synthetic_historical_data(self, symbol, days):
        """Generate synthetic historical data based on current price and reasonable volatility."""
        # Get current price or use simulation value
        current_price = self.get_current_price(symbol)
        if current_price is None:
            current_price = self.simulation_prices.get(symbol, 100)
            logger.warning(f"Using SIMULATED price for historical data generation")
        
        # Set volatility based on asset type
        if symbol.startswith("BTC"):
            daily_volatility = 0.03  # 3% daily volatility for BTC
        elif symbol.startswith("ETH"):
            daily_volatility = 0.04  # 4% daily volatility for ETH
        else:
            daily_volatility = 0.05  # 5% for other assets
        
        # Generate dates
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(days)][::-1]
        
        # Generate prices using geometric brownian motion (simplified)
        prices = [current_price]
        for i in range(1, days):
            daily_return = np.random.normal(0.0002, daily_volatility)  # Slight upward drift
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'price': prices
        })
        df.set_index('date', inplace=True)
        
        logger.info(f"Generated synthetic historical data for {symbol} starting at ${current_price:.2f}")
        return df
    
    def check_health(self):
        """Check the health of the data provider."""
        return self.health_status
    
    def _data_fetch_loop(self):
        """Main loop to fetch cryptocurrency data."""
        while self.running:
            try:
                current_time = time.time()
                
                # Fetch data for each subscribed symbol
                for symbol in self.subscribed_symbols:
                    # Check if we need to update (rate limiting)
                    last_update = self.last_update_time.get(symbol, 0)
                    if current_time - last_update < self.update_interval:
                        continue
                    
                    # Get the base symbol (BTC, ETH, etc.)
                    symbol_base = symbol.split('/')[0].lower()
                    
                    try:
                        # Fetch from CoinGecko API (free, no API key needed)
                        url = f"https://api.coingecko.com/api/v3/simple/price"
                        params = {
                            'ids': symbol_base,
                            'vs_currencies': 'usd',
                            'include_market_cap': 'true',
                            'include_24hr_vol': 'true',
                            'include_24hr_change': 'true'
                        }
                        response = requests.get(url, params=params)
                        data = response.json()
                        
                        if symbol_base in data:
                            price_data = data[symbol_base]
                            price = price_data.get('usd', 0)
                            market_cap = price_data.get('usd_market_cap', 0)
                            volume = price_data.get('usd_24h_vol', 0)
                            change_24h = price_data.get('usd_24h_change', 0)
                            
                            # Create market data object
                            market_data = {
                                'symbol': symbol,
                                'price': price,
                                'market_cap': market_cap,
                                'volume': volume,
                                'change_24h': change_24h,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Update cache
                            self.price_cache[symbol] = price
                            self.last_update_time[symbol] = current_time
                            
                            # Store the data
                            self.latest_data[symbol] = market_data
                            
                            # Add to the queue
                            self.data_queue.put({
                                'type': 'market_data',
                                'symbol': symbol,
                                'data': market_data
                            })
                            
                            logger.debug(f"Updated {symbol}: ${price:.2f}")
                    
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {e}")
                        if not self.failure_mode:
                            self.health_status = HealthStatus.DEGRADED
                
                # Sleep until next cycle
                time.sleep(5.0)  # 5 second interval between API calls
                    
            except Exception as e:
                logger.error(f"Error in data fetch loop: {e}")
                self.health_status = HealthStatus.DEGRADED
                time.sleep(10.0)  # Slow down on error

#=============================================================================
# Mock Sentiment Data Provider
#=============================================================================

class MockSentimentDataProvider:
    """Mock sentiment data provider for cryptocurrency assets."""
    
    def __init__(self):
        """Initialize the mock sentiment data provider."""
        self.subscribed_symbols = set()
        self.data_queue = queue.Queue(maxsize=1000)
        self.latest_data = {}
        self.sentiment_history = {}
        self.health_status = HealthStatus.HEALTHY
        self.failure_mode = False
        self.data_thread = None
        self.running = False
        self.update_interval = 60.0  # seconds - sentiment updates less frequently than price
        
        logger.info("Initialized Mock Sentiment Data Provider")
    
    def start(self):
        """Start the sentiment data provider."""
        if self.running:
            logger.warning("Sentiment data provider already running")
            return False
        
        self.running = True
        self.data_thread = threading.Thread(target=self._sentiment_data_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        logger.info("Started sentiment data provider")
        return True
    
    def stop(self):
        """Stop the sentiment data provider."""
        if not self.running:
            logger.warning("Sentiment data provider not running")
            return False
        
        self.running = False
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=3.0)
        
        logger.info("Stopped sentiment data provider")
        return True
    
    def subscribe(self, symbol):
        """Subscribe to sentiment updates for a symbol."""
        self.subscribed_symbols.add(symbol)
        return True
    
    def get_latest_data(self, timeout=0.1):
        """Get the latest sentiment data."""
        try:
            return self.data_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_historical_data(self, symbol, days=30):
        """Get historical sentiment data for a symbol (mocked)."""
        # Create some basic synthetic data
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        sentiment_scores = []
        
        # Create synthetic sentiment with some trends based on symbol
        if symbol.startswith('BTC'):
            # BTC gets more positive sentiment
            sentiment_scores = [random.uniform(-0.3, 0.8) for _ in range(days)]
        elif symbol.startswith('ETH'):
            # ETH gets slightly positive sentiment
            sentiment_scores = [random.uniform(-0.4, 0.6) for _ in range(days)]
        else:
            # Other assets get neutral sentiment
            sentiment_scores = [random.uniform(-0.5, 0.5) for _ in range(days)]
        
        # Add a sentiment trend (cycles of positive/negative)
        for i in range(days):
            cycle_adjust = 0.3 * np.sin(i / 10 * np.pi)
            sentiment_scores[i] += cycle_adjust
            # Clamp values
            sentiment_scores[i] = max(-1.0, min(1.0, sentiment_scores[i]))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'confidence': [random.uniform(0.7, 0.95) for _ in range(days)]
        })
        df.set_index('date', inplace=True)
        return df
    
    def check_health(self):
        """Check the health of the sentiment data provider."""
        return self.health_status
    
    def _sentiment_data_loop(self):
        """Main loop to generate sentiment data."""
        while self.running:
            try:
                # Skip if in failure mode
                if self.failure_mode:
                    time.sleep(1.0)
                    continue
                
                # Generate sentiment data for each subscribed symbol
                for symbol in self.subscribed_symbols:
                    # Generate sentiment based on symbol and add some randomness
                    base_sentiment = 0.0
                    if symbol.startswith('BTC'):
                        # BTC gets more positive sentiment
                        base_sentiment = 0.3
                    elif symbol.startswith('ETH'):
                        # ETH gets slightly positive sentiment
                        base_sentiment = 0.1
                    
                    # Add randomness
                    sentiment_score = base_sentiment + random.uniform(-0.5, 0.5)
                    # Clamp to valid range
                    sentiment_score = max(-1.0, min(1.0, sentiment_score))
                    
                    # Generate random sentiment data
                    sentiment_data = {
                        'sentiment_score': sentiment_score,
                        'confidence': random.uniform(0.7, 0.95),
                        'source': random.choice(['news', 'social_media', 'analyst_report']),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Store the data
                    self.latest_data[symbol] = sentiment_data
                    
                    # Keep track of sentiment history
                    if symbol not in self.sentiment_history:
                        self.sentiment_history[symbol] = []
                    self.sentiment_history[symbol].append((datetime.now(), sentiment_score))
                    
                    # Only keep the last 30 days of history
                    max_history = 30 * 24  # 30 days worth of hourly updates
                    if len(self.sentiment_history[symbol]) > max_history:
                        self.sentiment_history[symbol] = self.sentiment_history[symbol][-max_history:]
                    
                    # Add to the queue
                    self.data_queue.put({
                        'type': 'sentiment_data',
                        'symbol': symbol,
                        'data': sentiment_data
                    })
                    
                    logger.debug(f"Updated sentiment for {symbol}: {sentiment_score:.2f}")
                
                # Sleep for the update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in sentiment data loop: {e}")
                time.sleep(1.0)

#=============================================================================
# Paper Trading Engine
#=============================================================================

class PaperTradingEngine:
    """Paper trading engine for executing trades without real money."""
    
    def __init__(self, initial_cash=INITIAL_CASH):
        """Initialize the paper trading engine.
        
        Args:
            initial_cash: Starting cash amount for paper trading
        """
        self.cash = initial_cash
        self.positions = {}  # symbol -> quantity
        self.transactions = []
        self.orders = {}  # order_id -> order details
        self.next_order_id = 1
        self.data_provider = None
        self.health_status = HealthStatus.HEALTHY
        
        logger.info(f"Initialized Paper Trading Engine with ${initial_cash:.2f}")
    
    def set_data_provider(self, provider):
        """Set the market data provider for price information."""
        self.data_provider = provider
        logger.info("Set data provider for paper trading engine")
    
    def place_order(self, symbol, order_type, side, quantity, price=None):
        """Place a new order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            order_type: Type of order ('market' or 'limit')
            side: 'buy' or 'sell'
            quantity: Amount to buy/sell
            price: Limit price (only for limit orders)
        
        Returns:
            order_id: Unique ID of the placed order
        """
        # Generate order ID
        order_id = str(self.next_order_id)
        self.next_order_id += 1
        
        # Current timestamp
        timestamp = datetime.now()
        
        # Get current price if needed
        current_price = None
        if self.data_provider:  
            current_price = self.data_provider.get_current_price(symbol)
        
        # For market orders or if we can't get current price, use provided price or estimation
        if order_type == 'market' or current_price is None:
            if price is not None:
                current_price = price
            else:
                logger.warning(f"No price data available for {symbol}, using placeholder price")
                # Use a placeholder price if we have nothing better
                current_price = 50000 if symbol.startswith('BTC') else 3000  # Placeholder prices
        
        # Create the order
        order = {
            'id': order_id,
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'quantity': quantity,
            'price': price,  # Limit price if specified
            'executed_price': current_price if order_type == 'market' else None,
            'status': 'created',
            'created_at': timestamp.isoformat(),
            'updated_at': timestamp.isoformat(),
            'filled': 0.0
        }
        
        # Store the order
        self.orders[order_id] = order
        
        # For market orders, execute immediately
        if order_type == 'market':
            self._execute_order(order_id)
        
        logger.info(f"Placed {order_type} {side} order for {quantity} {symbol} at {price if price else 'market price'}")
        return order_id
    
    def cancel_order(self, order_id):
        """Cancel an existing order."""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        if order['status'] in ['filled', 'cancelled']:
            logger.warning(f"Cannot cancel order {order_id}, status is {order['status']}")
            return False
        
        order['status'] = 'cancelled'
        order['updated_at'] = datetime.now().isoformat()
        
        logger.info(f"Cancelled order {order_id}")
        return True
    
    def get_order(self, order_id):
        """Get details of an order."""
        return self.orders.get(order_id)
    
    def get_position(self, symbol):
        """Get current position for a symbol."""
        return self.positions.get(symbol, 0.0)
    
    def get_balance(self):
        """Get current cash balance."""
        return self.cash
    
    def get_portfolio_value(self):
        """Get total portfolio value (cash + positions)."""
        total = self.cash
        
        for symbol, quantity in self.positions.items():
            price = None
            if self.data_provider:
                price = self.data_provider.get_current_price(symbol)
            
            if price is not None:
                total += quantity * price
            else:
                logger.warning(f"Cannot value {symbol} position, no price data")
        
        return total
    
    def get_transaction_history(self):
        """Get history of all transactions."""
        return self.transactions
    
    def _execute_order(self, order_id):
        """Internal method to execute an order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        symbol = order['symbol']
        side = order['side']
        quantity = order['quantity']
        
        # Get current price
        current_price = None
        if self.data_provider:
            current_price = self.data_provider.get_current_price(symbol)
        
        # Use provided price if we don't have current price
        if current_price is None:
            if order['executed_price'] is not None:
                current_price = order['executed_price']
            elif order['price'] is not None:
                current_price = order['price']
            else:
                logger.error(f"Cannot execute order {order_id}, no price available")
                order['status'] = 'failed'
                order['updated_at'] = datetime.now().isoformat()
                return False
        
        # Calculate cost/proceeds
        cost = quantity * current_price
        
        # For buy orders, check if we have enough cash
        if side == 'buy':
            if cost > self.cash:
                logger.warning(f"Insufficient funds to execute buy order {order_id}: need ${cost:.2f}, have ${self.cash:.2f}")
                order['status'] = 'failed'
                order['updated_at'] = datetime.now().isoformat()
                return False
            
            # Deduct cash and add to position
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0.0) + quantity
        
        # For sell orders, check if we have enough of the asset
        elif side == 'sell':
            current_position = self.positions.get(symbol, 0.0)
            if quantity > current_position:
                logger.warning(f"Insufficient {symbol} to execute sell order {order_id}: need {quantity}, have {current_position}")
                order['status'] = 'failed'
                order['updated_at'] = datetime.now().isoformat()
                return False
            
            # Add cash and reduce position
            self.cash += cost
            self.positions[symbol] = current_position - quantity
        
        # Update order status
        order['status'] = 'filled'
        order['executed_price'] = current_price
        order['filled'] = quantity
        order['updated_at'] = datetime.now().isoformat()
        
        # Record the transaction
        transaction = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': current_price,
            'value': cost,
            'timestamp': datetime.now().isoformat()
        }
        self.transactions.append(transaction)
        
        logger.info(f"Executed {side} order for {quantity} {symbol} at ${current_price:.2f}, total ${cost:.2f}")
        return True

#=============================================================================
# Trading System Orchestrator
#=============================================================================

class TradingSystemOrchestrator:
    """Orchestrates all components of the trading system."""
    
    def __init__(self, market_data_provider=None, sentiment_provider=None, 
                 trading_engine=None, llm_oversight=None, initial_cash=100000):
        """Initialize the trading system orchestrator."""
        # Use provided components or create new ones
        self.market_data_provider = market_data_provider or CryptoDataProvider()
        self.sentiment_provider = sentiment_provider or MockSentimentDataProvider()
        self.trading_engine = trading_engine or PaperTradingEngine(initial_cash=initial_cash)
        self.llm_oversight = llm_oversight or MockLLMOversight()
        
        # Connect components if needed
        if not hasattr(self.trading_engine, 'data_provider') or not self.trading_engine.data_provider:
            self.trading_engine.set_data_provider(self.market_data_provider)
        
        # System state
        self.running = False
        self.control_thread = None
        self.start_time = None
        self.end_time = None
        self.cycle_count = 0
        self.max_cycles = 1000
        
        # Trading metrics
        self.initial_portfolio_value = initial_cash
        self.latest_portfolio_value = initial_cash
        self.peak_portfolio_value = initial_cash
        
        # Statistics and performance tracking
        self.trade_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit_loss': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0
        }
        
        logger.info("Initialized Trading System Orchestrator")
    
    def start(self, asset_list=['BTC/USD', 'ETH/USD'], duration_sec=1800):
        """Start the trading system."""
        if self.running:
            logger.warning("Trading system already running")
            return False
        
        # Record start time and calculate end time
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(seconds=duration_sec)
        
        # Start all components
        logger.info(f"Starting trading system for {len(asset_list)} assets, duration: {duration_sec} seconds")
        
        # Start market data provider
        self.market_data_provider.start()
        
        # Subscribe to assets
        for asset in asset_list:
            self.market_data_provider.subscribe(asset)
            self.sentiment_provider.subscribe(asset)
            logger.info(f"Subscribed to {asset}")
            
            # Initialize price history with initial prices
            # This ensures we have data to make trading decisions from the beginning
            if not hasattr(self, 'price_history'):
                self.price_history = {}
            
            # Get initial price
            initial_price = self.market_data_provider.get_current_price(asset)
            if initial_price:
                # Create slight variations of the price for history
                self.price_history[asset] = [
                    initial_price * (1 + random.uniform(-0.005, 0.005)) for _ in range(5)
                ]
                # Add the current price at the end
                self.price_history[asset].append(initial_price)
                logger.info(f"Initialized price history for {asset} with starting price: ${initial_price:.2f}")
        
        # Start sentiment provider and ensure we have initial sentiment values
        self.sentiment_provider.start()
        
        # Initialize sentiment values
        for asset in asset_list:
            # Generate initial sentiment
            if asset.startswith("BTC"):
                # Slightly positive sentiment for BTC
                sentiment_score = 0.3 + random.uniform(-0.1, 0.1)
            else:
                # Neutral sentiment for others
                sentiment_score = 0.1 + random.uniform(-0.2, 0.2)
                
            # Store in sentiment provider
            self.sentiment_provider.latest_data[asset] = {
                'sentiment_score': sentiment_score,
                'confidence': 0.85,
                'source': 'initialization',
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Initialized sentiment for {asset}: {sentiment_score:.2f}")
        
        # Start LLM oversight if available
        if self.llm_oversight:
            self.llm_oversight.start()
            
            # Configure LLM to be more approving in test mode
            if hasattr(self.llm_oversight, 'approval_rate'):
                self.llm_oversight.approval_rate = 0.95  # 95% approval in test mode
                logger.info(f"Set LLM oversight approval rate to 95% for testing")
        
        # Start main control loop
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info(f"Trading system started, will run until {self.end_time}")
        return True
    
    def stop(self):
        """Stop the trading system."""
        if not self.running:
            logger.warning("Trading system already stopped")
            return False
            
        logger.info("Stopping trading system...")
        self.running = False
        
        # Set stopping flag before joining threads to prevent deadlocks
        self._stopping = True
        
        # Gracefully terminate control thread
        if self.control_thread and self.control_thread.is_alive():
            logger.debug("Waiting for control thread to terminate...")
            try:
                self.control_thread.join(timeout=3.0)
                if self.control_thread.is_alive():
                    logger.warning("Control thread did not terminate in time, proceeding anyway")
            except Exception as e:
                logger.error(f"Error while stopping control thread: {e}")
        
        # Stop all components in reverse order of initialization
        if self.llm_oversight:
            try:
                logger.debug("Stopping LLM oversight...")
                self.llm_oversight.stop()
            except Exception as e:
                logger.error(f"Error stopping LLM oversight: {e}")
        
        try:
            logger.debug("Stopping sentiment provider...")
            self.sentiment_provider.stop()
        except Exception as e:
            logger.error(f"Error stopping sentiment provider: {e}")
            
        try:
            logger.debug("Stopping market data provider...")
            self.market_data_provider.stop()
        except Exception as e:
            logger.error(f"Error stopping market data provider: {e}")
            
        # Calculate final statistics
        self._calculate_final_stats()
        
        logger.info("Trading system stopped successfully")
        return True
    
    def get_status(self):
        """Get the current status of the trading system."""
        # Calculate portfolio value
        portfolio_value = self.trading_engine.get_portfolio_value()
        
        # Calculate performance metrics
        performance = {
            'portfolio_value': portfolio_value,
            'initial_value': self.initial_portfolio_value,
            'absolute_change': portfolio_value - self.initial_portfolio_value,
            'percent_change': ((portfolio_value / self.initial_portfolio_value) - 1.0) * 100,
            'run_time': str(datetime.now() - self.start_time) if self.start_time else "Not started",
            'trades': len(self.trading_engine.transactions),
            'positions': self.trading_engine.positions,
            'cash': self.trading_engine.cash
        }
        
        # Component health
        health = {
            'market_data': self.market_data_provider.check_health(),
            'sentiment': self.sentiment_provider.check_health(),
            'llm_oversight': self.llm_oversight.check_health() if self.llm_oversight else None
        }
        
        return {
            'running': self.running,
            'cycle': self.cycle_count,
            'max_cycles': self.max_cycles,
            'performance': performance,
            'health': health,
            'trade_stats': self.trade_stats
        }
    
    def _control_loop(self):
        """Main control loop for the trading system."""
        logger.info("Starting control loop")
        
        while self.running:
            try:
                # Check if test duration has elapsed
                now = datetime.now()
                if now >= self.end_time:
                    logger.info(f"Test duration elapsed at {now}, stopping system")
                    self.running = False
                    break
                
                # Increment cycle counter
                self.cycle_count += 1
                if self.cycle_count >= self.max_cycles:
                    logger.info(f"Reached maximum test cycles ({self.max_cycles}), stopping system")
                    self.running = False
                    break
                
                # Process market data
                market_data = self.market_data_provider.get_latest_data(timeout=0.1)
                if market_data:
                    # New market data received, process it
                    self._process_market_data(market_data)
                
                # Process sentiment data
                sentiment_data = self.sentiment_provider.get_latest_data(timeout=0.1)
                if sentiment_data:
                    # New sentiment data received, process it
                    self._process_sentiment_data(sentiment_data)
                
                # Run trading strategy
                self._run_trading_strategy()
                
                # Update portfolio value
                self._update_portfolio_metrics()
                
                # Sleep until next cycle
                time.sleep(CYCLE_INTERVAL_SEC)
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(5.0)  # Slow down on error
        
        logger.info("Control loop ended")
    
    def _process_market_data(self, data):
        """Process new market data."""
        # Extract relevant information
        data_type = data.get('type')
        symbol = data.get('symbol')
        market_data = data.get('data', {})
        
        if data_type == 'market_data' and symbol and market_data:
            price = market_data.get('price')
            logger.debug(f"Processing market data for {symbol}: ${price:.2f}")
            
            # Here we would update our market models, technical indicators, etc.
            # For this test, we'll just log the data
            if price:
                logger.info(f"Market data: {symbol} = ${price:.2f}")
    
    def _process_sentiment_data(self, data):
        """Process new sentiment data."""
        # Extract relevant information
        data_type = data.get('type')
        symbol = data.get('symbol')
        sentiment_data = data.get('data', {})
        
        if data_type == 'sentiment_data' and symbol and sentiment_data:
            sentiment_score = sentiment_data.get('sentiment_score')
            confidence = sentiment_data.get('confidence')
            
            if sentiment_score is not None:
                logger.info(f"Sentiment data: {symbol} = {sentiment_score:.2f} (confidence: {confidence:.2f})")
    
    def _run_trading_strategy(self):
        """Execute trading strategy based on available data."""
        # Enhanced strategy with:
        # 1. More realistic price thresholds based on current market conditions
        # 2. Selling logic for profit-taking and stop losses
        # 3. Technical indicators and sentiment integration
        
        # Store some historical prices for technical analysis
        # This would normally be done in a more sophisticated way
        if not hasattr(self, 'price_history'):
            self.price_history = {symbol: [] for symbol in ASSETS}
            self.last_trade_prices = {}
            self.position_entry_prices = {}
        
        # Check each symbol
        for symbol in ASSETS:
            # Get current price
            price = self.market_data_provider.get_current_price(symbol)
            if not price:
                continue
            
            # Update price history
            self.price_history[symbol].append(price)
            # Keep only last 20 prices for our simple technical analysis
            if len(self.price_history[symbol]) > 20:
                self.price_history[symbol] = self.price_history[symbol][-20:]
            
            # Get current position
            position = self.trading_engine.get_position(symbol)
            cash = self.trading_engine.get_balance()
            
            # Technical Analysis Indicators
            # ---------------------------
            
            # 1. Simple Moving Averages
            short_ma = np.mean(self.price_history[symbol][-5:]) if len(self.price_history[symbol]) >= 5 else price
            long_ma = np.mean(self.price_history[symbol][-15:]) if len(self.price_history[symbol]) >= 15 else price
            
            # 2. Simple momentum (current price vs price 5 periods ago)
            momentum = price - self.price_history[symbol][-5] if len(self.price_history[symbol]) >= 5 else 0
            
            # 3. Price volatility (standard deviation of recent prices)
            volatility = np.std(self.price_history[symbol][-10:]) if len(self.price_history[symbol]) >= 10 else 0
            
            # 4. Get sentiment data
            sentiment = 0
            for symbol_key in self.sentiment_provider.latest_data:
                if symbol in symbol_key:
                    sentiment_data = self.sentiment_provider.latest_data[symbol_key]
                    sentiment = sentiment_data.get('sentiment_score', 0)
                    break
            
            # Market regime detection (simple version)
            if short_ma > long_ma and momentum > 0:
                regime = 'bullish'
            elif short_ma < long_ma and momentum < 0:
                regime = 'bearish'
            elif abs(short_ma - long_ma) / long_ma < 0.01:
                regime = 'sideways'
            else:
                regime = 'mixed'
            
            logger.debug(f"Analysis for {symbol}: Price=${price:.2f}, Short MA=${short_ma:.2f}, Long MA=${long_ma:.2f}, Momentum=${momentum:.2f}, Sentiment={sentiment:.2f}, Regime={regime}")
            
            # Position sizing based on volatility
            # Lower position size in higher volatility environments
            base_position_value = 10000  # Default position value in USD
            if volatility > 0:
                vol_adjustment = min(1, 500 / volatility)  # Reduce position size as volatility increases
                position_value = base_position_value * vol_adjustment
            else:
                position_value = base_position_value
            
            # Calculate position size
            quantity = round(position_value / price, 8)  # Round to 8 decimal places for crypto
            
            # Ensure minimum position sizes for crypto
            min_btc = 0.001
            min_eth = 0.01
            if symbol == "BTC/USD" and quantity < min_btc:
                quantity = min_btc
            elif symbol == "ETH/USD" and quantity < min_eth:
                quantity = min_eth
            
            # Trading Strategy Logic
            # ----------------------
            
            # CASE 1: We have no position - check for entry conditions
            if position == 0:
                entry_signal = False
                entry_reason = ""
                
                # IMPORTANT: For demonstration purposes only, we'll force trades on specific cycles
                # This is to showcase the system functionality during a brief test
                # In a real system, you would use proper technical analysis and market conditions
                
                # Force trades on specific cycles for demonstration
                if self.cycle_count == 2 and symbol == "BTC/USD":
                    entry_signal = True
                    entry_reason = "DEMO: Forced BTC entry for demonstration purposes"
                elif self.cycle_count == 4 and symbol == "ETH/USD":
                    entry_signal = True
                    entry_reason = "DEMO: Forced ETH entry for demonstration purposes"
                elif self.cycle_count == 10 and symbol == "BTC/USD" and not position:
                    entry_signal = True
                    entry_reason = "DEMO: Second forced BTC entry for demonstration"
                    
                # Real strategy (would be used in actual trading)
                elif len(self.price_history[symbol]) > 2:
                    # BTC Strategy
                    if symbol == "BTC/USD":
                        if price > self.price_history[symbol][-2] and sentiment > 0:
                            entry_signal = True
                            entry_reason = f"BTC upward momentum with positive sentiment ({sentiment:.2f})"
                    
                    # ETH Strategy
                    elif symbol == "ETH/USD":
                        if price > self.price_history[symbol][-2] and sentiment >= -0.1:
                            entry_signal = True
                            entry_reason = f"ETH upward momentum with neutral/positive sentiment ({sentiment:.2f})"
                            
                # Log our analysis regardless of decisions
                logger.debug(f"Analysis for {symbol} - Price: ${price:.2f}, Position: {position}, Sentiment: {sentiment:.2f}, Cycle: {self.cycle_count}")
                
                # Make sure we display the real price even if entry_signal is False
                if not entry_signal and len(self.price_history[symbol]) > 1 and self.cycle_count % 3 == 0:
                    logger.info(f"Current market price for {symbol}: ${price:.2f}")
                
                # Special Logging for Demo
                if self.cycle_count <= 20 and self.cycle_count % 3 == 0:
                    logger.info(f"ANALYSIS - Cycle {self.cycle_count} - {symbol}: Price=${price:.2f}, Sentiment={sentiment:.2f}, Position={position}")
                
                # Execute buy if we have an entry signal and enough cash
                if entry_signal and cash >= price * quantity:
                    # Submit decision to LLM oversight if available
                    approved = True
                    if self.llm_oversight:
                        decision = {
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': quantity,
                            'price': price,
                            'reason': entry_reason,
                            'technical_indicators': {
                                'short_ma': short_ma,
                                'long_ma': long_ma, 
                                'momentum': momentum,
                                'volatility': volatility,
                                'sentiment': sentiment,
                                'regime': regime
                            }
                        }
                        result = self.llm_oversight.review_decision(decision)
                        approved = result.get('approved', False)
                    
                    if approved:
                        order_id = self.trading_engine.place_order(symbol, 'market', 'buy', quantity)
                        logger.info(f"BUY decision for {symbol}: {quantity} at ${price:.2f}, Reason: {entry_reason}")
                        
                        # Record entry price for position
                        self.position_entry_prices[symbol] = price
                        
                        # Store last trade for this symbol
                        self.last_trade_prices[symbol] = price
            
            # CASE 2: We have an existing position - check for exit conditions
            elif position > 0:
                exit_signal = False
                exit_reason = ""
                
                # Get entry price if available
                entry_price = self.position_entry_prices.get(symbol, price * 0.9)  # Assume 10% lower if unknown
                profit_pct = (price - entry_price) / entry_price * 100
                
                # Force exit trades on specific cycles for demonstration
                if self.cycle_count == 7 and symbol == "BTC/USD":
                    exit_signal = True
                    exit_reason = "DEMO: Forced BTC exit for demonstration purposes"
                elif self.cycle_count == 9 and symbol == "ETH/USD":
                    exit_signal = True
                    exit_reason = "DEMO: Forced ETH exit for demonstration purposes"
                elif self.cycle_count == 15 and symbol == "BTC/USD":
                    exit_signal = True
                    exit_reason = "DEMO: Second forced BTC exit for demonstration"
                
                # Regular exit strategy (if not forced)
                elif profit_pct >= 0.2:  # Very small profit target for demo
                    exit_signal = True
                    exit_reason = f"Take profit target reached: {profit_pct:.2f}% gain"
                elif profit_pct <= -0.3:  # Very tight stop loss for demo
                    exit_signal = True 
                    exit_reason = f"Stop loss triggered: {profit_pct:.2f}% loss"
                elif sentiment < -0.1:  # Exit on any negative sentiment
                    exit_signal = True
                    exit_reason = f"Exit on negative sentiment: {sentiment:.2f}"
                
                # 4. Rapid cycling for testing - exit after holding for a few cycles
                # Use a cycle counter instead of trying to find exact price match
                if not hasattr(self, 'position_age_counters'):
                    self.position_age_counters = {}
                
                # Initialize counter if new position
                if symbol not in self.position_age_counters and symbol in self.position_entry_prices:
                    self.position_age_counters[symbol] = 1
                elif symbol in self.position_age_counters:
                    self.position_age_counters[symbol] += 1
                
                # Exit after several cycles of holding
                if symbol in self.position_age_counters and self.position_age_counters[symbol] > 5:
                    exit_signal = True
                    exit_reason = f"Test cycle exit after holding for {self.position_age_counters[symbol]} cycles"
                    # Reset counter on exit
                    if exit_signal:
                        self.position_age_counters[symbol] = 0
                
                # Execute sell if we have an exit signal
                if exit_signal:
                    # Submit decision to LLM oversight if available
                    approved = True
                    if self.llm_oversight:
                        decision = {
                            'symbol': symbol,
                            'action': 'sell',
                            'quantity': position,  # Sell entire position
                            'price': price,
                            'reason': exit_reason,
                            'technical_indicators': {
                                'short_ma': short_ma,
                                'long_ma': long_ma, 
                                'momentum': momentum,
                                'volatility': volatility,
                                'sentiment': sentiment,
                                'regime': regime,
                                'profit_pct': profit_pct
                            }
                        }
                        result = self.llm_oversight.review_decision(decision)
                        approved = result.get('approved', False)
                    
                    if approved:
                        order_id = self.trading_engine.place_order(symbol, 'market', 'sell', position)
                        logger.info(f"SELL decision for {symbol}: {position} at ${price:.2f}, Reason: {exit_reason}, P&L: {profit_pct:.2f}%")
                        
                        # Store last trade for this symbol
                        self.last_trade_prices[symbol] = price
                        
                        # Clear entry price
                        if symbol in self.position_entry_prices:
                            del self.position_entry_prices[symbol]
    
    def _update_portfolio_metrics(self):
        """Update portfolio metrics."""
        portfolio_value = self.trading_engine.get_portfolio_value()
        self.latest_portfolio_value = portfolio_value
        
        # Update peak value
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        # Calculate drawdown
        drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100
        if drawdown > self.trade_stats['max_drawdown']:
            self.trade_stats['max_drawdown'] = drawdown
    
    def _calculate_final_stats(self):
        """Calculate final trading statistics."""
        # Get all transactions
        transactions = self.trading_engine.get_transaction_history()
        
        # Count trades
        self.trade_stats['total_trades'] = len(transactions)
        
        # Calculate win rate (for this test, we can't really determine wins/losses)
        self.trade_stats['win_rate'] = 0.0
        
        # Calculate total P&L
        final_value = self.trading_engine.get_portfolio_value()
        self.trade_stats['total_profit_loss'] = final_value - self.initial_portfolio_value
        
        logger.info(f"Final portfolio value: ${final_value:.2f} (Initial: ${self.initial_portfolio_value:.2f})")
        logger.info(f"P&L: ${self.trade_stats['total_profit_loss']:.2f} ({(self.trade_stats['total_profit_loss'] / self.initial_portfolio_value * 100):.2f}%)")
        logger.info(f"Max drawdown: {self.trade_stats['max_drawdown']:.2f}%")
        logger.info(f"Total trades: {self.trade_stats['total_trades']}")

#=============================================================================
# Entry Point
#=============================================================================

def run_test(duration_minutes=30, assets=None, initial_cash=INITIAL_CASH, debug=False):
    """Run the real data paper trading test.
    
    Args:
        duration_minutes: How long to run the test (in minutes)
        assets: List of assets to trade (default: BTC/USD, ETH/USD)
        initial_cash: Starting cash amount
        debug: Enable debug logging
    """
    # Set up logging level
    if debug:
        logger.setLevel(logging.DEBUG)
    
    if assets is None:
        assets = ASSETS
    
    # Create and start the trading system
    system = TradingSystemOrchestrator(initial_cash=initial_cash)
    
    try:
        # Start the system
        print(f"\n==== STARTING REAL DATA PAPER TRADING TEST - {duration_minutes} MINUTES ====\n")
        system.start(asset_list=assets, duration_sec=duration_minutes * 60)
        
        # Wait for completion (with status updates)
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time and system.running:
            # Print status every minute
            time.sleep(60)
            status = system.get_status()
            
            # Calculate time remaining
            remaining = end_time - datetime.now()
            remaining_min = remaining.total_seconds() / 60
            
            # Print current status
            print(f"\n---- Status Update ({remaining_min:.1f} minutes remaining) ----")
            print(f"Portfolio Value: ${status['performance']['portfolio_value']:.2f} ")
            print(f"Change: ${status['performance']['absolute_change']:.2f} ({status['performance']['percent_change']:.2f}%)")
            print(f"Positions: {status['performance']['positions']}")
            print(f"Cash: ${status['performance']['cash']:.2f}")
            print(f"Trades: {status['performance']['trades']}")
            print(f"System Health: {status['health']}")
            print("-" * 50)
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Stop the system
        system.stop()
        print("\n==== TEST COMPLETED ====\n")
        
        # Print final results
        status = system.get_status()
        print("\n==== FINAL RESULTS ====")
        print(f"Portfolio Value: ${status['performance']['portfolio_value']:.2f}")
        print(f"Initial Value: ${status['performance']['initial_value']:.2f}")
        print(f"Absolute Change: ${status['performance']['absolute_change']:.2f}")
        print(f"Percent Change: {status['performance']['percent_change']:.2f}%")
        print(f"Number of Trades: {status['performance']['trades']}")
        print(f"Final Positions: {status['performance']['positions']}")
        print(f"Remaining Cash: ${status['performance']['cash']:.2f}")
        print(f"Maximum Drawdown: {status['trade_stats']['max_drawdown']:.2f}%")
        print("\n==== TEST ENDED ====\n")

def display_market_dashboard(market_data_provider, sentiment_provider=None, trading_engine=None):
    """Display a dashboard of current market data with sources and confidence."""
    now = datetime.now().strftime("%H:%M:%S")
    # Use a clearly defined separator for better readability
    separator = "=" * 70
    dashboard = [f"\n{separator}\n== MARKET DATA DASHBOARD [{now}] =="]
    
    # Force refresh prices if the provider supports it
    if hasattr(market_data_provider, 'get_current_price'):
        # Explicitly update prices for more accurate dashboard
        market_data_provider.get_current_price("BTC/USD")
        market_data_provider.get_current_price("ETH/USD")
        btc_price = market_data_provider.get_current_price("BTC/USD")
        eth_price = market_data_provider.get_current_price("ETH/USD")
    
    # Get list of assets to display
    assets = list(market_data_provider.subscribed_symbols) if hasattr(market_data_provider, 'subscribed_symbols') else []
    if not assets and hasattr(market_data_provider, 'price_cache'):
        assets = list(market_data_provider.price_cache.keys())
    
    if not assets:
        dashboard.append("No assets subscribed")
    else:
        # Add a header for the price section
        dashboard.append("\nCURRENT MARKET PRICES:")
        dashboard.append("-" * 30)
        
        for asset in sorted(assets):  # Sort assets for consistent display
            # Get current price
            price = market_data_provider.price_cache.get(asset) if hasattr(market_data_provider, 'price_cache') else None
            source = market_data_provider.data_source.get(asset, "Unknown") if hasattr(market_data_provider, 'data_source') else "Unknown"
            
            if price is not None:
                # Get sentiment if available
                sentiment_desc = "N/A"
                sentiment_score = 0
                if sentiment_provider and hasattr(sentiment_provider, 'latest_data'):
                    sentiment_data = sentiment_provider.latest_data.get(asset, {})
                    sentiment_score = sentiment_data.get('sentiment_score', 0)
                    sentiment_desc = "Bearish" if sentiment_score < -0.2 else "Neutral" if sentiment_score < 0.2 else "Bullish"
                
                # Format the display line with clear visual separators
                dashboard.append(f"{asset:8} | ${price:,.2f} | Source: {source} | Sentiment: {sentiment_desc} ({sentiment_score:.2f})")
            else:
                dashboard.append(f"{asset:8} | No price data available")
    
    # Add portfolio status if trading engine is available
    if trading_engine and hasattr(trading_engine, 'get_portfolio_value') and hasattr(trading_engine, 'get_cash_balance'):
        portfolio_value = trading_engine.get_portfolio_value()
        cash = trading_engine.get_cash_balance()
        
        dashboard.append("\nPORTFOLIO STATUS:")
        dashboard.append("-" * 30)
        dashboard.append(f"Portfolio Value: ${portfolio_value:,.2f}")
        dashboard.append(f"Cash Balance: ${cash:,.2f}")
        
        # Show positions if available
        if hasattr(trading_engine, 'positions'):
            positions = trading_engine.positions
            if positions:
                dashboard.append("\nCURRENT POSITIONS:")
                dashboard.append("-" * 30)
                for symbol, quantity in positions.items():
                    if quantity > 0:
                        current_price = market_data_provider.get_current_price(symbol) if hasattr(market_data_provider, 'get_current_price') else None
                        position_value = quantity * current_price if current_price else 0
                        dashboard.append(f"{symbol:8} | {quantity:.6f} units | Value: ${position_value:,.2f}")
    
    # Show API stats summary if available
    if hasattr(market_data_provider, 'api_stats'):
        # Show successful providers
        dashboard.append("\nAPI PROVIDER STATISTICS:")
        dashboard.append("-" * 30)
        
        success_counts = [(name, stats['success'], stats['last_success']) 
                         for name, stats in market_data_provider.api_stats.items()]
        
        # Sort by success count (descending)
        success_counts.sort(key=lambda x: x[1], reverse=True)
        
        for name, count, last_success in success_counts:
            if count > 0:
                last_time = last_success.strftime("%H:%M:%S") if last_success else "Never"
                dashboard.append(f"{name:12} | Successful requests: {count} | Last success: {last_time}")
    
    dashboard.append(f"\n{separator}")
    
    # Print the dashboard with clear formatting
    dashboard_text = '\n'.join(dashboard)
    print(dashboard_text)
    return dashboard_text


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Trading Agent real data test")
    parser.add_argument("--duration", type=int, default=1800, help="Test duration in seconds (default: 1800)")
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash (default: $100,000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--use-api-keys", action="store_true", help="Use API keys from .env file")
    parser.add_argument("--dashboard", action="store_true", help="Show real-time market data dashboard")
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # Import from ai_trading_agent if available
    try:
        from ai_trading_agent.common.logging_config import setup_logging
        setup_logging()  # Call without level parameter
    except (ImportError, TypeError):
        # If import fails or function signature is different, just use basic logging
        pass
    
    logger.info("Starting Real Data Paper Trading Test")
    
    # Load API keys from .env file if requested
    api_keys = {}
    if args.use_api_keys:
        try:
            # Load environment variables from .env file
            dotenv.load_dotenv()
            
            # Get API keys from environment variables
            api_keys = {
                'cryptocompare': os.getenv('CRYPTOCOMPARE_API_KEY'),
                'coinapi': os.getenv('COINAPI_KEY')
            }
            
            valid_keys = [k for k, v in api_keys.items() if v]
            if valid_keys:
                logger.info(f"Loaded API keys: {', '.join(valid_keys)}")
            else:
                logger.warning("No valid API keys found in .env file. Using free APIs.")
                
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
    
    # Create data providers
    cryptocompare_key = api_keys.get('cryptocompare')
    coinapi_key = api_keys.get('coinapi')
    
    # Use the integrated MarketDataProvider from the core system if available,
    # otherwise fall back to the legacy CryptoDataProvider
    if actual_components_available:
        logger.info("Using MarketDataProvider from the core system")
        market_data_provider = MarketDataProvider(crypto_compare_api_key=cryptocompare_key, coinapi_key=coinapi_key)
    else:
        logger.info("Falling back to legacy CryptoDataProvider")
        market_data_provider = CryptoDataProvider(api_key=cryptocompare_key)
        # Add additional API keys if available
        if coinapi_key:
            market_data_provider.coinapi_key = coinapi_key
    
    sentiment_provider = MockSentimentDataProvider()
    
    # Create trading engine
    trading_engine = PaperTradingEngine(initial_cash=args.cash)
    trading_engine.set_data_provider(market_data_provider)
    
    # Create LLM oversight
    llm_oversight = MockLLMOversight()
    
    # Create and start trading system
    trading_system = TradingSystemOrchestrator(
        market_data_provider=market_data_provider,
        sentiment_provider=sentiment_provider,
        trading_engine=trading_engine,
        llm_oversight=llm_oversight
    )
    
    # Start the system and wait for completion
    trading_system.start()
    
    # Sleep until complete, showing the dashboard periodically
    end_time = datetime.now() + timedelta(seconds=args.duration)
    dashboard_interval = 10  # seconds between dashboard updates
    next_dashboard_time = datetime.now()
    
    logger.info(f"Running test until {end_time.strftime('%H:%M:%S')}, press Ctrl+C to stop early")
    
    try:
        while datetime.now() < end_time and trading_system.running:
            # Show dashboard if enabled and it's time
            if args.dashboard and datetime.now() >= next_dashboard_time:
                display_market_dashboard(market_data_provider, sentiment_provider, trading_engine)
                next_dashboard_time = datetime.now() + timedelta(seconds=dashboard_interval)
            
            # Shorter sleep to be more responsive
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # Show final dashboard
        if args.dashboard:
            display_market_dashboard(market_data_provider, sentiment_provider, trading_engine)
        
        # Stop the system and print results
        trading_system.stop()
        
        print("\n==== TEST ENDED ====")
