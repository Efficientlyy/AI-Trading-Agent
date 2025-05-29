"""
Cryptocurrency Data Provider Module.

This module provides a robust implementation of a cryptocurrency data provider
that can fetch real-time and historical cryptocurrency data from multiple sources
with fallback mechanisms, validation, and cache management.
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Import local modules
from .binance_client import BinanceClient
from .coingecko_client import CoinGeckoClient
from .crypto_compare_client import CryptoCompareClient

# Set up logging
logger = logging.getLogger(__name__)

class MarketDataProvider:
    """
    Real-time cryptocurrency data provider with multiple API fallbacks and validation.
    
    This class provides a reliable source of cryptocurrency data by:
    1. Supporting multiple API providers (CryptoCompare, CoinGecko, Binance)
    2. Implementing fallback mechanisms when APIs fail
    3. Validating data across multiple sources for accuracy
    4. Caching data to reduce API calls and improve response times
    5. Supporting both real-time and historical data
    """
    
    def __init__(self, crypto_compare_api_key=None, coin_api_key=None):
        """
        Initialize the market data provider.
        
        Args:
            crypto_compare_api_key: Optional API key for CryptoCompare
            coin_api_key: Optional API key for CoinAPI
        """
        # Load API keys from environment if not provided
        self.crypto_compare_api_key = crypto_compare_api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
        self.coin_api_key = coin_api_key or os.getenv("COINAPI_KEY")
        
        # Initialize clients
        self.binance_client = BinanceClient()
        self.coin_gecko_client = CoinGeckoClient()
        self.crypto_compare_client = CryptoCompareClient(api_key=self.crypto_compare_api_key)
        
        # Set up data structures
        self.subscribed_symbols = set()
        self.data_queue = queue.Queue(maxsize=1000)
        self.latest_data = {}
        self.historical_data = {}
        self.price_cache = {}
        self.cache_timeout = 30  # Cache timeout in seconds
        self.cache_timestamps = {}
        self.data_source = {}  # Track which source provided each price
        
        # API statistics tracking
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
        
        # Set up threading
        self.running = False
        self.data_thread = None
        self.lock = threading.Lock()
        
        # Config of which providers to try (in order of preference)
        # Will try these in order until one succeeds
        self.provider_configs = {
            'real_time': [
                ('CryptoCompare', self._fetch_from_crypto_compare),
                ('Binance', self._fetch_from_binance),
                ('CoinGecko', self._fetch_from_coin_gecko),
                ('Simulation', self._simulate_price)
            ],
            'historical': [
                ('CryptoCompare', self._fetch_historical_from_crypto_compare),
                ('CoinGecko', self._fetch_historical_from_coin_gecko),
                ('Simulation', self._simulate_historical_data)
            ]
        }
        
        # Market approximations to use as fallback
        # These values are updated whenever we get valid data
        self.fallback_prices = {
            'BTC/USD': 27000.0,  # Initial fallback prices (May 2023)
            'ETH/USD': 1800.0,
            'XRP/USD': 0.50,
            'ADA/USD': 0.40,
            'SOL/USD': 20.0,
            'DOGE/USD': 0.08
        }
        
        # Volatility values for each asset (for synthetic data generation)
        self.asset_volatilities = {
            'BTC/USD': 0.04,  # 4% daily volatility
            'ETH/USD': 0.05,  # 5% daily volatility
            'XRP/USD': 0.07,
            'ADA/USD': 0.08,
            'SOL/USD': 0.09,
            'DOGE/USD': 0.12
        }
        
        logger.info("Initialized MarketDataProvider with multiple API fallbacks")
    
    def start(self):
        """Start the data provider."""
        if self.running:
            logger.warning("MarketDataProvider is already running")
            return False
        
        self.running = True
        self.data_thread = threading.Thread(target=self._data_thread_func)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        logger.info("Started MarketDataProvider")
        return True
    
    def stop(self):
        """Stop the data provider and clean up resources."""
        if not self.running:
            logger.warning("MarketDataProvider is not running")
            return False
        
        self.running = False
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=3.0)
        
        logger.info("Stopped MarketDataProvider")
        return True
    
    def subscribe(self, symbol):
        """Subscribe to data updates for a symbol."""
        # Convert to standard format if needed
        if symbol == "BTC":
            symbol = "BTC/USD"
        elif symbol == "ETH":
            symbol = "ETH/USD"
        
        with self.lock:
            self.subscribed_symbols.add(symbol)
            # Force a refresh of data for this symbol
            self._update_price(symbol)
        
        logger.info(f"Subscribed to {symbol}")
        return True
    
    def unsubscribe(self, symbol):
        """Unsubscribe from data updates for a symbol."""
        # Convert to standard format if needed
        if symbol == "BTC":
            symbol = "BTC/USD"
        elif symbol == "ETH":
            symbol = "ETH/USD"
        
        with self.lock:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from {symbol}")
        return True
    
    def _data_thread_func(self):
        """Background thread to update data at regular intervals."""
        while self.running:
            try:
                # Update prices for all subscribed symbols
                for symbol in list(self.subscribed_symbols):
                    self._update_price(symbol)
                
                # Sleep for a short period
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Error in data thread: {e}")
                time.sleep(1.0)  # Sleep on error to prevent CPU spiking
    
    def _update_price(self, symbol):
        """Update the price for a symbol."""
        price, source, confidence = self._fetch_real_time_price(symbol)
        
        if price is not None:
            with self.lock:
                self.latest_data[symbol] = {
                    'price': price,
                    'timestamp': datetime.now(),
                    'source': source,
                    'confidence': confidence
                }
                self.price_cache[symbol] = price
                self.cache_timestamps[symbol] = time.time()
                self.data_source[symbol] = f"{source} ({confidence}% confidence)"
                
                # Update fallback price for this symbol when we get a good price
                if confidence >= 90:
                    self.fallback_prices[symbol] = price
                
                # Put data in the queue for subscribers
                try:
                    data_item = {
                        'symbol': symbol,
                        'price': price,
                        'timestamp': datetime.now().isoformat(),
                        'source': source
                    }
                    self.data_queue.put_nowait(data_item)
                except queue.Full:
                    # If queue is full, remove oldest item and try again
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(data_item)
                    except:
                        pass
    
    def _fetch_real_time_price(self, symbol, validate=True):
        """
        Fetch real-time price from multiple sources with fallbacks.
        
        Args:
            symbol: The symbol to fetch price for (e.g., 'BTC/USD')
            validate: Whether to validate the price across multiple sources
            
        Returns:
            Tuple of (price, source, confidence)
        """
        providers = []
        
        # Choose providers based on available API keys
        if self.crypto_compare_api_key:
            # If we have a CryptoCompare API key, use that as primary
            providers = self.provider_configs['real_time']
        else:
            # Otherwise, reorder to use free APIs first
            free_providers = [p for p in self.provider_configs['real_time'] 
                             if p[0] in ('CoinGecko', 'Binance', 'Simulation')]
            auth_providers = [p for p in self.provider_configs['real_time'] 
                             if p[0] not in ('CoinGecko', 'Binance', 'Simulation')]
            providers = free_providers + auth_providers
        
        # Try each provider in order
        price = None
        source = None
        all_prices = []
        
        for provider_name, provider_func in providers:
            try:
                # Add exponential backoff for retries on failure
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        if retry > 0:
                            backoff_time = 0.5 * (2 ** retry)
                            logger.debug(f"Retry {retry} for {provider_name}, backing off for {backoff_time}s")
                            time.sleep(backoff_time)
                            
                        price = provider_func(symbol)
                        if price is not None:
                            logger.debug(f"Got price from {provider_name}: ${price:.2f}")
                            self.api_stats[provider_name]['success'] += 1
                            self.api_stats[provider_name]['last_success'] = datetime.now()
                            source = provider_name
                            all_prices.append((price, source))
                            
                            # If not validating, return the first good price
                            if not validate:
                                return price, source, 95
                            
                            # If validating but this is the last provider, use it anyway
                            if provider_name == 'Simulation':
                                break
                                
                            # Otherwise continue to collect prices from other sources
                            break
                    except Exception as e:
                        if retry == max_retries - 1:
                            logger.warning(f"Error from {provider_name} after {max_retries} retries: {e}")
                            self.api_stats[provider_name]['failure'] += 1
            except Exception as e:
                logger.warning(f"Error from {provider_name}: {e}")
                self.api_stats[provider_name]['failure'] += 1
        
        # If we're validating and have multiple prices, compute a confidence score
        if validate and len(all_prices) > 1:
            return self._validate_prices(symbol, all_prices)
        elif all_prices:
            # Return the first price we found with 95% confidence
            return all_prices[0][0], all_prices[0][1], 95
        else:
            # If all sources failed, use the last known good price with a penalty to confidence
            if symbol in self.price_cache:
                logger.warning(f"All sources failed for {symbol}, using cached price")
                return self.price_cache[symbol], "Cache", 60
            
            # If no cached price, use fallback price
            if symbol in self.fallback_prices:
                logger.warning(f"No cached price for {symbol}, using fallback")
                # Add a small random variation to the fallback price
                variation = random.uniform(-0.005, 0.005)  # ±0.5%
                fallback_price = self.fallback_prices[symbol] * (1 + variation)
                return fallback_price, "Fallback", 40
            
            # If all else fails
            logger.error(f"Could not get price for {symbol} from any source")
            return None, None, 0
    
    def _validate_prices(self, symbol, prices):
        """
        Validate prices across multiple sources and compute a confidence score.
        
        Args:
            symbol: The symbol being validated
            prices: List of (price, source) tuples
            
        Returns:
            Tuple of (validated_price, source, confidence)
        """
        if not prices:
            return None, None, 0
        
        # If only one price, return it with medium confidence
        if len(prices) == 1:
            return prices[0][0], prices[0][1], 85
        
        # Extract just the prices and calculate stats
        price_values = [p[0] for p in prices]
        mean_price = np.mean(price_values)
        median_price = np.median(price_values)
        std_dev = np.std(price_values)
        
        # Calculate the coefficient of variation (CV) as a measure of agreement
        cv = std_dev / mean_price if mean_price > 0 else 1.0
        
        # Calculate z-scores to identify outliers
        z_scores = [(price - mean_price) / std_dev if std_dev > 0 else 0 for price in price_values]
        
        # Filter out outliers (z-score > 2)
        valid_prices = [(price, source) for (price, source), z in zip(prices, z_scores) 
                        if abs(z) <= 2.0]
        
        # If we have valid prices after filtering outliers, use those
        if valid_prices:
            filtered_price_values = [p[0] for p in valid_prices]
            filtered_mean = np.mean(filtered_price_values)
            
            # Use median unless it was an outlier
            if abs((median_price - mean_price) / std_dev) <= 2.0 if std_dev > 0 else True:
                final_price = median_price
            else:
                final_price = filtered_mean
            
            # Calculate confidence score based on agreement between sources
            if cv < 0.005:  # Very tight agreement (0.5%)
                confidence = 99
            elif cv < 0.01:  # Good agreement (1%)
                confidence = 95
            elif cv < 0.02:  # Acceptable agreement (2%)
                confidence = 90
            elif cv < 0.05:  # Some disagreement (5%)
                confidence = 80
            else:  # Significant disagreement
                confidence = 70
            
            # Include all sources that were used in the validation
            source_list = [source for _, source in valid_prices]
            source_str = source_list[0] if len(source_list) == 1 else f"{source_list[0]}+{len(source_list)-1}"
            
            logger.debug(f"Validated price for {symbol}: ${final_price:.2f} (confidence: {confidence}%)")
            self._log_validation_results(symbol, prices, valid_prices, final_price, confidence, cv)
            
            return final_price, source_str, confidence
        
        # If we filtered out all prices as outliers, use the median with lower confidence
        logger.warning(f"All prices for {symbol} were outliers, using median with low confidence")
        return median_price, prices[0][1], 65
    
    def _log_validation_results(self, symbol, all_prices, valid_prices, final_price, confidence, cv):
        """Log detailed validation results for debugging."""
        if logger.isEnabledFor(logging.DEBUG):
            price_str = ", ".join([f"{source}: ${price:.2f}" for price, source in all_prices])
            valid_str = ", ".join([f"{source}: ${price:.2f}" for price, source in valid_prices])
            
            logger.debug(f"Price validation for {symbol}:")
            logger.debug(f"  All prices: {price_str}")
            logger.debug(f"  Valid prices: {valid_str}")
            logger.debug(f"  Final price: ${final_price:.2f}")
            logger.debug(f"  Confidence: {confidence}%")
            logger.debug(f"  Coefficient of variation: {cv:.6f}")
    
    def _fetch_from_crypto_compare(self, symbol):
        """Fetch price from CryptoCompare."""
        return self.crypto_compare_client.get_price(symbol)
    
    def _fetch_from_coin_gecko(self, symbol):
        """Fetch price from CoinGecko."""
        return self.coin_gecko_client.get_price(symbol)
    
    def _fetch_from_binance(self, symbol):
        """Fetch price from Binance."""
        return self.binance_client.get_price(symbol)
    
    def _simulate_price(self, symbol):
        """
        Generate a simulated price for testing when APIs are unavailable.
        This uses the last known good price with random variations.
        """
        if symbol in self.fallback_prices:
            base_price = self.fallback_prices[symbol]
            # Volatility-based random walk around the base price
            volatility = self.asset_volatilities.get(symbol, 0.05)  # Default 5% daily volatility
            daily_factor = volatility / np.sqrt(252 * 24)  # Convert to hourly
            variation = random.normalvariate(0, daily_factor)
            simulated_price = base_price * (1 + variation)
            
            logger.debug(f"Generated simulated price for {symbol}: ${simulated_price:.2f}")
            return simulated_price
        return None
    
    def _fetch_historical_from_crypto_compare(self, symbol, days=30, interval='daily'):
        """Fetch historical data from CryptoCompare."""
        try:
            return self.crypto_compare_client.get_historical_data(symbol, days, interval)
        except Exception as e:
            logger.warning(f"Error fetching historical data from CryptoCompare: {e}")
            return None
    
    def _fetch_historical_from_coin_gecko(self, symbol, days=30, interval='daily'):
        """Fetch historical data from CoinGecko."""
        try:
            return self.coin_gecko_client.get_historical_data(symbol, days, interval)
        except Exception as e:
            logger.warning(f"Error fetching historical data from CoinGecko: {e}")
            return None
    
    def _simulate_historical_data(self, symbol, days=30, interval='daily'):
        """Generate simulated historical data using geometric Brownian motion."""
        if symbol not in self.fallback_prices:
            logger.warning(f"No fallback price available for {symbol}, cannot simulate historical data")
            return None
        
        # Current price as the end price
        end_price = self.fallback_prices[symbol]
        # Volatility for this asset
        volatility = self.asset_volatilities.get(symbol, 0.05)  # Default 5% daily volatility
        
        # Generate dates
        end_date = datetime.now()
        if interval == 'daily':
            dates = [end_date - timedelta(days=i) for i in range(days)]
            dates.reverse()  # So they're in ascending order
        elif interval == 'hourly':
            dates = [end_date - timedelta(hours=i) for i in range(days * 24)]
            dates.reverse()
        else:
            logger.warning(f"Unsupported interval {interval}, using daily")
            dates = [end_date - timedelta(days=i) for i in range(days)]
            dates.reverse()
        
        # Generate prices using GBM
        dt = 1.0 / 252  # Daily time step (252 trading days per year)
        if interval == 'hourly':
            dt = dt / 24  # Hourly time step
        
        annual_drift = 0.05  # 5% annual drift for crypto
        price_path = [end_price]
        
        # Generate the path backward from the current price
        for i in range(len(dates) - 1):
            drift = annual_drift * dt
            diffusion = volatility * np.sqrt(dt) * np.random.normal()
            prev_price = price_path[-1] / (1 + drift + diffusion)
            price_path.append(prev_price)
        
        price_path.reverse()  # To match dates order
        
        # Create a pandas DataFrame
        df = pd.DataFrame({
            'date': dates,
            'price': price_path,
            'volume': [random.uniform(1000, 10000) for _ in range(len(dates))]
        })
        
        logger.info(f"Generated simulated historical data for {symbol} ({interval}, {days} periods)")
        return df
    
    def get_current_price(self, symbol):
        """
        Get the current price for a symbol.
        
        Args:
            symbol: The symbol to get price for (e.g., 'BTC/USD')
            
        Returns:
            Current price or None if unavailable
        """
        # Check if we need to refresh the cache
        if (symbol in self.cache_timestamps and 
            time.time() - self.cache_timestamps.get(symbol, 0) < self.cache_timeout and
            symbol in self.price_cache):
            logger.debug(f"Returning cached price for {symbol}: ${self.price_cache[symbol]:.2f}")
            return self.price_cache[symbol]
        
        # Need to fetch a fresh price
        price, source, confidence = self._fetch_real_time_price(symbol)
        
        if price is not None:
            # Update cache
            self.price_cache[symbol] = price
            self.cache_timestamps[symbol] = time.time()
            self.data_source[symbol] = f"{source} ({confidence}% confidence)"
            logger.debug(f"Fetched new price for {symbol}: ${price:.2f} from {source}")
            return price
        
        # If we couldn't get a price, return the cached price if available
        if symbol in self.price_cache:
            logger.warning(f"Could not fetch fresh price for {symbol}, using cached value")
            return self.price_cache[symbol]
        
        # Last resort: use fallback price
        if symbol in self.fallback_prices:
            logger.warning(f"No cached price for {symbol}, using fallback")
            return self.fallback_prices[symbol]
        
        logger.error(f"Could not get price for {symbol}")
        return None
    
    def get_historical_data(self, symbol, days=30, interval='daily'):
        """
        Get historical data for a symbol.
        
        Args:
            symbol: The symbol to get data for (e.g., 'BTC/USD')
            days: Number of days of historical data
            interval: Data interval ('daily', 'hourly', etc.)
            
        Returns:
            Pandas DataFrame with historical data or None if unavailable
        """
        cache_key = f"{symbol}_{days}_{interval}"
        
        # Check if we have this data cached
        if cache_key in self.historical_data:
            logger.debug(f"Returning cached historical data for {cache_key}")
            return self.historical_data[cache_key]
        
        # Try each provider in order
        for provider_name, provider_func in self.provider_configs['historical']:
            try:
                logger.debug(f"Fetching historical data for {symbol} from {provider_name}")
                df = provider_func(symbol, days, interval)
                if df is not None and not df.empty:
                    logger.info(f"Got historical data for {symbol} from {provider_name}")
                    # Cache the data
                    self.historical_data[cache_key] = df
                    self.api_stats[provider_name]['success'] += 1
                    self.api_stats[provider_name]['last_success'] = datetime.now()
                    return df
            except Exception as e:
                logger.warning(f"Error getting historical data from {provider_name}: {e}")
                self.api_stats[provider_name]['failure'] += 1
        
        logger.error(f"Could not get historical data for {symbol}")
        return None
    
    def get_provider_stats(self):
        """Get statistics on API provider usage and success rates."""
        stats = {}
        
        for provider, data in self.api_stats.items():
            if data['success'] + data['failure'] > 0:
                success_rate = (data['success'] / (data['success'] + data['failure'])) * 100
            else:
                success_rate = 0
                
            stats[provider] = {
                'success': data['success'],
                'failure': data['failure'],
                'success_rate': success_rate,
                'last_success': data['last_success']
            }
            
        return stats
