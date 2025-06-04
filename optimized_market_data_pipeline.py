#!/usr/bin/env python
"""
Optimized Market Data Pipeline for MEXC API Rate Limit Compliance

This module optimizes the market data pipeline to comply with MEXC API rate limits
and focuses exclusively on BTCUSDC as requested. It implements request spacing,
exponential backoff, and detailed logging for API interactions.
"""

import os
import sys
import json
import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from multi_asset_data_service import MultiAssetDataService
from symbol_standardization import SymbolStandardizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("market_data_pipeline.log")
    ]
)

logger = logging.getLogger("market_data_pipeline")

class OptimizedMarketDataPipeline:
    """Optimized market data pipeline with MEXC rate limit compliance"""
    
    # MEXC API Rate Limits (as of March 25, 2025)
    SPOT_ORDER_RATE_LIMIT = 5  # orders per second
    BATCH_ORDER_RATE_LIMIT = 2  # batch requests per second
    MARKET_DATA_RATE_LIMIT = 10  # estimated safe limit for market data requests per second
    
    # Request spacing in seconds
    REQUEST_SPACING = 0.2  # 200ms between requests (5 per second max)
    
    def __init__(self, cache_dir=None):
        """Initialize optimized market data pipeline
        
        Args:
            cache_dir: Directory for caching data (default: "./cache")
        """
        # Initialize symbol standardizer
        self.standardizer = SymbolStandardizer()
        
        # Focus exclusively on BTCUSDC as requested
        self.symbol = "BTC/USDC"  # Internal format
        self.api_symbol = self.standardizer.for_api(self.symbol)  # API format
        self.mexc_symbol = self.standardizer.for_mexc(self.symbol)  # MEXC format
        
        # Initialize timeframes
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Initialize data service with single symbol
        self.data_service = MultiAssetDataService(supported_assets=[self.symbol])
        
        # Set up caching
        self.cache_dir = cache_dir or "./cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize data stores with TTL cache
        self.market_data_cache = {}
        self.cache_ttl = {}
        self.default_ttl = 60  # seconds
        
        # Initialize status tracking
        self.last_update = {}
        self.error_counts = {}
        self.max_errors = 3
        self.status = "initialized"
        
        # Rate limiting
        self.last_request_time = 0
        self.request_queue = queue.Queue()
        self.request_thread = threading.Thread(target=self._process_request_queue, daemon=True)
        self.request_thread.start()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Optimized market data pipeline initialized for {self.symbol} (API: {self.api_symbol}, MEXC: {self.mexc_symbol})")
    
    def _process_request_queue(self):
        """Process the request queue with rate limiting"""
        while True:
            try:
                # Get request from queue
                request_func, args, kwargs, result_queue = self.request_queue.get()
                
                # Calculate time since last request
                time_since_last = time.time() - self.last_request_time
                
                # If we need to wait to comply with rate limits, do so
                if time_since_last < self.REQUEST_SPACING:
                    sleep_time = self.REQUEST_SPACING - time_since_last
                    logger.debug(f"Rate limiting: Waiting {sleep_time:.3f}s before next request")
                    time.sleep(sleep_time)
                
                # Execute request
                try:
                    self.last_request_time = time.time()
                    result = request_func(*args, **kwargs)
                    result_queue.put((True, result))
                except Exception as e:
                    logger.error(f"Error in queued request: {str(e)}")
                    result_queue.put((False, str(e)))
                
                # Mark task as done
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in request queue processor: {str(e)}")
                time.sleep(1)  # Prevent tight loop in case of persistent errors
    
    def _queue_request(self, request_func, *args, **kwargs):
        """Queue a request with rate limiting
        
        Args:
            request_func: Function to call
            *args: Arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Any: Result of function call
        """
        # Create result queue
        result_queue = queue.Queue()
        
        # Add request to queue
        self.request_queue.put((request_func, args, kwargs, result_queue))
        
        # Wait for result
        success, result = result_queue.get()
        
        if success:
            return result
        else:
            raise Exception(f"Request failed: {result}")
    
    def get_market_data(self, timeframe="5m", limit=100, use_cache=True):
        """Get market data with rate limit compliance and enhanced error handling
        
        Args:
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to return
            use_cache: Whether to use cached data if available
            
        Returns:
            list: Market data as list of candles or empty list if data unavailable
        """
        # Generate cache key
        cache_key = f"{self.symbol}_{timeframe}_{limit}"
        
        # Check cache if enabled
        if use_cache and cache_key in self.market_data_cache:
            # Check if cache is still valid
            if time.time() < self.cache_ttl.get(cache_key, 0):
                logger.debug(f"Using cached data for {self.symbol} {timeframe}")
                return self.market_data_cache[cache_key]
        
        # Try to get live data with exponential backoff
        max_retries = 3
        base_delay = 1  # second
        
        for retry in range(max_retries + 1):
            try:
                # Get data from service with rate limiting
                with self.lock:
                    self.status = "fetching"
                    
                    # Use the queue system to enforce rate limits
                    data = self._queue_request(
                        self.data_service.get_klines,
                        self.api_symbol, timeframe, limit
                    )
                
                # Validate data
                if data and len(data) > 0:
                    # Update cache
                    self.market_data_cache[cache_key] = data
                    self.cache_ttl[cache_key] = time.time() + self.default_ttl
                    self.last_update[cache_key] = time.time()
                    self.error_counts[cache_key] = 0
                    self.status = "success"
                    
                    logger.info(f"Successfully fetched {len(data)} candles for {self.symbol} {timeframe}")
                    return data
                else:
                    logger.warning(f"Empty data returned for {self.symbol} {timeframe}")
                    
                    # If this is not the last retry, wait and try again
                    if retry < max_retries:
                        delay = base_delay * (2 ** retry)
                        logger.info(f"Retrying in {delay} seconds (attempt {retry + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        # Increment error count on final failure
                        self.error_counts[cache_key] = self.error_counts.get(cache_key, 0) + 1
            except Exception as e:
                logger.error(f"Error fetching data for {self.symbol} {timeframe}: {str(e)}")
                
                # If this is not the last retry, wait and try again
                if retry < max_retries:
                    delay = base_delay * (2 ** retry)
                    logger.info(f"Retrying in {delay} seconds (attempt {retry + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # Increment error count on final failure
                    self.error_counts[cache_key] = self.error_counts.get(cache_key, 0) + 1
        
        # If we reach here, live data failed after all retries
        self.status = "error"
        
        # If we still have cached data, return it even if expired, but with warning
        if cache_key in self.market_data_cache:
            logger.warning(f"WARNING: Using expired cached data for {self.symbol} {timeframe}")
            return self.market_data_cache[cache_key]
        
        # If all else fails, return empty list with clear error message
        logger.error(f"ERROR: No market data available for {self.symbol} {timeframe}. Please try again later or contact support.")
        return []
    
    def get_status(self):
        """Get pipeline status
        
        Returns:
            dict: Pipeline status
        """
        return {
            "status": self.status,
            "symbol": self.symbol,
            "api_symbol": self.api_symbol,
            "mexc_symbol": self.mexc_symbol,
            "timeframes": self.timeframes,
            "last_update": self.last_update,
            "error_counts": self.error_counts,
            "queue_size": self.request_queue.qsize()
        }
    
    def clear_cache(self):
        """Clear data cache"""
        with self.lock:
            self.market_data_cache = {}
            self.cache_ttl = {}
        
        logger.info("Cache cleared")
    
    def wait_for_queue_empty(self, timeout=None):
        """Wait for request queue to be empty
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if queue is empty, False if timeout occurred
        """
        try:
            self.request_queue.join(timeout=timeout)
            return True
        except queue.Empty:
            return False

# Example usage
if __name__ == "__main__":
    pipeline = OptimizedMarketDataPipeline()
    
    # Test with different timeframes
    timeframes = ["1m", "5m", "15m", "1h"]
    
    for timeframe in timeframes:
        print(f"Getting data for BTCUSDC {timeframe}...")
        data = pipeline.get_market_data(timeframe)
        print(f"Got {len(data)} candles")
        
        # Print first and last candle
        if data:
            print(f"First candle: {data[0]}")
            print(f"Last candle: {data[-1]}")
        
        print("---")
    
    # Wait for any pending requests to complete
    pipeline.wait_for_queue_empty()
