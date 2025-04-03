"""
Fallback Manager

This module provides a fallback strategy manager for handling API outages.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FallbackStrategy(Enum):
    """Fallback strategies."""
    RETRY = 'retry'  # Retry the request
    CACHE = 'cache'  # Use cached data
    MOCK = 'mock'    # Use mock data
    FAIL = 'fail'    # Fail with error
    
class FallbackTrigger(Enum):
    """Fallback triggers."""
    TIMEOUT = 'timeout'        # Request timeout
    CONNECTION_ERROR = 'connection_error'  # Connection error
    RATE_LIMIT = 'rate_limit'  # Rate limit exceeded
    API_ERROR = 'api_error'    # API error
    CIRCUIT_OPEN = 'circuit_open'  # Circuit breaker open
    
class FallbackManager:
    """
    Fallback strategy manager for handling API outages.
    
    This class provides a configurable fallback strategy manager
    for handling API outages and other error conditions.
    """
    
    def __init__(self):
        """Initialize fallback manager."""
        self.strategies = {}
        self.default_strategy = {
            FallbackTrigger.TIMEOUT: FallbackStrategy.RETRY,
            FallbackTrigger.CONNECTION_ERROR: FallbackStrategy.RETRY,
            FallbackTrigger.RATE_LIMIT: FallbackStrategy.CACHE,
            FallbackTrigger.API_ERROR: FallbackStrategy.CACHE,
            FallbackTrigger.CIRCUIT_OPEN: FallbackStrategy.CACHE
        }
        self.handlers = {
            FallbackStrategy.RETRY: self._handle_retry,
            FallbackStrategy.CACHE: self._handle_cache,
            FallbackStrategy.MOCK: self._handle_mock,
            FallbackStrategy.FAIL: self._handle_fail
        }
        self.cache_handler = None
        self.mock_handler = None
        self.stats = {
            'retries': 0,
            'cache_hits': 0,
            'mock_hits': 0,
            'failures': 0
        }
        self.lock = threading.RLock()
        
        logger.info("Initialized fallback manager")
        
    def set_strategy(self, source: str, trigger: FallbackTrigger, strategy: FallbackStrategy):
        """
        Set fallback strategy for a source and trigger.
        
        Args:
            source: Source name
            trigger: Fallback trigger
            strategy: Fallback strategy
        """
        with self.lock:
            # Initialize source if needed
            if source not in self.strategies:
                self.strategies[source] = {}
                
            # Set strategy
            self.strategies[source][trigger] = strategy
            
            logger.info(f"Set fallback strategy for {source}/{trigger.value}: {strategy.value}")
            
    def set_default_strategy(self, trigger: FallbackTrigger, strategy: FallbackStrategy):
        """
        Set default fallback strategy for a trigger.
        
        Args:
            trigger: Fallback trigger
            strategy: Fallback strategy
        """
        with self.lock:
            self.default_strategy[trigger] = strategy
            
            logger.info(f"Set default fallback strategy for {trigger.value}: {strategy.value}")
            
    def set_cache_handler(self, handler: Callable[[str, Dict[str, Any]], Any]):
        """
        Set cache handler.
        
        Args:
            handler: Cache handler function
        """
        self.cache_handler = handler
        logger.info("Set cache handler")
        
    def set_mock_handler(self, handler: Callable[[str, Dict[str, Any]], Any]):
        """
        Set mock handler.
        
        Args:
            handler: Mock handler function
        """
        self.mock_handler = handler
        logger.info("Set mock handler")
        
    def handle_fallback(self, source: str, trigger: FallbackTrigger, 
                        request_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Handle fallback for a request.
        
        Args:
            source: Source name
            trigger: Fallback trigger
            request_data: Request data
            
        Returns:
            Tuple[bool, Any]: (success, result)
        """
        with self.lock:
            # Get strategy
            strategy = self._get_strategy(source, trigger)
            
            # Get handler
            handler = self.handlers.get(strategy)
            
            if not handler:
                logger.error(f"No handler for strategy {strategy.value}")
                return False, None
                
        # Handle fallback (outside lock)
        try:
            return handler(source, request_data)
        except Exception as e:
            logger.error(f"Error handling fallback for {source}/{trigger.value}: {e}")
            return False, None
            
    def _get_strategy(self, source: str, trigger: FallbackTrigger) -> FallbackStrategy:
        """
        Get fallback strategy for a source and trigger.
        
        Args:
            source: Source name
            trigger: Fallback trigger
            
        Returns:
            FallbackStrategy: Fallback strategy
        """
        # Check source-specific strategy
        if source in self.strategies and trigger in self.strategies[source]:
            return self.strategies[source][trigger]
            
        # Use default strategy
        return self.default_strategy.get(trigger, FallbackStrategy.FAIL)
        
    def _handle_retry(self, source: str, request_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Handle retry strategy.
        
        Args:
            source: Source name
            request_data: Request data
            
        Returns:
            Tuple[bool, Any]: (success, result)
        """
        # Retry is handled by the caller
        self.stats['retries'] += 1
        logger.info(f"Retry strategy for {source}")
        return False, None
        
    def _handle_cache(self, source: str, request_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Handle cache strategy.
        
        Args:
            source: Source name
            request_data: Request data
            
        Returns:
            Tuple[bool, Any]: (success, result)
        """
        if not self.cache_handler:
            logger.warning(f"No cache handler for {source}")
            return False, None
            
        try:
            result = self.cache_handler(source, request_data)
            if result is not None:
                self.stats['cache_hits'] += 1
                logger.info(f"Cache hit for {source}")
                return True, result
            else:
                logger.info(f"Cache miss for {source}")
                return False, None
        except Exception as e:
            logger.error(f"Error in cache handler for {source}: {e}")
            return False, None
            
    def _handle_mock(self, source: str, request_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Handle mock strategy.
        
        Args:
            source: Source name
            request_data: Request data
            
        Returns:
            Tuple[bool, Any]: (success, result)
        """
        if not self.mock_handler:
            logger.warning(f"No mock handler for {source}")
            return False, None
            
        try:
            result = self.mock_handler(source, request_data)
            self.stats['mock_hits'] += 1
            logger.info(f"Mock data for {source}")
            return True, result
        except Exception as e:
            logger.error(f"Error in mock handler for {source}: {e}")
            return False, None
            
    def _handle_fail(self, source: str, request_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Handle fail strategy.
        
        Args:
            source: Source name
            request_data: Request data
            
        Returns:
            Tuple[bool, Any]: (success, result)
        """
        self.stats['failures'] += 1
        logger.info(f"Fail strategy for {source}")
        return False, None
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get fallback manager statistics.
        
        Returns:
            Dict with statistics
        """
        with self.lock:
            stats = {
                'retries': self.stats['retries'],
                'cache_hits': self.stats['cache_hits'],
                'mock_hits': self.stats['mock_hits'],
                'failures': self.stats['failures']
            }
            
            return stats


class BitvavoFallbackManager(FallbackManager):
    """
    Fallback manager for Bitvavo API.
    
    This class extends the base FallbackManager with Bitvavo-specific
    fallback strategies and handlers.
    """
    
    def __init__(self, cache_handler=None, mock_handler=None):
        """
        Initialize Bitvavo fallback manager.
        
        Args:
            cache_handler: Cache handler function
            mock_handler: Mock handler function
        """
        super().__init__()
        
        # Set handlers
        if cache_handler:
            self.set_cache_handler(cache_handler)
        if mock_handler:
            self.set_mock_handler(mock_handler)
            
        # Set Bitvavo-specific strategies
        self.set_strategy('bitvavo', FallbackTrigger.RATE_LIMIT, FallbackStrategy.CACHE)
        self.set_strategy('bitvavo', FallbackTrigger.API_ERROR, FallbackStrategy.RETRY)
        
        logger.info("Initialized Bitvavo fallback manager")
        
    def handle_bitvavo_error(self, error_code: int, request_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Handle Bitvavo API error.
        
        Args:
            error_code: Bitvavo error code
            request_data: Request data
            
        Returns:
            Tuple[bool, Any]: (success, result)
        """
        # Map error code to trigger
        if error_code == 429:
            trigger = FallbackTrigger.RATE_LIMIT
        elif error_code >= 500:
            trigger = FallbackTrigger.API_ERROR
        else:
            trigger = FallbackTrigger.API_ERROR
            
        # Handle fallback
        return self.handle_fallback('bitvavo', trigger, request_data)