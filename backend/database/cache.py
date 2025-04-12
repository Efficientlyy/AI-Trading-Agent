"""
Caching module for database operations.
"""

import time
import logging
import functools
from typing import Any, Dict, Optional, Tuple, Callable, TypeVar, cast
from threading import RLock

# Set up logger
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
CacheKey = Tuple[Any, ...]

class Cache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items in the cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[CacheKey, Tuple[Any, float]] = {}
        self._lock = RLock()
        logger.debug(f"Initialized cache with max_size={max_size}, default_ttl={default_ttl}")
    
    def get(self, key: CacheKey) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            
            # Check if expired
            if expiry < time.time():
                logger.debug(f"Cache miss (expired): {key}")
                del self._cache[key]
                return None
            
            logger.debug(f"Cache hit: {key}")
            return value
    
    def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds, uses default if None
        """
        with self._lock:
            # Evict items if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()
            
            # Calculate expiry time
            ttl = ttl if ttl is not None else self.default_ttl
            expiry = time.time() + ttl
            
            # Store in cache
            self._cache[key] = (value, expiry)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    def delete(self, key: CacheKey) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache delete: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            logger.debug("Cache cleared")
    
    def _evict(self) -> None:
        """Evict the oldest or expired items from the cache."""
        # First try to remove expired items
        now = time.time()
        expired_keys = [k for k, (_, exp) in self._cache.items() if exp < now]
        
        for key in expired_keys:
            del self._cache[key]
            logger.debug(f"Cache eviction (expired): {key}")
        
        # If we still need to evict, remove the oldest item
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
            logger.debug(f"Cache eviction (oldest): {oldest_key}")


# Global cache instance
_global_cache = Cache()

def cached(ttl: Optional[int] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds, uses default if None
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from function name, args, and kwargs
            key = (func.__module__, func.__name__, args, frozenset(kwargs.items()))
            
            # Try to get from cache
            cached_value = _global_cache.get(key)
            if cached_value is not None:
                return cast(T, cached_value)
            
            # Call function and cache result
            result = func(*args, **kwargs)
            _global_cache.set(key, result, ttl)
            return result
        
        return wrapper
    
    return decorator


def invalidate_cache(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """
    Invalidate cache for a specific function call.
    
    Args:
        func: Function whose cache to invalidate
        *args: Function arguments
        **kwargs: Function keyword arguments
    """
    key = (func.__module__, func.__name__, args, frozenset(kwargs.items()))
    _global_cache.delete(key)
    logger.debug(f"Cache invalidated for {func.__module__}.{func.__name__}")


def clear_cache() -> None:
    """Clear the entire cache."""
    _global_cache.clear()
