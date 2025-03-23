"""Caching module for sentiment analysis system.

This module provides a simple time-based caching system to improve performance
by reducing redundant computations and API calls.
"""

import time
from typing import Any, Dict, Optional, TypeVar, Generic

T = TypeVar('T')

class Cache(Generic[T]):
    """Simple time-based cache implementation."""
    
    def __init__(self, ttl: int = 300):
        """Initialize the cache.
        
        Args:
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        self.ttl = ttl
        self.data: Dict[str, Dict[str, Any]] = {}
        
    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        if key not in self.data:
            return None
            
        entry = self.data[key]
        if time.time() - entry["timestamp"] > self.ttl:
            # Expired
            del self.data[key]
            return None
            
        return entry["value"]
        
    def set(self, key: str, value: T) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        self.data[key] = {
            "value": value,
            "timestamp": time.time()
        }
        
    def invalidate(self, key: str) -> None:
        """Invalidate a cache entry.
        
        Args:
            key: The cache key
        """
        if key in self.data:
            del self.data[key]
            
    def clear(self) -> None:
        """Clear all cache entries."""
        self.data.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.data),
            "keys": list(self.data.keys()),
            "ttl": self.ttl
        }
