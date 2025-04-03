"""
Data Cache

This module provides a cache for API data.
"""

import os
import time
import json
import logging
import hashlib
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCache:
    """
    Cache for API data with disk persistence.
    
    This class provides a two-level caching system:
    1. In-memory cache for fast access
    2. Disk cache for persistence
    
    Features:
    - Configurable TTL (time-to-live) for cache entries
    - Automatic pruning of expired entries
    - Thread-safe operations
    - Statistics tracking
    """
    
    def __init__(self, cache_dir: str = None, default_ttl: int = 3600, max_memory_entries: int = 1000):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory for cache files
            default_ttl: Default time-to-live in seconds
            max_memory_entries: Maximum number of entries to keep in memory
        """
        self.cache_dir = cache_dir or os.path.join('data', 'cache')
        self.default_ttl = default_ttl
        self.max_memory_entries = max_memory_entries
        self.memory_cache = {}
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0,
            'prunes': 0
        }
        self.lock = threading.RLock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized data cache in {self.cache_dir}")
        
    def get(self, key: str, ttl: int = None) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds (default: use default_ttl)
            
        Returns:
            Cached data or None if not found or expired
        """
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_valid(entry, ttl):
                    self.stats['memory_hits'] += 1
                    logger.debug(f"Cache hit (memory): {key}")
                    return entry['data']
                    
            # Check file cache
            cache_file = self._get_cache_file(key)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        entry = json.load(f)
                        
                    if self._is_valid(entry, ttl):
                        # Update memory cache
                        self.memory_cache[key] = entry
                        self._prune_memory_cache()
                        
                        self.stats['disk_hits'] += 1
                        logger.debug(f"Cache hit (disk): {key}")
                        return entry['data']
                except Exception as e:
                    logger.error(f"Error reading cache file: {e}")
                    
            self.stats['misses'] += 1
            logger.debug(f"Cache miss: {key}")
            return None
        
    def set(self, key: str, data: Any, ttl: int = None) -> bool:
        """
        Set data in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (default: use default_ttl)
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            ttl = ttl or self.default_ttl
            
            entry = {
                'data': data,
                'timestamp': time.time(),
                'ttl': ttl
            }
            
            # Update memory cache
            self.memory_cache[key] = entry
            self._prune_memory_cache()
            
            # Update file cache
            cache_file = self._get_cache_file(key)
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                
                with open(cache_file, 'w') as f:
                    json.dump(entry, f)
                    
                self.stats['sets'] += 1
                logger.debug(f"Cache set: {key}")
                return True
            except Exception as e:
                logger.error(f"Error writing cache file: {e}")
                return False
            
    def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                
            # Remove from file cache
            cache_file = self._get_cache_file(key)
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    self.stats['invalidations'] += 1
                    logger.debug(f"Cache invalidated: {key}")
                    return True
                except Exception as e:
                    logger.error(f"Error removing cache file: {e}")
                    return False
            
            return True
        
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            # Clear memory cache
            self.memory_cache = {}
            
            # Clear file cache
            try:
                for root, dirs, files in os.walk(self.cache_dir):
                    for file in files:
                        if file.endswith('.cache'):
                            os.remove(os.path.join(root, file))
                            
                self.stats['invalidations'] += 1
                logger.info("Cache cleared")
                return True
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return False
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        with self.lock:
            total_hits = self.stats['memory_hits'] + self.stats['disk_hits']
            total_requests = total_hits + self.stats['misses']
            
            stats = {
                'memory_entries': len(self.memory_cache),
                'memory_hits': self.stats['memory_hits'],
                'disk_hits': self.stats['disk_hits'],
                'total_hits': total_hits,
                'misses': self.stats['misses'],
                'hit_ratio': total_hits / total_requests if total_requests > 0 else 0,
                'sets': self.stats['sets'],
                'invalidations': self.stats['invalidations'],
                'prunes': self.stats['prunes']
            }
            
            return stats
            
    def _get_cache_file(self, key: str) -> str:
        """
        Get cache file path for key.
        
        Args:
            key: Cache key
            
        Returns:
            str: Cache file path
        """
        # Create hash of key
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        # Use first 2 characters as directory
        dir_name = key_hash[:2]
        
        # Create path
        return os.path.join(self.cache_dir, dir_name, f"{key_hash}.cache")
        
    def _is_valid(self, entry: Dict[str, Any], ttl: int = None) -> bool:
        """
        Check if cache entry is valid.
        
        Args:
            entry: Cache entry
            ttl: Time-to-live in seconds (default: use entry's ttl)
            
        Returns:
            bool: True if valid, False if expired
        """
        ttl = ttl or entry.get('ttl', self.default_ttl)
        timestamp = entry.get('timestamp', 0)
        
        # Check if expired
        return (time.time() - timestamp) < ttl
        
    def _prune_memory_cache(self):
        """Prune memory cache if it exceeds max size."""
        if len(self.memory_cache) <= self.max_memory_entries:
            return
            
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Remove oldest entries
        entries_to_remove = len(self.memory_cache) - self.max_memory_entries
        for i in range(entries_to_remove):
            key, _ = sorted_entries[i]
            del self.memory_cache[key]
            
        self.stats['prunes'] += 1
        logger.debug(f"Pruned {entries_to_remove} entries from memory cache")
        
    def get_keys(self, pattern: str = None) -> List[str]:
        """
        Get all cache keys.
        
        Args:
            pattern: Pattern to match keys (not implemented yet)
            
        Returns:
            List of cache keys
        """
        keys = set()
        
        # Add keys from memory cache
        keys.update(self.memory_cache.keys())
        
        # Add keys from file cache
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                if file.endswith('.cache'):
                    # Extract key from filename
                    key_hash = file[:-6]  # Remove .cache
                    
                    # TODO: Implement reverse lookup for actual keys
                    # For now, just add the hash
                    keys.add(key_hash)
                    
        return list(keys)
        
    def get_many(self, keys: List[str], ttl: int = None) -> Dict[str, Any]:
        """
        Get multiple cache entries.
        
        Args:
            keys: List of cache keys
            ttl: Time-to-live in seconds (default: use default_ttl)
            
        Returns:
            Dict with cache entries
        """
        result = {}
        
        for key in keys:
            value = self.get(key, ttl)
            if value is not None:
                result[key] = value
                
        return result
        
    def set_many(self, entries: Dict[str, Any], ttl: int = None) -> bool:
        """
        Set multiple cache entries.
        
        Args:
            entries: Dict with cache entries
            ttl: Time-to-live in seconds (default: use default_ttl)
            
        Returns:
            bool: True if all successful, False otherwise
        """
        success = True
        
        for key, value in entries.items():
            if not self.set(key, value, ttl):
                success = False
                
        return success
        
    def invalidate_many(self, keys: List[str]) -> bool:
        """
        Invalidate multiple cache entries.
        
        Args:
            keys: List of cache keys
            
        Returns:
            bool: True if all successful, False otherwise
        """
        success = True
        
        for key in keys:
            if not self.invalidate(key):
                success = False
                
        return success
        
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match keys
            
        Returns:
            int: Number of invalidated entries
        """
        # TODO: Implement pattern matching
        logger.warning("Pattern invalidation not implemented yet")
        return 0