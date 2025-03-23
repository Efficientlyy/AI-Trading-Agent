"""Unit tests for the Cache class.

This module contains tests for the caching system used for performance optimization.
"""

import pytest
import time
from unittest.mock import patch

from src.common.caching import Cache


class TestCache:
    """Tests for the Cache class."""
    
    def test_cache_set_get(self):
        """Test setting and getting values from the cache."""
        # Create cache with default TTL
        cache = Cache()
        
        # Set a value
        cache.set("test_key", "test_value")
        
        # Get the value
        result = cache.get("test_key")
        
        # Verify result
        assert result == "test_value"
    
    def test_cache_expiration(self):
        """Test that cached values expire after TTL."""
        # Create cache with short TTL for testing
        cache = Cache(ttl=0.1)
        
        # Set a value
        cache.set("test_key", "test_value")
        
        # Get the value immediately (should be present)
        result1 = cache.get("test_key")
        assert result1 == "test_value"
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Get the value after expiration (should be None)
        result2 = cache.get("test_key")
        assert result2 is None
    
    def test_cache_invalidate(self):
        """Test invalidating a cache entry."""
        # Create cache
        cache = Cache()
        
        # Set multiple values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Invalidate one key
        cache.invalidate("key1")
        
        # Verify key1 is gone but key2 remains
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
    
    def test_cache_clear(self):
        """Test clearing all cache entries."""
        # Create cache
        cache = Cache()
        
        # Set multiple values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Clear the cache
        cache.clear()
        
        # Verify all keys are gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache.data) == 0
    
    def test_cache_nonexistent_key(self):
        """Test getting a nonexistent key."""
        # Create cache
        cache = Cache()
        
        # Get a key that doesn't exist
        result = cache.get("nonexistent")
        
        # Verify result is None
        assert result is None
    
    def test_cache_with_complex_values(self):
        """Test caching complex values like dictionaries and lists."""
        # Create cache
        cache = Cache()
        
        # Set complex values
        dict_value = {"name": "test", "value": 123}
        list_value = [1, 2, 3, 4, 5]
        
        cache.set("dict_key", dict_value)
        cache.set("list_key", list_value)
        
        # Get the values
        result_dict = cache.get("dict_key")
        result_list = cache.get("list_key")
        
        # Verify results
        assert result_dict == dict_value
        assert result_list == list_value
        
        # Verify they're the same objects (not copies)
        assert result_dict is dict_value
        assert result_list is list_value
    
    def test_cache_stats(self):
        """Test getting cache statistics."""
        # Create cache
        cache = Cache(ttl=300)
        
        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Get stats
        stats = cache.get_stats()
        
        # Verify stats
        assert stats["size"] == 2
        assert set(stats["keys"]) == {"key1", "key2"}
        assert stats["ttl"] == 300
    
    def test_cache_update_existing(self):
        """Test updating an existing cache entry."""
        # Create cache
        cache = Cache()
        
        # Set initial value
        cache.set("test_key", "initial_value")
        
        # Update with new value
        cache.set("test_key", "updated_value")
        
        # Get the value
        result = cache.get("test_key")
        
        # Verify result is the updated value
        assert result == "updated_value"
