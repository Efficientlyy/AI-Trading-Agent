"""
Performance Optimization Module for Technical Analysis

This module provides performance optimizations for the technical analysis components,
including caching, batching, and parallel processing capabilities.
"""

import functools
import time
import logging
from typing import Dict, Any, List, Callable, Optional, Tuple
import concurrent.futures
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger(__name__)

# Cache storage
_cache_store: Dict[str, Dict[str, Any]] = {
    "indicators": {},
    "patterns": {},
    "analysis": {}
}

# Cache configuration
_cache_config = {
    "indicators": {"ttl": 300},  # 5 minutes
    "patterns": {"ttl": 600},    # 10 minutes
    "analysis": {"ttl": 300}     # 5 minutes
}

def timed_lru_cache(
    maxsize: int = 128, 
    ttl_seconds: int = 300, 
    typed: bool = False, 
    include_args: Optional[List[str]] = None
):
    """
    Decorator that applies an LRU cache with time expiration to a function.
    
    Args:
        maxsize: Maximum size of the cache
        ttl_seconds: Time to live in seconds
        typed: Whether to cache different function argument types separately
        include_args: List of argument names to include in the cache key
        
    Returns:
        Decorated function with timed LRU cache
    """
    def decorator(func):
        # Apply standard lru_cache
        func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)
        
        # Store cache creation times
        creation_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from args and specific kwargs
            if include_args and kwargs:
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in include_args}
                key = (args, tuple(sorted(filtered_kwargs.items())))
            else:
                key = args
                
            # Check if key is in cache and not expired
            now = time.time()
            if key in creation_times:
                if now - creation_times[key] < ttl_seconds:
                    # Not expired, use cache
                    return func(*args, **kwargs)
                else:
                    # Expired, clear this key
                    func.cache_clear()
                    creation_times.clear()
            
            # Generate new result and store creation time
            result = func(*args, **kwargs)
            creation_times[key] = now
            return result
            
        # Add function to clear cache explicitly
        wrapper.cache_clear = func.cache_clear
        wrapper.cache_info = func.cache_info
        
        return wrapper
    
    return decorator

def cache_result(
    cache_type: str, 
    key_fields: List[str], 
    ttl_override: Optional[int] = None
):
    """
    Decorator that caches the results of a function with custom keys.
    
    Args:
        cache_type: Type of cache (indicators, patterns, analysis)
        key_fields: List of argument names to use in the cache key
        ttl_override: Optional override for TTL in seconds
        
    Returns:
        Decorated function with result caching
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from specified fields
            key_parts = []
            for field in key_fields:
                if field in kwargs:
                    key_parts.append(f"{field}:{kwargs[field]}")
            
            cache_key = "|".join(key_parts)
            
            # Get cache TTL
            ttl = ttl_override if ttl_override is not None else _cache_config[cache_type]["ttl"]
            
            # Check if result is in cache and not expired
            cache = _cache_store[cache_type]
            now = datetime.now()
            
            if cache_key in cache:
                entry = cache[cache_key]
                if now - entry["timestamp"] < timedelta(seconds=ttl):
                    logger.debug(f"Cache hit for {cache_type}: {cache_key}")
                    return entry["data"]
            
            # Not in cache or expired, call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = {
                "data": result,
                "timestamp": now
            }
            
            logger.debug(f"Cache miss for {cache_type}: {cache_key}, stored new result")
            return result
            
        return wrapper
    
    return decorator

def clear_cache(cache_type: Optional[str] = None):
    """
    Clear the cache for the specified type or all caches.
    
    Args:
        cache_type: Type of cache to clear or None for all
    """
    global _cache_store
    
    if cache_type is None:
        # Clear all caches
        for cache in _cache_store.values():
            cache.clear()
        logger.info("All caches cleared")
    elif cache_type in _cache_store:
        # Clear specific cache
        _cache_store[cache_type].clear()
        logger.info(f"Cache cleared: {cache_type}")
    else:
        logger.warning(f"Unknown cache type: {cache_type}")

def parallel_process(
    items: List[Any], 
    process_func: Callable[[Any], Any], 
    max_workers: int = 4
) -> List[Any]:
    """
    Process a list of items in parallel.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of worker threads
        
    Returns:
        List of processed results
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_func, items))
    
    return results

def batch_process(
    items: List[Any], 
    process_func: Callable[[List[Any]], List[Any]], 
    batch_size: int = 10
) -> List[Any]:
    """
    Process a list of items in batches.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each batch
        batch_size: Size of each batch
        
    Returns:
        List of processed results
    """
    results = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results

def optimize_dataframe_memory(df):
    """
    Optimize the memory usage of a pandas DataFrame.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    import pandas as pd
    import numpy as np
    
    # Make a copy of the dataframe
    result = df.copy()
    
    # Optimize numeric columns
    for col in result.select_dtypes(include=['int']).columns:
        col_min = result[col].min()
        col_max = result[col].max()
        
        # Convert to smallest possible integer type
        if col_min >= 0:
            if col_max < 255:
                result[col] = result[col].astype(np.uint8)
            elif col_max < 65535:
                result[col] = result[col].astype(np.uint16)
            elif col_max < 4294967295:
                result[col] = result[col].astype(np.uint32)
            else:
                result[col] = result[col].astype(np.uint64)
        else:
            if col_min > -128 and col_max < 127:
                result[col] = result[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                result[col] = result[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483647:
                result[col] = result[col].astype(np.int32)
            else:
                result[col] = result[col].astype(np.int64)
    
    # Optimize float columns
    for col in result.select_dtypes(include=['float']).columns:
        result[col] = result[col].astype(np.float32)
    
    # Optimize object columns (usually strings)
    for col in result.select_dtypes(include=['object']).columns:
        # If they are all strings, convert to category
        if result[col].apply(lambda x: isinstance(x, str)).all():
            if result[col].nunique() < len(result[col]) * 0.5:  # If less than 50% unique values
                result[col] = result[col].astype('category')
    
    return result

def memory_usage_report(obj) -> Dict[str, Any]:
    """
    Generate a memory usage report for an object.
    
    Args:
        obj: Object to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    import sys
    import pandas as pd
    
    # Get base memory usage
    memory_bytes = sys.getsizeof(obj)
    
    report = {
        "total_bytes": memory_bytes,
        "total_mb": memory_bytes / (1024 * 1024),
        "type": type(obj).__name__
    }
    
    # Additional details for DataFrames
    if isinstance(obj, pd.DataFrame):
        report.update({
            "shape": obj.shape,
            "columns": len(obj.columns),
            "rows": len(obj),
            "dtypes": dict(obj.dtypes.apply(lambda x: str(x))),
            "memory_usage_by_column": dict(obj.memory_usage(deep=True) / (1024 * 1024))
        })
    
    return report
