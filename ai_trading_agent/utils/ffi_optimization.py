"""
FFI Optimization Module

This module provides optimizations for the Rust-Python FFI (Foreign Function Interface)
boundary to improve performance when transferring data between the two languages.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import time
import logging
from functools import wraps
import ctypes
import os
import sys

# Setup logging
logger = logging.getLogger(__name__)

# Global shared memory cache for FFI operations
_shared_memory_cache = {}

def profile_ffi_call(func):
    """
    Decorator to profile FFI calls and log performance metrics.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Log performance for slow calls
        if elapsed > 0.01:  # Log calls taking more than 10ms
            logger.debug(f"FFI call to {func.__name__} took {elapsed * 1000:.2f}ms")
        
        return result
    return wrapper

def zero_copy_array(np_array):
    """
    Create a zero-copy view of a NumPy array for passing to Rust.
    
    Args:
        np_array: NumPy array to create a view of
        
    Returns:
        Tuple of (data pointer, shape, strides) for Rust FFI
    """
    # Ensure array is contiguous in memory
    if not np_array.flags['C_CONTIGUOUS']:
        np_array = np.ascontiguousarray(np_array)
    
    # Get data pointer, shape, and strides
    data_ptr = np_array.ctypes.data_as(ctypes.c_void_p)
    shape = np_array.shape
    strides = np_array.strides
    
    return (data_ptr, shape, strides)

def dataframe_to_rust_compatible(df):
    """
    Convert a pandas DataFrame to a format that can be efficiently passed to Rust.
    
    Args:
        df: pandas DataFrame to convert
        
    Returns:
        Dict with column arrays in a Rust-compatible format
    """
    result = {
        'index': zero_copy_array(df.index.values),
        'columns': df.columns.tolist(),
        'data': {}
    }
    
    # Convert each column to a zero-copy array
    for col in df.columns:
        result['data'][col] = zero_copy_array(df[col].values)
    
    return result

def rust_array_to_numpy(data_ptr, shape, dtype, owner=None):
    """
    Create a NumPy array from a data pointer received from Rust.
    
    Args:
        data_ptr: Pointer to the data
        shape: Shape of the array
        dtype: NumPy data type
        owner: Object that owns the data (to prevent garbage collection)
        
    Returns:
        NumPy array view of the data
    """
    # Create a buffer from the memory pointer
    buf = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_byte * (np.prod(shape) * np.dtype(dtype).itemsize)))
    
    # Create a NumPy array from the buffer
    arr = np.frombuffer(buf.contents, dtype=dtype).reshape(shape)
    
    # If an owner is provided, store a reference to prevent garbage collection
    if owner is not None:
        arr._owner = owner
    
    return arr

def cache_rust_computation(key_prefix, ttl_seconds=300):
    """
    Decorator to cache results of Rust computations.
    
    Args:
        key_prefix: Prefix for the cache key
        ttl_seconds: Time to live for the cache entry in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a cache key
            key_parts = [key_prefix]
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                elif isinstance(arg, np.ndarray):
                    # For arrays, use a hash of the first and last few elements
                    if len(arg) > 0:
                        key_parts.append(f"array:{hash(str(arg[:3].tolist() + arg[-3:].tolist()))}")
                elif isinstance(arg, pd.DataFrame):
                    # For DataFrames, use shape and column info
                    key_parts.append(f"df:{arg.shape}:{'-'.join(arg.columns)}")
            
            cache_key = ":".join(key_parts)
            
            # Check if result is in cache and not expired
            if cache_key in _shared_memory_cache:
                entry = _shared_memory_cache[cache_key]
                if time.time() - entry['timestamp'] < ttl_seconds:
                    logger.debug(f"FFI cache hit for {key_prefix}")
                    return entry['data']
            
            # Not in cache or expired, call the function
            result = func(*args, **kwargs)
            
            # Store in cache
            _shared_memory_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            return result
        return wrapper
    return decorator

def clear_ffi_cache(key_prefix=None):
    """
    Clear the FFI shared memory cache.
    
    Args:
        key_prefix: Optional prefix to clear only specific entries
    """
    global _shared_memory_cache
    
    if key_prefix is None:
        _shared_memory_cache.clear()
        logger.debug("Cleared entire FFI cache")
    else:
        keys_to_remove = [k for k in _shared_memory_cache if k.startswith(key_prefix)]
        for k in keys_to_remove:
            del _shared_memory_cache[k]
        logger.debug(f"Cleared {len(keys_to_remove)} entries with prefix {key_prefix} from FFI cache")

def adaptive_data_conversion(data, threshold_size=1000000):
    """
    Adaptively choose the most efficient data conversion method based on data size.
    
    Args:
        data: Data to convert (DataFrame or NumPy array)
        threshold_size: Size threshold for using zero-copy vs. serialization
        
    Returns:
        Converted data in the most efficient format
    """
    if isinstance(data, pd.DataFrame):
        # For large DataFrames, use zero-copy
        if data.size > threshold_size:
            return dataframe_to_rust_compatible(data)
        # For smaller DataFrames, simple conversion is faster
        return {
            'index': data.index.tolist(),
            'columns': data.columns.tolist(),
            'data': {col: data[col].tolist() for col in data.columns}
        }
    elif isinstance(data, np.ndarray):
        # For large arrays, use zero-copy
        if data.size > threshold_size:
            return zero_copy_array(data)
        # For smaller arrays, simple conversion is faster
        return data.tolist()
    
    # Return as is for other types
    return data

def optimize_pandas_for_rust(df):
    """
    Optimize a pandas DataFrame for efficient processing in Rust.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    # Ensure index is in an efficient format
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    
    # Convert string columns to categorical if they have few unique values
    for col in df.select_dtypes(include=['object']):
        num_unique = df[col].nunique()
        if num_unique < len(df) * 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Downcast numeric columns to save memory
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

class RustMemoryManager:
    """
    Memory manager for Rust-allocated memory to ensure proper cleanup.
    """
    def __init__(self):
        self.allocated_pointers = {}
    
    def register_pointer(self, ptr, size, description=""):
        """Register a pointer allocated by Rust."""
        self.allocated_pointers[ptr] = {
            'size': size,
            'description': description,
            'allocated_at': time.time()
        }
    
    def free_pointer(self, ptr):
        """Free a pointer allocated by Rust."""
        if ptr in self.allocated_pointers:
            # Call Rust deallocation function here
            # rust_lib.free_memory(ptr)
            del self.allocated_pointers[ptr]
            return True
        return False
    
    def cleanup_old_pointers(self, max_age_seconds=300):
        """Clean up pointers older than max_age_seconds."""
        now = time.time()
        ptrs_to_free = []
        
        for ptr, info in self.allocated_pointers.items():
            if now - info['allocated_at'] > max_age_seconds:
                ptrs_to_free.append(ptr)
        
        for ptr in ptrs_to_free:
            self.free_pointer(ptr)
        
        return len(ptrs_to_free)
    
    def __del__(self):
        """Ensure all pointers are freed when the manager is garbage collected."""
        for ptr in list(self.allocated_pointers.keys()):
            self.free_pointer(ptr)

# Create a global memory manager
memory_manager = RustMemoryManager()

def load_rust_library():
    """
    Load the Rust dynamic library based on the current platform.
    
    Returns:
        Loaded Rust library or None if not found
    """
    lib_name = None
    
    # Determine the appropriate library name based on the platform
    if sys.platform.startswith('win'):
        lib_name = "technical_analysis.dll"
    elif sys.platform.startswith('darwin'):
        lib_name = "libtechnical_analysis.dylib"
    else:  # Linux and others
        lib_name = "libtechnical_analysis.so"
    
    # Define possible paths for the library
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_paths = [
        os.path.join(base_dir, "rust_modules", lib_name),
        os.path.join(base_dir, "lib", lib_name),
        os.path.join(base_dir, "target", "release", lib_name),
        os.path.join(base_dir, "..", "target", "release", lib_name),
    ]
    
    # Try to load the library from each path
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except Exception as e:
                logger.error(f"Failed to load Rust library from {path}: {e}")
    
    logger.warning(f"Could not find Rust library {lib_name} in any of the expected locations")
    return None

# Try to load the Rust library
rust_lib = load_rust_library()
