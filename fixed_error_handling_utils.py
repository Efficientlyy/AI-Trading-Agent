#!/usr/bin/env python
"""
Fixed Error Handling Utilities for Trading Agent

This module provides standardized utilities for error handling, validation,
and safe access to potentially None or malformed API responses.
"""

import logging
import functools
import traceback
from typing import Any, Dict, List, Optional, Union, TypeVar, Callable

# Configure logging
logger = logging.getLogger("error_handling")

# Type variables for generic functions
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

def safe_get(data: Optional[Dict[K, V]], key: K, default: V = None) -> V:
    """
    Safely get a value from a dictionary with validation.
    
    Args:
        data: Dictionary to access, may be None
        key: Key to access
        default: Default value if key doesn't exist or data is None
        
    Returns:
        Value if exists, default otherwise
    """
    if data is None:
        return default
        
    if not isinstance(data, dict):
        logger.warning(f"Expected dict for safe_get, got {type(data)}")
        return default
        
    return data.get(key, default)

def safe_get_nested(data: Optional[Dict], path: List[str], default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary with validation.
    
    Args:
        data: Dictionary to access, may be None
        path: List of keys forming path to nested value
        default: Default value if path doesn't exist or data is None
        
    Returns:
        Nested value if exists, default otherwise
    """
    if data is None:
        return default
        
    if not isinstance(data, dict):
        logger.warning(f"Expected dict for safe_get_nested, got {type(data)}")
        return default
    
    current = data
    for key in path:
        if not isinstance(current, dict):
            return default
            
        if key not in current:
            return default
            
        current = current[key]
    
    return current

def safe_list_access(data: Optional[List[T]], index: int, default: T = None) -> T:
    """
    Safely access a list element with validation.
    
    Args:
        data: List to access, may be None
        index: Index to access
        default: Default value if index out of bounds or data is None
        
    Returns:
        Element if exists, default otherwise
    """
    if data is None:
        return default
        
    if not isinstance(data, list):
        logger.warning(f"Expected list for safe_list_access, got {type(data)}")
        return default
        
    if index < 0 or index >= len(data):
        return default
        
    return data[index]

def parse_float_safely(value: Any, default: float = 0.0) -> float:
    """
    Safely parse a value to float with validation.
    
    Args:
        value: Value to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed float if successful, default otherwise
    """
    if value is None:
        return default
        
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def parse_int_safely(value: Any, default: int = 0) -> int:
    """
    Safely parse a value to int with validation.
    
    Args:
        value: Value to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed int if successful, default otherwise
    """
    if value is None:
        return default
        
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def validate_api_response(response: Any, required_fields: List[str] = None) -> bool:
    """
    Validate API response structure.
    
    Args:
        response: API response to validate
        required_fields: List of required fields in response
        
    Returns:
        True if valid, False otherwise
    """
    if response is None:
        return False
        
    # Fixed: Removed the problematic isinstance check with dynamic type
    # Now we just check if it's a dict when needed
    
    if required_fields and isinstance(response, dict):
        for field in required_fields:
            if field not in response:
                logger.warning(f"Missing required field '{field}' in API response")
                return False
    
    return True

def log_exception(context: str, e: Exception) -> None:
    """
    Log exception with context and traceback.
    
    Args:
        context: Context description for the error
        e: Exception to log
    """
    logger.error(f"Error in {context}: {str(e)}")
    logger.debug(f"Exception traceback: {traceback.format_exc()}")

def handle_api_error(func: Callable) -> Callable:
    """
    Decorator for handling API errors with consistent patterns.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get function name for context
            context = func.__name__
            log_exception(context, e)
            
            # Return appropriate default based on function's return annotation
            return_type = func.__annotations__.get('return')
            
            if return_type is None:
                return None
            elif return_type is bool or return_type == bool:
                return False
            elif return_type is dict or return_type == dict:
                return {}
            elif return_type is list or return_type == list:
                return []
            elif return_type is int or return_type == int:
                return 0
            elif return_type is float or return_type == float:
                return 0.0
            elif return_type is str or return_type == str:
                return ""
            else:
                return None
    
    return wrapper
