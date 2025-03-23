"""Common utility functions used across the system."""

import functools
import hashlib
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import pandas as pd

from src.common.logging import get_logger

# Configure logger
logger = get_logger("system", "utils")

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


def timeit(func: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        func: The function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(
            "Function execution time",
            function=func.__name__,
            execution_time_ms=round((end_time - start_time) * 1000, 2),
        )
        
        return result
    
    return wrapper


def async_timeit(func: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator to measure and log the execution time of an async function.
    
    Args:
        func: The async function to measure
        
    Returns:
        Wrapped async function that logs execution time
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(
            "Async function execution time",
            function=func.__name__,
            execution_time_ms=round((end_time - start_time) * 1000, 2),
        )
        
        return result
    
    return wrapper


def retry(max_attempts: int = 3, delay_seconds: float = 1.0, 
          backoff: float = 2.0, exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator for retrying functions that may fail with specified exceptions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries in seconds
        backoff: Multiplier for the delay between retries
        exceptions: Tuple of exceptions that trigger a retry
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay_seconds
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                        logger.warning(
                            "Function failed, retrying",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay=current_delay,
                            error=str(e),
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            logger.exception(
                "Function failed after max retries",
                function=func.__name__,
                max_attempts=max_attempts,
                error=str(last_exception),
            )
            raise last_exception
        
        return wrapper
    
    return decorator


def generate_hash(data: Any) -> str:
    """
    Generate a deterministic hash of any serializable data.
    
    Args:
        data: Any JSON-serializable data
        
    Returns:
        SHA-256 hash of the data as a hex string
    """
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def format_timeframe(timeframe: str) -> str:
    """
    Format a timeframe string to a standardized format.
    
    Args:
        timeframe: String representing a timeframe (e.g., "1h", "4h", "1d")
        
    Returns:
        Standardized timeframe string
        
    Raises:
        ValueError: If the timeframe format is invalid
    """
    # Extract the numeric value and unit
    if not timeframe:
        raise ValueError("Timeframe cannot be empty")
    
    # Check for valid format (number followed by letter)
    if not any(char.isdigit() for char in timeframe) or not any(char.isalpha() for char in timeframe):
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    # Split numeric value and unit
    value = "".join(filter(str.isdigit, timeframe))
    unit = "".join(filter(str.isalpha, timeframe)).lower()
    
    # Standardize unit
    unit_mapping = {
        "m": "m",
        "min": "m",
        "minute": "m",
        "minutes": "m",
        "h": "h",
        "hr": "h",
        "hour": "h",
        "hours": "h",
        "d": "d",
        "day": "d",
        "days": "d",
        "w": "w",
        "wk": "w",
        "week": "w",
        "weeks": "w",
    }
    
    if unit not in unit_mapping:
        raise ValueError(f"Invalid timeframe unit: {unit}")
    
    standardized_unit = unit_mapping[unit]
    
    return f"{value}{standardized_unit}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if the denominator is zero.
    
    Args:
        numerator: The numerator in the division
        denominator: The denominator in the division
        default: The default value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator 