"""
Common utility functions used across the AI Trading Agent system.
"""

import logging
import os
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger with the specified name.
    
    Args:
        name: The name for the logger
        level: Optional logging level (defaults to INFO or value from LOG_LEVEL env var)
        
    Returns:
        Configured logger instance
    """
    # Get default log level from environment or use INFO
    default_level = os.environ.get('LOG_LEVEL', 'INFO')
    level = level or getattr(logging, default_level, logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Add a console handler if no handlers exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    return logger


def format_float(value: float, precision: int = 2) -> str:
    """
    Format a float value with the specified precision.
    
    Args:
        value: The float value to format
        precision: Number of decimal places
        
    Returns:
        Formatted string representation
    """
    return f"{value:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format a value as a percentage with the specified precision.
    
    Args:
        value: The value to format (0.01 = 1%)
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"
