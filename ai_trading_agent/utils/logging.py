"""
Logging utilities for AI Trading Agent

This module provides logging functions and configurations for the AI Trading Agent system.
"""

import logging
from typing import Optional, Union


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Name for the logger, typically the module name
        level: Optional logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level if provided
    if level is not None:
        logger.setLevel(level)
    
    # Add a console handler if no handlers exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)s - %(message)s')
        )
        logger.addHandler(console_handler)
    
    return logger
