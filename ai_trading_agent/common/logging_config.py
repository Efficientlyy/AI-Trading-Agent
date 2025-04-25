"""
Logging configuration for the AI Trading Agent.
Provides a centralized way to configure logging throughout the application.
"""
import os
import sys
from pathlib import Path
from loguru import logger

def setup_logging(config=None, log_level=None):
    """
    Set up logging configuration based on the provided config.
    If no config is provided, uses sensible defaults.

    Args:
        config (dict, optional): Configuration dictionary with logging settings.
        log_level (str, optional): Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logger: Configured logger instance
    """
    # Default configuration
    default_log_level = "INFO"
    log_file = "logs/trading_agent.log"

    # Override with config if provided
    if config and 'system' in config:
        default_log_level = config['system'].get('log_level', default_log_level)
        log_file = config['system'].get('log_file', log_file)
    
    # Override with explicit log_level if provided
    if log_level:
        default_log_level = log_level

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stderr,
        level=default_log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Add file logger
    logger.add(
        log_file,
        level=default_log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="1 week"
    )

    logger.info(f"Logging initialized with level {default_log_level}")
    return logger

# Export the logger instance for use throughout the application
logger = setup_logging(log_level="DEBUG") # Set to DEBUG for detailed output
