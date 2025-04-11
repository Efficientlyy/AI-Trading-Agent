"""
Logging configuration for the AI Trading Agent.
Provides a centralized way to configure logging throughout the application.
"""
import os
import sys
from pathlib import Path
from loguru import logger

def setup_logging(config=None):
    """
    Set up logging configuration based on the provided config.
    If no config is provided, uses sensible defaults.

    Args:
        config (dict, optional): Configuration dictionary with logging settings.

    Returns:
        logger: Configured logger instance
    """
    # Default configuration
    log_level = "INFO"
    log_file = "logs/trading_agent.log"

    # Override with config if provided
    if config and 'system' in config:
        log_level = config['system'].get('log_level', log_level)
        log_file = config['system'].get('log_file', log_file)

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Add file logger
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="1 week"
    )

    logger.info(f"Logging initialized with level {log_level}")
    return logger

# Export the logger instance for use throughout the application
logger = setup_logging()
