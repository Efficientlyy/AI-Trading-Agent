"""
Logging configuration for the AI Trading Agent.
Provides a centralized way to configure logging throughout the application.
"""
import os
import sys
from pathlib import Path
from loguru import logger

# Remove any default handlers loguru might have added on import
logger.remove()

def setup_logging(config=None, log_level=None, colorize_console: bool = True):
    """
    Set up logging configuration based on the provided config.
    If no config is provided, uses sensible defaults.

    Args:
        config (dict, optional): Configuration dictionary with logging settings.
        log_level (str, optional): Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        colorize_console (bool, optional): Whether to colorize console output. Defaults to True.

    Returns:
        logger: Configured logger instance
    """
    # --- Try printing a very simple message first --- 
    # print("OK_STDOUT_PRIME", file=sys.stdout, flush=True)
    # --- End simple message ---

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
    logger.remove() # This is fine, ensures a clean slate for this specific setup call

    # Define format strings
    if colorize_console:
        console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    else:
        console_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}" # Plain format

    # Add console logger
    logger.add(
        sys.stdout,
        level=default_log_level,
        format=console_format,  # Use the conditional format string
        colorize=colorize_console # Still pass colorize for loguru's internal handling of <level> etc.
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
# logger = setup_logging(log_level="DEBUG") # Set to DEBUG for detailed output

# Function to get a logger for a specific module
def get_logger(name=None):
    """
    Get a logger for a specific module.
    
    Args:
        name (str, optional): Name of the module requesting the logger. 
                             Defaults to None, which uses the caller's module name.
    
    Returns:
        logger: Logger instance configured with proper context
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals['__name__']
    
    return logger.bind(name=name)
