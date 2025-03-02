"""Logging system for the entire application.

This module sets up structured logging with proper formatting,
rotation, and component-specific loggers.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytz
import structlog
from structlog.types import Processor

from src.common.config import config

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Default log file
DEFAULT_LOG_FILE = LOG_DIR / "crypto_trading.log"


def configure_logging() -> None:
    """Configure the logging system based on the application configuration."""
    # Get configuration values
    log_level_name = config.get("system.logging.level", "INFO")
    log_format = config.get("system.logging.format", "json")
    include_timestamps = config.get("system.logging.include_timestamps", True)
    include_line_numbers = config.get("system.logging.include_line_numbers", True)
    
    # Convert log level string to constant
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    
    # Set up standard logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(DEFAULT_LOG_FILE),
        ],
    )
    
    # Configure processors for structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
    ]
    
    # Add optional processors based on configuration
    if include_timestamps:
        processors.append(_add_timestamp)
    
    if include_line_numbers:
        processors.append(structlog.processors.CallsiteParameterAdder(
            parameters=["filename", "lineno", "func_name"]
        ))
    
    # Add standard processors for all formats
    processors.extend([
        structlog.processors.dict_tracebacks,
        structlog.processors.ExceptionPrettyPrinter(),
    ])
    
    # Add format-specific processors
    if log_format.lower() == "json":
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _add_timestamp(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add a formatted timestamp to the log event."""
    tz = pytz.timezone(config.get("system.general.timezone", "UTC"))
    now = datetime.now(tz)
    event_dict["timestamp"] = now.isoformat()
    return event_dict


def get_logger(component: str, subcomponent: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger for a specific component.
    
    Args:
        component: The main component name
        subcomponent: Optional subcomponent name
        
    Returns:
        A bound logger with component context
    """
    logger = structlog.get_logger()
    
    # Add component and subcomponent to the logger context
    if subcomponent:
        return logger.bind(component=component, subcomponent=subcomponent)
    
    return logger.bind(component=component)


# Configure logging when the module is imported
configure_logging()

# Default system logger
system_logger = get_logger("system") 