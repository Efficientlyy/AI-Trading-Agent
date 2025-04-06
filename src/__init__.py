"""
AI Trading Agent - Main package.
This package contains the core functionality of the AI Trading Agent.
"""

from .common import logger, setup_logging, get_config, get_config_value

# Initialize configuration and logging early
try:
    config = get_config()
    logger = setup_logging(config)
    logger.info("AI Trading Agent initialized")
except Exception as e:
    # Fallback logging if config fails
    import sys
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    logger.critical(f"Failed to initialize configuration or logging: {e}")
    # Depending on severity, might want to exit or raise

__version__ = config.get('system', {}).get('version', '2.0.0') if 'config' in locals() else '2.0.0'

__all__ = [
    'logger',
    'setup_logging',
    'get_config',
    'get_config_value',
    '__version__',
]
