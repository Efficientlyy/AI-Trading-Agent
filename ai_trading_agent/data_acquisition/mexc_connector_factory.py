"""
MEXC Connector Factory

This module provides a factory for creating MEXC connectors,
with a fallback to a mock connector if the real connection fails.
"""

import logging
from typing import Optional, Union, Any

from ..common import get_logger
from ..config.mexc_config import MEXC_CONFIG

# Configure logger
logger = get_logger(__name__)

async def create_mexc_connector(use_mock: bool = False) -> Any:
    """
    Create a MEXC connector with fallback to mock if real connection fails.
    
    Args:
        use_mock: Force using mock connector
        
    Returns:
        MexcConnector or MockMexcConnector instance
    """
    api_key = MEXC_CONFIG.get("API_KEY", "")
    api_secret = MEXC_CONFIG.get("API_SECRET", "")
    
    # Check if API credentials are available
    if not api_key or not api_secret:
        logger.warning("MEXC API credentials not found, using mock connector")
        use_mock = True
    
    if use_mock:
        # Import mock connector only when needed (avoid circular imports)
        try:
            from .mock_mexc_connector import MockMexcConnector
            
            # Use mock connector
            logger.info("Using mock MEXC connector")
            connector = MockMexcConnector(api_key=api_key, api_secret=api_secret)
            await connector.connect()
            return connector
        except Exception as e:
            logger.error(f"Error creating mock MEXC connector: {e}")
            return None
    
    # Try real connector first
    try:
        # Import real connector only when needed
        from .mexc_connector import MexcConnector
        
        logger.info("Attempting to use real MEXC connector")
        connector = MexcConnector(api_key=api_key, api_secret=api_secret)
        
        # Try to connect
        success = await connector.connect()
        if success:
            logger.info("Successfully connected to real MEXC API")
            return connector
        
        # If connection fails, fall back to mock
        logger.warning("Failed to connect to real MEXC API, falling back to mock connector")
        await connector.disconnect()
        
        # Import mock connector only when needed
        from .mock_mexc_connector import MockMexcConnector
        mock_connector = MockMexcConnector(api_key=api_key, api_secret=api_secret)
        await mock_connector.connect()
        return mock_connector
    except Exception as e:
        logger.error(f"Error creating real MEXC connector: {e}")
        logger.warning("Falling back to mock MEXC connector")
        
        try:
            # Import mock connector only when needed
            from .mock_mexc_connector import MockMexcConnector
            
            # Use mock connector as fallback
            mock_connector = MockMexcConnector(api_key=api_key, api_secret=api_secret)
            await mock_connector.connect()
            return mock_connector
        except Exception as mock_error:
            logger.error(f"Even mock connector failed: {mock_error}")
            return None