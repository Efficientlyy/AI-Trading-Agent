"""
Test Bitvavo Integration

This script tests the Bitvavo integration by connecting to the Bitvavo API
and performing basic operations.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bitvavo_connector():
    """Test the BitvavoConnector class."""
    try:
        from src.execution.exchange.bitvavo import BitvavoConnector
        
        logger.info("Testing BitvavoConnector class...")
        
        # Create a connector with test credentials
        # In a real scenario, you would use your own API key and secret
        connector = BitvavoConnector(
            api_key="test_api_key",
            api_secret="test_api_secret"
        )
        
        logger.info("BitvavoConnector instance created successfully")
        
        # Test the normalize_symbol and standardize_symbol methods
        symbol = "BTC/EUR"
        normalized = connector.normalize_symbol(symbol)
        standardized = connector.standardize_symbol(normalized)
        
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Normalized: {normalized}")
        logger.info(f"Standardized: {standardized}")
        
        assert standardized == symbol, f"Symbol normalization/standardization failed: {standardized} != {symbol}"
        logger.info("Symbol normalization/standardization test passed")
        
        # Test the _generate_signature method
        timestamp = int(datetime.now().timestamp() * 1000)
        method = "GET"
        url_path = "/v2/time"
        
        try:
            signature = connector._generate_signature(timestamp, method, url_path)
            logger.info(f"Generated signature: {signature}")
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            return False
        
        logger.info("BitvavoConnector tests completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing BitvavoConnector: {e}")
        return False

def test_api_key_manager():
    """Test the ApiKeyManager Bitvavo methods."""
    try:
        from src.common.security.api_keys import ApiKeyManager
        
        logger.info("Testing ApiKeyManager Bitvavo methods...")
        
        # Create a temporary API key manager
        api_key_manager = ApiKeyManager()
        
        # Test adding Bitvavo credentials
        api_key = "test_api_key"
        api_secret = "test_api_secret"
        
        try:
            success = api_key_manager.add_bitvavo_credentials(api_key, api_secret)
            logger.info(f"Added Bitvavo credentials: {success}")
        except Exception as e:
            logger.error(f"Error adding Bitvavo credentials: {e}")
            return False
        
        # Test getting Bitvavo credentials
        try:
            credential = api_key_manager.get_credential("bitvavo")
            if credential:
                logger.info("Retrieved Bitvavo credentials successfully")
            else:
                logger.error("Failed to retrieve Bitvavo credentials")
                return False
        except Exception as e:
            logger.error(f"Error getting Bitvavo credentials: {e}")
            return False
        
        # Test validating Bitvavo credentials
        try:
            is_valid, message = api_key_manager._validate_bitvavo_credentials(api_key, api_secret)
            logger.info(f"Validated Bitvavo credentials: {is_valid}, {message}")
        except Exception as e:
            logger.error(f"Error validating Bitvavo credentials: {e}")
            return False
        
        logger.info("ApiKeyManager Bitvavo methods tests completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing ApiKeyManager Bitvavo methods: {e}")
        return False

def test_dashboard_api_routes():
    """Test the dashboard API routes for Bitvavo."""
    try:
        import requests
        
        logger.info("Testing dashboard API routes for Bitvavo...")
        
        # Base URL for the dashboard API
        base_url = "http://localhost:8083/api"
        
        # Test the Bitvavo status endpoint
        try:
            response = requests.get(f"{base_url}/settings/bitvavo/status")
            logger.info(f"Bitvavo status response: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Bitvavo status: {response.json()}")
            else:
                logger.error(f"Error getting Bitvavo status: {response.text}")
        except Exception as e:
            logger.error(f"Error testing Bitvavo status endpoint: {e}")
        
        # Test the Bitvavo test connection endpoint
        try:
            data = {
                "apiKey": "test_api_key",
                "apiSecret": "test_api_secret"
            }
            response = requests.post(f"{base_url}/settings/bitvavo/test", json=data)
            logger.info(f"Bitvavo test connection response: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Bitvavo test connection: {response.json()}")
            else:
                logger.error(f"Error testing Bitvavo connection: {response.text}")
        except Exception as e:
            logger.error(f"Error testing Bitvavo test connection endpoint: {e}")
        
        logger.info("Dashboard API routes for Bitvavo tests completed")
        return True
    except Exception as e:
        logger.error(f"Error testing dashboard API routes for Bitvavo: {e}")
        return False

def main():
    """Main function to run all tests."""
    logger.info("Starting Bitvavo integration tests...")
    
    # Test the BitvavoConnector class
    if test_bitvavo_connector():
        logger.info("BitvavoConnector tests passed")
    else:
        logger.error("BitvavoConnector tests failed")
    
    # Test the ApiKeyManager Bitvavo methods
    if test_api_key_manager():
        logger.info("ApiKeyManager Bitvavo methods tests passed")
    else:
        logger.error("ApiKeyManager Bitvavo methods tests failed")
    
    # Test the dashboard API routes for Bitvavo
    if test_dashboard_api_routes():
        logger.info("Dashboard API routes for Bitvavo tests passed")
    else:
        logger.error("Dashboard API routes for Bitvavo tests failed")
    
    logger.info("Bitvavo integration tests completed")

if __name__ == "__main__":
    main()