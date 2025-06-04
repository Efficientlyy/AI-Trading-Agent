#!/usr/bin/env python
"""
Monkey Patch for OptimizedMEXCClient

This module adds the missing get_order_book method to OptimizedMEXCClient
to ensure compatibility with the SignalGenerator.
"""

import logging
import importlib
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monkey_patch")

def apply_monkey_patch():
    """Apply monkey patch to OptimizedMEXCClient"""
    try:
        # Import the module containing OptimizedMEXCClient
        from optimized_mexc_client import OptimizedMEXCClient
        
        # Check if get_order_book already exists
        if hasattr(OptimizedMEXCClient, 'get_order_book'):
            logger.info("get_order_book method already exists in OptimizedMEXCClient")
            return True
        
        # Add get_order_book method that delegates to get_orderbook
        def get_order_book(self, symbol, limit=20):
            """
            Get order book for specified symbol (alias for get_orderbook)
            
            Args:
                symbol: Symbol in API format (e.g., BTCUSDC)
                limit: Number of entries to return
                
            Returns:
                dict: Order book data
            """
            logger.debug(f"Delegating get_order_book to get_orderbook for {symbol}")
            return self.get_orderbook(symbol, limit)
        
        # Add the method to the class
        OptimizedMEXCClient.get_order_book = get_order_book
        
        logger.info("Successfully added get_order_book method to OptimizedMEXCClient")
        return True
    except Exception as e:
        logger.error(f"Failed to apply monkey patch: {str(e)}")
        return False

if __name__ == "__main__":
    # Apply the monkey patch
    success = apply_monkey_patch()
    
    # Print result
    if success:
        print("✅ Successfully applied monkey patch to OptimizedMEXCClient")
    else:
        print("❌ Failed to apply monkey patch to OptimizedMEXCClient")
        sys.exit(1)
