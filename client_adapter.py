#!/usr/bin/env python
"""
Client Adapter for Trading-Agent System

This module provides an adapter that ensures compatibility between
different client interfaces used in the Trading-Agent system.
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("client_adapter")

class ClientAdapter:
    """
    Adapter class to ensure compatibility between different client interfaces
    in the Trading-Agent system.
    """
    
    def __init__(self, client_instance):
        """
        Initialize the client adapter with a client instance.
        
        Args:
            client_instance: The client instance to adapt (MultiAssetDataService, OptimizedMEXCClient, etc.)
        """
        self.client = client_instance
        logger.info(f"Initialized ClientAdapter with client of type: {type(client_instance).__name__}")
    
    def get_order_book(self, symbol, limit=20):
        """
        Get order book for the specified symbol.
        Adapts between get_order_book and get_orderbook method names.
        
        Args:
            symbol: Symbol to get order book for
            limit: Number of entries to return
            
        Returns:
            dict: Order book data with 'bids' and 'asks' lists
        """
        logger.debug(f"ClientAdapter: Getting order book for {symbol}")
        
        # Check which method the client implements
        if hasattr(self.client, 'get_order_book'):
            logger.debug("Using client's native get_order_book method")
            return self.client.get_order_book(symbol, limit)
        elif hasattr(self.client, 'get_orderbook'):
            logger.debug("Adapting client's get_orderbook method")
            orderbook = self.client.get_orderbook(symbol, limit)
            
            # Convert to the format expected by SignalGenerator if needed
            if isinstance(orderbook, dict) and 'bids' in orderbook and 'asks' in orderbook:
                # Check if bids/asks are already in the expected format
                if orderbook['bids'] and isinstance(orderbook['bids'][0], dict):
                    # Convert from dict format to list format
                    bids = [[bid['price'], bid['amount']] for bid in orderbook['bids']]
                    asks = [[ask['price'], ask['amount']] for ask in orderbook['asks']]
                    return {'bids': bids, 'asks': asks}
                else:
                    # Already in the expected format
                    return orderbook
            else:
                logger.error(f"Unexpected orderbook format: {orderbook}")
                return {'bids': [], 'asks': []}
        else:
            logger.error("Client does not implement get_order_book or get_orderbook")
            return {'bids': [], 'asks': []}
    
    def __getattr__(self, name):
        """
        Pass through any other method calls to the underlying client.
        
        Args:
            name: Method name
            
        Returns:
            The method from the underlying client
        """
        return getattr(self.client, name)
