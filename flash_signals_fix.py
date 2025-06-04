#!/usr/bin/env python
"""
Direct Fix for FlashTradingSignals Integration

This module directly modifies the FlashTradingSignals class to accept
any client type and ensure the get_order_book method is always available.
"""

import logging
import importlib
import sys
import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_signals_fix")

def apply_flash_signals_fix():
    """Apply direct fix to FlashTradingSignals class"""
    try:
        # Import the module containing FlashTradingSignals
        from flash_trading_signals import FlashTradingSignals
        
        # Store the original __init__ method
        original_init = FlashTradingSignals.__init__
        
        # Define a new __init__ method that accepts any client type
        def new_init(self, client_instance=None, api_key=None, api_secret=None, env_path=None):
            """Modified initializer that accepts any client type and ensures get_order_book is available"""
            # Use existing client instance if provided, otherwise create new one
            if client_instance is not None:
                self.api_client = client_instance
                logger.info(f"Using provided client instance of type {type(client_instance).__name__} for SignalGenerator")
                
                # Ensure get_order_book method is available
                if not hasattr(self.api_client, 'get_order_book'):
                    # If client has get_orderbook (note the case), create an adapter method
                    if hasattr(self.api_client, 'get_orderbook'):
                        logger.info("Adding get_order_book adapter method to client")
                        
                        def get_order_book(client_self, symbol, limit=20):
                            """Adapter method that calls get_orderbook"""
                            return client_self.get_orderbook(symbol, limit)
                        
                        # Add the method to the client instance
                        self.api_client.get_order_book = types.MethodType(get_order_book, self.api_client)
                    else:
                        logger.warning("Client has neither get_order_book nor get_orderbook method")
            else:
                # Import here to avoid circular imports
                from optimized_mexc_client import OptimizedMEXCClient
                
                # Initialize API client with direct credentials
                self.api_client = OptimizedMEXCClient(api_key, api_secret, env_path)
                logger.info("Created new client instance for SignalGenerator")
                
                # Ensure get_order_book method is available
                if not hasattr(self.api_client, 'get_order_book'):
                    if hasattr(self.api_client, 'get_orderbook'):
                        logger.info("Adding get_order_book adapter method to new client")
                        
                        def get_order_book(client_self, symbol, limit=20):
                            """Adapter method that calls get_orderbook"""
                            return client_self.get_orderbook(symbol, limit)
                        
                        # Add the method to the client instance
                        self.api_client.get_order_book = types.MethodType(get_order_book, self.api_client)
            
            # Initialize the rest of the object using the original code
            # Initialize session manager
            self.session_manager = TradingSessionManager()
            
            # Signal history
            self.signals = []
            self.max_signals = 1000
            
            # Market state cache
            self.market_states = {}
            
            # Thread safety for client access
            self.client_lock = RLock()
            
            # Thread safety for market state updates
            self.market_state_lock = RLock()
            
            # Configuration dictionary for compatibility with flash_trading.py
            self.config = {
                "imbalance_threshold": 0.2,
                "momentum_threshold": 0.005,
                "volatility_threshold": 0.002,
                "min_signal_strength": 0.1,
                "position_size": 0.1
            }
            
            # Thread management for compatibility with flash_trading.py
            self.running = False
            self.symbols = []
            self.update_thread = None
            self.stop_event = Event()
        
        # Import required modules for the new __init__ method
        from threading import RLock, Event
        from trading_session_manager import TradingSessionManager
        
        # Replace the __init__ method
        FlashTradingSignals.__init__ = new_init
        
        logger.info("Successfully applied fix to FlashTradingSignals.__init__")
        return True
    except Exception as e:
        logger.error(f"Failed to apply fix to FlashTradingSignals: {str(e)}")
        return False

if __name__ == "__main__":
    # Apply the fix
    success = apply_flash_signals_fix()
    
    # Print result
    if success:
        print("✅ Successfully applied fix to FlashTradingSignals")
    else:
        print("❌ Failed to apply fix to FlashTradingSignals")
        sys.exit(1)
