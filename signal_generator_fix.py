#!/usr/bin/env python
"""
Signal Generator Fix for Trading-Agent System

This module provides a wrapper and compatibility layer for the FlashTradingSignals class,
exposing it as SignalGenerator for compatibility with the rest of the system.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("signal_generator_fix.log")
    ]
)

logger = logging.getLogger("signal_generator_fix")

# Import the original signals module
try:
    from flash_trading_signals import FlashTradingSignals
    logger.info("Successfully imported FlashTradingSignals")
except ImportError as e:
    logger.error(f"Failed to import FlashTradingSignals: {str(e)}")
    raise

class SignalGenerator:
    """Wrapper class for FlashTradingSignals to provide compatibility with expected interface"""
    
    def __init__(self, client_instance=None, api_key=None, api_secret=None, env_path=None, use_mock_data=False):
        """Initialize signal generator
        
        Args:
            client_instance: Existing OptimizedMEXCClient instance to use (preferred)
            api_key: API key for MEXC (used only if client_instance is None)
            api_secret: API secret for MEXC (used only if client_instance is None)
            env_path: Path to .env file (used only if client_instance is None)
            use_mock_data: Whether to use mock data (ignored in production mode)
        """
        # Store use_mock_data flag but ignore it in production
        self.use_mock_data = False  # Always set to False for production mode
        
        if use_mock_data:
            logger.warning("Mock data requested but disabled in production mode")
        
        # Initialize underlying signal generator
        self.signal_generator = FlashTradingSignals(
            client_instance=client_instance,
            api_key=api_key,
            api_secret=api_secret,
            env_path=env_path
        )
        
        logger.info("SignalGenerator initialized with FlashTradingSignals")
    
    def generate_signals(self, symbol):
        """Generate trading signals for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDC')
            
        Returns:
            list: List of trading signals
        """
        logger.info(f"Generating signals for {symbol}")
        
        try:
            # Call underlying signal generator
            signals = self.signal_generator.generate_signals(symbol)
            
            logger.info(f"Generated {len(signals)} signals for {symbol}")
            return signals
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
    
    def generate_signal(self, symbol, market_data=None):
        """Generate a single trading signal for a symbol
        
        This method is added for compatibility with the integrated pipeline
        which expects a generate_signal method.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            market_data: Optional market data dictionary
            
        Returns:
            dict: Trading signal with type, strength, price, etc.
        """
        logger.info(f"Generating single signal for {symbol} with market data: {market_data is not None}")
        
        try:
            # If we have market data, use it to create a signal
            if market_data:
                price = market_data.get('price', 0)
                momentum = market_data.get('momentum', 0)
                
                # Determine signal type based on momentum
                signal_type = "BUY" if momentum > 0 else "SELL"
                
                # Calculate signal strength (0.5-1.0) based on momentum
                strength = min(1.0, max(0.5, 0.5 + abs(momentum) * 5))
                
                # Create signal
                signal = {
                    "symbol": symbol,
                    "type": signal_type,
                    "strength": strength,
                    "price": price,
                    "source": "SignalGenerator",
                    "timestamp": int(time.time() * 1000)
                }
                
                logger.info(f"Generated signal for {symbol}: {signal}")
                return signal
            
            # Otherwise, try to get signals from the underlying generator
            signals = self.generate_signals(symbol)
            
            # Return the first signal if available
            if signals and len(signals) > 0:
                logger.info(f"Returning first signal from generate_signals: {signals[0]}")
                return signals[0]
            
            # If no signals, create a default one
            default_signal = {
                "symbol": symbol,
                "type": "NEUTRAL",
                "strength": 0.5,
                "price": 0.0,
                "source": "SignalGenerator Default",
                "timestamp": int(time.time() * 1000)
            }
            
            logger.info(f"No signals available, returning default: {default_signal}")
            return default_signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            
            # Return a fallback signal on error
            fallback_signal = {
                "symbol": symbol,
                "type": "NEUTRAL",
                "strength": 0.5,
                "price": 0.0,
                "source": "SignalGenerator Fallback",
                "timestamp": int(time.time() * 1000),
                "error": str(e)
            }
            
            logger.info(f"Returning fallback signal: {fallback_signal}")
            return fallback_signal
    
    def start(self, symbols):
        """Start signal generation for specified symbols
        
        Args:
            symbols: List of trading pair symbols (e.g., ['BTC/USDC', 'ETH/USDC'])
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        logger.info(f"Starting signal generation for {symbols}")
        
        try:
            # Call underlying signal generator
            result = self.signal_generator.start(symbols)
            
            logger.info(f"Signal generation started: {result}")
            return result
        except Exception as e:
            logger.error(f"Error starting signal generation: {str(e)}")
            return False
    
    def stop(self):
        """Stop signal generation
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        logger.info("Stopping signal generation")
        
        try:
            # Call underlying signal generator
            result = self.signal_generator.stop()
            
            logger.info(f"Signal generation stopped: {result}")
            return result
        except Exception as e:
            logger.error(f"Error stopping signal generation: {str(e)}")
            return False

# Add missing import
import time

# Make SignalGenerator available when importing from flash_trading_signals
sys.modules['flash_trading_signals'].SignalGenerator = SignalGenerator
logger.info("SignalGenerator class added to flash_trading_signals module")

if __name__ == "__main__":
    # Test the fix
    try:
        from flash_trading_signals import SignalGenerator
        
        print("SignalGenerator import successful")
        
        # Create instance
        signal_gen = SignalGenerator()
        
        print("SignalGenerator instance created")
        
        # Test signal generation
        print("Testing signal generation for BTCUSDT...")
        signals = signal_gen.generate_signals("BTCUSDT")
        
        print(f"Generated {len(signals)} signals")
        if signals:
            print("Sample signal:")
            print(signals[0])
            
        # Test single signal generation
        print("\nTesting single signal generation for BTCUSDT...")
        market_data = {
            "price": 65000,
            "momentum": 0.02,
            "volatility": 0.01,
            "volume": 1000000
        }
        signal = signal_gen.generate_signal("BTCUSDT", market_data)
        
        print("Generated signal:")
        print(signal)
    except Exception as e:
        print(f"Error testing SignalGenerator: {str(e)}")
