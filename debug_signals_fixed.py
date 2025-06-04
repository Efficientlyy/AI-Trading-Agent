# Updated Signal Generation Debug Script (Fixed Method Signature)

import os
import sys
import time
import logging
from optimized_mexc_client import OptimizedMEXCClient
from flash_trading_signals import FlashTradingSignals, MarketState

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("signal_debug")

def debug_signal_generation():
    """Debug the signal generation process with verbose logging"""
    logger.info("Starting signal generation debug")
    
    # Initialize client with API credentials
    client = OptimizedMEXCClient()
    logger.info("MEXC client initialized")
    
    # Test basic market data retrieval
    symbols = ["BTCUSDC", "ETHUSDC"]
    for symbol in symbols:
        try:
            # Get ticker data
            ticker = client.get_ticker(symbol)
            logger.info(f"Ticker for {symbol}: {ticker}")
            
            # Get order book
            order_book = client.get_order_book(symbol, limit=10)
            logger.info(f"Order book for {symbol} - Bids: {len(order_book['bids'])}, Asks: {len(order_book['asks'])}")
            logger.info(f"Top bid: {order_book['bids'][0]}, Top ask: {order_book['asks'][0]}")
            
            # Calculate mid price and spread
            top_bid = float(order_book['bids'][0][0])
            top_ask = float(order_book['asks'][0][0])
            mid_price = (top_bid + top_ask) / 2
            spread = top_ask - top_bid
            spread_bps = (spread / mid_price) * 10000
            
            logger.info(f"Mid price: {mid_price}, Spread: {spread} ({spread_bps:.2f} bps)")
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {e}")
    
    # Initialize signal generator with debug mode
    logger.info("Initializing signal generator")
    signal_generator = FlashTradingSignals(client, symbols)
    
    # Manually update market state and check for signals
    logger.info("Manually updating market state and checking for signals")
    for symbol in symbols:
        try:
            # Get fresh order book
            order_book = client.get_order_book(symbol, limit=20)
            
            # Create market state
            market_state = MarketState(symbol)
            
            # Update market state with order book
            bids = order_book['bids']
            asks = order_book['asks']
            
            # Convert string values to float
            bids = [[float(price), float(qty)] for price, qty in bids]
            asks = [[float(price), float(qty)] for price, qty in asks]
            
            # Update market state
            market_state.update_order_book(bids, asks)
            
            # Print market state details - only use attributes we know exist
            logger.info(f"Market state for {symbol}:")
            logger.info(f"  Mid price: {market_state.mid_price}")
            logger.info(f"  Spread: {market_state.spread}")
            
            # Inspect all available attributes of MarketState
            logger.info(f"Available attributes of MarketState for {symbol}:")
            for attr in dir(market_state):
                if not attr.startswith('_') and not callable(getattr(market_state, attr)):
                    try:
                        value = getattr(market_state, attr)
                        logger.info(f"  {attr}: {value}")
                    except Exception as e:
                        logger.error(f"  Error accessing {attr}: {e}")
            
            # Try to generate signals using the correct method signature
            logger.info("Attempting to generate signals")
            
            # Temporarily modify session parameters for testing with different thresholds
            original_params = signal_generator.session_manager.get_session_parameters("US")
            logger.info(f"Original session parameters: {original_params}")
            
            # Try with progressively lower thresholds
            for factor in [1.0, 0.5, 0.25, 0.1, 0.05]:
                try:
                    # Temporarily update session parameters
                    session_name = signal_generator.session_manager.get_current_session_name()
                    logger.info(f"Current session: {session_name}")
                    
                    # Update market state in signal generator
                    signal_generator.market_states[symbol] = market_state
                    
                    # Modify session parameters for testing
                    test_params = signal_generator.session_manager.get_session_parameters(session_name).copy()
                    test_params["imbalance_threshold"] = 0.2 * factor
                    test_params["volatility_threshold"] = 0.1 * factor
                    test_params["momentum_threshold"] = 0.05 * factor
                    
                    logger.info(f"Testing with thresholds: imbalance={test_params['imbalance_threshold']}, "
                               f"volatility={test_params['volatility_threshold']}, "
                               f"momentum={test_params['momentum_threshold']}")
                    
                    # Temporarily replace session parameters
                    signal_generator.session_manager.session_parameters[session_name] = test_params
                    
                    # Generate signals using the correct method signature
                    signals = signal_generator.generate_signals(symbol)
                    
                    if signals:
                        logger.info(f"Generated {len(signals)} signals with threshold factor {factor}")
                        for signal in signals:
                            logger.info(f"  Signal: {signal}")
                    else:
                        logger.info(f"No signals generated with threshold factor {factor}")
                        
                except Exception as e:
                    logger.error(f"Error generating signals with factor {factor}: {e}")
                    
            # Restore original session parameters
            signal_generator.session_manager.session_parameters["US"] = original_params
            
        except Exception as e:
            logger.error(f"Error in manual signal generation for {symbol}: {e}")
    
    logger.info("Signal generation debug completed")

if __name__ == "__main__":
    debug_signal_generation()
