# Signal Generation Debug Script

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
            
            # Print market state details
            logger.info(f"Market state for {symbol}:")
            logger.info(f"  Mid price: {market_state.mid_price}")
            logger.info(f"  Spread: {market_state.spread}")
            logger.info(f"  Bid depth: {market_state.bid_depth}")
            logger.info(f"  Ask depth: {market_state.ask_depth}")
            logger.info(f"  Imbalance: {market_state.imbalance}")
            
            # Try to generate signals with different thresholds
            logger.info("Attempting to generate signals with different thresholds")
            
            # Default thresholds from config
            thresholds = {
                "imbalance": 0.2,
                "volatility": 0.1,
                "momentum": 0.05
            }
            
            # Try with progressively lower thresholds
            for factor in [1.0, 0.5, 0.25, 0.1]:
                adjusted_thresholds = {k: v * factor for k, v in thresholds.items()}
                logger.info(f"Trying with thresholds: {adjusted_thresholds}")
                
                # Generate signals
                signals = signal_generator.generate_signals(
                    symbol, 
                    market_state,
                    imbalance_threshold=adjusted_thresholds["imbalance"],
                    volatility_threshold=adjusted_thresholds["volatility"],
                    momentum_threshold=adjusted_thresholds["momentum"]
                )
                
                if signals:
                    logger.info(f"Generated {len(signals)} signals with threshold factor {factor}")
                    for signal in signals:
                        logger.info(f"  Signal: {signal}")
                else:
                    logger.info(f"No signals generated with threshold factor {factor}")
            
        except Exception as e:
            logger.error(f"Error in manual signal generation for {symbol}: {e}")
    
    logger.info("Signal generation debug completed")

if __name__ == "__main__":
    debug_signal_generation()
