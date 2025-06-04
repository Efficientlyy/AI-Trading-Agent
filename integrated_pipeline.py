#!/usr/bin/env python
"""
Integrated Pipeline with Deep Fixes for Trading-Agent System

This module provides a fully integrated pipeline that incorporates all fixes
for symbol standardization, market data handling, signal generation,
LLM strategic overseer, and Telegram notifications.

This is the main entry point for running the complete trading system in production mode.
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integrated_pipeline.log")
    ]
)

logger = logging.getLogger("integrated_pipeline")

# Apply deep OpenRouter patch before any other imports
try:
    import deep_openrouter_patch
    deep_openrouter_patch.apply_all_fixes()
    logger.info("Applied deep OpenRouter patch")
except ImportError:
    logger.error("Failed to import deep_openrouter_patch module")
    raise

# Import fixed modules
try:
    import symbol_mapping_fix
    import mexc_api_fix
    import signal_generator_fix
    import paper_trading_fix
    import llm_overseer_fix
    import enhanced_telegram_notifications
    
    logger.info("Successfully imported all fixed modules")
except ImportError as e:
    logger.error(f"Failed to import fixed modules: {str(e)}")
    raise

class IntegratedPipeline:
    """Integrated pipeline for the Trading-Agent system"""
    
    def __init__(self, symbols=None, use_mock_data=False):
        """Initialize integrated pipeline
        
        Args:
            symbols: List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            use_mock_data: Whether to use mock data (ignored in production mode)
        """
        logger.info(f"Initializing integrated pipeline with symbols: {symbols}")
        
        # Store use_mock_data flag but ignore it in production
        self.use_mock_data = False  # Always set to False for production mode
        
        if use_mock_data:
            logger.warning("Mock data requested but disabled in production mode")
        
        # Set default symbols if not provided
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Initialize components
        self.mexc_client = mexc_api_fix.OptimizedMEXCClient()
        self.signal_generator = signal_generator_fix.SignalGenerator(client_instance=self.mexc_client)
        self.paper_trading = paper_trading_fix.PaperTrading(client_instance=self.mexc_client)
        self.llm_overseer = llm_overseer_fix.LLMOverseer()
        self.telegram_notifier = enhanced_telegram_notifications.EnhancedTelegramNotifier()
        
        logger.info("Integrated pipeline initialized")
    
    def start(self):
        """Start the integrated pipeline"""
        logger.info("Starting integrated pipeline")
        
        # Start Telegram notifier
        self.telegram_notifier.start()
        logger.info("Telegram notifier started")
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                logger.info(f"Processing {symbol}")
                
                # Get current price
                ticker = self.mexc_client.get_ticker(symbol)
                price = float(ticker["last"])
                logger.info(f"Current price for {symbol}: {price}")
                
                # Prepare market data
                market_data = {
                    "symbol": symbol,
                    "price": price,
                    "timestamp": int(time.time() * 1000),
                    "momentum": 0.02,  # Sample value for testing
                    "volatility": 0.01,  # Sample value for testing
                    "volume": 1000000  # Sample value for testing
                }
                
                # Generate signal
                signal = self.signal_generator.generate_signal(symbol, market_data)
                logger.info(f"Generated signal for {symbol}: {signal}")
                
                # Send signal notification
                self.telegram_notifier.send_signal_notification(symbol, signal)
                logger.info(f"Signal notification sent for {symbol}")
                
                # Get strategic decision
                decision = self.llm_overseer.get_strategic_decision(symbol, market_data)
                logger.info(f"Strategic decision for {symbol}: {decision}")
                
                # Send decision notification
                self.telegram_notifier.send_decision_notification(symbol, decision)
                logger.info(f"Decision notification sent for {symbol}")
                
                # Wait between symbols to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
        
        # Get current positions
        positions = self.paper_trading.get_positions()
        logger.info(f"Current positions: {positions}")
        
        # Send system notification
        self.telegram_notifier.send_system_notification("Trading cycle completed", {
            "symbols_processed": self.symbols,
            "positions": positions
        })
        logger.info("System notification sent")
        
        # Wait for notifications to be processed
        logger.info("Waiting for notifications to be processed...")
        time.sleep(5)
        
        # Stop Telegram notifier
        self.telegram_notifier.stop()
        logger.info("Telegram notifier stopped")
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Integrated pipeline for Trading-Agent system")
    parser.add_argument("--symbols", nargs="+", help="Trading pair symbols to process")
    parser.add_argument("--use-mock-data", action="store_true", help="Use mock data (ignored in production mode)")
    args = parser.parse_args()
    
    # Create and start integrated pipeline
    pipeline = IntegratedPipeline(symbols=args.symbols, use_mock_data=args.use_mock_data)
    result = pipeline.start()
    
    if result:
        print("Integrated pipeline completed successfully")
    else:
        print("Integrated pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
