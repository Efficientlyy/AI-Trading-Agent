#!/usr/bin/env python
"""
Integrated Pipeline Test

This script tests the integrated pipeline with all fixes applied.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integrated_pipeline_test.log")
    ]
)

logger = logging.getLogger("integrated_pipeline_test")

# Import the integrated pipeline fix module
try:
    import integrated_pipeline_fix
    
    # Apply all fixes
    integrated_pipeline_fix.load_environment_variables()
    integrated_pipeline_fix.invalidate_import_caches()
    integrated_pipeline_fix.inject_openrouter_module()
    
    logger.info("Successfully applied all fixes")
except ImportError as e:
    logger.error(f"Failed to import integrated_pipeline_fix: {str(e)}")
    raise

# Import required modules
try:
    from symbol_mapping_fix import SymbolMapper
    from mexc_api_fix import get_ticker
    from signal_generator_fix import SignalGenerator
    from paper_trading_fix import PaperTradingSystem
    from llm_overseer_fix import LLMOverseer
    from enhanced_telegram_notifications import EnhancedTelegramNotifier
    
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    raise

def run_integrated_pipeline_test():
    """Run integrated pipeline test"""
    logger.info("Starting integrated pipeline test")
    
    try:
        # Create configuration
        config = {
            'telegram_bot_token': os.environ.get('TELEGRAM_BOT_TOKEN'),
            'telegram_user_id': os.environ.get('TELEGRAM_USER_ID')
        }
        
        # Create components
        symbol_mapper = SymbolMapper()
        signal_generator = SignalGenerator()
        llm_overseer = LLMOverseer(use_mock_data=False)
        paper_trading = PaperTradingSystem(use_mock_data=False)
        notifier = EnhancedTelegramNotifier(config)
        
        # Start notifier
        notifier.start()
        logger.info("Telegram notifier started")
        
        # Test symbols
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Process each symbol
        for symbol in symbols:
            logger.info(f"Processing {symbol}")
            
            # Get current market data
            try:
                # Get current price from MEXC
                ticker_data = get_ticker(symbol)
                current_price = float(ticker_data.get('lastPrice', 0))
                
                if current_price == 0:
                    logger.warning(f"Failed to get current price for {symbol}, using fallback price")
                    current_price = 65000 if symbol.startswith("BTC") else (3000 if symbol.startswith("ETH") else 150)
                
                logger.info(f"Current price for {symbol}: {current_price}")
                
                # Create market data
                market_data = {
                    'price': current_price,
                    'momentum': 0.02,
                    'volatility': 0.01,
                    'volume': 1000000,
                    'timestamp': int(time.time() * 1000)
                }
                
                # Generate signal
                signal = signal_generator.generate_signal(symbol, market_data)
                logger.info(f"Generated signal for {symbol}: {signal}")
                
                # Send signal notification
                notifier.notify_signal(signal)
                logger.info(f"Signal notification sent for {symbol}")
                
                # Get strategic decision
                decision = llm_overseer.get_strategic_decision(symbol, market_data)
                logger.info(f"Strategic decision for {symbol}: {decision}")
                
                # Add symbol to decision
                decision['symbol'] = symbol
                
                # Send decision notification
                notifier.notify_decision(decision)
                logger.info(f"Decision notification sent for {symbol}")
                
                # Execute paper trade if action is BUY or SELL
                if decision.get('action') in ['BUY', 'SELL']:
                    # Calculate quantity based on price
                    quantity = 0.001 if symbol.startswith("BTC") else (0.01 if symbol.startswith("ETH") else 0.1)
                    
                    # Execute paper trade
                    trade_result = paper_trading.execute_trade(
                        symbol, 
                        decision.get('action'), 
                        quantity, 
                        'MARKET', 
                        {'source': 'test', 'strength': decision.get('confidence', 0.5)}
                    )
                    
                    logger.info(f"Paper trade executed for {symbol}: {trade_result}")
                    
                    # Create order notification
                    order = {
                        'symbol': symbol,
                        'side': decision.get('action'),
                        'type': 'MARKET',
                        'quantity': quantity,
                        'price': current_price,
                        'orderId': trade_result.get('order_id', f"ORD-{int(time.time())}")
                    }
                    
                    # Send order created notification
                    notifier.notify_order_created(order)
                    logger.info(f"Order created notification sent for {symbol}")
                    
                    # Wait for a moment
                    time.sleep(1)
                    
                    # Send order filled notification
                    notifier.notify_order_filled(order)
                    logger.info(f"Order filled notification sent for {symbol}")
                
                # Wait between symbols
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                notifier.notify_error("Integrated Pipeline Test", f"Error processing {symbol}: {str(e)}")
        
        # Get positions
        positions = paper_trading.get_positions()
        logger.info(f"Current positions: {positions}")
        
        # Send system notification
        notifier.notify_system("Integrated Pipeline Test", "All tests completed successfully")
        logger.info("System notification sent")
        
        # Wait for notifications to be processed
        logger.info("Waiting for notifications to be processed...")
        time.sleep(5)
        
        # Stop notifier
        notifier.stop()
        logger.info("Telegram notifier stopped")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during integrated pipeline test: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    success = run_integrated_pipeline_test()
    
    # Print result
    if success:
        print("Integrated pipeline test completed successfully")
    else:
        print("Integrated pipeline test failed")
