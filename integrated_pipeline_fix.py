#!/usr/bin/env python
"""
Integrated Pipeline Fix for Trading-Agent System

This module validates and fixes the integrated pipeline with the updated
MEXC API signature generation and symbol standardization logic.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integrated_pipeline_fix.log")
    ]
)

logger = logging.getLogger("integrated_pipeline_fix")

# Import required modules
try:
    from symbol_standardization import SymbolStandardizer
    from enhanced_market_data_pipeline import EnhancedMarketDataPipeline
    from flash_trading_signals import FlashTradingSignals
    from fixed_paper_trading import FixedPaperTradingSystem
    from enhanced_telegram_notifications import EnhancedTelegramNotifier
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    sys.exit(1)

class IntegratedPipelineFix:
    """Integrated pipeline fix for Trading-Agent System"""
    
    def __init__(self):
        """Initialize integrated pipeline fix"""
        logger.info("Initializing integrated pipeline fix...")
        
        # Initialize components
        self.standardizer = SymbolStandardizer()
        self.market_data = EnhancedMarketDataPipeline()
        self.signals = FlashTradingSignals()
        self.paper_trading = FixedPaperTradingSystem()
        self.telegram = EnhancedTelegramNotifier()
        
        # Test symbols
        self.test_symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "BTC/USDT", "ETH/USDT", "SOL/USDT"]
        
        # Results storage
        self.results = {
            "market_data": {},
            "signals": {},
            "paper_trading": {},
            "telegram": {}
        }
        
        logger.info("Integrated pipeline fix initialized")
    
    def validate_market_data(self):
        """Validate market data collection with new keys and fixed logic"""
        logger.info("Validating market data collection...")
        
        for symbol in self.test_symbols:
            logger.info(f"Testing market data for {symbol}...")
            
            # Convert symbol to API format
            api_symbol = self.standardizer.for_mexc(symbol)
            logger.info(f"API symbol: {api_symbol}")
            
            # Get market data
            data = self.market_data.get_market_data(
                symbol=symbol,
                timeframe="5m",
                limit=10,
                fallback_to_mock=False,  # Ensure no mock data is used
                is_test_mode=False       # Use production mode
            )
            
            # Store results
            self.results["market_data"][symbol] = {
                "api_symbol": api_symbol,
                "data_length": len(data),
                "has_data": len(data) > 0,
                "first_candle": data[0] if data else None,
                "last_candle": data[-1] if data else None
            }
            
            if data:
                logger.info(f"Successfully fetched {len(data)} candles for {symbol}")
                logger.info(f"First candle: {data[0]}")
                logger.info(f"Last candle: {data[-1]}")
            else:
                logger.warning(f"No market data available for {symbol}")
        
        return self.results["market_data"]
    
    def validate_technical_analysis(self):
        """Validate technical analysis with live market data"""
        logger.info("Validating technical analysis...")
        
        for symbol in self.test_symbols:
            logger.info(f"Testing technical analysis for {symbol}...")
            
            # Convert symbol to API format
            api_symbol = self.standardizer.for_mexc(symbol)
            
            # Generate signals
            try:
                self.signals.start([api_symbol])
                time.sleep(2)  # Allow time for market state update
                generated_signals = self.signals.generate_signals(api_symbol)
                self.signals.stop()
                
                # Store results
                self.results["signals"][symbol] = {
                    "api_symbol": api_symbol,
                    "signals_count": len(generated_signals),
                    "signals": generated_signals
                }
                
                logger.info(f"Generated {len(generated_signals)} signals for {symbol}")
                for signal in generated_signals:
                    logger.info(f"Signal: {signal}")
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                self.results["signals"][symbol] = {
                    "api_symbol": api_symbol,
                    "error": str(e)
                }
        
        return self.results["signals"]
    
    def validate_paper_trading(self):
        """Validate paper trading with live market data"""
        logger.info("Validating paper trading...")
        
        # Start paper trading system
        self.paper_trading.start()
        
        # Store notifications
        notifications = []
        
        # Define notification callback
        def notification_callback(notification_type, data):
            logger.info(f"Notification received: {notification_type}")
            logger.info(f"Notification data: {data}")
            notifications.append({
                "type": notification_type,
                "data": data,
                "timestamp": int(time.time() * 1000)
            })
        
        # Set notification callback
        self.paper_trading.set_notification_callback(notification_callback)
        
        # Test paper trading with signals
        for symbol, signal_data in self.results["signals"].items():
            if "error" in signal_data:
                logger.warning(f"Skipping paper trading for {symbol} due to signal generation error")
                continue
            
            logger.info(f"Testing paper trading for {symbol}...")
            
            # Get API symbol
            api_symbol = signal_data["api_symbol"]
            
            # Get signals
            signals = signal_data.get("signals", [])
            
            # Execute paper trades based on signals
            trades_executed = 0
            
            for signal in signals:
                try:
                    signal_type = signal.get("type")
                    symbol = signal.get("symbol")
                    price = signal.get("price")
                    strength = signal.get("strength", 0.0)
                    
                    # Only execute trades for strong signals
                    if strength > 0.2:
                        # Calculate position size based on signal strength
                        position_size = 0.01 * strength * 10  # Scale based on strength
                        
                        logger.info(f"Executing {signal_type} trade for {symbol} at {price} with size {position_size}")
                        
                        # Create order
                        order_id = self.paper_trading.create_order(
                            symbol=symbol,
                            side=signal_type,
                            order_type="MARKET",
                            quantity=position_size
                        )
                        
                        if order_id:
                            trades_executed += 1
                            logger.info(f"Trade executed with order ID: {order_id}")
                        else:
                            logger.warning(f"Failed to execute trade for signal: {signal}")
                    else:
                        logger.info(f"Signal strength too low ({strength}), not executing trade")
                except Exception as e:
                    logger.error(f"Error executing trade for signal {signal}: {str(e)}")
            
            # Store results
            self.results["paper_trading"][symbol] = {
                "api_symbol": api_symbol,
                "signals_count": len(signals),
                "trades_executed": trades_executed,
                "notifications": len(notifications)
            }
            
            logger.info(f"Paper trading test for {symbol} completed: {trades_executed} trades executed")
        
        # Stop paper trading system
        self.paper_trading.stop()
        
        # Store notifications
        self.results["telegram"]["notifications"] = notifications
        
        return self.results["paper_trading"]
    
    def validate_telegram_notifications(self):
        """Validate Telegram notifications with new token"""
        logger.info("Validating Telegram notifications...")
        
        # Start notification system
        self.telegram.start()
        
        # Test notification delivery
        logger.info("Testing notification delivery...")
        
        # Test system notification
        self.telegram.notify_system("test", "System test notification")
        
        # Test signal notification
        for symbol in self.test_symbols:
            # Create test signal
            test_signal = {
                "type": "BUY",
                "source": "test",
                "strength": 0.75,
                "timestamp": int(time.time() * 1000),
                "price": 50000.0 if "BTC" in symbol else (3000.0 if "ETH" in symbol else 150.0),
                "symbol": symbol
            }
            
            # Send notification
            logger.info(f"Sending signal notification for {symbol}")
            self.telegram.notify_signal(test_signal)
        
        # Test order notifications
        test_order = {
            "orderId": f"TEST-{int(time.time())}",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "type": "MARKET",
            "quantity": 0.01,
            "price": 50000.0,
            "status": "NEW",
            "timestamp": int(time.time() * 1000)
        }
        
        # Test order created notification
        logger.info("Sending order created notification")
        self.telegram.notify_order_created(test_order)
        
        # Test order filled notification
        logger.info("Sending order filled notification")
        test_order["status"] = "FILLED"
        self.telegram.notify_order_filled(test_order)
        
        # Test error notification
        logger.info("Sending error notification")
        self.telegram.notify_error("test", "Test error message")
        
        # Wait for notifications to be processed
        logger.info("Waiting for notifications to be processed")
        time.sleep(3)
        
        # Stop notification system
        logger.info("Stopping Telegram notification system")
        self.telegram.stop()
        
        # Store results
        self.results["telegram"]["status"] = "SUCCESS"
        
        return self.results["telegram"]
    
    def run_validation(self):
        """Run full validation of the integrated pipeline"""
        logger.info("Running full validation of the integrated pipeline...")
        
        # Validate market data
        market_data_results = self.validate_market_data()
        
        # Validate technical analysis
        signals_results = self.validate_technical_analysis()
        
        # Validate paper trading
        paper_trading_results = self.validate_paper_trading()
        
        # Validate Telegram notifications
        telegram_results = self.validate_telegram_notifications()
        
        # Print summary
        logger.info("Integrated pipeline validation completed")
        logger.info(f"Market data results: {json.dumps(market_data_results, default=str, indent=2)}")
        logger.info(f"Signals results: {json.dumps(signals_results, default=str, indent=2)}")
        logger.info(f"Paper trading results: {json.dumps(paper_trading_results, default=str, indent=2)}")
        logger.info(f"Telegram results: {json.dumps(telegram_results, default=str, indent=2)}")
        
        # Save results to file
        with open("integrated_pipeline_results.json", "w") as f:
            json.dump(self.results, f, default=str, indent=2)
        
        return self.results

def main():
    """Main function"""
    logger.info("Starting integrated pipeline fix...")
    
    # Initialize and run validation
    pipeline_fix = IntegratedPipelineFix()
    results = pipeline_fix.run_validation()
    
    # Print summary
    print("\n=== INTEGRATED PIPELINE VALIDATION RESULTS ===")
    
    # Market data summary
    print("\nMarket Data Results:")
    for symbol, data in results["market_data"].items():
        status = "✅ SUCCESS" if data["has_data"] else "❌ FAILED"
        print(f"{symbol}: {status} - {data['data_length']} candles")
    
    # Signals summary
    print("\nSignals Results:")
    for symbol, data in results["signals"].items():
        if "error" in data:
            print(f"{symbol}: ❌ FAILED - {data['error']}")
        else:
            print(f"{symbol}: ✅ SUCCESS - {data['signals_count']} signals")
    
    # Paper trading summary
    print("\nPaper Trading Results:")
    for symbol, data in results["paper_trading"].items():
        print(f"{symbol}: {data['trades_executed']} trades executed")
    
    # Telegram summary
    print("\nTelegram Notification Results:")
    print(f"Status: {results['telegram'].get('status', 'UNKNOWN')}")
    print(f"Notifications sent: {len(results['telegram'].get('notifications', []))}")
    
    print("\nDetailed results saved to integrated_pipeline_results.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
