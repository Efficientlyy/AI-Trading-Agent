#!/usr/bin/env python
"""
End-to-End Pipeline Test for BTCUSDC

This module tests the complete trading pipeline with BTCUSDC only,
using all optimized and enhanced components.
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("end_to_end_test.log")
    ]
)

logger = logging.getLogger("end_to_end_test")

# Import required modules
try:
    from enhanced_mexc_client import EnhancedMEXCClient, MEXCAPIError
    from fixed_multi_asset_data_service import MultiAssetDataService
    from optimized_market_data_pipeline import OptimizedMarketDataPipeline
    from flash_trading_signals import SignalGenerator
    from fixed_paper_trading import PaperTradingSystem
    from enhanced_telegram_notifications import TelegramNotifier
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    sys.exit(1)

class EndToEndPipeline:
    """End-to-end trading pipeline for BTCUSDC"""
    
    def __init__(self, results_dir="./pipeline_results"):
        """Initialize end-to-end pipeline
        
        Args:
            results_dir: Directory to save pipeline results
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing pipeline components")
        
        # MEXC client
        self.mexc_client = EnhancedMEXCClient()
        
        # Market data pipeline
        self.market_data = OptimizedMarketDataPipeline()
        
        # Signal generator
        self.signal_generator = SignalGenerator()
        
        # Paper trading system
        self.paper_trading = PaperTradingSystem()
        
        # Telegram notifier
        self.telegram = TelegramNotifier()
        
        # Pipeline status
        self.status = "initialized"
        self.last_run_time = None
        self.last_signal = None
        self.last_trade = None
        
        # Results storage
        self.results = {
            "symbol": "BTCUSDC",
            "test_time": datetime.now().isoformat(),
            "market_data": {},
            "signals": [],
            "trades": [],
            "notifications": [],
            "errors": []
        }
        
        logger.info("End-to-end pipeline initialized")
    
    def collect_market_data(self, timeframes=None):
        """Collect market data for BTCUSDC
        
        Args:
            timeframes: List of timeframes to collect (default: ["1m", "5m", "15m", "1h"])
            
        Returns:
            dict: Collected market data
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m", "1h"]
        
        logger.info(f"Collecting market data for timeframes: {timeframes}")
        
        market_data = {}
        
        for timeframe in timeframes:
            try:
                # Get market data from optimized pipeline
                data = self.market_data.get_market_data(timeframe=timeframe)
                
                market_data[timeframe] = {
                    "candles": len(data),
                    "first_candle": data[0] if data else None,
                    "last_candle": data[-1] if data else None,
                    "success": len(data) > 0
                }
                
                logger.info(f"Collected {len(data)} candles for {timeframe}")
                
            except Exception as e:
                error_msg = f"Error collecting market data for {timeframe}: {str(e)}"
                logger.error(error_msg)
                
                market_data[timeframe] = {
                    "candles": 0,
                    "error": str(e),
                    "success": False
                }
                
                self.results["errors"].append({
                    "component": "market_data",
                    "timeframe": timeframe,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        self.results["market_data"] = market_data
        return market_data
    
    def generate_signals(self):
        """Generate trading signals based on market data
        
        Returns:
            list: Generated signals
        """
        logger.info("Generating trading signals")
        
        signals = []
        
        try:
            # Get market data for signal generation
            candles_5m = self.market_data.get_market_data(timeframe="5m", limit=100)
            candles_1h = self.market_data.get_market_data(timeframe="1h", limit=24)
            
            # Generate signals
            signal = self.signal_generator.generate_signal(
                symbol="BTC/USDC",
                candles_short=candles_5m,
                candles_long=candles_1h
            )
            
            if signal:
                signals.append(signal)
                self.last_signal = signal
                
                logger.info(f"Generated signal: {signal['direction']} for {signal['symbol']} at {signal['timestamp']}")
                
                # Send notification
                self.telegram.send_signal_notification(signal)
                
                self.results["notifications"].append({
                    "type": "signal",
                    "signal": signal,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                logger.info("No trading signals generated")
        
        except Exception as e:
            error_msg = f"Error generating signals: {str(e)}"
            logger.error(error_msg)
            
            self.results["errors"].append({
                "component": "signal_generator",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        self.results["signals"] = signals
        return signals
    
    def execute_paper_trades(self, signals):
        """Execute paper trades based on signals
        
        Args:
            signals: Trading signals
            
        Returns:
            list: Executed trades
        """
        logger.info("Executing paper trades")
        
        trades = []
        
        for signal in signals:
            try:
                # Execute paper trade
                trade = self.paper_trading.execute_trade(signal)
                
                if trade:
                    trades.append(trade)
                    self.last_trade = trade
                    
                    logger.info(f"Executed paper trade: {trade['action']} {trade['symbol']} at {trade['price']}")
                    
                    # Send notification
                    self.telegram.send_trade_notification(trade)
                    
                    self.results["notifications"].append({
                        "type": "trade",
                        "trade": trade,
                        "timestamp": datetime.now().isoformat()
                    })
            
            except Exception as e:
                error_msg = f"Error executing paper trade for signal {signal['id']}: {str(e)}"
                logger.error(error_msg)
                
                self.results["errors"].append({
                    "component": "paper_trading",
                    "signal_id": signal['id'],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        self.results["trades"] = trades
        return trades
    
    def run_pipeline(self):
        """Run the complete end-to-end pipeline
        
        Returns:
            dict: Pipeline results
        """
        logger.info("Starting end-to-end pipeline run")
        
        self.status = "running"
        self.last_run_time = datetime.now().isoformat()
        
        # Step 1: Collect market data
        market_data = self.collect_market_data()
        
        # Check if market data collection was successful
        if not any(data.get("success", False) for data in market_data.values()):
            logger.error("Market data collection failed for all timeframes")
            self.status = "failed"
            
            # Send error notification
            self.telegram.send_error_notification(
                "Market data collection failed for all timeframes",
                component="market_data"
            )
            
            self.results["notifications"].append({
                "type": "error",
                "message": "Market data collection failed for all timeframes",
                "component": "market_data",
                "timestamp": datetime.now().isoformat()
            })
            
            self.save_results()
            return self.results
        
        # Step 2: Generate signals
        signals = self.generate_signals()
        
        # Step 3: Execute paper trades
        trades = self.execute_paper_trades(signals)
        
        # Update status
        if len(self.results["errors"]) == 0:
            self.status = "success"
        else:
            self.status = "completed_with_errors"
        
        # Send system notification
        self.telegram.send_system_notification(
            f"Pipeline run completed with status: {self.status}",
            details={
                "signals": len(signals),
                "trades": len(trades),
                "errors": len(self.results["errors"])
            }
        )
        
        self.results["notifications"].append({
            "type": "system",
            "message": f"Pipeline run completed with status: {self.status}",
            "details": {
                "signals": len(signals),
                "trades": len(trades),
                "errors": len(self.results["errors"])
            },
            "timestamp": datetime.now().isoformat()
        })
        
        # Save results
        self.save_results()
        
        logger.info(f"Pipeline run completed with status: {self.status}")
        
        return self.results
    
    def save_results(self):
        """Save pipeline results to file"""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Add final status
        self.results["status"] = self.status
        self.results["end_time"] = datetime.now().isoformat()
        
        # Add component status
        self.results["component_status"] = {
            "mexc_client": self.mexc_client.get_status(),
            "market_data": self.market_data.get_status(),
            "paper_trading": self.paper_trading.get_status() if hasattr(self.paper_trading, "get_status") else {}
        }
        
        # Save results
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Pipeline results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving pipeline results: {str(e)}")
    
    def print_summary(self):
        """Print pipeline summary"""
        print("\n=== END-TO-END PIPELINE SUMMARY ===")
        print(f"Symbol: {self.results['symbol']}")
        print(f"Test Time: {self.results['test_time']}")
        print(f"Status: {self.results.get('status', 'unknown')}")
        
        print("\nMarket Data:")
        for timeframe, data in self.results["market_data"].items():
            success = "✅" if data.get("success", False) else "❌"
            candles = data.get("candles", 0)
            
            print(f"{timeframe}: {success} {candles} candles")
        
        print("\nSignals:")
        if self.results["signals"]:
            for signal in self.results["signals"]:
                print(f"- {signal['direction']} signal for {signal['symbol']} at {signal['timestamp']}")
        else:
            print("- No signals generated")
        
        print("\nTrades:")
        if self.results["trades"]:
            for trade in self.results["trades"]:
                print(f"- {trade['action']} {trade['symbol']} at {trade['price']}")
        else:
            print("- No trades executed")
        
        print("\nErrors:")
        if self.results["errors"]:
            for error in self.results["errors"]:
                print(f"- {error['component']}: {error['error']}")
        else:
            print("- No errors")
        
        print("\nDetailed results saved to JSON file in pipeline_results directory")

def main():
    """Main function"""
    logger.info("Starting end-to-end pipeline test")
    
    # Create and run pipeline
    pipeline = EndToEndPipeline()
    results = pipeline.run_pipeline()
    
    # Print summary
    pipeline.print_summary()
    
    # Return success status
    return 0 if results.get("status") == "success" else 1

if __name__ == "__main__":
    sys.exit(main())
