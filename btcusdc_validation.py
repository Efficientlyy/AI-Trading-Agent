#!/usr/bin/env python
"""
BTCUSDC Market Data Validation Script

This script validates the fixed MultiAssetDataService with BTCUSDC
to ensure proper symbol handling and data retrieval.
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
        logging.FileHandler("btcusdc_validation.log")
    ]
)

logger = logging.getLogger("btcusdc_validation")

# Import fixed data service
try:
    from fixed_multi_asset_data_service import MultiAssetDataService
except ImportError as e:
    logger.error(f"Error importing fixed MultiAssetDataService: {str(e)}")
    sys.exit(1)

class BTCUSDCValidator:
    """Validator for BTCUSDC market data collection"""
    
    def __init__(self, results_dir="./validation_results"):
        """Initialize BTCUSDC validator
        
        Args:
            results_dir: Directory to save validation results
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize data service with BTCUSDC only
        self.data_service = MultiAssetDataService(supported_assets=["BTC/USDC"])
        
        # Test parameters
        self.timeframes = ["1m", "5m", "15m", "60m", "4h", "1d"]
        self.limits = {
            "1m": 60,    # 1 hour
            "5m": 60,    # 5 hours
            "15m": 40,   # 10 hours
            "60m": 24,   # 1 day
            "4h": 30,    # 5 days
            "1d": 10     # 10 days
        }
        
        # Results storage
        self.results = {
            "symbol": "BTC/USDC",
            "api_symbol": "BTCUSDC",
            "validation_time": datetime.now().isoformat(),
            "endpoints": {},
            "timeframes": {}
        }
        
        logger.info("BTCUSDC validator initialized")
    
    def validate_ticker(self):
        """Validate ticker endpoint
        
        Returns:
            dict: Validation results
        """
        logger.info("Validating ticker endpoint")
        
        # Record start time
        start_time = time.time()
        
        # Get ticker
        try:
            ticker = self.data_service.get_ticker()
            success = ticker and ticker.get("price", 0) > 0
            
            result = {
                "success": success,
                "duration_seconds": time.time() - start_time,
                "data": ticker if success else None,
                "error": None if success else "Invalid ticker data"
            }
            
            logger.info(f"Ticker validation {'successful' if success else 'failed'}")
            
        except Exception as e:
            result = {
                "success": False,
                "duration_seconds": time.time() - start_time,
                "data": None,
                "error": str(e)
            }
            logger.error(f"Error validating ticker: {str(e)}")
        
        self.results["endpoints"]["ticker"] = result
        return result
    
    def validate_orderbook(self):
        """Validate orderbook endpoint
        
        Returns:
            dict: Validation results
        """
        logger.info("Validating orderbook endpoint")
        
        # Record start time
        start_time = time.time()
        
        # Get orderbook
        try:
            orderbook = self.data_service.get_orderbook()
            success = orderbook and len(orderbook.get("asks", [])) > 0 and len(orderbook.get("bids", [])) > 0
            
            result = {
                "success": success,
                "duration_seconds": time.time() - start_time,
                "data_summary": {
                    "asks_count": len(orderbook.get("asks", [])),
                    "bids_count": len(orderbook.get("bids", [])),
                    "top_ask": orderbook.get("asks", [{}])[0].get("price") if orderbook.get("asks") else None,
                    "top_bid": orderbook.get("bids", [{}])[0].get("price") if orderbook.get("bids") else None
                } if success else None,
                "error": None if success else "Invalid orderbook data"
            }
            
            logger.info(f"Orderbook validation {'successful' if success else 'failed'}")
            
        except Exception as e:
            result = {
                "success": False,
                "duration_seconds": time.time() - start_time,
                "data_summary": None,
                "error": str(e)
            }
            logger.error(f"Error validating orderbook: {str(e)}")
        
        self.results["endpoints"]["orderbook"] = result
        return result
    
    def validate_trades(self):
        """Validate trades endpoint
        
        Returns:
            dict: Validation results
        """
        logger.info("Validating trades endpoint")
        
        # Record start time
        start_time = time.time()
        
        # Get trades
        try:
            trades = self.data_service.get_trades()
            success = trades and len(trades) > 0
            
            result = {
                "success": success,
                "duration_seconds": time.time() - start_time,
                "data_summary": {
                    "trades_count": len(trades),
                    "latest_trade_price": trades[0].get("price") if trades else None,
                    "latest_trade_time": datetime.fromtimestamp(trades[0].get("time", 0) / 1000).isoformat() if trades else None
                } if success else None,
                "error": None if success else "Invalid trades data"
            }
            
            logger.info(f"Trades validation {'successful' if success else 'failed'}")
            
        except Exception as e:
            result = {
                "success": False,
                "duration_seconds": time.time() - start_time,
                "data_summary": None,
                "error": str(e)
            }
            logger.error(f"Error validating trades: {str(e)}")
        
        self.results["endpoints"]["trades"] = result
        return result
    
    def validate_klines(self, timeframe):
        """Validate klines endpoint for a specific timeframe
        
        Args:
            timeframe: Timeframe to validate
            
        Returns:
            dict: Validation results
        """
        logger.info(f"Validating klines endpoint for timeframe {timeframe}")
        
        # Get limit from test parameters
        limit = self.limits.get(timeframe, 10)
        
        # Record start time
        start_time = time.time()
        
        # Get klines
        try:
            klines = self.data_service.get_klines(interval=timeframe, limit=limit)
            success = klines and len(klines) > 0
            
            # Calculate time range if data available
            time_range = None
            if success and len(klines) >= 2:
                first_time = datetime.fromtimestamp(klines[0]["time"] / 1000)
                last_time = datetime.fromtimestamp(klines[-1]["time"] / 1000)
                time_diff = last_time - first_time
                time_range = {
                    "first_time": first_time.isoformat(),
                    "last_time": last_time.isoformat(),
                    "days": time_diff.days,
                    "hours": time_diff.seconds // 3600,
                    "minutes": (time_diff.seconds % 3600) // 60
                }
            
            result = {
                "timeframe": timeframe,
                "success": success,
                "duration_seconds": time.time() - start_time,
                "requested_candles": limit,
                "received_candles": len(klines),
                "data_summary": {
                    "first_candle": {
                        "time": datetime.fromtimestamp(klines[0]["time"] / 1000).isoformat(),
                        "open": klines[0]["open"],
                        "high": klines[0]["high"],
                        "low": klines[0]["low"],
                        "close": klines[0]["close"],
                        "volume": klines[0]["volume"]
                    } if klines else None,
                    "last_candle": {
                        "time": datetime.fromtimestamp(klines[-1]["time"] / 1000).isoformat(),
                        "open": klines[-1]["open"],
                        "high": klines[-1]["high"],
                        "low": klines[-1]["low"],
                        "close": klines[-1]["close"],
                        "volume": klines[-1]["volume"]
                    } if klines else None,
                    "time_range": time_range
                } if success else None,
                "error": None if success else "Invalid klines data"
            }
            
            logger.info(f"Klines validation for {timeframe} {'successful' if success else 'failed'}: {len(klines)}/{limit} candles")
            
        except Exception as e:
            result = {
                "timeframe": timeframe,
                "success": False,
                "duration_seconds": time.time() - start_time,
                "requested_candles": limit,
                "received_candles": 0,
                "data_summary": None,
                "error": str(e)
            }
            logger.error(f"Error validating klines for {timeframe}: {str(e)}")
        
        self.results["timeframes"][timeframe] = result
        return result
    
    def validate_exchange_info(self):
        """Validate exchange info endpoint
        
        Returns:
            dict: Validation results
        """
        logger.info("Validating exchange info endpoint")
        
        # Record start time
        start_time = time.time()
        
        # Get exchange info
        try:
            exchange_info = self.data_service.get_exchange_info()
            success = exchange_info and "symbols" in exchange_info
            
            # Find BTCUSDC symbol info
            btcusdc_info = None
            if success:
                for symbol_info in exchange_info.get("symbols", []):
                    if symbol_info.get("symbol") == "BTCUSDC":
                        btcusdc_info = symbol_info
                        break
            
            result = {
                "success": success and btcusdc_info is not None,
                "duration_seconds": time.time() - start_time,
                "data_summary": {
                    "symbols_count": len(exchange_info.get("symbols", [])),
                    "btcusdc_found": btcusdc_info is not None,
                    "btcusdc_status": btcusdc_info.get("status") if btcusdc_info else None,
                    "btcusdc_base_asset": btcusdc_info.get("baseAsset") if btcusdc_info else None,
                    "btcusdc_quote_asset": btcusdc_info.get("quoteAsset") if btcusdc_info else None
                } if success else None,
                "error": None if success and btcusdc_info else "BTCUSDC symbol not found in exchange info"
            }
            
            logger.info(f"Exchange info validation {'successful' if result['success'] else 'failed'}")
            
        except Exception as e:
            result = {
                "success": False,
                "duration_seconds": time.time() - start_time,
                "data_summary": None,
                "error": str(e)
            }
            logger.error(f"Error validating exchange info: {str(e)}")
        
        self.results["endpoints"]["exchange_info"] = result
        return result
    
    def run_validation(self):
        """Run all validation tests
        
        Returns:
            dict: Complete validation results
        """
        logger.info("Starting BTCUSDC validation")
        
        # Validate basic endpoints
        self.validate_ticker()
        time.sleep(0.5)  # Add delay to respect rate limits
        
        self.validate_orderbook()
        time.sleep(0.5)
        
        self.validate_trades()
        time.sleep(0.5)
        
        self.validate_exchange_info()
        time.sleep(0.5)
        
        # Validate klines for each timeframe
        for timeframe in self.timeframes:
            self.validate_klines(timeframe)
            time.sleep(1)  # Add longer delay between kline requests
        
        # Calculate overall success
        endpoint_success = all(result.get("success", False) for result in self.results["endpoints"].values())
        timeframe_success = all(result.get("success", False) for result in self.results["timeframes"].values())
        
        self.results["overall_success"] = endpoint_success and timeframe_success
        
        # Save results
        self.save_results()
        
        logger.info(f"All validation tests completed: {'SUCCESS' if self.results['overall_success'] else 'FAILURE'}")
        
        return self.results
    
    def save_results(self):
        """Save validation results to file"""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"btcusdc_validation_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Save results
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Validation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving validation results: {str(e)}")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n=== BTCUSDC VALIDATION SUMMARY ===")
        print(f"Symbol: {self.results['symbol']} (API: {self.results['api_symbol']})")
        print(f"Validation Time: {self.results['validation_time']}")
        print(f"Overall Success: {'✅' if self.results['overall_success'] else '❌'}")
        
        print("\nEndpoint Results:")
        for endpoint, result in self.results["endpoints"].items():
            success = "✅" if result.get("success", False) else "❌"
            duration = f"{result.get('duration_seconds', 0):.2f}s"
            
            print(f"{endpoint}: {success} in {duration}")
        
        print("\nTimeframe Results:")
        for timeframe, result in self.results["timeframes"].items():
            success = "✅" if result.get("success", False) else "❌"
            candles = f"{result.get('received_candles', 0)}/{result.get('requested_candles', 0)}"
            duration = f"{result.get('duration_seconds', 0):.2f}s"
            
            print(f"{timeframe}: {success} {candles} candles in {duration}")
        
        print("\nDetailed results saved to JSON file in validation_results directory")

def main():
    """Main function"""
    logger.info("Starting BTCUSDC validation")
    
    # Create and run validator
    validator = BTCUSDCValidator()
    results = validator.run_validation()
    
    # Print summary
    validator.print_summary()
    
    # Return success status
    return 0 if results["overall_success"] else 1

if __name__ == "__main__":
    sys.exit(main())
