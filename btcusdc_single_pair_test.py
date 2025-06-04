#!/usr/bin/env python
"""
Single-Pair BTCUSDC Testing Module

This module tests the optimized market data pipeline with BTCUSDC,
ensuring rate limit compliance and proper data collection.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("btcusdc_test.log")
    ]
)

logger = logging.getLogger("btcusdc_test")

# Import optimized pipeline
try:
    from optimized_market_data_pipeline import OptimizedMarketDataPipeline
except ImportError as e:
    logger.error(f"Error importing OptimizedMarketDataPipeline: {str(e)}")
    sys.exit(1)

class BTCUSDCTester:
    """Tester for BTCUSDC market data collection"""
    
    def __init__(self, results_dir="./test_results"):
        """Initialize BTCUSDC tester
        
        Args:
            results_dir: Directory to save test results
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = OptimizedMarketDataPipeline()
        
        # Test parameters
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.test_durations = {
            "1m": 60,     # 1 hour
            "5m": 288,    # 24 hours
            "15m": 192,   # 48 hours
            "1h": 168,    # 7 days
            "4h": 180,    # 30 days
            "1d": 90      # 90 days
        }
        
        # Results storage
        self.results = {
            "symbol": "BTCUSDC",
            "test_time": datetime.now().isoformat(),
            "timeframe_results": {}
        }
        
        logger.info("BTCUSDC tester initialized")
    
    def test_timeframe(self, timeframe):
        """Test market data collection for a specific timeframe
        
        Args:
            timeframe: Timeframe to test
            
        Returns:
            dict: Test results for timeframe
        """
        logger.info(f"Testing timeframe: {timeframe}")
        
        # Get limit from test durations
        limit = self.test_durations.get(timeframe, 100)
        
        # Record start time
        start_time = time.time()
        
        # Get market data
        data = self.pipeline.get_market_data(timeframe=timeframe, limit=limit, use_cache=False)
        
        # Record end time
        end_time = time.time()
        
        # Calculate request duration
        duration = end_time - start_time
        
        # Analyze results
        result = {
            "timeframe": timeframe,
            "requested_candles": limit,
            "received_candles": len(data),
            "success": len(data) > 0,
            "duration_seconds": duration,
            "rate_per_second": 1 / duration if duration > 0 else 0,
            "first_candle": data[0] if data else None,
            "last_candle": data[-1] if data else None,
            "time_range": None
        }
        
        # Calculate time range if data available
        if data and len(data) >= 2:
            first_time = datetime.fromtimestamp(data[0]["time"] / 1000)
            last_time = datetime.fromtimestamp(data[-1]["time"] / 1000)
            time_diff = last_time - first_time
            result["time_range"] = {
                "first_time": first_time.isoformat(),
                "last_time": last_time.isoformat(),
                "days": time_diff.days,
                "hours": time_diff.seconds // 3600,
                "minutes": (time_diff.seconds % 3600) // 60
            }
        
        logger.info(f"Timeframe {timeframe} test completed: {result['received_candles']}/{result['requested_candles']} candles received in {result['duration_seconds']:.2f}s")
        
        return result
    
    def run_tests(self):
        """Run all timeframe tests
        
        Returns:
            dict: Complete test results
        """
        logger.info("Starting BTCUSDC tests")
        
        # Test each timeframe
        for timeframe in self.timeframes:
            try:
                result = self.test_timeframe(timeframe)
                self.results["timeframe_results"][timeframe] = result
                
                # Add delay between tests to respect rate limits
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error testing timeframe {timeframe}: {str(e)}")
                self.results["timeframe_results"][timeframe] = {
                    "timeframe": timeframe,
                    "error": str(e),
                    "success": False
                }
        
        # Wait for any pending requests to complete
        logger.info("Waiting for any pending requests to complete")
        self.pipeline.wait_for_queue_empty(timeout=10)
        
        # Add pipeline status to results
        self.results["pipeline_status"] = self.pipeline.get_status()
        
        # Save results
        self.save_results()
        
        logger.info("All tests completed")
        
        return self.results
    
    def save_results(self):
        """Save test results to file"""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"btcusdc_test_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Save results
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving test results: {str(e)}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n=== BTCUSDC TEST SUMMARY ===")
        print(f"Symbol: {self.results['symbol']}")
        print(f"Test Time: {self.results['test_time']}")
        print("\nTimeframe Results:")
        
        for timeframe, result in self.results["timeframe_results"].items():
            success = "✅" if result.get("success", False) else "❌"
            candles = f"{result.get('received_candles', 0)}/{result.get('requested_candles', 0)}"
            duration = f"{result.get('duration_seconds', 0):.2f}s"
            
            print(f"{timeframe}: {success} {candles} candles in {duration}")
        
        print("\nPipeline Status:")
        status = self.results.get("pipeline_status", {})
        print(f"Status: {status.get('status', 'unknown')}")
        print(f"Queue Size: {status.get('queue_size', 0)}")
        print(f"Error Counts: {status.get('error_counts', {})}")
        
        print("\nDetailed results saved to JSON file in test_results directory")

def main():
    """Main function"""
    logger.info("Starting BTCUSDC single-pair test")
    
    # Create and run tester
    tester = BTCUSDCTester()
    results = tester.run_tests()
    
    # Print summary
    tester.print_summary()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
