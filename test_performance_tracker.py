"""
Test script for verifying the PerformanceTracker functionality.

This script tests the PerformanceTracker's ability to load and process trade data
from the data/trades directory.
"""

import json
import logging
from pathlib import Path
from src.common.performance import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_performance_tracker():
    """Test the PerformanceTracker's ability to load and process trade data."""
    logger.info("Starting PerformanceTracker test")
    
    # Create a PerformanceTracker instance
    tracker = PerformanceTracker()
    
    # Check if trade files exist
    trades_dir = Path("data/trades")
    trade_files = list(trades_dir.glob("*.json"))
    logger.info(f"Found {len(trade_files)} trade files in {trades_dir}")
    
    if not trade_files:
        logger.warning("No trade files found. Test will be limited.")
    
    # Get performance summary
    logger.info("Getting performance summary...")
    summary = tracker.get_performance_summary()
    logger.info(f"Performance summary: {json.dumps(summary, indent=2)}")
    
    # Get detailed metrics
    logger.info("Getting detailed performance metrics...")
    metrics = tracker.get_performance_metrics()
    
    # Log key metrics
    if metrics.get("strategy_performance"):
        logger.info(f"Strategy performance: {len(metrics['strategy_performance'])} strategies found")
        for strategy in metrics["strategy_performance"]:
            logger.info(f"  - {strategy['strategy']}: {strategy['win_rate']:.2f}% win rate, {strategy['total_trades']} trades")
    
    if metrics.get("asset_performance"):
        logger.info(f"Asset performance: {len(metrics['asset_performance'])} assets found")
        for asset in metrics["asset_performance"]:
            logger.info(f"  - {asset['asset']}: {asset['win_rate']:.2f}% win rate, {asset['total_trades']} trades")
    
    if metrics.get("recent_trades"):
        logger.info(f"Recent trades: {len(metrics['recent_trades'])} trades found")
    
    if metrics.get("equity_curve") and metrics["equity_curve"].get("equity"):
        logger.info(f"Equity curve: {len(metrics['equity_curve']['equity'])} data points")
        
    logger.info("PerformanceTracker test completed successfully")
    return True

if __name__ == "__main__":
    try:
        success = test_performance_tracker()
        print(f"\nTest {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        print("\nTest FAILED with exception")
