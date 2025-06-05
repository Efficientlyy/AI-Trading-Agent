#!/usr/bin/env python
"""
Visualization Plugin Integration Test

This script tests the integration of the Visualization Plugin with the System Overseer.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization_plugin_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("visualization_plugin_test")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import system overseer components
from system_overseer.config_registry import ConfigRegistry
from system_overseer.plugins.visualization.visualization_plugin import VisualizationPlugin
from system_overseer.plugins.visualization.data_providers.mexc import MexcDataProvider

class MockSystemCore:
    """Mock System Core for testing."""
    
    def __init__(self):
        """Initialize Mock System Core."""
        self.config_registry = ConfigRegistry()
        self.services = {
            "config_registry": self.config_registry
        }
        self.data_dir = "./data"
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_service(self, service_name):
        """Get service by name."""
        return self.services.get(service_name)

def test_visualization_plugin():
    """Test Visualization Plugin integration."""
    logger.info("Starting Visualization Plugin integration test")
    
    try:
        # Create mock system core
        system_core = MockSystemCore()
        
        # Initialize visualization plugin
        plugin = VisualizationPlugin()
        result = plugin.initialize(system_core)
        
        if not result:
            logger.error("Failed to initialize Visualization Plugin")
            return False
        
        logger.info("Visualization Plugin initialized successfully")
        
        # Start plugin
        result = plugin.start()
        if not result:
            logger.error("Failed to start Visualization Plugin")
            return False
        
        logger.info("Visualization Plugin started successfully")
        
        # Test chart generation
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        test_chart_types = ["candlestick", "line", "volume"]
        test_intervals = ["1m", "15m", "1h"]
        
        for symbol in test_symbols:
            for chart_type in test_chart_types:
                for interval in test_intervals:
                    logger.info(f"Testing chart generation: {symbol} {chart_type} {interval}")
                    
                    # Generate chart
                    chart_data = plugin.get_chart(symbol, chart_type, interval, ["sma", "ema"])
                    
                    if chart_data:
                        logger.info(f"Successfully generated {chart_type} chart for {symbol} {interval}")
                        
                        # Save chart to file for inspection
                        output_dir = "./test_results"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        filename = f"{symbol}_{chart_type}_{interval}.png"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(chart_data)
                        
                        logger.info(f"Saved chart to {filepath}")
                    else:
                        logger.error(f"Failed to generate {chart_type} chart for {symbol} {interval}")
        
        # Stop plugin
        plugin.stop()
        logger.info("Visualization Plugin stopped successfully")
        
        logger.info("Visualization Plugin integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in Visualization Plugin integration test: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualization Plugin Integration Test")
    args = parser.parse_args()
    
    # Run test
    success = test_visualization_plugin()
    
    if success:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
