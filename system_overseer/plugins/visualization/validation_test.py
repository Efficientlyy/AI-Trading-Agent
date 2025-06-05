#!/usr/bin/env python
"""
Visualization Plugin Validation Test

This script performs comprehensive validation tests for the Visualization Plugin.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization_plugin_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("visualization_plugin_validation")

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

def validate_chart_image(image_data, min_size_kb=10):
    """Validate chart image data.
    
    Args:
        image_data: PNG image data
        min_size_kb: Minimum size in KB
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check if image data is not None
        if image_data is None:
            logger.error("Image data is None")
            return False
        
        # Check minimum size
        if len(image_data) < min_size_kb * 1024:
            logger.error(f"Image data too small: {len(image_data)} bytes")
            return False
        
        # Try to open image with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Check image format
        if image.format != "PNG":
            logger.error(f"Invalid image format: {image.format}")
            return False
        
        # Check image size
        if image.width < 100 or image.height < 100:
            logger.error(f"Image too small: {image.width}x{image.height}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False

def test_all_supported_intervals():
    """Test all supported intervals."""
    logger.info("Testing all supported intervals")
    
    # Create mock system core
    system_core = MockSystemCore()
    
    # Initialize visualization plugin
    plugin = VisualizationPlugin()
    result = plugin.initialize(system_core)
    
    if not result:
        logger.error("Failed to initialize Visualization Plugin")
        return False
    
    # Start plugin
    plugin.start()
    
    # Test intervals
    test_symbol = "BTCUSDT"
    test_chart_type = "candlestick"
    test_intervals = ["1m", "5m", "15m", "30m", "4h", "1d", "1w", "1M"]
    
    results = {}
    
    for interval in test_intervals:
        logger.info(f"Testing interval: {interval}")
        
        # Generate chart
        chart_data = plugin.get_chart(test_symbol, test_chart_type, interval, ["sma", "ema"])
        
        # Validate chart
        is_valid = chart_data is not None
        
        if is_valid:
            # Validate image
            is_valid = validate_chart_image(chart_data)
            
            if is_valid:
                logger.info(f"Successfully generated chart for interval: {interval}")
                
                # Save chart to file for inspection
                output_dir = "./test_results"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = f"{test_symbol}_{test_chart_type}_{interval}_validation.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(chart_data)
                
                logger.info(f"Saved chart to {filepath}")
            else:
                logger.error(f"Invalid chart image for interval: {interval}")
        else:
            logger.error(f"Failed to generate chart for interval: {interval}")
        
        results[interval] = is_valid
    
    # Stop plugin
    plugin.stop()
    
    # Log results
    logger.info("Interval validation results:")
    for interval, is_valid in results.items():
        logger.info(f"{interval}: {'SUCCESS' if is_valid else 'FAILED'}")
    
    # Check if all required intervals are supported
    required_intervals = ["1m", "15m"]
    for interval in required_intervals:
        if not results.get(interval, False):
            logger.error(f"Required interval not supported: {interval}")
            return False
    
    return True

def test_all_chart_types():
    """Test all chart types."""
    logger.info("Testing all chart types")
    
    # Create mock system core
    system_core = MockSystemCore()
    
    # Initialize visualization plugin
    plugin = VisualizationPlugin()
    result = plugin.initialize(system_core)
    
    if not result:
        logger.error("Failed to initialize Visualization Plugin")
        return False
    
    # Start plugin
    plugin.start()
    
    # Test chart types
    test_symbol = "BTCUSDT"
    test_interval = "15m"
    test_chart_types = ["candlestick", "line", "volume"]
    
    results = {}
    
    for chart_type in test_chart_types:
        logger.info(f"Testing chart type: {chart_type}")
        
        # Generate chart
        chart_data = plugin.get_chart(test_symbol, chart_type, test_interval, ["sma", "ema"])
        
        # Validate chart
        is_valid = chart_data is not None
        
        if is_valid:
            # Validate image
            is_valid = validate_chart_image(chart_data)
            
            if is_valid:
                logger.info(f"Successfully generated chart for type: {chart_type}")
                
                # Save chart to file for inspection
                output_dir = "./test_results"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = f"{test_symbol}_{chart_type}_{test_interval}_validation.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(chart_data)
                
                logger.info(f"Saved chart to {filepath}")
            else:
                logger.error(f"Invalid chart image for type: {chart_type}")
        else:
            logger.error(f"Failed to generate chart for type: {chart_type}")
        
        results[chart_type] = is_valid
    
    # Stop plugin
    plugin.stop()
    
    # Log results
    logger.info("Chart type validation results:")
    for chart_type, is_valid in results.items():
        logger.info(f"{chart_type}: {'SUCCESS' if is_valid else 'FAILED'}")
    
    # Check if all required chart types are supported
    for chart_type in test_chart_types:
        if not results.get(chart_type, False):
            logger.error(f"Required chart type not supported: {chart_type}")
            return False
    
    return True

def test_all_trading_pairs():
    """Test all trading pairs."""
    logger.info("Testing all trading pairs")
    
    # Create mock system core
    system_core = MockSystemCore()
    
    # Initialize visualization plugin
    plugin = VisualizationPlugin()
    result = plugin.initialize(system_core)
    
    if not result:
        logger.error("Failed to initialize Visualization Plugin")
        return False
    
    # Start plugin
    plugin.start()
    
    # Test trading pairs
    test_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BTCUSDC", "ETHUSDC", "SOLUSDC"]
    test_chart_type = "candlestick"
    test_interval = "15m"
    
    results = {}
    
    for pair in test_pairs:
        logger.info(f"Testing trading pair: {pair}")
        
        # Generate chart
        chart_data = plugin.get_chart(pair, test_chart_type, test_interval, ["sma", "ema"])
        
        # Validate chart
        is_valid = chart_data is not None
        
        if is_valid:
            # Validate image
            is_valid = validate_chart_image(chart_data)
            
            if is_valid:
                logger.info(f"Successfully generated chart for pair: {pair}")
                
                # Save chart to file for inspection
                output_dir = "./test_results"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = f"{pair}_{test_chart_type}_{test_interval}_validation.png"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(chart_data)
                
                logger.info(f"Saved chart to {filepath}")
            else:
                logger.error(f"Invalid chart image for pair: {pair}")
        else:
            logger.error(f"Failed to generate chart for pair: {pair}")
        
        results[pair] = is_valid
    
    # Stop plugin
    plugin.stop()
    
    # Log results
    logger.info("Trading pair validation results:")
    for pair, is_valid in results.items():
        logger.info(f"{pair}: {'SUCCESS' if is_valid else 'FAILED'}")
    
    # Check if at least one pair from each base currency is supported
    required_bases = ["BTC", "ETH", "SOL"]
    for base in required_bases:
        if not (results.get(f"{base}USDT", False) or results.get(f"{base}USDC", False)):
            logger.error(f"No supported pair found for base currency: {base}")
            return False
    
    return True

def test_error_handling():
    """Test error handling."""
    logger.info("Testing error handling")
    
    # Create mock system core
    system_core = MockSystemCore()
    
    # Initialize visualization plugin
    plugin = VisualizationPlugin()
    result = plugin.initialize(system_core)
    
    if not result:
        logger.error("Failed to initialize Visualization Plugin")
        return False
    
    # Start plugin
    plugin.start()
    
    # Test invalid symbol
    logger.info("Testing invalid symbol")
    chart_data = plugin.get_chart("INVALIDPAIR", "candlestick", "15m")
    if chart_data is not None:
        logger.error("Failed to handle invalid symbol")
        return False
    logger.info("Successfully handled invalid symbol")
    
    # Test invalid chart type
    logger.info("Testing invalid chart type")
    chart_data = plugin.get_chart("BTCUSDT", "invalid_type", "15m")
    if chart_data is not None:
        logger.error("Failed to handle invalid chart type")
        return False
    logger.info("Successfully handled invalid chart type")
    
    # Test invalid interval
    logger.info("Testing invalid interval")
    chart_data = plugin.get_chart("BTCUSDT", "candlestick", "invalid_interval")
    if chart_data is not None:
        logger.error("Failed to handle invalid interval")
        return False
    logger.info("Successfully handled invalid interval")
    
    # Stop plugin
    plugin.stop()
    
    logger.info("Error handling tests passed")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualization Plugin Validation Test")
    args = parser.parse_args()
    
    logger.info("Starting Visualization Plugin validation tests")
    
    # Run tests
    interval_test = test_all_supported_intervals()
    chart_type_test = test_all_chart_types()
    trading_pair_test = test_all_trading_pairs()
    error_handling_test = test_error_handling()
    
    # Check results
    all_passed = interval_test and chart_type_test and trading_pair_test and error_handling_test
    
    if all_passed:
        logger.info("All validation tests passed!")
        sys.exit(0)
    else:
        logger.error("Validation tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
