#!/usr/bin/env python
"""
Validate USDC Pairs Chart Generation

This script validates chart generation for USDC trading pairs.
"""

import os
import logging
from system_overseer.plugins.visualization.visualization_plugin import VisualizationPlugin
from system_overseer.config_registry import ConfigRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("usdc_pairs_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("usdc_pairs_validation")

class MockCore:
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

def validate_usdc_pairs():
    """Validate chart generation for USDC pairs."""
    logger.info("Starting USDC pairs validation")
    
    # Create mock system core
    core = MockCore()
    
    # Initialize visualization plugin
    plugin = VisualizationPlugin()
    result = plugin.initialize(core)
    
    if not result:
        logger.error("Failed to initialize Visualization Plugin")
        return False
    
    # Start plugin
    plugin.start()
    
    # Test USDC pairs
    usdc_pairs = ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
    chart_types = ["candlestick", "line", "volume"]
    intervals = ["15m"]
    
    results = {}
    
    for pair in usdc_pairs:
        pair_results = {}
        for chart_type in chart_types:
            for interval in intervals:
                logger.info(f"Testing {pair} {chart_type} {interval}...")
                
                # Generate chart
                chart_data = plugin.get_chart(pair, chart_type, interval, ["sma", "ema"])
                
                # Validate chart
                is_valid = chart_data is not None
                
                if is_valid:
                    logger.info(f"Successfully generated {chart_type} chart for {pair} {interval}")
                    
                    # Save chart to file for inspection
                    output_dir = "./test_results"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    filename = f"usdc_focus_{pair}_{chart_type}_{interval}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(chart_data)
                    
                    logger.info(f"Saved chart to {filepath}")
                else:
                    logger.error(f"Failed to generate {chart_type} chart for {pair} {interval}")
                
                pair_results[f"{chart_type}_{interval}"] = is_valid
        
        results[pair] = pair_results
    
    # Stop plugin
    plugin.stop()
    
    # Log results
    logger.info("USDC pairs validation results:")
    all_passed = True
    for pair, pair_results in results.items():
        logger.info(f"{pair}:")
        for test, passed in pair_results.items():
            logger.info(f"  {test}: {'SUCCESS' if passed else 'FAILED'}")
            if not passed:
                all_passed = False
    
    if all_passed:
        logger.info("All USDC pair tests passed!")
        return True
    else:
        logger.error("Some USDC pair tests failed!")
        return False

if __name__ == "__main__":
    validate_usdc_pairs()
