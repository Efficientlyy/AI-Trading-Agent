"""
Test script to verify the Rust integration with our simplified lag engine.
This should be run after the Rust module has been rebuilt.
"""

import pandas as pd
import numpy as np
from simplified_lag_engine import SimplifiedLagEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

def test_lag_features_integration():
    """Test the lag features integration with Rust."""
    logger.info("Starting lag features integration test")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'close': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    }, index=dates)
    
    # Create the engine
    engine = SimplifiedLagEngine()
    
    # Test Python implementation
    logger.info("Testing Python implementation")
    python_results = engine.calculate_lag_features_python(data, [1, 3, 5])
    
    # Test Rust implementation
    if engine.rust_available and engine.has_rust_lag_features:
        logger.info("Testing Rust implementation")
        rust_results = engine.calculate_lag_features_rust(data, [1, 3, 5])
        
        # Compare results
        logger.info("Comparing Python and Rust results")
        match_count = 0
        total_count = len(python_results)
        
        for key in python_results.keys():
            match = python_results[key].equals(rust_results[key])
            if match:
                match_count += 1
            logger.info(f"{key}: {'Match' if match else 'Mismatch'}")
        
        # Print overall result
        if match_count == total_count:
            logger.info("✅ All results match! Rust integration successful.")
        else:
            logger.warning(f"❌ Only {match_count}/{total_count} results match.")
        
        return match_count == total_count
    else:
        logger.warning("❌ Rust implementation not available, integration test failed.")
        return False

if __name__ == "__main__":
    success = test_lag_features_integration()
    
    if success:
        print("\n✅ INTEGRATION TEST PASSED: Rust and Python implementations match!")
    else:
        print("\n❌ INTEGRATION TEST FAILED: See logs for details.")
