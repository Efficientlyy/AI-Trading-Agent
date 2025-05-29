"""
This script replicates the test_calculate_lag_features_rs test with detailed logging.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the indicator engine
from ai_trading_agent.agent.indicator_engine import IndicatorEngine

# Create test data similar to the test
data = pd.DataFrame({
    'close': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    'timestamp': pd.to_datetime([f'2023-01-01T00:0{i}:00Z' for i in range(6)])
}).set_index('timestamp')

# The lags we want to test
lags_to_test = [1, 3, 5]

def run_test():
    print("\n==== Starting Test Replication ====")
    
    # Set up the configuration for the engine, exactly as in the test
    test_config = {
        "features": {
            "LAG_FEATURES_RS_test": {
                "enabled": True,
                "lags_to_calculate": lags_to_test,
                "source_column": "close"
            }
        },
        "logging": {"log_level": "DEBUG"}
    }
    
    # Create the engine
    print("Creating IndicatorEngine...")
    test_engine = IndicatorEngine(test_config)
    
    # Print detailed info about the registered indicators
    print(f"Registered indicators: {list(test_engine.indicators.keys())}")
    for name, settings in test_engine.indicators.items():
        print(f"Indicator: {name}")
        for key, value in settings.items():
            if key != 'calculator':  # Skip the calculator function itself
                print(f"  {key}: {value}")
    
    # Verify the indicator is registered
    if "LAG_FEATURES_RS_test" in test_engine.indicators:
        print("✓ LAG_FEATURES_RS_test is registered in indicators")
    else:
        print("✗ LAG_FEATURES_RS_test is NOT registered in indicators")
    
    # Prepare the data
    test_data = {"TEST_SYMBOL": data.copy()}
    
    # Add detailed logging to the engine
    test_engine.logger.setLevel(logging.DEBUG)
    
    # Add a hook to print the results before they're returned
    original_calculate = test_engine.calculate_all_indicators
    
    def debug_calculate(*args, **kwargs):
        print("\n==== Before calculate_all_indicators ====")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        
        results = original_calculate(*args, **kwargs)
        
        print("\n==== After calculate_all_indicators ====")
        print(f"Results keys: {list(results.keys())}")
        for symbol, indicators in results.items():
            print(f"Symbol: {symbol}")
            print(f"  Indicator keys: {list(indicators.keys())}")
        
        return results
    
    # Replace the method with our debug version
    test_engine.calculate_all_indicators = debug_calculate
    
    # Run calculate_all_indicators as in the test
    print("\nRunning calculate_all_indicators...")
    results = test_engine.calculate_all_indicators(test_data, ["TEST_SYMBOL"])
    
    # Check the results
    print("\n==== Results Check ====")
    if "TEST_SYMBOL" in results:
        print("✓ TEST_SYMBOL found in results")
        if "LAG_FEATURES_RS_test" in results["TEST_SYMBOL"]:
            print("✓ LAG_FEATURES_RS_test found in results[TEST_SYMBOL]")
            result_df = results["TEST_SYMBOL"]["LAG_FEATURES_RS_test"]
            print(f"  Result type: {type(result_df)}")
            if hasattr(result_df, 'columns'):
                print(f"  Columns: {result_df.columns.tolist()}")
        else:
            print("✗ LAG_FEATURES_RS_test NOT found in results[TEST_SYMBOL]")
            print(f"  Available keys: {list(results['TEST_SYMBOL'].keys())}")
    else:
        print("✗ TEST_SYMBOL NOT found in results")
    
    # Debug symbol_indicators creation
    print("\n==== Debugging Symbol Indicators Creation ====")
    # Create a minimal direct test of the LAG_FEATURES_RS_ indicator calculation
    df = data.copy()
    if "LAG_FEATURES_RS_test" in test_engine.indicators:
        settings = test_engine.indicators["LAG_FEATURES_RS_test"]
        calculator = settings.get("calculator")
        if calculator:
            print(f"Found calculator for LAG_FEATURES_RS_test: {calculator.__name__ if hasattr(calculator, '__name__') else 'unknown'}")
            source_column = settings.get("source_column", "close")
            lags = settings.get("lags_to_calculate", settings.get("lags", []))
            
            print(f"Calling calculator directly with lags={lags}, source_column={source_column}")
            try:
                # This is the key part - test direct calculation
                result = calculator(df, lags, source_column)
                print(f"Direct calculation result type: {type(result)}")
                if isinstance(result, dict):
                    print(f"Result keys: {list(result.keys())}")
                    for k, v in result.items():
                        print(f"  {k}: {type(v)}, value: {v}")
                    
                    # Convert to DataFrame as in the code
                    result_df = pd.DataFrame(result)
                    print(f"Converted to DataFrame with shape: {result_df.shape}")
                    print(f"DataFrame columns: {result_df.columns.tolist()}")
                    print(f"First few rows:\n{result_df.head(3)}")
                else:
                    print(f"Result is not a dict: {result}")
            except Exception as e:
                print(f"Error during direct calculation: {e}")
        else:
            print("No calculator found for LAG_FEATURES_RS_test")
    else:
        print("LAG_FEATURES_RS_test not found in indicators")
    
    print("\n==== Test Complete ====")

if __name__ == "__main__":
    run_test()
