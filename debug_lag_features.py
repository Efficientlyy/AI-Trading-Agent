"""
This script tests the lag features implementation directly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the indicator engine
from ai_trading_agent.agent.indicator_engine import IndicatorEngine

# Create test data
data = pd.DataFrame({
    'close': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    'timestamp': pd.to_datetime([f'2023-01-01T00:0{i}:00Z' for i in range(6)])
}).set_index('timestamp')

# The lags we want to test
lags_to_test = [1, 3, 5]

# Set up the configuration for the engine
config = {
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
engine = IndicatorEngine(config)

# Check what indicators were registered
print(f"Registered indicators: {list(engine.indicators.keys())}")
if "LAG_FEATURES_RS_test" in engine.indicators:
    print(f"LAG_FEATURES_RS_test indicator details: {engine.indicators['LAG_FEATURES_RS_test']}")

# Test direct method call
print("\nTesting direct method call...")
lag_features = engine._calculate_lag_features(data, lags_to_test, 'close')
print(f"Direct method result keys: {list(lag_features.keys())}")
for key, series in lag_features.items():
    print(f"{key} shape: {series.shape}, first few values: {series.head(3).tolist()}")

# Test the calculate_all_indicators method
print("\nTesting calculate_all_indicators...")
sample_data = {"TEST_SYMBOL": data.copy()}
symbols = ["TEST_SYMBOL"]

# Register the LAG_FEATURES_RS_test indicator manually if needed
if "LAG_FEATURES_RS_test" not in engine.indicators:
    print("Manually registering LAG_FEATURES_RS_test indicator...")
    engine.indicators["LAG_FEATURES_RS_test"] = {
        "calculator": engine._calculate_lag_features,
        "lags_to_calculate": lags_to_test,
        "source_column": "close",
        "enabled": True
    }
    print(f"Updated indicators: {list(engine.indicators.keys())}")

results = engine.calculate_all_indicators(sample_data, symbols)
print(f"Result keys for TEST_SYMBOL: {list(results['TEST_SYMBOL'].keys())}")

# If LAG_FEATURES_RS_test is in the results, print some details
if "LAG_FEATURES_RS_test" in results["TEST_SYMBOL"]:
    lag_df = results["TEST_SYMBOL"]["LAG_FEATURES_RS_test"]
    print(f"LAG_FEATURES_RS_test DataFrame shape: {lag_df.shape}")
    print(f"LAG_FEATURES_RS_test DataFrame columns: {lag_df.columns.tolist()}")
else:
    print("LAG_FEATURES_RS_test not found in results.")
    
    # Try to debug why it's not being included
    print("\nDebugging...")
    for symbol, indicators in results.items():
        print(f"Symbol: {symbol}, available indicators: {list(indicators.keys())}")
    
    # Check if any indicator names start with LAG_FEATURES_RS_
    for symbol, indicators in results.items():
        lag_indicators = [k for k in indicators.keys() if k.startswith("LAG_FEATURES_RS_")]
        if lag_indicators:
            print(f"Found lag indicators for {symbol}: {lag_indicators}")
