"""
Test script to verify that the MockDataGenerator and pattern detection work properly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ai_trading_agent.agent.mock_data_generator import MockDataGenerator, MarketPattern
from ai_trading_agent.agent.pattern_detector import PatternDetector
from datetime import datetime, timedelta
import os
import sys

# Suppress warnings about missing Rust extensions
import warnings
warnings.filterwarnings("ignore", message=".*Rust functions.*")
warnings.filterwarnings("ignore", message=".*metadata.*")

# Create output directory for plots if it doesn't exist
os.makedirs("test_outputs", exist_ok=True)

# Initialize mock data generator and pattern detector
mock_generator = MockDataGenerator()
pattern_detector = PatternDetector({
    "peak_prominence": 0.005,  # 0.5% of price level
    "peak_distance": 5,        # Minimum 5 bars between peaks
    "shoulder_height_diff_pct": 0.10,  # Maximum 10% difference between shoulders
    "triangle_min_points": 5,
    "min_pattern_bars": 10
})

def test_pattern(pattern_type, symbol="TEST"):
    """Generate mock data with a specific pattern and detect patterns in it."""
    print(f"\nTesting pattern: {pattern_type}")
    
    # Generate mock data with the specified pattern
    start_date = datetime.now() - timedelta(days=200)
    df = mock_generator.generate_ohlcv_data(
        symbol=symbol,
        start_date=start_date,
        periods=200,
        interval="1d",
        base_price=100.0,
        pattern_type=pattern_type
    )
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'])
    plt.title(f"Mock {pattern_type} Pattern")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig(f"test_outputs/{pattern_type}_mock.png")
    plt.close()
    
    # Detect patterns in the data
    market_data = {symbol: df}
    detected_patterns = pattern_detector.detect_patterns(market_data, [symbol])
    
    # Print detected patterns
    if symbol in detected_patterns and detected_patterns[symbol]:
        print(f"Detected patterns in {pattern_type} data:")
        for pattern in detected_patterns[symbol]:
            print(f"  - {pattern['pattern_type']} with confidence {pattern['confidence']:.2f}")
    else:
        print(f"No patterns detected in {pattern_type} data")
    
    return detected_patterns[symbol] if symbol in detected_patterns else []

# Test all pattern types
pattern_types = [
    "head_shoulders",
    "inverse_head_shoulders", 
    "double_top", 
    "double_bottom",
    "triangle_ascending", 
    "triangle_descending", 
    "triangle_symmetrical",
    "flag_bullish",
    "flag_bearish",
    "cup_and_handle",
    "wedge_rising",
    "wedge_falling"
]

print("Testing pattern generation and detection:")
for pattern in pattern_types:
    detected = test_pattern(pattern)

print("\nPattern detection test completed.")
