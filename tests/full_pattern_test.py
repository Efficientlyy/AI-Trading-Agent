"""
Comprehensive pattern detection test for all implemented patterns.

This script tests all pattern detectors including the newly added
cup and handle and wedge patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import pattern detection modules
from ai_trading_agent.agent.pattern_detector import PatternDetector
from ai_trading_agent.agent.mock_data_generator import MockDataGenerator, MarketPattern
from ai_trading_agent.agent.pattern_types import PatternType

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs("test_outputs", exist_ok=True)

# Initialize mock data generator and pattern detector
mock_generator = MockDataGenerator(config={'random_seed': 42})
detector = PatternDetector({
    "peak_prominence": 0.002,  # 0.2% of price level (more sensitive)
    "peak_distance": 3,        # Minimum 3 bars between peaks
    "cup_depth_threshold": 0.005,  # Cup depth at least 0.5% of price
    "cup_symmetry_threshold": 0.2,  # Cup should be 20% symmetrical
    "min_handle_size": 2,      # Minimum 2 periods for handle
    "handle_retrace_threshold": 0.5  # Handle should retrace at most 50% of cup depth
})

def test_all_patterns():
    """
    Generate and test detection for all pattern types.
    """
    pattern_types = [
        'head_shoulders',
        'inverse_head_shoulders',
        'double_top',
        'double_bottom',
        'triangle_ascending',
        'triangle_descending',
        'triangle_symmetrical',
        'flag_bullish',
        'flag_bearish',
        'cup_and_handle',
        'wedge_rising',
        'wedge_falling'
    ]
    
    results = {}
    
    print("\nTesting all pattern detection algorithms:")
    
    for pattern_type in pattern_types:
        print(f"\nGenerating and detecting {pattern_type} pattern...")
        
        # Generate mock data
        symbol = f"TEST_{pattern_type.upper()}"
        start_date = datetime.now() - timedelta(days=200)
        df = mock_generator.generate_ohlcv_data(
            symbol=symbol,
            start_date=start_date,
            periods=200,
            interval="1d",
            base_price=100.0,
            pattern_type=pattern_type
        )
        
        # Detect patterns using the pattern detector
        market_data = {symbol: df}
        detected_patterns = detector.detect_patterns(market_data, [symbol])
        
        # Save results
        results[pattern_type] = detected_patterns[symbol] if symbol in detected_patterns else []
        
        # Plot the data with detected patterns
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close Price')
        
        # Mark detected patterns
        if symbol in detected_patterns and detected_patterns[symbol]:
            for pattern in detected_patterns[symbol]:
                if 'start_idx' in pattern and 'end_idx' in pattern:
                    start_idx = pattern['start_idx']
                    end_idx = pattern['end_idx']
                    
                    # Highlight pattern region
                    plt.axvspan(df.index[start_idx], df.index[end_idx], alpha=0.2, color='yellow')
                    
                    # Add pattern label
                    mid_idx = (start_idx + end_idx) // 2
                    y_pos = df['high'].iloc[mid_idx] * 1.05
                    plt.text(df.index[mid_idx], y_pos, 
                             f"{pattern['pattern']}\nConf: {pattern['confidence']:.2f}", 
                             ha='center', va='bottom', 
                             bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title(f"{pattern_type.replace('_', ' ').title()} Pattern Detection")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.savefig(f"test_outputs/{pattern_type}_complete_test.png")
        plt.close()
        
        # Print detection results
        if symbol in detected_patterns and detected_patterns[symbol]:
            patterns = detected_patterns[symbol]
            print(f"  Detected {len(patterns)} patterns:")
            for i, pattern in enumerate(patterns):
                print(f"    Pattern {i+1}: {pattern['pattern']} with confidence {pattern['confidence']:.2f}")
        else:
            print(f"  No {pattern_type} patterns detected.")
    
    # Overall summary
    print("\nSummary of Pattern Detection:")
    for pattern_type, patterns in results.items():
        print(f"  {pattern_type.replace('_', ' ').title()}: {len(patterns)} patterns detected")
    
    return results

# Run the test
test_results = test_all_patterns()
print("\nAll pattern detection tests completed.")
