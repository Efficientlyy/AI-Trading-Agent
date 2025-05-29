"""
Basic pattern detection test script.

This simple script tests the cup and handle and wedge pattern detectors
without requiring Rust extensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import pattern detection modules
from ai_trading_agent.agent.mock_data_generator import MockDataGenerator, MarketPattern
from ai_trading_agent.agent.cup_handle_detector import detect_cup_and_handle
from ai_trading_agent.agent.wedge_detector import detect_wedges
from ai_trading_agent.agent.pattern_types import PatternType, PatternDetectionResult

# Create output directory
os.makedirs("test_outputs", exist_ok=True)

# Initialize mock data generator
mock_generator = MockDataGenerator()

def test_pattern_detection(pattern_type: str, detector_func, symbol="TEST"):
    """
    Test a specific pattern detector with mock data.
    
    Args:
        pattern_type: Type of pattern to generate and test
        detector_func: Pattern detector function to test
        symbol: Symbol to use for the test
    """
    print(f"\nTesting {pattern_type} pattern detection")
    
    # Generate mock data
    start_date = datetime.now() - timedelta(days=200)
    df = mock_generator.generate_ohlcv_data(
        symbol=symbol,
        start_date=start_date,
        periods=200,
        interval="1d",
        base_price=100.0,
        pattern_type=pattern_type
    )
    
    # Extract price data for pattern detection
    # Find peak and trough indices for pattern detection
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # Parameters
    params = {
        "peak_prominence": 0.005,  # 0.5% of price level
        "peak_distance": 5,        # Minimum 5 bars between peaks
        "cup_depth_threshold": 0.03,  # Cup depth at least 3% of price
        "cup_symmetry_threshold": 0.7,  # Cup should be 70% symmetrical
        "wedge_slope_diff_threshold": 0.2,  # Maximum 20% difference in wedge slopes
        "wedge_touches_threshold": 3,  # Minimum 3 touches of trendlines
    }
    
    # Find peaks (local maxima)
    from scipy import signal
    peak_indices = signal.find_peaks(
        high_prices,
        prominence=params["peak_prominence"] * np.mean(high_prices),
        distance=params["peak_distance"]
    )[0]
    
    # Find troughs (local minima)
    trough_indices = signal.find_peaks(
        -low_prices,  # Negate to find local minima
        prominence=params["peak_prominence"] * np.mean(low_prices),
        distance=params["peak_distance"]
    )[0]
    
    # Detect patterns
    detected_patterns = detector_func(df, symbol, peak_indices, trough_indices, params)
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price')
    
    # Mark peaks and troughs
    plt.scatter(df.index[peak_indices], high_prices[peak_indices], color='green', label='Peaks', marker='^')
    plt.scatter(df.index[trough_indices], low_prices[trough_indices], color='red', label='Troughs', marker='v')
    
    # Mark detected patterns
    if detected_patterns:
        for pattern in detected_patterns:
            start_idx = pattern.start_idx
            end_idx = pattern.end_idx
            
            # Highlight pattern region
            plt.axvspan(df.index[start_idx], df.index[end_idx], alpha=0.2, color='yellow')
            
            # Add pattern label
            mid_idx = (start_idx + end_idx) // 2
            y_pos = df['high'].iloc[mid_idx] * 1.05
            plt.text(df.index[mid_idx], y_pos, 
                     f"{pattern.pattern_type.name}\nConf: {pattern.confidence:.2f}", 
                     ha='center', va='bottom', 
                     bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f"{pattern_type} Pattern Test")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"test_outputs/{pattern_type}_detection_test.png")
    plt.close()
    
    # Print results
    if detected_patterns:
        print(f"Detected {len(detected_patterns)} {pattern_type} patterns:")
        for i, pattern in enumerate(detected_patterns):
            print(f"  Pattern {i+1}: {pattern.pattern_type.name} with confidence {pattern.confidence:.2f}")
            if hasattr(pattern, 'metadata') and pattern.metadata:
                for key, value in pattern.metadata.items():
                    print(f"    {key}: {value}")
    else:
        print(f"No {pattern_type} patterns detected.")
    
    return detected_patterns

# Test cup and handle detection
test_pattern_detection('cup_and_handle', detect_cup_and_handle)

# Test wedge pattern detection
test_pattern_detection('wedge_rising', detect_wedges)
test_pattern_detection('wedge_falling', detect_wedges)

print("\nPattern detection tests completed.")
