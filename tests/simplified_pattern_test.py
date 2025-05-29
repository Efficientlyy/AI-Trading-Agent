"""
Simple test script for pattern detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import pattern detection modules
from ai_trading_agent.agent.mock_data_generator import MockDataGenerator
from ai_trading_agent.agent.pattern_detector import PatternDetector
from ai_trading_agent.agent.pattern_types import PatternType

# Generate mock data for different pattern types
mock_generator = MockDataGenerator(config={'random_seed': 42})
start_date = datetime.now() - timedelta(days=200)

# Generate various pattern data
symbols = ["TEST_CUP_HANDLE", "TEST_WEDGE_RISING", "TEST_WEDGE_FALLING"]
market_data = {}

# Generate cup and handle pattern
market_data["TEST_CUP_HANDLE"] = mock_generator.generate_ohlcv_data(
    symbol="TEST_CUP_HANDLE",
    start_date=start_date,
    periods=200,
    interval="1d",
    base_price=100.0,
    pattern_type='cup_and_handle'
)

# Generate rising wedge pattern
market_data["TEST_WEDGE_RISING"] = mock_generator.generate_ohlcv_data(
    symbol="TEST_WEDGE_RISING",
    start_date=start_date,
    periods=200,
    interval="1d",
    base_price=100.0,
    pattern_type='wedge_rising'
)

# Generate falling wedge pattern
market_data["TEST_WEDGE_FALLING"] = mock_generator.generate_ohlcv_data(
    symbol="TEST_WEDGE_FALLING",
    start_date=start_date,
    periods=200,
    interval="1d",
    base_price=100.0,
    pattern_type='wedge_falling'
)

# Create pattern detector
detector = PatternDetector(config={
    "parameters": {
        "cup_depth_threshold": 0.001,        # Cup depth at least 0.1% of price
        "cup_symmetry_threshold": 0.2,       # Cup should be 20% symmetrical
        "min_handle_size": 1,                # Minimum 1 period for handle
        "max_handle_size": 60,               # Maximum 60 periods for handle
        "handle_retrace_threshold": 0.7,     # Handle should retrace at most 70% of cup depth
        "min_cup_duration": 3,               # Minimum cup duration
        "cup_lookback_window": 180,          # Look at most 180 bars back
        "wedge_slope_diff_threshold": 0.2,   # Maximum 20% difference in wedge slopes
        "wedge_touches_threshold": 3,        # Minimum 3 touches of trendlines
        "min_pattern_bars": 10,              # Minimum pattern duration
        "confidence_threshold": 0.5          # Minimum confidence threshold
    }
})

# Test each pattern type individually first
try:
    print("Testing individual pattern detection...")
    
    # Test cup and handle
    print("\nTesting cup and handle pattern detection:")
    cup_handle_df = market_data["TEST_CUP_HANDLE"]
    # Extract price data
    high_prices = cup_handle_df['high'].values
    low_prices = cup_handle_df['low'].values
    
    # Find peaks and troughs
    from scipy import signal
    prominence = 0.002  # 0.2% of price level
    min_distance = 3    # Minimum 3 bars between peaks
    
    peak_indices = signal.find_peaks(
        high_prices,
        prominence=prominence * np.mean(high_prices),
        distance=min_distance
    )[0]
    
    trough_indices = signal.find_peaks(
        -low_prices,  # Negate to find local minima
        prominence=prominence * np.mean(low_prices),
        distance=min_distance
    )[0]
    
    print(f"Found {len(peak_indices)} peaks and {len(trough_indices)} troughs")
    
    from ai_trading_agent.agent.cup_handle_detector import detect_cup_and_handle
    cup_handle_patterns = detect_cup_and_handle(cup_handle_df, "TEST_CUP_HANDLE", peak_indices, trough_indices, detector.params)
    
    if cup_handle_patterns:
        print(f"Detected {len(cup_handle_patterns)} cup and handle patterns")
        for i, pattern in enumerate(cup_handle_patterns):
            print(f"  Pattern {i+1}: {pattern.pattern_type.name} with confidence {pattern.confidence:.2f}")
    else:
        print("No cup and handle patterns detected")
    
    # Test wedge patterns
    print("\nTesting wedge pattern detection:")
    wedge_df = market_data["TEST_WEDGE_RISING"]
    
    # Extract price data
    high_prices = wedge_df['high'].values
    low_prices = wedge_df['low'].values
    
    # Find peaks and troughs
    peak_indices = signal.find_peaks(
        high_prices,
        prominence=prominence * np.mean(high_prices),
        distance=min_distance
    )[0]
    
    trough_indices = signal.find_peaks(
        -low_prices,  # Negate to find local minima
        prominence=prominence * np.mean(low_prices),
        distance=min_distance
    )[0]
    
    print(f"Found {len(peak_indices)} peaks and {len(trough_indices)} troughs")
    
    from ai_trading_agent.agent.wedge_detector import detect_wedges
    wedge_patterns = detect_wedges(wedge_df, "TEST_WEDGE_RISING", peak_indices, trough_indices, detector.params)
    
    if wedge_patterns:
        print(f"Detected {len(wedge_patterns)} wedge patterns")
        for i, pattern in enumerate(wedge_patterns):
            print(f"  Pattern {i+1}: {pattern.pattern_type.name} with confidence {pattern.confidence:.2f}")
    else:
        print("No wedge patterns detected")
    
    # Now test the main detector method with all pattern types
    print("\nTesting full pattern detector with all patterns:")
    try:
        # Detect patterns for all symbols
        results = detector.detect_patterns(market_data, symbols)
        
        # Print results
        print("\nPattern detection results:")
        for symbol, patterns in results.items():
            print(f"\n{symbol}:")
            if patterns:
                print(f"  Detected {len(patterns)} patterns:")
                for i, pattern in enumerate(patterns):
                    pattern_type = pattern.get("pattern_type", "Unknown")
                    confidence = pattern.get("confidence", 0.0)
                    print(f"    Pattern {i+1}: {pattern_type} with confidence {confidence:.2f}")
            else:
                print("  No patterns detected")
    
    except Exception as e:
        print(f"Error in pattern detector: {str(e)}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Error testing pattern detection: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nSimplified pattern detection test completed.")
