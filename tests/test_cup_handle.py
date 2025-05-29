"""
Specific test for cup and handle pattern detection.

This script focuses on generating and detecting cup and handle patterns
with visualizations to help debug and improve the detection algorithm.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import pattern detection modules
from ai_trading_agent.agent.mock_data_generator import MockDataGenerator
from ai_trading_agent.agent.cup_handle_detector import detect_cup_and_handle
from ai_trading_agent.agent.pattern_types import PatternType, PatternDetectionResult

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs("test_outputs", exist_ok=True)

# Initialize mock data generator
mock_generator = MockDataGenerator(config={'random_seed': 42})

def test_cup_handle_detection(symbol="TEST"):
    """
    Generate cup and handle pattern and test detection algorithm with
    progressively relaxed parameters until detection works.
    """
    print("\nTesting cup and handle pattern detection with debug parameters")
    
    # Generate mock data
    start_date = datetime.now() - timedelta(days=200)
    df = mock_generator.generate_ohlcv_data(
        symbol=symbol,
        start_date=start_date,
        periods=200,
        interval="1d",
        base_price=100.0,
        pattern_type='cup_and_handle'
    )
    
    # Plot the raw price data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.title("Cup and Handle Mock Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig("test_outputs/cup_handle_raw.png")
    plt.close()
    
    # Extract price data for pattern detection
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # Find peak and trough indices
    from scipy import signal
    
    # Try different prominence values to find peaks and troughs
    for prominence in [0.001, 0.002, 0.005, 0.01, 0.02]:
        print(f"\nUsing prominence={prominence}")
        
        # Find peaks (local maxima)
        peak_indices = signal.find_peaks(
            high_prices,
            prominence=prominence * np.mean(high_prices),
            distance=3  # Minimum 3 bars between peaks
        )[0]
        
        # Find troughs (local minima)
        trough_indices = signal.find_peaks(
            -low_prices,  # Negate to find local minima
            prominence=prominence * np.mean(low_prices),
            distance=3  # Minimum 3 bars between troughs
        )[0]
        
        print(f"Found {len(peak_indices)} peaks and {len(trough_indices)} troughs")
        
        # Plot with peaks and troughs
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close Price')
        plt.scatter(df.index[peak_indices], high_prices[peak_indices], color='green', label='Peaks', marker='^')
        plt.scatter(df.index[trough_indices], low_prices[trough_indices], color='red', label='Troughs', marker='v')
        plt.title(f"Cup and Handle with Peaks/Troughs (prominence={prominence})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"test_outputs/cup_handle_peaks_{prominence}.png")
        plt.close()
        
        # Try different parameter sets for detection
        parameter_sets = [
            {
                "cup_depth_threshold": 0.01,  # Cup depth at least 1% of price
                "cup_symmetry_threshold": 0.3,  # Cup should be 30% symmetrical
                "min_handle_size": 3,         # Minimum 3 periods for handle
                "max_handle_size": 30,        # Maximum 30 periods for handle
                "handle_retrace_threshold": 0.4,  # Handle should retrace at most 40% of cup depth
                "min_cup_duration": 5,        # Minimum cup duration
                "cup_lookback_window": 150    # Look at most 150 bars back
            },
            {
                "cup_depth_threshold": 0.005,  # Even more relaxed depth requirement
                "cup_symmetry_threshold": 0.2,  # Even more relaxed symmetry
                "min_handle_size": 2,
                "max_handle_size": 40,
                "handle_retrace_threshold": 0.5,
                "min_cup_duration": 3,
                "cup_lookback_window": 180
            }
        ]
        
        for i, params in enumerate(parameter_sets):
            print(f"\nParameter set {i+1}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            # Detect patterns
            detected_patterns = detect_cup_and_handle(df, symbol, peak_indices, trough_indices, params)
            
            # Plot the data with detected patterns
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['close'], label='Close Price')
            plt.scatter(df.index[peak_indices], high_prices[peak_indices], color='green', label='Peaks', marker='^', alpha=0.5)
            plt.scatter(df.index[trough_indices], low_prices[trough_indices], color='red', label='Troughs', marker='v', alpha=0.5)
            
            # Mark detected patterns
            if detected_patterns:
                for pattern in detected_patterns:
                    start_idx = pattern.start_idx
                    end_idx = pattern.end_idx
                    
                    # Highlight pattern region
                    plt.axvspan(df.index[start_idx], df.index[end_idx], alpha=0.2, color='yellow')
                    
                    # Mark cup components if metadata exists
                    if hasattr(pattern, 'metadata') and pattern.metadata:
                        meta = pattern.metadata
                        if 'left_rim_idx' in meta and 'cup_bottom_idx' in meta and 'right_rim_idx' in meta:
                            left_rim_idx = meta['left_rim_idx']
                            cup_bottom_idx = meta['cup_bottom_idx']
                            right_rim_idx = meta['right_rim_idx']
                            
                            plt.plot([df.index[left_rim_idx], df.index[cup_bottom_idx], df.index[right_rim_idx]], 
                                     [high_prices[left_rim_idx], low_prices[cup_bottom_idx], high_prices[right_rim_idx]],
                                     'b-', linewidth=2, alpha=0.7)
                    
                    # Add pattern label
                    mid_idx = (start_idx + end_idx) // 2
                    y_pos = df['high'].iloc[mid_idx] * 1.05
                    plt.text(df.index[mid_idx], y_pos, 
                             f"{pattern.pattern_type.name}\nConf: {pattern.confidence:.2f}", 
                             ha='center', va='bottom', 
                             bbox=dict(facecolor='white', alpha=0.7))
            
            plt.title(f"Cup and Handle Detection (prom={prominence}, params={i+1})")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"test_outputs/cup_handle_detection_prom{prominence}_params{i+1}.png")
            plt.close()
            
            # Print results
            if detected_patterns:
                print(f"Detected {len(detected_patterns)} cup and handle patterns:")
                for i, pattern in enumerate(detected_patterns):
                    print(f"  Pattern {i+1}: {pattern.pattern_type.name} with confidence {pattern.confidence:.2f}")
                    if hasattr(pattern, 'metadata') and pattern.metadata:
                        for key, value in pattern.metadata.items():
                            print(f"    {key}: {value}")
            else:
                print("No cup and handle patterns detected with these parameters.")
    
    return detected_patterns

# Run the test
test_cup_handle_detection()
print("\nCup and handle pattern detection testing completed.")
