"""
Simple test script for cup and handle pattern detection.
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
from ai_trading_agent.agent.cup_handle_detector import detect_cup_and_handle
from ai_trading_agent.agent.pattern_types import PatternType

# Generate mock data
mock_generator = MockDataGenerator(config={'random_seed': 42})
start_date = datetime.now() - timedelta(days=200)
symbol = "TEST_CUP_HANDLE"

df = mock_generator.generate_ohlcv_data(
    symbol=symbol,
    start_date=start_date,
    periods=200,
    interval="1d",
    base_price=100.0,
    pattern_type='cup_and_handle'
)

# Extract price data for pattern detection
high_prices = df['high'].values
low_prices = df['low'].values

# Find peak and trough indices
from scipy import signal

# Set peak detection parameters
prominence = 0.002  # 0.2% of price level
min_distance = 3    # Minimum 3 bars between peaks

# Find peaks (local maxima)
peak_indices = signal.find_peaks(
    high_prices,
    prominence=prominence * np.mean(high_prices),
    distance=min_distance
)[0]

# Find troughs (local minima)
trough_indices = signal.find_peaks(
    -low_prices,  # Negate to find local minima
    prominence=prominence * np.mean(low_prices),
    distance=min_distance
)[0]

print(f"Found {len(peak_indices)} peaks and {len(trough_indices)} troughs")

# Define detection parameters
params = {
    "cup_depth_threshold": 0.005,  # Cup depth at least 0.5% of price
    "cup_symmetry_threshold": 0.2,  # Cup should be 20% symmetrical
    "min_handle_size": 2,          # Minimum 2 periods for handle
    "max_handle_size": 30,         # Maximum 30 periods for handle
    "handle_retrace_threshold": 0.5  # Handle should retrace at most 50% of cup depth
}

# Detect cup and handle patterns
detected_patterns = detect_cup_and_handle(df, symbol, peak_indices, trough_indices, params)

# Print detection results
if detected_patterns:
    print(f"Detected {len(detected_patterns)} cup and handle patterns:")
    for i, pattern in enumerate(detected_patterns):
        print(f"  Pattern {i+1}: {pattern.pattern_type.name} with confidence {pattern.confidence:.2f}")
        if hasattr(pattern, 'additional_info') and pattern.additional_info:
            for key, value in pattern.additional_info.items():
                print(f"    {key}: {value}")
else:
    print("No cup and handle patterns detected.")

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
        
        # Add pattern label
        mid_idx = (start_idx + end_idx) // 2
        y_pos = df['high'].iloc[mid_idx] * 1.05
        plt.text(df.index[mid_idx], y_pos, 
                 f"{pattern.pattern_type.name}\nConf: {pattern.confidence:.2f}", 
                 ha='center', va='bottom', 
                 bbox=dict(facecolor='white', alpha=0.7))

plt.title("Cup and Handle Pattern Detection")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# Save the plot
import os
os.makedirs("test_outputs", exist_ok=True)
plt.savefig("test_outputs/cup_handle_test.png")
plt.close()

print("\nCup and handle pattern detection test completed.")
