"""
Simple test script for wedge pattern detection.
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
from ai_trading_agent.agent.wedge_detector import detect_wedges
from ai_trading_agent.agent.pattern_types import PatternType

# Generate mock data
mock_generator = MockDataGenerator(config={'random_seed': 42})
start_date = datetime.now() - timedelta(days=200)
symbol = "TEST_WEDGE_RISING"

df = mock_generator.generate_ohlcv_data(
    symbol=symbol,
    start_date=start_date,
    periods=200,
    interval="1d",
    base_price=100.0,
    pattern_type='wedge_rising'
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
    "wedge_slope_diff_threshold": 0.2,  # Maximum 20% difference in wedge slopes
    "wedge_touches_threshold": 3,      # Minimum 3 touches of trendlines
    "min_pattern_bars": 10,            # Minimum pattern duration
    "confidence_threshold": 0.5        # Minimum confidence threshold
}

# Detect wedge patterns
detected_patterns = detect_wedges(df, symbol, peak_indices, trough_indices, params)

# Print detection results
if detected_patterns:
    print(f"Detected {len(detected_patterns)} wedge patterns:")
    for i, pattern in enumerate(detected_patterns):
        print(f"  Pattern {i+1}: {pattern.pattern_type.name} with confidence {pattern.confidence:.2f}")
        if hasattr(pattern, 'additional_info') and pattern.additional_info:
            for key, value in pattern.additional_info.items():
                print(f"    {key}: {value}")
else:
    print("No wedge patterns detected.")

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

plt.title("Wedge Pattern Detection")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# Save the plot
import os
os.makedirs("test_outputs", exist_ok=True)
plt.savefig("test_outputs/wedge_test.png")
plt.close()

print("\nWedge pattern detection test completed.")
