"""
Debug test for pattern detector issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Import pattern detection modules
from ai_trading_agent.agent.mock_data_generator import MockDataGenerator
from ai_trading_agent.agent.pattern_detector import PatternDetector
from ai_trading_agent.agent.pattern_types import PatternType

def debug_detect_patterns():
    """Debug implementation of detect_patterns to find the issue"""
    try:
        print("Debugging pattern detector...")
        
        # Generate mock data
        mock_generator = MockDataGenerator(config={'random_seed': 42})
        start_date = datetime.now() - timedelta(days=200)
        
        # Generate only wedge pattern data which we know works
        symbol = "TEST_WEDGE_RISING"
        
        df = mock_generator.generate_ohlcv_data(
            symbol=symbol,
            start_date=start_date,
            periods=200,
            interval="1d",
            base_price=100.0,
            pattern_type='wedge_rising'
        )
        
        market_data = {symbol: df}
        symbols = [symbol]
        
        # Create pattern detector with debug prints
        detector = PatternDetector(config={
            "parameters": {
                "wedge_slope_diff_threshold": 0.2,
                "wedge_touches_threshold": 3,
                "min_pattern_bars": 10,
                "confidence_threshold": 0.5
            }
        })
        
        print("\nStep 1: Initialize variables")
        start_time = datetime.now()
        results = {}
        total_patterns = 0
        total_confidence = 0.0
        
        print("\nStep 2: Process each symbol")
        for symbol in symbols:
            print(f"  Processing symbol: {symbol}")
            
            if symbol not in market_data:
                print(f"  No data for symbol {symbol}")
                continue
                
            df = market_data[symbol].copy()
            
            if len(df) < 20:
                print(f"  Insufficient data for {symbol} (length={len(df)})")
                continue
            
            print("  Finding peaks and troughs...")
            # Find peaks and troughs for pattern detection
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            from scipy import signal
            
            peak_prominence = np.mean(high_prices) * detector.params["peak_prominence"]
            peak_indices, _ = signal.find_peaks(
                high_prices, 
                prominence=peak_prominence,
                distance=detector.params["peak_distance"]
            )
            
            trough_prominence = np.mean(low_prices) * detector.params["peak_prominence"]
            trough_indices, _ = signal.find_peaks(
                -low_prices,  # Invert to find troughs
                prominence=trough_prominence,
                distance=detector.params["peak_distance"]
            )
            
            print(f"  Found {len(peak_indices)} peaks and {len(trough_indices)} troughs")
            
            # Initialize empty patterns list
            patterns = []
            
            # Skip all pattern detections except wedge which we know works
            print("  Detecting wedge patterns...")
            from ai_trading_agent.agent.wedge_detector import detect_wedges
            
            wedge_patterns = detect_wedges(df, symbol, peak_indices, trough_indices, detector.params)
            print(f"  Found {len(wedge_patterns)} wedge patterns")
            
            if wedge_patterns:
                for pattern in wedge_patterns:
                    print(f"    {pattern.pattern_type.name} with confidence {pattern.confidence:.2f}")
            
            # Add the wedge patterns to our patterns list
            patterns.extend(wedge_patterns)
            
            print(f"  Total patterns found: {len(patterns)}")
            
            # Add each pattern to results dictionary
            if patterns:
                print("  Converting patterns to dictionaries...")
                # Convert PatternDetectionResult objects to dictionaries
                pattern_dicts = []
                for pattern in patterns:
                    if hasattr(pattern, 'to_dict'):
                        pattern_dict = pattern.to_dict()
                        pattern_dicts.append(pattern_dict)
                        print(f"    Converted {pattern.pattern_type.name} to dict")
                    else:
                        pattern_dicts.append(pattern)
                        print(f"    Pattern was already a dict")
                        
                results[symbol] = pattern_dicts
            else:
                print("  No patterns found")
                results[symbol] = []
            
            # Update metrics
            print(f"  Updating metrics with {len(patterns)} patterns")
            total_patterns += len(patterns)
            
            if patterns:
                # Handle both PatternDetectionResult objects and dictionaries
                pattern_confidence_sum = 0
                for p in patterns:
                    if hasattr(p, 'confidence'):
                        pattern_confidence_sum += p.confidence
                        print(f"    Added confidence {p.confidence} from object")
                    elif isinstance(p, dict) and "confidence" in p:
                        pattern_confidence_sum += p["confidence"]
                        print(f"    Added confidence {p['confidence']} from dict")
                    else:
                        print(f"    Unknown pattern type: {type(p)}")
                
                total_confidence += pattern_confidence_sum
                print(f"  Total confidence sum: {pattern_confidence_sum}")
        
        # Update overall metrics
        print("\nStep 3: Finalize metrics")
        detector.metrics["patterns_detected"] = total_patterns
        detector.metrics["detection_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        
        if total_patterns > 0:
            detector.metrics["avg_confidence"] = total_confidence / total_patterns
        
        print("\nResults:")
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
                
        return results
        
    except Exception as e:
        print(f"Exception in debug_detect_patterns: {str(e)}")
        traceback.print_exc()
        return None

# Run the debug function
if __name__ == "__main__":
    print("Starting debug test...")
    result = debug_detect_patterns()
    print("\nDebug test completed.")
