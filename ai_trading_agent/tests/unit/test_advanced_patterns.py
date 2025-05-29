"""
Unit tests for Advanced Pattern Detection.

Tests the pattern detection capabilities of the advanced pattern detector.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai_trading_agent.patterns.advanced_patterns import AdvancedPatternDetector

class TestAdvancedPatternDetector(unittest.TestCase):
    """Test the Advanced Pattern Detector functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = AdvancedPatternDetector({
            "confidence_threshold": 0.6
        })
        
    def generate_test_data(self, pattern_type=None):
        """
        Generate test data with specific patterns.
        
        Args:
            pattern_type: Type of pattern to generate
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate base data
        dates = pd.date_range(start='2025-01-01', periods=100)
        data = {
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(101, 5, 100),
            'volume': np.random.normal(1000000, 500000, 100),
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Modify data to create specific patterns
        if pattern_type == "head_and_shoulders":
            # Create head and shoulders pattern in the last 30 days
            for i in range(70, 100):
                if i < 75:  # Left shoulder
                    df.loc[dates[i], 'close'] = 110 + (i - 70) * 2
                elif i < 80:  # Down to head
                    df.loc[dates[i], 'close'] = 120 - (i - 75) * 2
                elif i < 85:  # Head
                    df.loc[dates[i], 'close'] = 110 + (i - 80) * 3
                elif i < 90:  # Down to right shoulder
                    df.loc[dates[i], 'close'] = 125 - (i - 85) * 3
                else:  # Right shoulder
                    df.loc[dates[i], 'close'] = 110 + (i - 90) * 2 - (i - 95) * 4 if i >= 95 else 110 + (i - 90) * 2
        
        elif pattern_type == "double_bottom":
            # Create double bottom pattern in the last 30 days
            for i in range(70, 100):
                if i < 80:  # First bottom
                    df.loc[dates[i], 'close'] = 110 - (i - 70) * 2 if i < 75 else 100 + (i - 75) * 2
                elif i < 90:  # Second bottom
                    df.loc[dates[i], 'close'] = 110 - (i - 80) * 2 if i < 85 else 100 + (i - 85) * 2
                else:  # Confirmation rise
                    df.loc[dates[i], 'close'] = 110 + (i - 90) * 1.5
        
        elif pattern_type == "ascending_triangle":
            # Create ascending triangle pattern in the last 30 days
            resistance = 120
            for i in range(70, 100):
                if i % 5 < 3:  # Approaching resistance
                    df.loc[dates[i], 'close'] = resistance - (3 - i % 5) * 2
                    df.loc[dates[i], 'high'] = resistance
                else:  # Falling to higher lows
                    support = 105 + (i - 70) // 5 * 2
                    df.loc[dates[i], 'close'] = support + (i % 5 - 3) * 3
                    df.loc[dates[i], 'low'] = support
        
        # Adjust high/low values to be consistent with open/close
        for i in range(len(df)):
            df.iloc[i, df.columns.get_loc('high')] = max(
                df.iloc[i, df.columns.get_loc('high')],
                df.iloc[i, df.columns.get_loc('open')],
                df.iloc[i, df.columns.get_loc('close')]
            )
            df.iloc[i, df.columns.get_loc('low')] = min(
                df.iloc[i, df.columns.get_loc('low')],
                df.iloc[i, df.columns.get_loc('open')],
                df.iloc[i, df.columns.get_loc('close')]
            )
            
        return df
    
    def test_pattern_detection_initialization(self):
        """Test that the pattern detector initializes correctly."""
        # Check default configuration
        self.assertEqual(self.detector.config['confidence_threshold'], 0.6)
        
        # Create with custom configuration
        custom_detector = AdvancedPatternDetector({
            "confidence_threshold": 0.8
        })
        self.assertEqual(custom_detector.config['confidence_threshold'], 0.8)
    
    def test_head_and_shoulders_detection(self):
        """Test detection of head and shoulders pattern."""
        # Generate data with head and shoulders pattern
        df = self.generate_test_data("head_and_shoulders")
        
        # Detect patterns
        patterns = self.detector.detect_all_patterns(df)
        
        # Filter for head and shoulders patterns
        head_and_shoulders = [p for p in patterns if p['pattern'] == 'head_and_shoulders']
        
        # Should detect at least one instance
        self.assertGreaterEqual(len(head_and_shoulders), 1)
        
        # Check pattern details
        pattern = head_and_shoulders[0]
        self.assertEqual(pattern['direction'], 'bearish')
        self.assertGreaterEqual(pattern['confidence'], 0.6)
        self.assertGreaterEqual(len(pattern['candles']), 20)  # Pattern spans at least 20 candles
    
    def test_double_bottom_detection(self):
        """Test detection of double bottom pattern."""
        # Generate data with double bottom pattern
        df = self.generate_test_data("double_bottom")
        
        # Detect patterns
        patterns = self.detector.detect_all_patterns(df)
        
        # Filter for double bottom patterns
        double_bottoms = [p for p in patterns if p['pattern'] == 'double_bottom']
        
        # Should detect at least one instance
        self.assertGreaterEqual(len(double_bottoms), 1)
        
        # Check pattern details
        pattern = double_bottoms[0]
        self.assertEqual(pattern['direction'], 'bullish')
        self.assertGreaterEqual(pattern['confidence'], 0.6)
        self.assertGreaterEqual(len(pattern['candles']), 15)  # Pattern spans at least 15 candles
    
    def test_ascending_triangle_detection(self):
        """Test detection of ascending triangle pattern."""
        # Generate data with ascending triangle pattern
        df = self.generate_test_data("ascending_triangle")
        
        # Detect patterns
        patterns = self.detector.detect_all_patterns(df)
        
        # Filter for ascending triangle patterns
        triangles = [p for p in patterns if p['pattern'] == 'ascending_triangle']
        
        # Should detect at least one instance
        self.assertGreaterEqual(len(triangles), 1)
        
        # Check pattern details
        pattern = triangles[0]
        self.assertEqual(pattern['direction'], 'bullish')
        self.assertGreaterEqual(pattern['confidence'], 0.6)
        self.assertGreaterEqual(len(pattern['candles']), 10)  # Pattern spans at least 10 candles
    
    def test_no_pattern_detection(self):
        """Test that no patterns are detected in random data."""
        # Generate random data with no specific patterns
        df = self.generate_test_data()
        
        # Set a high confidence threshold to ensure random patterns aren't detected
        high_confidence_detector = AdvancedPatternDetector({
            "confidence_threshold": 0.9
        })
        
        # Detect patterns
        patterns = high_confidence_detector.detect_all_patterns(df)
        
        # Should detect few or no patterns in random data with high threshold
        self.assertLessEqual(len(patterns), 2)
    
    def test_pattern_confidence_scoring(self):
        """Test that pattern confidence scoring works correctly."""
        # Generate data with a clear pattern
        df = self.generate_test_data("head_and_shoulders")
        
        # Detect with different confidence thresholds
        low_confidence = AdvancedPatternDetector({"confidence_threshold": 0.4})
        medium_confidence = AdvancedPatternDetector({"confidence_threshold": 0.7})
        high_confidence = AdvancedPatternDetector({"confidence_threshold": 0.9})
        
        low_patterns = low_confidence.detect_all_patterns(df)
        medium_patterns = medium_confidence.detect_all_patterns(df)
        high_patterns = high_confidence.detect_all_patterns(df)
        
        # Higher thresholds should result in fewer patterns
        self.assertGreaterEqual(len(low_patterns), len(medium_patterns))
        self.assertGreaterEqual(len(medium_patterns), len(high_patterns))
    
    def test_pattern_metadata(self):
        """Test that pattern metadata is correctly generated."""
        # Generate data with a pattern
        df = self.generate_test_data("double_bottom")
        
        # Detect patterns
        patterns = self.detector.detect_all_patterns(df)
        
        # Check that each pattern has required metadata
        for pattern in patterns:
            self.assertIn('pattern', pattern)
            self.assertIn('position', pattern)
            self.assertIn('direction', pattern)
            self.assertIn('confidence', pattern)
            self.assertIn('candles', pattern)
            self.assertIn('metadata', pattern)
            
            # Check metadata details
            metadata = pattern['metadata']
            self.assertIn('detection_method', metadata)
            self.assertIn('confirmation_candles', metadata)
            self.assertIn('measurement', metadata)
            
            # Check measurement details
            measurement = metadata['measurement']
            self.assertIn('price_target', measurement)
            self.assertIn('risk_level', measurement)
            self.assertIn('pattern_height', measurement)


if __name__ == "__main__":
    unittest.main()
