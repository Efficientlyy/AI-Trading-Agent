"""
Mock Data Generator for Technical Analysis

This module provides functionality to generate realistic market data with
predefined patterns for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import random
from enum import Enum, auto


class MarketPattern(Enum):
    """Enumeration of market patterns that can be generated for testing."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    CUP_AND_HANDLE = "cup_and_handle"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"

class MockDataGenerator:
    """
    Generates realistic OHLCV data with embedded technical patterns
    for testing and demonstration purposes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mock data generator with configuration.
        
        Args:
            config: Configuration dictionary with parameters for data generation
        """
        self.config = config or {}
        self.default_volatility = self.config.get('default_volatility', 0.02)
        self.trend_strength = self.config.get('trend_strength', 0.6)
        self.random_seed = self.config.get('random_seed', None)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
    
    def generate_ohlcv_data(
        self, 
        symbol: str, 
        start_date: datetime,
        periods: int = 200,
        interval: str = '1d',
        base_price: float = 100.0,
        pattern_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate OHLCV data for a given symbol with optional embedded patterns.
        
        Args:
            symbol: Trading symbol
            start_date: Starting date for the data
            periods: Number of periods to generate
            interval: Time interval ('1m', '5m', '1h', '1d', etc.)
            base_price: Starting price level
            pattern_type: Type of pattern to embed in the data
                (None, 'uptrend', 'downtrend', 'sideways', 'head_shoulders',
                'inverse_head_shoulders', 'double_top', 'double_bottom',
                'triangle_ascending', 'triangle_descending', 'triangle_symmetrical',
                'flag_bullish', 'flag_bearish')
                
        Returns:
            DataFrame with OHLCV data
        """
        # Determine date range based on interval
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            date_range = [start_date + timedelta(minutes=i*minutes) for i in range(periods)]
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            date_range = [start_date + timedelta(hours=i*hours) for i in range(periods)]
        elif interval.endswith('d'):
            days = int(interval[:-1])
            date_range = [start_date + timedelta(days=i*days) for i in range(periods)]
        else:
            # Default to daily
            date_range = [start_date + timedelta(days=i) for i in range(periods)]
        
        # Generate base price series with appropriate pattern
        if pattern_type == 'uptrend':
            closes = self._generate_uptrend(periods, base_price)
        elif pattern_type == 'downtrend':
            closes = self._generate_downtrend(periods, base_price)
        elif pattern_type == 'sideways':
            closes = self._generate_sideways(periods, base_price)
        elif pattern_type == 'head_shoulders':
            closes = self._generate_head_shoulders(periods, base_price)
        elif pattern_type == 'cup_and_handle':
            closes = self._generate_cup_and_handle(periods, base_price)
        elif pattern_type == 'wedge_rising':
            closes = self._generate_wedge_rising(periods, base_price)
        elif pattern_type == 'wedge_falling':
            closes = self._generate_wedge_falling(periods, base_price)
        elif pattern_type == 'inverse_head_shoulders':
            closes = self._generate_inverse_head_shoulders(periods, base_price)
        elif pattern_type == 'double_top':
            closes = self._generate_double_top(periods, base_price)
        elif pattern_type == 'double_bottom':
            closes = self._generate_double_bottom(periods, base_price)
        elif pattern_type == 'triangle_ascending':
            closes = self._generate_ascending_triangle(periods, base_price)
        elif pattern_type == 'triangle_descending':
            closes = self._generate_descending_triangle(periods, base_price)
        elif pattern_type == 'triangle_symmetrical':
            closes = self._generate_symmetrical_triangle(periods, base_price)
        elif pattern_type == 'flag_bullish':
            closes = self._generate_bullish_flag(periods, base_price)
        elif pattern_type == 'flag_bearish':
            closes = self._generate_bearish_flag(periods, base_price)
        else:
            # Default to sideways with some randomness
            drift = np.random.choice([-0.0001, 0, 0.0001])
            random_walk = np.random.normal(drift, self.default_volatility, periods).cumsum()
            closes = base_price * (1 + random_walk)
        
        # Generate OHLCV data based on close prices
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        # Generate high and low prices
        intraday_volatility = self.default_volatility * 0.5
        highs = np.array([max(o, c) for o, c in zip(opens, closes)])
        lows = np.array([min(o, c) for o, c in zip(opens, closes)])
        
        # Add some randomness to highs and lows
        high_offsets = np.abs(np.random.normal(0, intraday_volatility, periods))
        low_offsets = np.abs(np.random.normal(0, intraday_volatility, periods))
        
        highs = highs + high_offsets
        lows = lows - low_offsets
        
        # Generate volumes with some correlation to price changes
        price_changes = np.abs(np.diff(np.append([base_price], closes)))
        base_volume = 1000000  # Base daily volume
        volume_volatility = 0.3
        volumes = base_volume * (1 + 2 * price_changes + np.random.normal(0, volume_volatility, periods))
        volumes = volumes.astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=date_range)
        
        return df
    
    def _generate_uptrend(self, periods: int, base_price: float) -> np.ndarray:
        """Generate an uptrend pattern."""
        # Linear uptrend with noise
        trend_factor = 0.001 * self.trend_strength
        linear_trend = np.array([i * trend_factor for i in range(periods)])
        noise = np.random.normal(0, self.default_volatility * 0.7, periods)
        
        return base_price * (1 + linear_trend + noise)
    
    def _generate_downtrend(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a downtrend pattern."""
        # Linear downtrend with noise
        trend_factor = 0.001 * self.trend_strength
        linear_trend = np.array([-i * trend_factor for i in range(periods)])
        noise = np.random.normal(0, self.default_volatility * 0.7, periods)
        
        return base_price * (1 + linear_trend + noise)
    
    def _generate_sideways(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a sideways/ranging pattern."""
        # Oscillating pattern with noise
        oscillation_factor = self.default_volatility * 2
        oscillation = np.sin(np.linspace(0, 3 * np.pi, periods)) * oscillation_factor
        noise = np.random.normal(0, self.default_volatility * 0.3, periods)
        
        return base_price * (1 + oscillation + noise)
    
    def _generate_head_shoulders(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a head and shoulders pattern."""
        if periods < 60:
            # Fall back to simpler pattern for small periods
            return self._generate_double_top(periods, base_price)
        
        # Parameters for the head and shoulders pattern
        left_shoulder_peak = int(periods * 0.25)
        head_peak = int(periods * 0.5)
        right_shoulder_peak = int(periods * 0.75)
        
        left_trough = int(periods * 0.35)
        right_trough = int(periods * 0.65)
        
        shoulder_height = base_price * 0.08
        head_height = base_price * 0.12
        
        # Start with slight uptrend before pattern
        prices = base_price * (1 + 0.0005 * np.arange(periods))
        
        # Create the pattern
        for i in range(periods):
            # Left shoulder
            if i < left_shoulder_peak:
                dist = 1 - abs(i - left_shoulder_peak) / left_shoulder_peak
                prices[i] += shoulder_height * dist
            # Head
            elif i < head_peak:
                dist = 1 - abs(i - head_peak) / (head_peak - left_trough)
                prices[i] += head_height * dist
            # Right shoulder
            elif i < right_shoulder_peak:
                dist = 1 - abs(i - right_shoulder_peak) / (right_shoulder_peak - right_trough)
                prices[i] += shoulder_height * dist
            # Breakdown after pattern
            else:
                breakdown_rate = 0.0008
                prices[i] -= base_price * breakdown_rate * (i - right_shoulder_peak)
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.5, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_inverse_head_shoulders(self, periods: int, base_price: float) -> np.ndarray:
        """Generate an inverse head and shoulders pattern."""
        if periods < 60:
            # Fall back to simpler pattern for small periods
            return self._generate_double_bottom(periods, base_price)
        
        # Parameters for the inverse head and shoulders pattern
        left_shoulder_trough = int(periods * 0.25)
        head_trough = int(periods * 0.5)
        right_shoulder_trough = int(periods * 0.75)
        
        left_peak = int(periods * 0.35)
        right_peak = int(periods * 0.65)
        
        shoulder_depth = base_price * 0.08
        head_depth = base_price * 0.12
        
        # Start with slight downtrend before pattern
        prices = base_price * (1 - 0.0005 * np.arange(periods))
        
        # Create the pattern
        for i in range(periods):
            # Left shoulder
            if i < left_shoulder_trough:
                dist = 1 - abs(i - left_shoulder_trough) / left_shoulder_trough
                prices[i] -= shoulder_depth * dist
            # Head
            elif i < head_trough:
                dist = 1 - abs(i - head_trough) / (head_trough - left_peak)
                prices[i] -= head_depth * dist
            # Right shoulder
            elif i < right_shoulder_trough:
                dist = 1 - abs(i - right_shoulder_trough) / (right_shoulder_trough - right_peak)
                prices[i] -= shoulder_depth * dist
            # Breakout after pattern
            else:
                breakout_rate = 0.0008
                prices[i] += base_price * breakout_rate * (i - right_shoulder_trough)
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.5, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_double_top(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a double top pattern."""
        if periods < 40:
            # Fall back to simpler pattern for small periods
            return self._generate_uptrend(periods, base_price)
        
        # Parameters for the double top pattern
        first_peak = int(periods * 0.3)
        second_peak = int(periods * 0.6)
        middle_trough = int((first_peak + second_peak) / 2)
        
        peak_height = base_price * 0.1
        trough_depth = base_price * 0.03
        
        # Start with slight uptrend before pattern
        prices = base_price * (1 + 0.0005 * np.arange(periods))
        
        # Create the pattern
        for i in range(periods):
            # First peak
            if i < first_peak:
                dist = 1 - abs(i - first_peak) / first_peak
                prices[i] += peak_height * dist
            # Middle trough
            elif i < middle_trough:
                dist = 1 - abs(i - middle_trough) / (middle_trough - first_peak)
                prices[i] += peak_height - (peak_height + trough_depth) * dist
            # Second peak
            elif i < second_peak:
                dist = 1 - abs(i - second_peak) / (second_peak - middle_trough)
                prices[i] += peak_height * dist
            # Breakdown after pattern
            else:
                breakdown_rate = 0.001
                prices[i] -= base_price * breakdown_rate * (i - second_peak)
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.5, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_double_bottom(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a double bottom pattern."""
        if periods < 40:
            # Fall back to simpler pattern for small periods
            return self._generate_downtrend(periods, base_price)
        
        # Parameters for the double bottom pattern
        first_trough = int(periods * 0.3)
        second_trough = int(periods * 0.6)
        middle_peak = int((first_trough + second_trough) / 2)
        
        trough_depth = base_price * 0.1
        peak_height = base_price * 0.03
        
        # Start with slight downtrend before pattern
        prices = base_price * (1 - 0.0005 * np.arange(periods))
        
        # Create the pattern
        for i in range(periods):
            # First trough
            if i < first_trough:
                dist = 1 - abs(i - first_trough) / first_trough
                prices[i] -= trough_depth * dist
            # Middle peak
            elif i < middle_peak:
                dist = 1 - abs(i - middle_peak) / (middle_peak - first_trough)
                prices[i] -= trough_depth - (trough_depth + peak_height) * dist
            # Second trough
            elif i < second_trough:
                dist = 1 - abs(i - second_trough) / (second_trough - middle_peak)
                prices[i] -= trough_depth * dist
            # Breakout after pattern
            else:
                breakout_rate = 0.001
                prices[i] += base_price * breakout_rate * (i - second_trough)
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.5, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_ascending_triangle(self, periods: int, base_price: float) -> np.ndarray:
        """Generate an ascending triangle pattern."""
        if periods < 40:
            # Fall back to simpler pattern for small periods
            return self._generate_sideways(periods, base_price)
        
        # Parameters for the ascending triangle
        triangle_start = int(periods * 0.1)
        triangle_end = int(periods * 0.8)
        breakout_point = triangle_end
        
        resistance_level = base_price * 1.08
        initial_support_level = base_price * 0.96
        
        # Create base prices
        prices = np.ones(periods) * base_price
        
        # Create the pattern
        for i in range(periods):
            if i < triangle_start:
                # Pre-pattern buildup
                prices[i] = base_price * (1 + 0.0002 * i)
            elif i < breakout_point:
                # Triangle pattern
                progress = (i - triangle_start) / (triangle_end - triangle_start)
                
                # Rising support line
                support_level = initial_support_level + (resistance_level - initial_support_level) * progress
                
                # Oscillate between support and resistance
                cycle_position = (i % 10) / 10.0  # Oscillation within the triangle
                if cycle_position < 0.5:
                    # Moving towards resistance
                    position = cycle_position * 2
                    prices[i] = support_level + (resistance_level - support_level) * position
                else:
                    # Moving towards support
                    position = (cycle_position - 0.5) * 2
                    prices[i] = resistance_level - (resistance_level - support_level) * position
            else:
                # Breakout
                breakout_height = resistance_level * 0.05
                progress = min(1.0, (i - breakout_point) / 10)  # Limit the breakout height
                prices[i] = resistance_level + breakout_height * progress
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.3, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_descending_triangle(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a descending triangle pattern."""
        if periods < 40:
            # Fall back to simpler pattern for small periods
            return self._generate_sideways(periods, base_price)
        
        # Parameters for the descending triangle
        triangle_start = int(periods * 0.1)
        triangle_end = int(periods * 0.8)
        breakdown_point = triangle_end
        
        support_level = base_price * 0.92
        initial_resistance_level = base_price * 1.04
        
        # Create base prices
        prices = np.ones(periods) * base_price
        
        # Create the pattern
        for i in range(periods):
            if i < triangle_start:
                # Pre-pattern buildup
                prices[i] = base_price * (1 - 0.0002 * i)
            elif i < breakdown_point:
                # Triangle pattern
                progress = (i - triangle_start) / (triangle_end - triangle_start)
                
                # Falling resistance line
                resistance_level = initial_resistance_level - (initial_resistance_level - support_level) * progress
                
                # Oscillate between support and resistance
                cycle_position = (i % 10) / 10.0  # Oscillation within the triangle
                if cycle_position < 0.5:
                    # Moving towards support
                    position = cycle_position * 2
                    prices[i] = resistance_level - (resistance_level - support_level) * position
                else:
                    # Moving towards resistance
                    position = (cycle_position - 0.5) * 2
                    prices[i] = support_level + (resistance_level - support_level) * position
            else:
                # Breakdown
                breakdown_depth = support_level * 0.05
                progress = min(1.0, (i - breakdown_point) / 10)  # Limit the breakdown depth
                prices[i] = support_level - breakdown_depth * progress
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.3, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_symmetrical_triangle(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a symmetrical triangle pattern."""
        if periods < 40:
            # Fall back to simpler pattern for small periods
            return self._generate_sideways(periods, base_price)
        
        # Parameters for the symmetrical triangle
        triangle_start = int(periods * 0.1)
        triangle_end = int(periods * 0.8)
        breakout_point = triangle_end
        
        initial_resistance_level = base_price * 1.06
        initial_support_level = base_price * 0.94
        
        # Create base prices
        prices = np.ones(periods) * base_price
        
        # Decide if breakout will be up or down
        breakout_up = random.choice([True, False])
        
        # Create the pattern
        for i in range(periods):
            if i < triangle_start:
                # Pre-pattern price movement
                prices[i] = base_price * (1 + 0.0001 * (i - triangle_start))
            elif i < breakout_point:
                # Triangle pattern
                progress = (i - triangle_start) / (triangle_end - triangle_start)
                
                # Calculate converging support and resistance
                resistance_level = initial_resistance_level - (initial_resistance_level - base_price) * progress
                support_level = initial_support_level + (base_price - initial_support_level) * progress
                
                # Oscillate between support and resistance
                cycle_position = (i % 10) / 10.0  # Oscillation within the triangle
                if cycle_position < 0.5:
                    # Moving towards resistance
                    position = cycle_position * 2
                    prices[i] = support_level + (resistance_level - support_level) * position
                else:
                    # Moving towards support
                    position = (cycle_position - 0.5) * 2
                    prices[i] = resistance_level - (resistance_level - support_level) * position
            else:
                # Breakout
                midpoint = (initial_resistance_level + initial_support_level) / 2
                breakout_magnitude = base_price * 0.04
                progress = min(1.0, (i - breakout_point) / 10)  # Limit the breakout magnitude
                
                if breakout_up:
                    prices[i] = midpoint + breakout_magnitude * progress
                else:
                    prices[i] = midpoint - breakout_magnitude * progress
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.3, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_bullish_flag(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a bullish flag pattern."""
        if periods < 30:
            # Fall back to simpler pattern for small periods
            return self._generate_uptrend(periods, base_price)
        
        # Parameters for the bullish flag
        pole_start = 0
        pole_end = int(periods * 0.3)
        flag_end = int(periods * 0.7)
        
        pole_height = base_price * 0.15
        flag_channel_width = base_price * 0.03
        
        # Create base prices
        prices = np.ones(periods) * base_price
        
        # Create the pattern
        for i in range(periods):
            if i < pole_end:
                # Flagpole (sharp upward move)
                progress = i / pole_end
                prices[i] = base_price + pole_height * progress
            elif i < flag_end:
                # Flag (consolidation with slight downward bias)
                flag_progress = (i - pole_end) / (flag_end - pole_end)
                flag_base = base_price + pole_height - flag_channel_width * flag_progress
                
                # Oscillate within the flag channel
                cycle_position = ((i - pole_end) % 6) / 6.0
                channel_position = np.sin(cycle_position * 2 * np.pi) * 0.5 + 0.5
                prices[i] = flag_base + flag_channel_width * channel_position
            else:
                # Breakout (continuation of the prior trend)
                breakout_progress = (i - flag_end) / (periods - flag_end)
                continuation_height = pole_height * 0.7
                prices[i] = (base_price + pole_height) + continuation_height * breakout_progress
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.4, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_bearish_flag(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a bearish flag pattern."""
        if periods < 30:
            # Fall back to simpler pattern for small periods
            return self._generate_downtrend(periods, base_price)
        
        # Parameters for the bearish flag
        pole_start = 0
        pole_end = int(periods * 0.3)
        flag_end = int(periods * 0.7)
        
        pole_depth = base_price * 0.15
        flag_channel_width = base_price * 0.03
        
        # Create base prices
        prices = np.ones(periods) * base_price
        
        # Create the pattern
        for i in range(periods):
            if i < pole_end:
                # Flagpole (sharp downward move)
                progress = i / pole_end
                prices[i] = base_price - pole_depth * progress
            elif i < flag_end:
                # Flag (consolidation with slight upward bias)
                flag_progress = (i - pole_end) / (flag_end - pole_end)
                flag_base = base_price - pole_depth + flag_channel_width * flag_progress
                
                # Oscillate within the flag channel
                cycle_position = ((i - pole_end) % 6) / 6.0
                channel_position = np.sin(cycle_position * 2 * np.pi) * 0.5 + 0.5
                prices[i] = flag_base - flag_channel_width * channel_position
            else:
                # Breakdown (continuation of the prior trend)
                breakdown_progress = (i - flag_end) / (periods - flag_end)
                continuation_depth = pole_depth * 0.7
                prices[i] = (base_price - pole_depth) - continuation_depth * breakdown_progress
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.4, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_cup_and_handle(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a cup and handle pattern."""
        if periods < 60:
            # Fall back to simpler pattern for small periods
            return self._generate_uptrend(periods, base_price)
        
        # Define pattern segments
        left_rim_end = int(periods * 0.2)  # Left cup rim
        cup_bottom = int(periods * 0.5)    # Bottom of the cup
        right_rim_end = int(periods * 0.8) # Right cup rim
        
        # Pattern parameters
        cup_depth = base_price * 0.12       # Depth of the cup
        handle_depth = base_price * 0.04    # Depth of the handle pullback
        breakout_height = base_price * 0.08 # Height of the final breakout
        
        # Create base prices
        prices = np.ones(periods) * base_price
        
        # Create the pattern
        for i in range(periods):
            if i < left_rim_end:
                # Left rim (initial price level)
                left_progress = i / left_rim_end
                # Slight downward slope at the end of left rim
                if left_progress > 0.7:
                    rim_completion = (left_progress - 0.7) / 0.3
                    prices[i] = base_price - cup_depth * 0.2 * rim_completion
                else:
                    prices[i] = base_price
            elif i < cup_bottom:
                # Cup formation (U-shaped)
                cup_progress = (i - left_rim_end) / (cup_bottom - left_rim_end)
                # U-shape using sine function
                u_shape = np.sin(cup_progress * np.pi) 
                prices[i] = base_price - cup_depth * u_shape
            elif i < right_rim_end:
                # Right rim formation
                rim_progress = (i - cup_bottom) / (right_rim_end - cup_bottom)
                # Rise back to starting level
                prices[i] = (base_price - cup_depth) + cup_depth * rim_progress
            else:
                # Handle formation and breakout
                handle_progress = (i - right_rim_end) / (periods - right_rim_end)
                
                if handle_progress < 0.4:
                    # Handle pullback (shallow dip)
                    pullback = np.sin(handle_progress * 2.5 * np.pi) * handle_depth
                    prices[i] = base_price - pullback
                else:
                    # Breakout (upward movement)
                    breakout_progress = (handle_progress - 0.4) / 0.6
                    prices[i] = base_price + breakout_height * breakout_progress
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.3, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_wedge_rising(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a rising wedge pattern (bearish reversal)."""
        if periods < 40:
            # Fall back to simpler pattern for small periods
            return self._generate_uptrend(periods, base_price)
        
        # Define pattern segments
        wedge_formation_end = int(periods * 0.8)  # Wedge formation period
        
        # Pattern parameters
        wedge_height = base_price * 0.15        # Height of the wedge
        breakdown_depth = base_price * 0.08     # Depth of the breakdown
        converge_factor = 0.7                   # How much the channel narrows
        
        # Create base prices
        prices = np.ones(periods) * base_price
        
        # Create the pattern
        for i in range(periods):
            if i < wedge_formation_end:
                # Wedge formation with higher lows and higher highs, but converging
                wedge_progress = i / wedge_formation_end
                
                # Calculate the upper and lower boundaries of the wedge
                # Both rising but at different rates (converging)
                lower_boundary = base_price + wedge_progress * wedge_height * 0.8
                upper_boundary = base_price + wedge_progress * wedge_height
                
                # Make the channel narrower as wedge progresses
                channel_width = (upper_boundary - lower_boundary) * (1 - wedge_progress * converge_factor)
                
                # Oscillate within the wedge channel
                cycle_position = (i % 8) / 8.0
                channel_position = np.sin(cycle_position * 2 * np.pi) * 0.5 + 0.5
                prices[i] = lower_boundary + channel_width * channel_position
            else:
                # Breakdown (bearish reversal)
                breakdown_progress = (i - wedge_formation_end) / (periods - wedge_formation_end)
                wedge_top = base_price + wedge_height
                prices[i] = wedge_top - breakdown_progress * breakdown_depth
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.3, periods)
        prices += base_price * noise
        
        return prices
    
    def _generate_wedge_falling(self, periods: int, base_price: float) -> np.ndarray:
        """Generate a falling wedge pattern (bullish reversal)."""
        if periods < 40:
            # Fall back to simpler pattern for small periods
            return self._generate_downtrend(periods, base_price)
        
        # Define pattern segments
        wedge_formation_end = int(periods * 0.8)  # Wedge formation period
        
        # Pattern parameters
        wedge_depth = base_price * 0.15        # Depth of the wedge
        breakout_height = base_price * 0.08    # Height of the breakout
        converge_factor = 0.7                  # How much the channel narrows
        
        # Create base prices
        prices = np.ones(periods) * base_price
        
        # Create the pattern
        for i in range(periods):
            if i < wedge_formation_end:
                # Wedge formation with lower highs and lower lows, but converging
                wedge_progress = i / wedge_formation_end
                
                # Calculate the upper and lower boundaries of the wedge
                # Both falling but at different rates (converging)
                upper_boundary = base_price - wedge_progress * wedge_depth * 0.8
                lower_boundary = base_price - wedge_progress * wedge_depth
                
                # Make the channel narrower as wedge progresses
                channel_width = (upper_boundary - lower_boundary) * (1 - wedge_progress * converge_factor)
                
                # Oscillate within the wedge channel
                cycle_position = (i % 8) / 8.0
                channel_position = np.sin(cycle_position * 2 * np.pi) * 0.5 + 0.5
                prices[i] = lower_boundary + channel_width * channel_position
            else:
                # Breakout (bullish reversal)
                breakout_progress = (i - wedge_formation_end) / (periods - wedge_formation_end)
                wedge_bottom = base_price - wedge_depth
                prices[i] = wedge_bottom + breakout_progress * breakout_height
        
        # Add noise
        noise = np.random.normal(0, self.default_volatility * 0.3, periods)
        prices += base_price * noise
        
        return prices