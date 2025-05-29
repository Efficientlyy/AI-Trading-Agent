"""
Mock Data Generator - Generates realistic market data for testing and demonstrations.

This module creates synthetic market data that can simulate various market conditions,
patterns, and scenarios for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import random
from datetime import datetime, timedelta
from enum import Enum

from ..common.utils import get_logger


class TrendType(Enum):
    """Types of trends that can be simulated in mock data."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class PatternType(Enum):
    """Types of patterns that can be embedded in mock data."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    CUP_AND_HANDLE = "cup_and_handle"
    BULLISH_FLAG = "bullish_flag"
    BEARISH_FLAG = "bearish_flag"
    NONE = "none"


class MockDataGenerator:
    """
    Generates synthetic market data for testing and demonstrations.
    
    This class can create realistic OHLCV data with embedded patterns,
    trends, and other market characteristics to simulate various conditions
    for testing technical analysis algorithms.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the MockDataGenerator with optional seed for reproducibility.
        
        Args:
            seed: Optional random seed for reproducible data generation
        """
        self.logger = get_logger("MockDataGenerator")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.logger.info(f"MockDataGenerator initialized with seed: {seed}")
    
    def generate_data(
        self,
        symbol: str,
        periods: int = 200,
        trend_type: TrendType = TrendType.SIDEWAYS,
        pattern: PatternType = PatternType.NONE,
        volatility: float = 0.02,
        end_date: Optional[datetime] = None,
        ensure_enough_data: bool = True
    ) -> pd.DataFrame:
        """Generate mock market data with specified characteristics.
        
        Args:
            symbol: The ticker symbol for the mock data
            periods: Number of trading days to generate
            trend_type: Type of trend to simulate (bullish, bearish, sideways)
            pattern: Chart pattern to embed in the data
            volatility: Daily volatility as a decimal (e.g., 0.02 = 2%)
            end_date: The ending date for the data (defaults to today)
            ensure_enough_data: If True, ensures at least 250 periods to calculate indicators properly
            
        Returns:
            DataFrame with OHLCV data
        """
        # Ensure we have enough data for indicators (RSI typically needs 14+ periods)
        if ensure_enough_data and periods < 250:  # 250 is a full trading year and ensures enough data for all indicators
            self.logger.info(f"Increasing periods from {periods} to 250 to ensure enough data for indicators")
            periods = 250
            
        if end_date is None:
            end_date = datetime.now()
            
        # Generate dates working backward from end_date - generate more than needed to account for weekends
        # and holidays (we'll filter them out)
        extra_days = int(periods * 1.4)  # Add 40% more days to account for weekends and holidays
        all_dates = [end_date - timedelta(days=i) for i in range(extra_days)]
        all_dates.reverse()  # Now in chronological order
        
        # Filter to only include trading days (Mon-Fri)
        dates = [date for date in all_dates if date.weekday() < 5]
        
        # Trim to the requested number of periods
        dates = dates[:periods]
        
        # Adjust periods if we don't have enough days
        periods = len(dates)
        
        # Initialize price with a random starting point between $10 and $1000
        # Note: seed was already set in __init__
        starting_price = np.random.uniform(10, 1000)
        
        # Create arrays for our OHLCV data
        close_prices = np.zeros(periods)
        open_prices = np.zeros(periods)
        high_prices = np.zeros(periods)
        low_prices = np.zeros(periods)
        volumes = np.zeros(periods)
        
        # Set base trend - this determines the overall price direction
        trend_factor = self._get_trend_factor(trend_type)
        
        # Generate baseline prices with the selected trend
        for i in range(periods):
            if i == 0:
                close_prices[i] = starting_price
            else:
                # Daily change based on trend and random volatility
                daily_return = trend_factor * (1/periods) + np.random.normal(0, volatility)
                close_prices[i] = close_prices[i-1] * (1 + daily_return)
        
        # Create DataFrame first
        df = pd.DataFrame({
            'date': dates,
            'open': np.zeros(periods),  # Temporary values, will update after pattern
            'high': np.zeros(periods),  # Temporary values, will update after pattern
            'low': np.zeros(periods),   # Temporary values, will update after pattern
            'close': close_prices,
            'volume': np.zeros(periods)  # Temporary values, will update after pattern
        })
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Apply the selected pattern to the generated data if not NONE
        if pattern != PatternType.NONE:
            pattern_start = np.random.randint(0, max(1, len(df) - self._get_pattern_length(pattern, len(df))))
            if pattern == PatternType.HEAD_AND_SHOULDERS:
                self._apply_head_and_shoulders(df, pattern_start, 1.0)
            elif pattern == PatternType.DOUBLE_TOP:
                self._apply_double_top(df, pattern_start, 1.0)
            elif pattern == PatternType.DOUBLE_BOTTOM:
                self._apply_double_bottom(df, pattern_start, 1.0)
            elif pattern == PatternType.ASCENDING_TRIANGLE:
                self._apply_ascending_triangle(df, pattern_start, 1.0)
            elif pattern == PatternType.DESCENDING_TRIANGLE:
                self._apply_descending_triangle(df, pattern_start, 1.0)
            elif pattern == PatternType.CUP_AND_HANDLE:
                self._apply_cup_and_handle(df, pattern_start, 1.0)
            elif pattern == PatternType.BULLISH_FLAG:
                self._apply_bullish_flag(df, pattern_start, 1.0)
            elif pattern == PatternType.BEARISH_FLAG:
                self._apply_bearish_flag(df, pattern_start, 1.0)
            
            # Update close_prices from the dataframe after pattern application
            close_prices = df['close'].values
        


        # Update the DataFrame with all OHLCV data
        for i in range(periods):
            # For the first day, base everything on the first close price
            if i == 0:
                daily_volatility = close_prices[i] * volatility
                open_prices[i] = close_prices[i] * (1 + np.random.normal(0, volatility/2))
                high_prices[i] = max(open_prices[i], close_prices[i]) + abs(np.random.normal(0, daily_volatility))
                low_prices[i] = min(open_prices[i], close_prices[i]) - abs(np.random.normal(0, daily_volatility))
            else:
                # For subsequent days, open is based on previous close
                daily_volatility = close_prices[i] * volatility
                open_prices[i] = close_prices[i-1] * (1 + np.random.normal(0, volatility/2))
                high_prices[i] = max(open_prices[i], close_prices[i]) + abs(np.random.normal(0, daily_volatility))
                low_prices[i] = min(open_prices[i], close_prices[i]) - abs(np.random.normal(0, daily_volatility))
                
            # Generate volumes - higher on trend changes, lower on sideways movement
            volume_base = np.random.uniform(100000, 1000000)
            price_change_factor = abs(close_prices[i] / close_prices[i-1] - 1) if i > 0 else 0
            volumes[i] = volume_base * (1 + 10 * price_change_factor)
        
        # Create final DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes.astype(int)
        })
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Add the symbol as a column (useful when combining multiple symbols)
        df['symbol'] = symbol
        
        self.logger.info(f"Generated mock data for {symbol}: {periods} periods, trend: {trend_type.value}, pattern: {pattern.value}")
        self.logger.info(
            f"Generated mock data for {symbol}: {periods} periods, "
            f"trend: {trend_type.value}, pattern: {pattern.value}"
        )
        
        return df
    
    def _get_trend_factor(self, trend_type: TrendType) -> float:
        """Get the trend factor based on trend type."""
        if trend_type == TrendType.BULLISH:
            return 0.3  # Positive factor for bullish trend
        elif trend_type == TrendType.BEARISH:
            return -0.3  # Negative factor for bearish trend
        elif trend_type == TrendType.SIDEWAYS:
            return 0.0  # No trend
        elif trend_type == TrendType.VOLATILE:
            # For volatile, we'll return 0 but increase volatility elsewhere
            return 0.0
        
    def _apply_pattern(
        self, 
        df: pd.DataFrame, 
        pattern: PatternType, 
        pattern_start: Optional[int] = None,
        strength: float = 1.0
    ) -> pd.DataFrame:
        """
        Apply a specific price pattern to the data.
        
        Args:
            df: DataFrame with OHLCV data
            pattern: Pattern to apply
            pattern_start: Starting index for pattern (if None, randomly placed)
            strength: Strength of the pattern (0.0 to 1.0)
            
        Returns:
            DataFrame with pattern applied
        """
        # This is a placeholder implementation
        # Full pattern implementation will be added in Phase 4
        
        periods = len(df)
        
        # If pattern_start is not specified, choose a random starting point
        # that allows the pattern to fit within the data
        if pattern_start is None:
            # Estimate pattern length based on the pattern type
            pattern_length = self._get_pattern_length(pattern, periods)
            max_start = periods - pattern_length
            if max_start <= 0:
                self.logger.warning(f"Not enough periods ({periods}) to fit pattern {pattern.value}")
                return df
                
            pattern_start = random.randint(0, max_start)
        
        # Log the pattern application
        self.logger.info(f"Applying {pattern.value} pattern starting at index {pattern_start} with strength {strength}")
        
        # Create a copy of the DataFrame to modify
        result_df = df.copy()
        
        # Apply the selected pattern
        if pattern == PatternType.HEAD_AND_SHOULDERS:
            self._apply_head_and_shoulders(result_df, pattern_start, strength)
        elif pattern == PatternType.DOUBLE_TOP:
            self._apply_double_top(result_df, pattern_start, strength)
        elif pattern == PatternType.DOUBLE_BOTTOM:
            self._apply_double_bottom(result_df, pattern_start, strength)
        elif pattern == PatternType.ASCENDING_TRIANGLE:
            self._apply_ascending_triangle(result_df, pattern_start, strength)
        elif pattern == PatternType.DESCENDING_TRIANGLE:
            self._apply_descending_triangle(result_df, pattern_start, strength)
        # Add other patterns in Phase 4
        
        return result_df
    
    def _apply_cup_and_handle(self, df: pd.DataFrame, start: int, strength: float) -> None:
        """Apply a cup and handle pattern."""
        pattern_length = self._get_pattern_length(PatternType.CUP_AND_HANDLE, len(df))
        if start + pattern_length >= len(df):
            pattern_length = len(df) - start - 1
            
        self.logger.info(f"Applying cup_and_handle pattern starting at index {start} with length {pattern_length}")
        # Basic implementation that will be enhanced in Phase 4
        # The cup and handle pattern consists of a rounded bottom (cup) followed by
        # a small downward drift (handle) before an eventual breakout
        
        # Get the base price at start
        base_price = df.iloc[start]['close']
        
        # Cup depth (percentage of price)
        cup_depth = 0.08 * strength
        
        # Divide the pattern into segments
        cup_length = int(pattern_length * 0.7)  # Cup is 70% of the pattern
        handle_length = pattern_length - cup_length  # Rest is the handle
        
        # Create the cup (U-shaped)
        for i in range(cup_length):
            idx = start + i
            if idx >= len(df):
                break
                
            # Position in the cup (0 to 1)
            pos = i / cup_length
            
            # Cup shape - parabolic function creates U shape
            # Lowest at the middle (pos = 0.5)
            cup_factor = cup_depth * (1 - 4 * (pos - 0.5) ** 2)
            
            # Apply to close price
            df.loc[df.index[idx], 'close'] = base_price * (1 - cup_factor)
            
            # Adjust high/low based on close
            current_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
            df.loc[df.index[idx], 'high'] = df.iloc[idx]['close'] + current_range * 0.4
            df.loc[df.index[idx], 'low'] = df.iloc[idx]['close'] - current_range * 0.4
        
        # Create the handle (small drift down then consolidation)
        for i in range(handle_length):
            idx = start + cup_length + i
            if idx >= len(df):
                break
                
            # Position in handle (0 to 1)
            pos = i / handle_length
            
            # Handle shape - slight drift down then consolidation
            if pos < 0.3:
                # Initial drift down
                handle_factor = 0.03 * strength * (pos / 0.3)
            else:
                # Consolidation with slight upward bias at the end
                handle_factor = 0.03 * strength * (1 - (pos - 0.3) / 0.7)
            
            # Apply to close price (at end of cup, price is back to base_price)
            df.loc[df.index[idx], 'close'] = base_price * (1 - handle_factor)
            
            # Less volatility in handle
            current_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
            df.loc[df.index[idx], 'high'] = df.iloc[idx]['close'] + current_range * 0.2
            df.loc[df.index[idx], 'low'] = df.iloc[idx]['close'] - current_range * 0.2
            
    def _apply_bullish_flag(self, df: pd.DataFrame, start: int, strength: float) -> None:
        """Apply a bullish flag pattern."""
        # Will be implemented in Phase 4
        pass
        
    def _apply_bearish_flag(self, df: pd.DataFrame, start: int, strength: float) -> None:
        """Apply a bearish flag pattern."""
        # Will be implemented in Phase 4
        pass
    
    def _parse_interval(self, interval: str) -> float:
        """Convert interval string to number of days."""
        if interval.endswith('m'):
            # Minutes - convert to fraction of day
            return float(interval[:-1]) / (60 * 24)
        elif interval.endswith('h'):
            # Hours - convert to fraction of day
            return float(interval[:-1]) / 24
        elif interval.endswith('d'):
            # Days
            return float(interval[:-1])
        elif interval.endswith('w'):
            # Weeks
            return float(interval[:-1]) * 7
        elif interval.endswith('M'):
            # Months (approximate)
            return float(interval[:-1]) * 30
        else:
            # Default to days
            return float(interval)
    
    def _generate_trend(self, periods: int, trend_type: TrendType, strength: float) -> np.ndarray:
        """Generate trend component based on trend type and strength."""
        if trend_type == TrendType.BULLISH:
            # Bullish trend - gradually increasing returns
            return np.linspace(0, strength, periods)
        elif trend_type == TrendType.BEARISH:
            # Bearish trend - gradually decreasing returns
            return np.linspace(0, -strength, periods)
        elif trend_type == TrendType.SIDEWAYS:
            # Sideways trend - no consistent direction
            return np.zeros(periods)
        elif trend_type == TrendType.VOLATILE:
            # Volatile trend - larger random movements
            return np.random.normal(0, strength, periods)
        else:
            return np.zeros(periods)
    
    def _generate_ohlc(self, close_prices: np.ndarray, volatility: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate open, high, low prices based on close prices."""
        periods = len(close_prices)
        
        # Generate daily ranges based on volatility
        daily_ranges = np.random.uniform(0.5 * volatility, 2.0 * volatility, periods) * close_prices
        
        # Generate high and low prices
        high_offsets = np.random.uniform(0.4, 1.0, periods) * daily_ranges
        low_offsets = np.random.uniform(0.4, 1.0, periods) * daily_ranges
        
        high_prices = close_prices + high_offsets
        low_prices = close_prices - low_offsets
        
        # Ensure low prices are always lower than high prices
        low_prices = np.minimum(low_prices, high_prices * 0.9999)
        
        # Generate open prices between high and low
        open_ratio = np.random.uniform(0, 1, periods)
        open_prices = low_prices + open_ratio * (high_prices - low_prices)
        
        return high_prices, low_prices, open_prices
    
    def _generate_volumes(self, prices: np.ndarray, base_volume: float, volatility: float) -> np.ndarray:
        """Generate trading volumes correlated with price movements."""
        periods = len(prices)
        
        # Base volumes with random noise
        volumes = base_volume * (1 + np.random.normal(0, volatility, periods))
        
        # Add correlation with price changes
        price_changes = np.zeros(periods)
        price_changes[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # Amplify volume on large price moves
        volume_amplification = 1 + 2 * np.abs(price_changes)
        volumes = volumes * volume_amplification
        
        # Ensure volumes are positive
        volumes = np.maximum(volumes, base_volume * 0.2)
        
        return volumes.astype(int)
    
    def _apply_pattern(
        self, 
        df: pd.DataFrame, 
        pattern: PatternType, 
        pattern_start: Optional[int] = None,
        strength: float = 1.0
    ) -> pd.DataFrame:
        """
        Apply a specific price pattern to the data.
        
        Args:
            df: DataFrame with OHLCV data
            pattern: Pattern to apply
            pattern_start: Starting index for pattern (if None, randomly placed)
            strength: Strength of the pattern (0.0 to 1.0)
            
        Returns:
            DataFrame with pattern applied
        """
        # This is a placeholder implementation
        # Full pattern implementation will be added in Phase 4
        
        periods = len(df)
        
        # If pattern_start is not specified, choose a random starting point
        # that allows the pattern to fit within the data
        if pattern_start is None:
            # Estimate pattern length based on the pattern type
            pattern_length = self._get_pattern_length(pattern, periods)
            max_start = periods - pattern_length
            if max_start <= 0:
                self.logger.warning(f"Not enough periods ({periods}) to fit pattern {pattern.value}")
                return df
                
            pattern_start = random.randint(0, max_start)
        
        # Log the pattern application
        self.logger.info(f"Applying {pattern.value} pattern starting at index {pattern_start} with strength {strength}")
        
        # Create a copy of the DataFrame to modify
        result_df = df.copy()
        
        # Apply the selected pattern
        if pattern == PatternType.HEAD_AND_SHOULDERS:
            self._apply_head_and_shoulders(result_df, pattern_start, strength)
        elif pattern == PatternType.DOUBLE_TOP:
            self._apply_double_top(result_df, pattern_start, strength)
        elif pattern == PatternType.DOUBLE_BOTTOM:
            self._apply_double_bottom(result_df, pattern_start, strength)
        elif pattern == PatternType.ASCENDING_TRIANGLE:
            self._apply_ascending_triangle(result_df, pattern_start, strength)
        elif pattern == PatternType.DESCENDING_TRIANGLE:
            self._apply_descending_triangle(result_df, pattern_start, strength)
        # Add other patterns in Phase 4
        
        return result_df
    
    def _get_pattern_length(self, pattern: PatternType, total_periods: int) -> int:
        """Estimate the length needed for a pattern based on pattern type."""
        # Return pattern length as a fraction of total periods
        if pattern == PatternType.HEAD_AND_SHOULDERS:
            return min(int(total_periods * 0.4), 40)
        elif pattern in [PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM]:
            return min(int(total_periods * 0.3), 30)
        elif pattern in [PatternType.ASCENDING_TRIANGLE, PatternType.DESCENDING_TRIANGLE]:
            return min(int(total_periods * 0.25), 25)
        elif pattern == PatternType.CUP_AND_HANDLE:
            return min(int(total_periods * 0.4), 40)
        elif pattern in [PatternType.BULLISH_FLAG, PatternType.BEARISH_FLAG]:
            return min(int(total_periods * 0.15), 15)
        else:
            return min(int(total_periods * 0.2), 20)
    
    # Pattern application methods will be implemented in Phase 4
    # These are placeholder implementations with basic functionality
    
    def _apply_head_and_shoulders(self, df: pd.DataFrame, start: int, strength: float) -> None:
        """Apply a head and shoulders pattern."""
        # Pattern parameters
        pattern_length = self._get_pattern_length(PatternType.HEAD_AND_SHOULDERS, len(df))
        if start + pattern_length >= len(df):
            pattern_length = len(df) - start - 1
            
        # Divide the pattern into segments
        segment_length = pattern_length // 5  # 5 segments: left shoulder, uptrend, head, downtrend, right shoulder
        
        # Get the price at the start of the pattern
        base_price = df.iloc[start]['close']
        
        # Define relative heights for each part
        left_shoulder_height = 0.05 * strength
        head_height = 0.08 * strength
        right_shoulder_height = 0.045 * strength
        
        # Create the pattern
        for i in range(pattern_length):
            idx = start + i
            if idx >= len(df):
                break
                
            # Determine which segment we're in
            segment = i // segment_length
            progress = (i % segment_length) / segment_length  # 0 to 1 within segment
            
            if segment == 0:  # Left shoulder formation
                # Rising to left shoulder peak
                if progress < 0.5:
                    modifier = left_shoulder_height * (progress * 2)
                else:  # Falling from left shoulder
                    modifier = left_shoulder_height * (1 - (progress - 0.5) * 2)
            elif segment == 1:  # Uptrend to head
                modifier = left_shoulder_height * (1 - progress) + head_height * progress
            elif segment == 2:  # Head formation
                if progress < 0.5:
                    modifier = head_height
                else:  # Falling from head
                    modifier = head_height * (1 - (progress - 0.5) * 2)
            elif segment == 3:  # Downtrend to right shoulder
                modifier = head_height * (1 - progress) * 0.5 + right_shoulder_height * progress
            elif segment == 4:  # Right shoulder formation
                if progress < 0.5:
                    modifier = right_shoulder_height * (progress * 2)
                else:  # Falling from right shoulder
                    modifier = right_shoulder_height * (1 - (progress - 0.5) * 2)
            
            # Apply the pattern modifier to prices
            price_adjustment = base_price * modifier
            df.loc[df.index[idx], 'close'] = base_price * (1 + modifier)
            
            # Adjust high and low prices based on the close adjustment
            current_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
            df.loc[df.index[idx], 'high'] = max(df.iloc[idx]['close'], df.iloc[idx]['open']) + current_range * 0.4
            df.loc[df.index[idx], 'low'] = min(df.iloc[idx]['close'], df.iloc[idx]['open']) - current_range * 0.4
            
        # After pattern, add a downtrend to simulate the breakdown
        breakdown_length = min(pattern_length // 3, len(df) - (start + pattern_length))
        for i in range(breakdown_length):
            idx = start + pattern_length + i
            if idx >= len(df):
                break
                
            progress = i / breakdown_length
            # Gradual decline after pattern completion
            decline = 0.06 * strength * progress
            df.loc[df.index[idx], 'close'] = base_price * (1 - decline)
            
            # Adjust high and low prices
            current_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
            df.loc[df.index[idx], 'high'] = max(df.iloc[idx]['close'], df.iloc[idx]['open']) + current_range * 0.3
            df.loc[df.index[idx], 'low'] = min(df.iloc[idx]['close'], df.iloc[idx]['open']) - current_range * 0.3
    
    def _apply_double_top(self, df: pd.DataFrame, start: int, strength: float) -> None:
        """Apply a double top pattern."""
        # Basic implementation - will be enhanced in Phase 4
        pattern_length = self._get_pattern_length(PatternType.DOUBLE_TOP, len(df))
        if start + pattern_length >= len(df):
            pattern_length = len(df) - start - 1
            
        # Get the price at the start of the pattern
        base_price = df.iloc[start]['close']
        peak_height = 0.07 * strength
        
        # Divide into segments
        segment_length = pattern_length // 4  # 4 segments: rise, first top, middle, second top
        
        # Create the pattern
        for i in range(pattern_length):
            idx = start + i
            if idx >= len(df):
                break
                
            segment = i // segment_length
            progress = (i % segment_length) / segment_length
            
            if segment == 0:  # Initial rise
                modifier = peak_height * progress
            elif segment == 1:  # First top
                modifier = peak_height
            elif segment == 2:  # Middle pullback
                modifier = peak_height * (1 - 0.3 * progress)
            elif segment == 3:  # Second top
                if progress < 0.5:
                    modifier = peak_height * (0.7 + 0.3 * progress)
                else:
                    modifier = peak_height * (1 - 0.2 * (progress - 0.5))
            
            # Apply the pattern modifier
            df.loc[df.index[idx], 'close'] = base_price * (1 + modifier)
            
            # Adjust high and low
            current_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
            df.loc[df.index[idx], 'high'] = max(df.iloc[idx]['close'], df.iloc[idx]['open']) + current_range * 0.4
            df.loc[df.index[idx], 'low'] = min(df.iloc[idx]['close'], df.iloc[idx]['open']) - current_range * 0.4
        
        # Add breakdown after pattern
        breakdown_length = min(pattern_length // 2, len(df) - (start + pattern_length))
        for i in range(breakdown_length):
            idx = start + pattern_length + i
            if idx >= len(df):
                break
                
            progress = i / breakdown_length
            df.loc[df.index[idx], 'close'] = base_price * (1 + peak_height * (0.8 - progress * 1.3))
    
    def _apply_double_bottom(self, df: pd.DataFrame, start: int, strength: float) -> None:
        """Apply a double bottom pattern."""
        # Placeholder implementation
        # Will be fully implemented in Phase 4
        pass
    
    def _apply_ascending_triangle(self, df: pd.DataFrame, start: int, strength: float) -> None:
        """Apply an ascending triangle pattern."""
        # Placeholder implementation
        # Will be fully implemented in Phase 4
        pass
    
    def _apply_descending_triangle(self, df: pd.DataFrame, start: int, strength: float) -> None:
        """Apply a descending triangle pattern."""
        # Placeholder implementation
        # Will be fully implemented in Phase 4
        pass
