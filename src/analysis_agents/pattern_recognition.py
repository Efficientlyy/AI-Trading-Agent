"""Pattern recognition analysis agent.

This module provides an analysis agent that detects technical chart patterns
from market data, including reversal patterns, continuation patterns, and
candlestick patterns.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from src.analysis_agents.base_agent import AnalysisAgent
from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import CandleData, TimeFrame


class PatternRecognitionAgent(AnalysisAgent):
    """Analysis agent for chart pattern recognition.
    
    This agent identifies common chart patterns in market data and
    publishes them as events with confidence scores, target prices,
    and invalidation levels.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the pattern recognition agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "pattern_recognition")
        
        # Pattern categories to detect
        self.pattern_categories = config.get(
            f"analysis_agents.{agent_id}.pattern_categories", 
            ["reversal", "continuation", "candlestick"]
        )
        
        # Pattern types within each category
        self.patterns = {}
        for category in self.pattern_categories:
            self.patterns[category] = config.get(
                f"analysis_agents.{agent_id}.patterns.{category}.types", 
                []
            )
        
        # Pattern parameters
        self.params = config.get(f"analysis_agents.{agent_id}.patterns", {})
        
        # Minimum required candles for pattern detection
        self.min_candles = config.get(f"analysis_agents.{agent_id}.min_candles", 50)
        
        # Minimum pattern quality threshold (0-100)
        self.min_pattern_quality = config.get(
            f"analysis_agents.{agent_id}.min_pattern_quality", 
            75.0
        )
        
        # Whether to require volume confirmation
        self.require_volume_confirmation = config.get(
            f"analysis_agents.{agent_id}.require_volume_confirmation", 
            True
        )
        
        # Cached data for pattern detection
        self.candle_cache: Dict[Tuple[str, str, TimeFrame], List[CandleData]] = {}
        self.max_cache_size = config.get(f"analysis_agents.{agent_id}.max_cache_size", 500)
        
        # Detected patterns cache to avoid duplicate detections
        self.detected_patterns: Dict[Tuple[str, str, TimeFrame, str], datetime] = {}
        
        # Historical pattern reliability tracking
        self.pattern_reliability: Dict[str, Dict[str, float]] = {}
        
    def _candles_to_dataframe(self, candles: List[CandleData]) -> pd.DataFrame:
        """Convert a list of candles to a pandas DataFrame.
        
        Args:
            candles: The list of candles to convert
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if not candles:
            return pd.DataFrame()
        
        # Extract OHLCV data from candles
        data = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'timestamp': []
        }
        
        for candle in candles:
            data['open'].append(candle.open)
            data['high'].append(candle.high)
            data['low'].append(candle.low)
            data['close'].append(candle.close)
            data['volume'].append(candle.volume)
            data['timestamp'].append(candle.timestamp)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Set timestamp as index and sort
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        Args:
            candle: The candle data to process
        """
        # Add to cache
        key = (candle.exchange, candle.symbol, candle.timeframe)
        if key not in self.candle_cache:
            self.candle_cache[key] = []
        
        # Check if we already have this candle or a candle with same timestamp
        for i, existing_candle in enumerate(self.candle_cache[key]):
            if existing_candle.timestamp == candle.timestamp:
                # Replace the existing candle with updated data
                self.candle_cache[key][i] = candle
                return
        
        # Add new candle and enforce cache size limit
        self.candle_cache[key].append(candle)
        if len(self.candle_cache[key]) > self.max_cache_size:
            self.candle_cache[key].pop(0)
        
        # Run pattern detection if we have enough candles
        if len(self.candle_cache[key]) >= self.min_candles:
            await self._detect_patterns(
                candle.exchange,
                candle.symbol,
                candle.timeframe,
                self.candle_cache[key]
            )
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data for a symbol, exchange, and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        if not candles or len(candles) < self.min_candles:
            return
        
        # Update the cache with these candles
        key = (exchange, symbol, timeframe)
        
        # If we don't have anything in the cache yet, just use these candles
        if key not in self.candle_cache or not self.candle_cache[key]:
            # Initialize cache with a copy of candles (up to max cache size)
            self.candle_cache[key] = candles[-self.max_cache_size:]
        else:
            # Merge new candles with existing cache
            existing_timestamps = {c.timestamp for c in self.candle_cache[key]}
            for candle in candles:
                if candle.timestamp not in existing_timestamps:
                    self.candle_cache[key].append(candle)
                    existing_timestamps.add(candle.timestamp)
            
            # Sort by timestamp and limit size
            self.candle_cache[key].sort(key=lambda c: c.timestamp)
            if len(self.candle_cache[key]) > self.max_cache_size:
                self.candle_cache[key] = self.candle_cache[key][-self.max_cache_size:]
        
        # Run pattern detection on the complete dataset
        await self._detect_patterns(exchange, symbol, timeframe, self.candle_cache[key])
    
    async def _detect_patterns(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Detect patterns in market data.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        if not candles or len(candles) < self.min_candles:
            return
        
        # Convert candles to pandas DataFrame for pattern detection
        df = self._candles_to_dataframe(candles)
        if df.empty:
            return
        
        # Get the latest price
        latest_price = df['close'].iloc[-1]
        
        # Detect patterns by category
        for category in self.pattern_categories:
            if category == "reversal":
                await self._detect_reversal_patterns(exchange, symbol, timeframe, df, latest_price)
            elif category == "continuation":
                await self._detect_continuation_patterns(exchange, symbol, timeframe, df, latest_price)
            elif category == "candlestick":
                await self._detect_candlestick_patterns(exchange, symbol, timeframe, df, latest_price)
    
    async def _detect_reversal_patterns(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect reversal patterns in market data.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        patterns = self.patterns.get("reversal", [])
        
        for pattern_name in patterns:
            try:
                if pattern_name == "HeadAndShoulders":
                    await self._detect_head_and_shoulders(exchange, symbol, timeframe, df, latest_price)
                elif pattern_name == "InverseHeadAndShoulders":
                    await self._detect_inverse_head_and_shoulders(exchange, symbol, timeframe, df, latest_price)
                elif pattern_name == "DoubleTop":
                    await self._detect_double_top(exchange, symbol, timeframe, df, latest_price)
                elif pattern_name == "DoubleBottom":
                    await self._detect_double_bottom(exchange, symbol, timeframe, df, latest_price)
                # Add more reversal patterns as needed
            except Exception as e:
                self.logger.error("Error detecting reversal pattern", 
                               pattern=pattern_name,
                               symbol=symbol,
                               exchange=exchange,
                               timeframe=timeframe.value,
                               error=str(e))
    
    async def _detect_continuation_patterns(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect continuation patterns in market data.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        patterns = self.patterns.get("continuation", [])
        
        for pattern_name in patterns:
            try:
                if pattern_name == "Flag":
                    await self._detect_flag_pattern(exchange, symbol, timeframe, df, latest_price)
                elif pattern_name == "Pennant":
                    await self._detect_pennant_pattern(exchange, symbol, timeframe, df, latest_price)
                elif pattern_name == "Triangle":
                    await self._detect_triangle_pattern(exchange, symbol, timeframe, df, latest_price)
                # Add more continuation patterns as needed
            except Exception as e:
                self.logger.error("Error detecting continuation pattern", 
                               pattern=pattern_name,
                               symbol=symbol,
                               exchange=exchange,
                               timeframe=timeframe.value,
                               error=str(e))
    
    async def _detect_candlestick_patterns(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect candlestick patterns in market data.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        patterns = self.patterns.get("candlestick", [])
        
        for pattern_name in patterns:
            try:
                if pattern_name == "Engulfing":
                    await self._detect_engulfing_pattern(exchange, symbol, timeframe, df, latest_price)
                elif pattern_name == "Hammer":
                    await self._detect_hammer_pattern(exchange, symbol, timeframe, df, latest_price)
                elif pattern_name == "ShootingStar":
                    await self._detect_shooting_star_pattern(exchange, symbol, timeframe, df, latest_price)
                # Add more candlestick patterns as needed
            except Exception as e:
                self.logger.error("Error detecting candlestick pattern", 
                               pattern=pattern_name,
                               symbol=symbol,
                               exchange=exchange,
                               timeframe=timeframe.value,
                               error=str(e))
    
    async def _detect_head_and_shoulders(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect head and shoulders pattern (bearish reversal).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Minimum formation bars from config
        min_formation_bars = self.params.get("reversal", {}).get(
            "min_formation_bars", 15
        )
        
        # Need at least this many bars for the pattern
        if len(df) < min_formation_bars:
            return
        
        # Get recent price data for analysis (last N bars)
        window = min(len(df), 100)  # Look at up to 100 recent bars
        closes = df['close'].values[-window:]
        
        # Pattern detection logic:
        # 1. Find 3 peaks, with middle peak (head) higher than the others
        # 2. The two shoulders should be at similar heights
        # 3. Neckline connects the lows between peaks
        # 4. Price should recently break below neckline
        
        # Simplified detection algorithm for proof of concept
        # (In production, would use more sophisticated peak detection and validation)
        
        # Find local peaks
        peaks = []
        for i in range(2, len(closes) - 2):
            if (closes[i] > closes[i-1] and 
                closes[i] > closes[i-2] and 
                closes[i] > closes[i+1] and 
                closes[i] > closes[i+2]):
                peaks.append((i, closes[i]))
        
        # Need at least 3 peaks
        if len(peaks) < 3:
            return
        
        # Look for head and shoulders formations among the peaks
        for i in range(len(peaks) - 2):
            # Get three consecutive peaks
            left_idx, left_peak = peaks[i]
            head_idx, head_peak = peaks[i+1]
            right_idx, right_peak = peaks[i+2]
            
            # Check if middle peak is higher than the others
            if (head_peak > left_peak and 
                head_peak > right_peak and
                abs(left_peak - right_peak) / ((left_peak + right_peak) / 2) < 0.1):  # Shoulders within 10% height
                
                # Find the lows between peaks
                left_trough = min(closes[left_idx:head_idx])
                right_trough = min(closes[head_idx:right_idx])
                
                # Calculate neckline
                neckline_start = left_trough
                neckline_end = right_trough
                
                # Check if price has broken below neckline
                if latest_price < neckline_end:
                    # Calculate pattern quality and confidence
                    quality = self._calculate_hs_pattern_quality(
                        left_peak, head_peak, right_peak,
                        left_trough, right_trough,
                        latest_price
                    )
                    
                    # If quality meets threshold, publish the pattern
                    if quality >= self.min_pattern_quality:
                        # Calculate target and invalidation levels
                        pattern_height = head_peak - ((left_trough + right_trough) / 2)
                        target_price = neckline_end - pattern_height
                        invalidation_price = head_peak
                        
                        # Calculate confidence based on quality and other factors
                        confidence = quality / 100.0
                        
                        # Check if this is a new pattern or recently detected
                        pattern_key = (exchange, symbol, timeframe, "HeadAndShoulders")
                        current_time = datetime.utcnow()
                        
                        # Only publish if new pattern or >12h since last detection
                        if (pattern_key not in self.detected_patterns or
                            (current_time - self.detected_patterns[pattern_key]).total_seconds() > 43200):
                            
                            # Update detection timestamp
                            self.detected_patterns[pattern_key] = current_time
                            
                            # Publish pattern event
                            await self.publish_pattern(
                                symbol=symbol,
                                timeframe=timeframe,
                                pattern_name="HeadAndShoulders",
                                confidence=confidence,
                                target_price=target_price,
                                invalidation_price=invalidation_price
                            )
                            
                            self.logger.info("Detected Head and Shoulders pattern", 
                                           symbol=symbol,
                                           timeframe=timeframe.value,
                                           confidence=confidence,
                                           target_price=target_price)
    
    async def _detect_inverse_head_and_shoulders(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect inverse head and shoulders pattern (bullish reversal).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Minimum formation bars from config
        min_formation_bars = self.params.get("reversal", {}).get(
            "min_formation_bars", 15
        )
        
        # Need at least this many bars for the pattern
        if len(df) < min_formation_bars:
            return
        
        # Get recent price data for analysis (last N bars)
        window = min(len(df), 100)  # Look at up to 100 recent bars
        closes = df['close'].values[-window:]
        
        # Pattern detection logic:
        # 1. Find 3 troughs, with middle trough (head) lower than the others
        # 2. The two shoulders should be at similar heights
        # 3. Neckline connects the highs between troughs
        # 4. Price should recently break above neckline
        
        # Simplified detection algorithm for proof of concept
        
        # Find local troughs
        troughs = []
        for i in range(2, len(closes) - 2):
            if (closes[i] < closes[i-1] and 
                closes[i] < closes[i-2] and 
                closes[i] < closes[i+1] and 
                closes[i] < closes[i+2]):
                troughs.append((i, closes[i]))
        
        # Need at least 3 troughs
        if len(troughs) < 3:
            return
        
        # Look for inverse head and shoulders formations among the troughs
        for i in range(len(troughs) - 2):
            # Get three consecutive troughs
            left_idx, left_trough = troughs[i]
            head_idx, head_trough = troughs[i+1]
            right_idx, right_trough = troughs[i+2]
            
            # Check if middle trough is lower than the others
            if (head_trough < left_trough and 
                head_trough < right_trough and
                abs(left_trough - right_trough) / ((left_trough + right_trough) / 2) < 0.1):  # Shoulders within 10% height
                
                # Find the highs between troughs
                left_peak = max(closes[left_idx:head_idx])
                right_peak = max(closes[head_idx:right_idx])
                
                # Calculate neckline
                neckline_start = left_peak
                neckline_end = right_peak
                
                # Check if price has broken above neckline
                if latest_price > neckline_end:
                    # Calculate pattern quality and confidence
                    quality = self._calculate_ihs_pattern_quality(
                        left_trough, head_trough, right_trough,
                        left_peak, right_peak,
                        latest_price
                    )
                    
                    # If quality meets threshold, publish the pattern
                    if quality >= self.min_pattern_quality:
                        # Calculate target and invalidation levels
                        pattern_height = ((left_peak + right_peak) / 2) - head_trough
                        target_price = neckline_end + pattern_height
                        invalidation_price = head_trough
                        
                        # Calculate confidence based on quality and other factors
                        confidence = quality / 100.0
                        
                        # Check if this is a new pattern or recently detected
                        pattern_key = (exchange, symbol, timeframe, "InverseHeadAndShoulders")
                        current_time = datetime.utcnow()
                        
                        # Only publish if new pattern or >12h since last detection
                        if (pattern_key not in self.detected_patterns or
                            (current_time - self.detected_patterns[pattern_key]).total_seconds() > 43200):
                            
                            # Update detection timestamp
                            self.detected_patterns[pattern_key] = current_time
                            
                            # Publish pattern event
                            await self.publish_pattern(
                                symbol=symbol,
                                timeframe=timeframe,
                                pattern_name="InverseHeadAndShoulders",
                                confidence=confidence,
                                target_price=target_price,
                                invalidation_price=invalidation_price
                            )
                            
                            self.logger.info("Detected Inverse Head and Shoulders pattern", 
                                           symbol=symbol,
                                           timeframe=timeframe.value,
                                           confidence=confidence,
                                           target_price=target_price)
    
    def _calculate_hs_pattern_quality(
        self,
        left_peak: float,
        head_peak: float,
        right_peak: float,
        left_trough: float,
        right_trough: float,
        current_price: float
    ) -> float:
        """Calculate quality score for head and shoulders pattern.
        
        Args:
            left_peak: Left shoulder peak
            head_peak: Head peak
            right_peak: Right shoulder peak
            left_trough: Left trough
            right_trough: Right trough
            current_price: Current price
            
        Returns:
            Pattern quality score (0-100)
        """
        # Calculate various pattern quality factors
        
        # 1. Symmetry of shoulders
        shoulder_diff = abs(left_peak - right_peak)
        shoulder_avg = (left_peak + right_peak) / 2
        shoulder_symmetry = 100 - min(100, (shoulder_diff / shoulder_avg) * 100)
        
        # 2. Head prominence
        head_prominence = min(100, (((head_peak / shoulder_avg) - 1) * 100) * 5)
        
        # 3. Neckline flatness
        neckline_diff = abs(left_trough - right_trough)
        neckline_avg = (left_trough + right_trough) / 2
        neckline_flatness = 100 - min(100, (neckline_diff / neckline_avg) * 100)
        
        # 4. Neckline break strength
        break_strength = min(100, ((neckline_avg - current_price) / neckline_avg) * 100 * 5)
        break_strength = max(0, break_strength)  # Ensure non-negative
        
        # Combine factors with weights
        quality = (
            shoulder_symmetry * 0.3 +
            head_prominence * 0.2 +
            neckline_flatness * 0.25 +
            break_strength * 0.25
        )
        
        return quality
    
    def _calculate_ihs_pattern_quality(
        self,
        left_trough: float,
        head_trough: float,
        right_trough: float,
        left_peak: float,
        right_peak: float,
        current_price: float
    ) -> float:
        """Calculate quality score for inverse head and shoulders pattern.
        
        Args:
            left_trough: Left shoulder trough
            head_trough: Head trough
            right_trough: Right shoulder trough
            left_peak: Left peak
            right_peak: Right peak
            current_price: Current price
            
        Returns:
            Pattern quality score (0-100)
        """
        # Calculate various pattern quality factors
        
        # 1. Symmetry of shoulders
        shoulder_diff = abs(left_trough - right_trough)
        shoulder_avg = (left_trough + right_trough) / 2
        shoulder_symmetry = 100 - min(100, (shoulder_diff / shoulder_avg) * 100)
        
        # 2. Head prominence
        head_prominence = min(100, ((1 - (head_trough / shoulder_avg)) * 100) * 5)
        
        # 3. Neckline flatness
        neckline_diff = abs(left_peak - right_peak)
        neckline_avg = (left_peak + right_peak) / 2
        neckline_flatness = 100 - min(100, (neckline_diff / neckline_avg) * 100)
        
        # 4. Neckline break strength
        break_strength = min(100, ((current_price - neckline_avg) / neckline_avg) * 100 * 5)
        break_strength = max(0, break_strength)  # Ensure non-negative
        
        # Combine factors with weights
        quality = (
            shoulder_symmetry * 0.3 +
            head_prominence * 0.2 +
            neckline_flatness * 0.25 +
            break_strength * 0.25
        )
        
        return quality
        
    async def _detect_double_top(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect double top pattern (bearish reversal).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Implementation placeholder - would follow similar approach to head and shoulders
        # This is a simplified placeholder for the method structure
        pass
        
    async def _detect_double_bottom(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect double bottom pattern (bullish reversal).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Implementation placeholder - would follow similar approach to inverse head and shoulders
        # This is a simplified placeholder for the method structure
        pass
    
    async def _detect_flag_pattern(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect flag pattern (continuation).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Implementation placeholder for flag pattern detection
        # This is a simplified placeholder for the method structure
        pass
    
    async def _detect_pennant_pattern(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect pennant pattern (continuation).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Implementation placeholder for pennant pattern detection
        # This is a simplified placeholder for the method structure
        pass
    
    async def _detect_triangle_pattern(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect triangle pattern (continuation).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Implementation placeholder for triangle pattern detection
        # This is a simplified placeholder for the method structure
        pass
    
    async def _detect_engulfing_pattern(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect engulfing candlestick pattern.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Need at least 2 bars
        if len(df) < 2:
            return
        
        # Get the last two candles
        prev_open = df['open'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        curr_open = df['open'].iloc[-1]
        curr_close = df['close'].iloc[-1]
        
        # Detect bullish engulfing
        if prev_close < prev_open and curr_close > curr_open:  # Previous bearish, current bullish
            if curr_open < prev_close and curr_close > prev_open:  # Current candle engulfs previous
                confidence = min(100, ((curr_close - curr_open) / (prev_open - prev_close)) * 70)
                confidence = min(95, max(70, confidence))  # Range: 70-95
                
                # Target and invalidation
                pattern_height = curr_close - curr_open
                target_price = curr_close + pattern_height
                invalidation_price = min(prev_close, curr_open) * 0.99  # Slightly below the pattern
                
                # Check if this is a new pattern or recently detected
                pattern_key = (exchange, symbol, timeframe, "BullishEngulfing")
                current_time = datetime.utcnow()
                
                # Only publish if new pattern or >6h since last detection
                if (pattern_key not in self.detected_patterns or
                    (current_time - self.detected_patterns[pattern_key]).total_seconds() > 21600):
                    
                    # Update detection timestamp
                    self.detected_patterns[pattern_key] = current_time
                    
                    # Publish pattern event
                    await self.publish_pattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_name="BullishEngulfing",
                        confidence=confidence / 100.0,
                        target_price=target_price,
                        invalidation_price=invalidation_price
                    )
                    
                    self.logger.info("Detected Bullish Engulfing pattern", 
                                   symbol=symbol,
                                   timeframe=timeframe.value,
                                   confidence=confidence / 100.0,
                                   target_price=target_price)
        
        # Detect bearish engulfing
        elif prev_close > prev_open and curr_close < curr_open:  # Previous bullish, current bearish
            if curr_open > prev_close and curr_close < prev_open:  # Current candle engulfs previous
                confidence = min(100, ((curr_open - curr_close) / (prev_close - prev_open)) * 70)
                confidence = min(95, max(70, confidence))  # Range: 70-95
                
                # Target and invalidation
                pattern_height = curr_open - curr_close
                target_price = curr_close - pattern_height
                invalidation_price = max(prev_close, curr_open) * 1.01  # Slightly above the pattern
                
                # Check if this is a new pattern or recently detected
                pattern_key = (exchange, symbol, timeframe, "BearishEngulfing")
                current_time = datetime.utcnow()
                
                # Only publish if new pattern or >6h since last detection
                if (pattern_key not in self.detected_patterns or
                    (current_time - self.detected_patterns[pattern_key]).total_seconds() > 21600):
                    
                    # Update detection timestamp
                    self.detected_patterns[pattern_key] = current_time
                    
                    # Publish pattern event
                    await self.publish_pattern(
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_name="BearishEngulfing",
                        confidence=confidence / 100.0,
                        target_price=target_price,
                        invalidation_price=invalidation_price
                    )
                    
                    self.logger.info("Detected Bearish Engulfing pattern", 
                                   symbol=symbol,
                                   timeframe=timeframe.value,
                                   confidence=confidence / 100.0,
                                   target_price=target_price)
    
    async def _detect_hammer_pattern(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect hammer candlestick pattern (bullish reversal).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Need at least 5 bars for context
        if len(df) < 5:
            return
        
        # Get the latest candle
        open_price = df['open'].iloc[-1]
        high_price = df['high'].iloc[-1]
        low_price = df['low'].iloc[-1]
        close_price = df['close'].iloc[-1]
        
        # Calculate body and shadow sizes
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        # For a hammer:
        # 1. Lower shadow should be at least 2x the body size
        # 2. Upper shadow should be small (< 10% of body)
        # 3. Price should be in a downtrend
        
        # Check for downtrend (lower lows and lower highs)
        recent_closes = df['close'].values[-6:-1]  # Previous 5 bars before current
        downtrend = all(recent_closes[i] > recent_closes[i+1] for i in range(len(recent_closes)-1))
        
        if (lower_shadow >= 2 * body_size and 
            upper_shadow <= 0.1 * body_size and
            downtrend):
            
            # Calculate confidence based on pattern quality
            lower_shadow_ratio = lower_shadow / body_size
            confidence = min(95, max(70, 70 + (lower_shadow_ratio - 2) * 10))
            
            # Target and invalidation
            target_price = close_price + (2 * body_size)
            invalidation_price = low_price * 0.99  # Slightly below the hammer's low
            
            # Check if this is a new pattern or recently detected
            pattern_key = (exchange, symbol, timeframe, "Hammer")
            current_time = datetime.utcnow()
            
            # Only publish if new pattern or >6h since last detection
            if (pattern_key not in self.detected_patterns or
                (current_time - self.detected_patterns[pattern_key]).total_seconds() > 21600):
                
                # Update detection timestamp
                self.detected_patterns[pattern_key] = current_time
                
                # Publish pattern event
                await self.publish_pattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_name="Hammer",
                    confidence=confidence / 100.0,
                    target_price=target_price,
                    invalidation_price=invalidation_price
                )
                
                self.logger.info("Detected Hammer pattern", 
                               symbol=symbol,
                               timeframe=timeframe.value,
                               confidence=confidence / 100.0,
                               target_price=target_price)
    
    async def _detect_shooting_star_pattern(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: TimeFrame, 
        df: pd.DataFrame,
        latest_price: float
    ) -> None:
        """Detect shooting star candlestick pattern (bearish reversal).
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            df: DataFrame of market data
            latest_price: The latest closing price
        """
        # Need at least 5 bars for context
        if len(df) < 5:
            return
        
        # Get the latest candle
        open_price = df['open'].iloc[-1]
        high_price = df['high'].iloc[-1]
        low_price = df['low'].iloc[-1]
        close_price = df['close'].iloc[-1]
        
        # Calculate body and shadow sizes
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        # For a shooting star:
        # 1. Upper shadow should be at least 2x the body size
        # 2. Lower shadow should be small (< 10% of body)
        # 3. Price should be in an uptrend
        
        # Check for uptrend (higher highs and higher lows)
        recent_closes = df['close'].values[-6:-1]  # Previous 5 bars before current
        uptrend = all(recent_closes[i] < recent_closes[i+1] for i in range(len(recent_closes)-1))
        
        if (upper_shadow >= 2 * body_size and 
            lower_shadow <= 0.1 * body_size and
            uptrend):
            
            # Calculate confidence based on pattern quality
            upper_shadow_ratio = upper_shadow / body_size
            confidence = min(95, max(70, 70 + (upper_shadow_ratio - 2) * 10))
            
            # Target and invalidation
            target_price = close_price - (2 * body_size)
            invalidation_price = high_price * 1.01  # Slightly above the shooting star's high
            
            # Check if this is a new pattern or recently detected
            pattern_key = (exchange, symbol, timeframe, "ShootingStar")
            current_time = datetime.utcnow()
            
            # Only publish if new pattern or >6h since last detection
            if (pattern_key not in self.detected_patterns or
                (current_time - self.detected_patterns[pattern_key]).total_seconds() > 21600):
                
                # Update detection timestamp
                self.detected_patterns[pattern_key] = current_time
                
                # Publish pattern event
                await self.publish_pattern(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_name="ShootingStar",
                    confidence=confidence / 100.0,
                    target_price=target_price,
                    invalidation_price=invalidation_price
                )
                
                self.logger.info("Detected Shooting Star pattern", 
                               symbol=symbol,
                               timeframe=timeframe.value,
                               confidence=confidence / 100.0,
                               target_price=target_price)
