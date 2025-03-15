"""Moving average crossover strategy for the AI Crypto Trading System.

This module implements a simple moving average crossover strategy, which
generates entry and exit signals based on crosses between fast and slow
moving averages.
"""

import asyncio
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple

from src.common.config import config
from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData
from src.models.signals import Signal, SignalType
from src.strategy.base_strategy import Strategy


class MovingAverageCrossoverStrategy(Strategy):
    """Strategy that generates signals based on moving average crossovers.
    
    This strategy tracks fast and slow moving averages (SMA or EMA) and
    generates entry signals when the fast MA crosses above the slow MA,
    and exit signals when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, strategy_id: str = "ma_crossover"):
        """Initialize the moving average crossover strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = get_logger("strategy", "ma_crossover")
        
        # Strategy-specific configuration
        self.fast_ma_type = config.get(f"strategies.{strategy_id}.fast_ma.type", "EMA")
        self.fast_ma_period = config.get(f"strategies.{strategy_id}.fast_ma.period", 12)
        self.slow_ma_type = config.get(f"strategies.{strategy_id}.slow_ma.type", "EMA")
        self.slow_ma_period = config.get(f"strategies.{strategy_id}.slow_ma.period", 26)
        
        # Minimum confidence to generate signals
        self.min_confidence = config.get(f"strategies.{strategy_id}.min_confidence", 0.6)
        
        # Position management
        self.use_stop_loss = config.get(f"strategies.{strategy_id}.use_stop_loss", True)
        self.stop_loss_pct = config.get(f"strategies.{strategy_id}.stop_loss_pct", 0.02)  # 2%
        self.use_take_profit = config.get(f"strategies.{strategy_id}.use_take_profit", True)
        self.take_profit_pct = config.get(f"strategies.{strategy_id}.take_profit_pct", 0.05)  # 5%
        
        # Strategy state
        self.latest_ma_values: Dict[Tuple[str, TimeFrame], Dict[datetime, Dict[str, float]]] = {}
        self.signal_sent: Dict[Tuple[str, TimeFrame, str], datetime] = {}  # Prevent signal spamming
        self.min_signal_interval = config.get(f"strategies.{strategy_id}.min_signal_interval", 3600)  # seconds
    
    async def _strategy_initialize(self) -> None:
        """Strategy-specific initialization."""
        self.logger.info("Initializing moving average crossover strategy",
                      fast_ma=f"{self.fast_ma_type}{self.fast_ma_period}",
                      slow_ma=f"{self.slow_ma_type}{self.slow_ma_period}")
    
    async def _strategy_start(self) -> None:
        """Strategy-specific startup."""
        self.logger.info("Starting moving average crossover strategy")
        
        # Publish status
        await self.publish_status(
            "Moving average crossover strategy started",
            {
                "fast_ma": f"{self.fast_ma_type}{self.fast_ma_period}",
                "slow_ma": f"{self.slow_ma_type}{self.slow_ma_period}",
                "symbols": list(self.symbols) if self.symbols else "all",
                "timeframes": [tf.value for tf in self.timeframes] if self.timeframes else "all"
            }
        )
    
    async def _strategy_stop(self) -> None:
        """Strategy-specific shutdown."""
        self.logger.info("Stopping moving average crossover strategy")
        
        # Publish status
        await self.publish_status("Moving average crossover strategy stopped")
    
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        Args:
            candle: The candle data to process
        """
        # We handle market data analysis in the process_indicator method
        # since this strategy is based on indicators, not raw candles
        pass
    
    async def process_trade(self, trade: TradeData) -> None:
        """Process a new trade data event.
        
        Args:
            trade: The trade data to process
        """
        # This strategy doesn't use trade data
        pass
    
    async def process_orderbook(self, orderbook: OrderBookData) -> None:
        """Process a new order book event.
        
        Args:
            orderbook: The order book data to process
        """
        # This strategy doesn't use order book data
        pass
    
    async def process_indicator(
        self,
        symbol: str,
        timeframe: TimeFrame,
        indicator_name: str,
        values: Dict
    ) -> None:
        """Process a technical indicator update.
        
        Args:
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            indicator_name: The name of the indicator
            values: The indicator values
        """
        # We only care about the moving averages we're tracking
        key = (symbol, timeframe)
        
        # Store the MA values for later analysis
        if key not in self.latest_ma_values:
            self.latest_ma_values[key] = {}
        
        # Update our stored MA values
        if indicator_name in [self.fast_ma_type, self.slow_ma_type]:
            for timestamp, value_dict in values.items():
                if timestamp not in self.latest_ma_values[key]:
                    self.latest_ma_values[key][timestamp] = {}
                
                # Handle both simple values and dict values
                if isinstance(value_dict, dict):
                    # Find the right period we need
                    for period_name, ma_value in value_dict.items():
                        if indicator_name == self.fast_ma_type and f"{indicator_name}{self.fast_ma_period}" in period_name:
                            self.latest_ma_values[key][timestamp]["fast_ma"] = ma_value
                        elif indicator_name == self.slow_ma_type and f"{indicator_name}{self.slow_ma_period}" in period_name:
                            self.latest_ma_values[key][timestamp]["slow_ma"] = ma_value
                else:
                    # Simple value case (less common)
                    if indicator_name == self.fast_ma_type:
                        self.latest_ma_values[key][timestamp]["fast_ma"] = value_dict
                    elif indicator_name == self.slow_ma_type:
                        self.latest_ma_values[key][timestamp]["slow_ma"] = value_dict
        
            # Analyze for crossovers after updating values
            await self._analyze_crossovers(symbol, timeframe)
    
    async def process_pattern(
        self,
        symbol: str,
        timeframe: TimeFrame,
        pattern_name: str,
        confidence: float,
        target_price: Optional[float],
        invalidation_price: Optional[float]
    ) -> None:
        """Process a pattern detection.
        
        Args:
            symbol: The trading pair symbol
            timeframe: The candle timeframe
            pattern_name: The name of the pattern
            confidence: The confidence score for the pattern
            target_price: Optional price target for the pattern
            invalidation_price: Optional price level that would invalidate the pattern
        """
        # This strategy doesn't use pattern detections
        pass
    
    async def _analyze_crossovers(self, symbol: str, timeframe: TimeFrame) -> None:
        """Analyze for moving average crossovers.
        
        Args:
            symbol: The trading pair symbol
            timeframe: The timeframe to analyze
        """
        key = (symbol, timeframe)
        if key not in self.latest_ma_values:
            return
        
        # Sort timestamps in ascending order
        timestamps = sorted(self.latest_ma_values[key].keys())
        if len(timestamps) < 2:
            return  # Need at least two points to detect a crossover
        
        # Get the two most recent data points
        current = timestamps[-1]
        previous = timestamps[-2]
        
        # Check if we have all necessary values for both timestamps
        if ("fast_ma" not in self.latest_ma_values[key][current] or 
            "slow_ma" not in self.latest_ma_values[key][current] or
            "fast_ma" not in self.latest_ma_values[key][previous] or
            "slow_ma" not in self.latest_ma_values[key][previous]):
            return
        
        # Get the MA values
        current_fast = self.latest_ma_values[key][current]["fast_ma"]
        current_slow = self.latest_ma_values[key][current]["slow_ma"]
        previous_fast = self.latest_ma_values[key][previous]["fast_ma"]
        previous_slow = self.latest_ma_values[key][previous]["slow_ma"]
        
        # Detect crossovers
        # Bullish crossover: fast MA crosses above slow MA
        if previous_fast <= previous_slow and current_fast > current_slow:
            await self._generate_signal(
                symbol, 
                timeframe, 
                "long", 
                current_fast,  # use fast MA as approximate price
                "Bullish MA crossover detected"
            )
        
        # Bearish crossover: fast MA crosses below slow MA
        elif previous_fast >= previous_slow and current_fast < current_slow:
            await self._generate_signal(
                symbol, 
                timeframe, 
                "short", 
                current_fast,  # use fast MA as approximate price
                "Bearish MA crossover detected"
            )
    
    async def _generate_signal(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        direction: str,
        price: float,
        reason: str
    ) -> None:
        """Generate a trading signal based on analysis.
        
        Args:
            symbol: The trading pair symbol
            timeframe: The timeframe of the signal
            direction: The direction of the signal ("long" or "short")
            price: The current price
            reason: The reason for the signal
        """
        # Check if we should generate a signal
        signal_key = (symbol, timeframe, direction)
        now = utc_now()
        
        # Check if we've sent a similar signal recently
        if signal_key in self.signal_sent:
            time_since_last = (now - self.signal_sent[signal_key]).total_seconds()
            if time_since_last < self.min_signal_interval:
                self.logger.debug(
                    "Skipping signal (sent recently)",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    direction=direction,
                    seconds_ago=time_since_last
                )
                return
        
        # Check if we already have an active signal for this symbol
        if symbol in self.active_signals:
            active_direction = self.active_signals[symbol].direction
            
            # If the new direction is the same as the active direction, skip
            if active_direction == direction:
                return
            
            # Otherwise, this is an exit signal for the opposite direction
            await self.publish_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT,
                direction=active_direction,
                timeframe=timeframe,
                price=price,
                confidence=0.8,  # Crossing in the opposite direction is a strong exit signal
                reason=f"MA crossover in opposite direction",
                take_profit=None,
                stop_loss=None
            )
        
        # Calculate confidence based on the MA difference
        key = (symbol, timeframe)
        latest_timestamp = sorted(self.latest_ma_values[key].keys())[-1]
        fast_ma = self.latest_ma_values[key][latest_timestamp]["fast_ma"]
        slow_ma = self.latest_ma_values[key][latest_timestamp]["slow_ma"]
        
        # Calculate confidence based on the MA difference
        ma_diff_pct = abs(fast_ma - slow_ma) / slow_ma
        confidence = min(0.5 + ma_diff_pct * 10, 0.95)  # Scale to 0.5-0.95 range
        
        # Skip if confidence is too low
        if confidence < self.min_confidence:
            self.logger.debug(
                "Skipping signal (low confidence)",
                symbol=symbol,
                timeframe=timeframe.value,
                direction=direction,
                confidence=confidence
            )
            return
        
        # Calculate stop loss and take profit levels
        stop_loss = None
        take_profit = None
        
        if direction == "long":
            if self.use_stop_loss:
                stop_loss = price * (1 - self.stop_loss_pct)
            if self.use_take_profit:
                take_profit = price * (1 + self.take_profit_pct)
        else:  # short
            if self.use_stop_loss:
                stop_loss = price * (1 + self.stop_loss_pct)
            if self.use_take_profit:
                take_profit = price * (1 - self.take_profit_pct)
        
        # Generate the entry signal
        await self.publish_signal(
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            direction=direction,
            timeframe=timeframe,
            price=price,
            confidence=confidence,
            reason=reason,
            take_profit=take_profit,
            stop_loss=stop_loss
        )
        
        # Record that we sent this signal
        self.signal_sent[signal_key] = now 