"""Market event detection module.

This module provides functionality for detecting significant market events
in real-time, such as volatility spikes, trend changes, and liquidity events.
"""

from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

class EventType(Enum):
    """Types of market events."""
    VOLATILITY_SPIKE = auto()
    TREND_CHANGE = auto()
    SUPPORT_BREAK = auto()
    RESISTANCE_BREAK = auto()
    VOLUME_SPIKE = auto()
    LIQUIDITY_DROP = auto()
    MOMENTUM_SHIFT = auto()
    CORRELATION_BREAK = auto()
    NEWS_IMPACT = auto()

@dataclass
class MarketEvent:
    """Details of a detected market event."""
    event_type: EventType
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    significance: float  # 0 to 1 scale
    description: str
    metadata: Dict[str, Union[float, str]]  # Allow both float and string values

class EventDetector:
    """Detect significant market events in real-time."""
    
    def __init__(
        self,
        volatility_threshold: float = 2.0,
        volume_threshold: float = 3.0,
        trend_threshold: float = 0.1,
        correlation_threshold: float = 0.3,
        lookback_window: int = 100
    ):
        """Initialize event detector.
        
        Args:
            volatility_threshold: Standard deviations for volatility events
            volume_threshold: Standard deviations for volume events
            trend_threshold: Minimum price change for trend events
            correlation_threshold: Correlation change threshold
            lookback_window: Number of periods for baseline calculations
        """
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.trend_threshold = trend_threshold
        self.correlation_threshold = correlation_threshold
        self.lookback_window = lookback_window
        
        # Event history
        self.events: List[MarketEvent] = []
        
        # Price and volume history
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.timestamp_history: Dict[str, List[datetime]] = {}
        
        # Technical levels
        self.support_levels: Dict[str, List[float]] = {}
        self.resistance_levels: Dict[str, List[float]] = {}
        
        # Correlation tracking
        self.correlation_baseline: Dict[Tuple[str, str], float] = {}
        
        # Event counters for rate limiting
        self.event_counters: Dict[str, Dict[EventType, int]] = {}
    
    def add_data_point(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: float,
        correlations: Optional[Dict[str, float]] = None
    ) -> List[MarketEvent]:
        """Add a new data point and check for events.
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            price: Current price
            volume: Current volume
            correlations: Optional correlation data
            
        Returns:
            List of detected events
        """
        # Initialize history for new symbols
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.timestamp_history[symbol] = []
            self.event_counters[symbol] = {event_type: 0 for event_type in EventType}
        
        # Update history
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        self.timestamp_history[symbol].append(timestamp)
        
        # Maintain lookback window
        if len(self.price_history[symbol]) > self.lookback_window:
            self.price_history[symbol].pop(0)
            self.volume_history[symbol].pop(0)
            self.timestamp_history[symbol].pop(0)
        
        # Detect events
        new_events = []
        
        # Only check for events if we have enough history
        if len(self.price_history[symbol]) >= 20:  # Minimum history requirement
            volatility_events = self._check_volatility(symbol)
            volume_events = self._check_volume(symbol)
            trend_events = self._check_trend(symbol)
            technical_events = self._check_technical_levels(symbol)
            momentum_events = self._check_momentum(symbol)
            
            new_events.extend(volatility_events)
            new_events.extend(volume_events)
            new_events.extend(trend_events)
            new_events.extend(technical_events)
            new_events.extend(momentum_events)
            
            # Check correlation breaks if data is provided
            if correlations:
                correlation_events = self._check_correlations(symbol, correlations)
                new_events.extend(correlation_events)
        
        # Update event history
        self.events.extend(new_events)
        
        # Update technical levels periodically
        if len(self.price_history[symbol]) >= self.lookback_window:
            self._update_technical_levels(symbol)
        
        return new_events
    
    def _check_volatility(self, symbol: str) -> List[MarketEvent]:
        """Check for volatility-related events.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of detected events
        """
        events = []
        prices = np.array(self.price_history[symbol])
        
        # Calculate rolling volatility
        returns = np.diff(np.log(prices))
        current_vol = float(np.std(returns[-20:]))  # 20-period volatility
        baseline_vol = float(np.std(returns[:-20]))  # Historical volatility
        
        # Check for volatility spike
        if current_vol > baseline_vol * self.volatility_threshold:
            # Rate limit volatility events
            if self._can_trigger_event(symbol, EventType.VOLATILITY_SPIKE):
                events.append(MarketEvent(
                    event_type=EventType.VOLATILITY_SPIKE,
                    symbol=symbol,
                    timestamp=self.timestamp_history[symbol][-1],
                    price=float(prices[-1]),
                    volume=float(self.volume_history[symbol][-1]),
                    significance=float(min(
                        current_vol / baseline_vol / self.volatility_threshold,
                        1.0
                    )),
                    description=f"Volatility spike detected: {current_vol:.2%} vs {baseline_vol:.2%}",
                    metadata={
                        'current_volatility': float(current_vol),
                        'baseline_volatility': float(baseline_vol),
                        'threshold': float(self.volatility_threshold)
                    }
                ))
        
        return events
    
    def _check_volume(self, symbol: str) -> List[MarketEvent]:
        """Check for volume-related events.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of detected events
        """
        events = []
        volumes = np.array(self.volume_history[symbol])
        
        # Calculate volume metrics
        current_vol = float(volumes[-1])
        avg_vol = float(np.mean(volumes[:-1]))
        vol_std = float(np.std(volumes[:-1]))
        
        # Check for volume spike
        if current_vol > avg_vol + vol_std * self.volume_threshold:
            if self._can_trigger_event(symbol, EventType.VOLUME_SPIKE):
                events.append(MarketEvent(
                    event_type=EventType.VOLUME_SPIKE,
                    symbol=symbol,
                    timestamp=self.timestamp_history[symbol][-1],
                    price=float(self.price_history[symbol][-1]),
                    volume=current_vol,
                    significance=float(min(
                        (current_vol - avg_vol) / (vol_std * self.volume_threshold),
                        1.0
                    )),
                    description=f"Volume spike detected: {current_vol:,.0f} vs avg {avg_vol:,.0f}",
                    metadata={
                        'current_volume': float(current_vol),
                        'average_volume': float(avg_vol),
                        'volume_std': float(vol_std)
                    }
                ))
        
        # Check for liquidity drop
        if current_vol < avg_vol - vol_std * 2:
            if self._can_trigger_event(symbol, EventType.LIQUIDITY_DROP):
                events.append(MarketEvent(
                    event_type=EventType.LIQUIDITY_DROP,
                    symbol=symbol,
                    timestamp=self.timestamp_history[symbol][-1],
                    price=float(self.price_history[symbol][-1]),
                    volume=current_vol,
                    significance=float(min((avg_vol - current_vol) / (vol_std * 2), 1.0)),
                    description=f"Liquidity drop detected: {current_vol:,.0f} vs avg {avg_vol:,.0f}",
                    metadata={
                        'current_volume': float(current_vol),
                        'average_volume': float(avg_vol),
                        'volume_std': float(vol_std)
                    }
                ))
        
        return events
    
    def _check_trend(self, symbol: str) -> List[MarketEvent]:
        """Check for trend-related events.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of detected events
        """
        events = []
        prices = np.array(self.price_history[symbol])
        
        # Calculate trend metrics
        short_ma = np.mean(prices[-20:])
        long_ma = np.mean(prices[-50:]) if len(prices) >= 50 else short_ma
        price_change = (prices[-1] / prices[0] - 1)
        
        # Check for trend change
        if abs(price_change) > self.trend_threshold:
            if self._can_trigger_event(symbol, EventType.TREND_CHANGE):
                events.append(MarketEvent(
                    event_type=EventType.TREND_CHANGE,
                    symbol=symbol,
                    timestamp=self.timestamp_history[symbol][-1],
                    price=float(prices[-1]),
                    volume=float(self.volume_history[symbol][-1]),
                    significance=float(min(abs(price_change) / self.trend_threshold, 1.0)),
                    description=f"Trend change detected: {price_change:.2%} move",
                    metadata={
                        'price_change': float(price_change),
                        'short_ma': float(short_ma),
                        'long_ma': float(long_ma)
                    }
                ))
        
        return events
    
    def _check_technical_levels(self, symbol: str) -> List[MarketEvent]:
        """Check for technical level breaks.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of detected events
        """
        events = []
        current_price = self.price_history[symbol][-1]
        
        # Check support breaks
        if symbol in self.support_levels:
            for level in self.support_levels[symbol]:
                if (current_price < level and 
                    self._can_trigger_event(symbol, EventType.SUPPORT_BREAK)):
                    events.append(MarketEvent(
                        event_type=EventType.SUPPORT_BREAK,
                        symbol=symbol,
                        timestamp=self.timestamp_history[symbol][-1],
                        price=float(current_price),
                        volume=float(self.volume_history[symbol][-1]),
                        significance=float(min(abs(level - current_price) / level, 1.0)),
                        description=f"Support break at {level:.2f}",
                        metadata={'support_level': float(level)}
                    ))
        
        # Check resistance breaks
        if symbol in self.resistance_levels:
            for level in self.resistance_levels[symbol]:
                if (current_price > level and 
                    self._can_trigger_event(symbol, EventType.RESISTANCE_BREAK)):
                    events.append(MarketEvent(
                        event_type=EventType.RESISTANCE_BREAK,
                        symbol=symbol,
                        timestamp=self.timestamp_history[symbol][-1],
                        price=float(current_price),
                        volume=float(self.volume_history[symbol][-1]),
                        significance=float(min(abs(current_price - level) / level, 1.0)),
                        description=f"Resistance break at {level:.2f}",
                        metadata={'resistance_level': float(level)}
                    ))
        
        return events
    
    def _check_momentum(self, symbol: str) -> List[MarketEvent]:
        """Check for momentum shifts.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of detected events
        """
        events = []
        prices = np.array(self.price_history[symbol])
        
        # Calculate momentum metrics
        returns = np.diff(np.log(prices))
        current_momentum = np.sum(returns[-5:])
        prev_momentum = np.sum(returns[-10:-5])
        
        # Check for momentum shift
        if abs(current_momentum - prev_momentum) > self.trend_threshold:
            if self._can_trigger_event(symbol, EventType.MOMENTUM_SHIFT):
                events.append(MarketEvent(
                    event_type=EventType.MOMENTUM_SHIFT,
                    symbol=symbol,
                    timestamp=self.timestamp_history[symbol][-1],
                    price=float(prices[-1]),
                    volume=float(self.volume_history[symbol][-1]),
                    significance=float(min(
                        abs(current_momentum - prev_momentum) / self.trend_threshold,
                        1.0
                    )),
                    description=f"Momentum shift detected",
                    metadata={
                        'current_momentum': float(current_momentum),
                        'previous_momentum': float(prev_momentum)
                    }
                ))
        
        return events
    
    def _check_correlations(
        self,
        symbol: str,
        correlations: Dict[str, float]
    ) -> List[MarketEvent]:
        """Check for correlation breaks."""
        events = []
        
        # Update correlation baseline
        for other_symbol, corr in correlations.items():
            # Ensure we always have exactly 2 strings in the tuple
            sorted_symbols = sorted([symbol, other_symbol])
            if len(sorted_symbols) != 2:
                continue
            pair = (sorted_symbols[0], sorted_symbols[1])
            
            if pair not in self.correlation_baseline:
                self.correlation_baseline[pair] = corr
            else:
                # Check for correlation break
                baseline = self.correlation_baseline[pair]
                if abs(corr - baseline) > self.correlation_threshold:
                    if self._can_trigger_event(symbol, EventType.CORRELATION_BREAK):
                        events.append(MarketEvent(
                            event_type=EventType.CORRELATION_BREAK,
                            symbol=symbol,
                            timestamp=self.timestamp_history[symbol][-1],
                            price=float(self.price_history[symbol][-1]),
                            volume=float(self.volume_history[symbol][-1]),
                            significance=float(min(
                                abs(corr - baseline) / self.correlation_threshold,
                                1.0
                            )),
                            description=(
                                f"Correlation break with {other_symbol}: "
                                f"{corr:.2f} vs baseline {baseline:.2f}"
                            ),
                            metadata={
                                'current_correlation': float(corr),
                                'baseline_correlation': float(baseline),
                                'other_symbol': other_symbol
                            }
                        ))
                # Update baseline with exponential smoothing
                self.correlation_baseline[pair] = 0.95 * baseline + 0.05 * corr
        
        return events
    
    def _update_technical_levels(self, symbol: str) -> None:
        """Update support and resistance levels.
        
        Args:
            symbol: Trading symbol
        """
        prices = np.array(self.price_history[symbol])
        
        # Find local minima and maxima
        peaks = []
        troughs = []
        
        for i in range(2, len(prices) - 2):
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                peaks.append(prices[i])
            elif (prices[i] < prices[i-1] and prices[i] < prices[i-2] and
                  prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                troughs.append(prices[i])
        
        # Update levels
        self.resistance_levels[symbol] = sorted(set(peaks))[-3:]  # Keep top 3
        self.support_levels[symbol] = sorted(set(troughs))[:3]    # Keep bottom 3
    
    def _can_trigger_event(self, symbol: str, event_type: EventType) -> bool:
        """Check if an event can be triggered (rate limiting).
        
        Args:
            symbol: Trading symbol
            event_type: Type of event
            
        Returns:
            True if event can be triggered
        """
        # Rate limit: maximum 1 event of each type per 20 periods
        if self.event_counters[symbol][event_type] < 20:
            self.event_counters[symbol][event_type] = 0
            return True
        
        self.event_counters[symbol][event_type] += 1
        return False
    
    def get_recent_events(
        self,
        symbol: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 10
    ) -> List[MarketEvent]:
        """Get recent market events.
        
        Args:
            symbol: Optional symbol filter
            event_type: Optional event type filter
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        filtered_events = self.events
        
        if symbol:
            filtered_events = [e for e in filtered_events if e.symbol == symbol]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        return sorted(
            filtered_events,
            key=lambda e: e.timestamp,
            reverse=True
        )[:limit] 