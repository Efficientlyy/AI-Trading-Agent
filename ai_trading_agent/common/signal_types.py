"""
Signal Types for Trading System

This module defines the signal types and related classes used throughout the trading system.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import json

class SignalType(str, Enum):
    """Enumeration of signal types."""
    PATTERN = "pattern"
    INDICATOR = "indicator"
    PRICE_ACTION = "price_action"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    REGIME_CHANGE = "regime_change"
    CUSTOM = "custom"
    UNKNOWN = "unknown"

class SignalDirection(str, Enum):
    """Enumeration of signal directions."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class SignalConfidence(float):
    """
    Signal confidence level, represented as a float between 0.0 and 1.0.
    """
    
    def __new__(cls, value: float):
        """Create a new SignalConfidence instance."""
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return float.__new__(cls, value)
    
    @property
    def as_percentage(self) -> float:
        """Get confidence as a percentage."""
        return self * 100.0
    
    @property
    def label(self) -> str:
        """Get a text label for the confidence level."""
        if self < 0.3:
            return "low"
        elif self < 0.7:
            return "medium"
        else:
            return "high"

class Signal:
    """
    Trading signal representation.
    
    A signal represents a detected market condition or recommendation
    from a trading agent or system component.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        signal_type: SignalType,
        direction: SignalDirection,
        confidence: SignalConfidence,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new signal.
        
        Args:
            symbol: The trading symbol (e.g., "BTC/USD")
            timeframe: The timeframe (e.g., "1h", "4h", "1d")
            timestamp: When the signal was generated
            signal_type: Type of signal
            direction: Direction of the signal (bullish, bearish, etc.)
            confidence: Confidence level (0.0 to 1.0)
            source: Source of the signal (e.g., agent name)
            metadata: Additional signal-specific information
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.direction = direction
        self.confidence = confidence
        self.source = source
        self.metadata = metadata or {}
        
        # Generate a unique ID for the signal
        self.id = f"{self.source}_{self.signal_type}_{self.symbol}_{self.timeframe}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary representation."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type,
            "direction": self.direction,
            "confidence": float(self.confidence),
            "confidence_label": self.confidence.label,
            "source": self.source,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert signal to JSON string."""
        signal_dict = self.to_dict()
        return json.dumps(signal_dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create a signal from a dictionary."""
        # Convert string timestamp to datetime
        if isinstance(data.get("timestamp"), str):
            timestamp = datetime.fromisoformat(data["timestamp"])
        else:
            timestamp = data.get("timestamp", datetime.now())
        
        return cls(
            symbol=data.get("symbol", ""),
            timeframe=data.get("timeframe", ""),
            timestamp=timestamp,
            signal_type=SignalType(data.get("signal_type", SignalType.UNKNOWN)),
            direction=SignalDirection(data.get("direction", SignalDirection.NEUTRAL)),
            confidence=SignalConfidence(data.get("confidence", 0.5)),
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Signal':
        """Create a signal from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation of the signal."""
        return (
            f"Signal({self.signal_type}, {self.direction}, "
            f"{self.symbol}@{self.timeframe}, "
            f"confidence: {self.confidence:.2f}, "
            f"source: {self.source})"
        )
    
    def __repr__(self) -> str:
        """Representation of the signal."""
        return self.__str__()

class SignalCollection:
    """
    Collection of trading signals with filtering and aggregation capabilities.
    """
    
    def __init__(self, signals: Optional[List[Signal]] = None):
        """
        Initialize a signal collection.
        
        Args:
            signals: Initial list of signals
        """
        self.signals = signals or []
    
    def add(self, signal: Signal):
        """Add a signal to the collection."""
        self.signals.append(signal)
    
    def filter(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        signal_type: Optional[Union[SignalType, List[SignalType]]] = None,
        direction: Optional[Union[SignalDirection, List[SignalDirection]]] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> 'SignalCollection':
        """
        Filter signals based on criteria.
        
        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            signal_type: Filter by signal type(s)
            direction: Filter by direction(s)
            source: Filter by source
            min_confidence: Filter by minimum confidence
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            A new SignalCollection with filtered signals
        """
        # Convert single items to lists for consistent handling
        if signal_type and not isinstance(signal_type, list):
            signal_type = [signal_type]
        
        if direction and not isinstance(direction, list):
            direction = [direction]
        
        filtered = []
        
        for signal in self.signals:
            # Apply filters
            if symbol and signal.symbol != symbol:
                continue
                
            if timeframe and signal.timeframe != timeframe:
                continue
                
            if signal_type and signal.signal_type not in signal_type:
                continue
                
            if direction and signal.direction not in direction:
                continue
                
            if source and signal.source != source:
                continue
                
            if min_confidence and signal.confidence < min_confidence:
                continue
                
            if start_time and signal.timestamp < start_time:
                continue
                
            if end_time and signal.timestamp > end_time:
                continue
            
            filtered.append(signal)
        
        return SignalCollection(filtered)
    
    def get_latest(self, count: int = 1) -> List[Signal]:
        """
        Get the latest signals.
        
        Args:
            count: Number of latest signals to retrieve
            
        Returns:
            List of latest signals
        """
        sorted_signals = sorted(self.signals, key=lambda s: s.timestamp, reverse=True)
        return sorted_signals[:count]
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all signals to a list of dictionaries."""
        return [signal.to_dict() for signal in self.signals]
    
    def to_json(self) -> str:
        """Convert all signals to a JSON string."""
        return json.dumps(self.to_dict_list())
    
    def __len__(self) -> int:
        """Get the number of signals in the collection."""
        return len(self.signals)
    
    def __iter__(self):
        """Iterate through signals."""
        return iter(self.signals)
    
    def __getitem__(self, index):
        """Get a signal by index."""
        return self.signals[index]
