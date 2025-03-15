"""Models for the decision engine.

This module defines the data models used by the decision engine,
including predictions, aggregated predictions, and decision outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class PredictionSource(Enum):
    """Sources of predictions."""
    TECHNICAL = "technical"
    PATTERN = "pattern"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    ENSEMBLE = "ensemble"


class Direction(Enum):
    """Trading direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalType(Enum):
    """Types of trading signals."""
    ENTRY = "entry"
    EXIT = "exit"
    ADJUST = "adjust"
    NO_ACTION = "no_action"


class RiskLevel(Enum):
    """Risk levels for trades."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Prediction:
    """A prediction from an analysis agent."""
    
    # Core prediction information
    id: str  # Unique prediction ID
    source: PredictionSource  # Source of the prediction
    agent_id: str  # ID of the agent making the prediction
    symbol: str  # Trading pair symbol
    timestamp: datetime  # When the prediction was made
    
    # Prediction details
    direction: Direction  # Predicted market direction
    confidence: float  # Confidence score (0.0-1.0)
    timeframe: str  # Timeframe of the prediction (e.g., "1h", "4h")
    signal_type: SignalType  # Type of trading signal
    
    # Fields with default values
    expiration: Optional[datetime] = None  # When the prediction expires
    
    # Price targets
    entry_price: Optional[float] = None  # Suggested entry price
    stop_loss: Optional[float] = None  # Suggested stop loss price
    take_profit: Optional[float] = None  # Suggested take profit price
    
    # Additional information
    rationale: str = ""  # Explanation for the prediction
    tags: List[str] = field(default_factory=list)  # Tags for filtering/grouping
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    def is_valid(self) -> bool:
        """Check if the prediction is valid and not expired."""
        now = datetime.utcnow()
        return (self.expiration is None or now < self.expiration) and self.confidence > 0.0
        
    def __eq__(self, other) -> bool:
        """Check if two predictions are equal based on ID."""
        if not isinstance(other, Prediction):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)


@dataclass
class AggregatedPrediction:
    """An aggregated prediction from multiple sources."""
    
    # Core prediction information
    id: str  # Unique prediction ID
    symbol: str  # Trading pair symbol
    timestamp: datetime  # When the aggregation was done
    timeframe: str  # Consensus timeframe
    
    # Aggregated prediction details
    direction: Direction  # Aggregated direction
    confidence: float  # Aggregated confidence (0.0-1.0)
    signal_type: SignalType  # Aggregated signal type
    
    # Price targets
    entry_price: Optional[float] = None  # Consensus entry price
    stop_loss: Optional[float] = None  # Consensus stop loss price
    take_profit: Optional[float] = None  # Consensus take profit price
    
    # Component predictions
    predictions: List[Prediction] = field(default_factory=list)  # Contributing predictions
    weights: Dict[str, float] = field(default_factory=dict)  # Weights for each prediction
    
    # Additional information
    rationale: str = ""  # Explanation for the aggregated prediction
    tags: List[str] = field(default_factory=list)  # Tags for filtering/grouping
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    def is_high_confidence(self, threshold: float = 0.85) -> bool:
        """Check if this is a high-confidence prediction."""
        return self.confidence >= threshold
    
    def get_reward_risk_ratio(self) -> Optional[float]:
        """Calculate the reward-to-risk ratio if targets are available."""
        if (self.entry_price is None or 
            self.stop_loss is None or 
            self.take_profit is None):
            return None
            
        # Calculate based on direction
        if self.direction == Direction.BULLISH:
            reward = self.take_profit - self.entry_price
            risk = self.entry_price - self.stop_loss
        else:  # BEARISH
            reward = self.entry_price - self.take_profit
            risk = self.stop_loss - self.entry_price
            
        if risk <= 0:
            return None
            
        return reward / risk


@dataclass
class TradingDecision:
    """A trading decision output from the decision engine."""
    
    # Core decision information
    id: str  # Unique decision ID
    symbol: str  # Trading pair symbol
    timestamp: datetime  # When the decision was made
    
    # Decision details
    decision_type: SignalType  # Type of trading decision
    direction: Direction  # Trading direction
    confidence: float  # Decision confidence (0.0-1.0)
    risk_level: RiskLevel  # Risk level for this trade
    
    # Position sizing
    position_size: Optional[float] = None  # Position size as percentage or units
    position_value: Optional[float] = None  # Position size in quote currency
    
    # Price targets
    entry_price: Optional[float] = None  # Entry price
    entry_valid_until: Optional[datetime] = None  # Entry validity time
    stop_loss: Optional[float] = None  # Stop loss price
    take_profit_levels: List[Tuple[float, float]] = field(default_factory=list)  # (price, percentage)
    
    # Source information
    source_predictions: List[str] = field(default_factory=list)  # IDs of source predictions
    aggregated_prediction: Optional[str] = None  # ID of aggregated prediction
    
    # Additional information
    rationale: str = ""  # Explanation for the decision
    tags: List[str] = field(default_factory=list)  # Tags for filtering/grouping
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
