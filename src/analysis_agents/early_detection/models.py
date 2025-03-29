"""
Models for the Early Event Detection System.

This module defines the data models used throughout the early event detection system.
"""

from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union


class SourceType(Enum):
    """Types of information sources."""
    OFFICIAL = "official"  # Central banks, government publications
    SOCIAL_MEDIA = "social_media"  # Twitter, Reddit, etc.
    NEWS = "news"  # News articles, blogs
    FINANCIAL_DATA = "financial_data"  # Market data, alternative data
    OTHER = "other"  # Other sources


class EventCategory(Enum):
    """Categories of detected events."""
    MONETARY_POLICY = "monetary_policy"  # Interest rates, central bank actions
    REGULATION = "regulation"  # Regulatory changes, legal developments
    TRADE_WAR = "trade_war"  # Trade tensions, tariffs
    GEOPOLITICAL = "geopolitical"  # Conflicts, international relations
    ECONOMIC_DATA = "economic_data"  # Economic indicators, reports
    CORPORATE = "corporate"  # Corporate events, earnings
    MARKET = "market"  # Market-specific events
    TECHNOLOGY = "technology"  # Technology developments
    SOCIAL = "social"  # Social trends, public sentiment
    OTHER = "other"  # Other events


class ConfidenceLevel(Enum):
    """Confidence levels for detected events."""
    VERY_HIGH = 5  # Multiple confirming sources, very reliable
    HIGH = 4  # Multiple sources, mostly reliable
    MEDIUM = 3  # Some corroboration, moderate reliability
    LOW = 2  # Limited corroboration, questionable reliability
    VERY_LOW = 1  # Uncorroborated, low reliability


class ImpactMagnitude(Enum):
    """Magnitude of potential market impact."""
    CRITICAL = 5  # Immediate, large-scale impact
    SIGNIFICANT = 4  # Significant impact expected
    MODERATE = 3  # Moderate impact expected
    MINOR = 2  # Minor impact expected
    NEGLIGIBLE = 1  # Little to no impact expected


class ImpactTimeframe(Enum):
    """Timeframe for potential market impact."""
    IMMEDIATE = "immediate"  # Within hours
    SHORT_TERM = "short_term"  # Days to a week
    MEDIUM_TERM = "medium_term"  # Weeks to a month
    LONG_TERM = "long_term"  # Months or longer


class ImpactDirection(Enum):
    """Expected direction of market impact."""
    POSITIVE = "positive"  # Positive for markets/prices
    NEGATIVE = "negative"  # Negative for markets/prices
    MIXED = "mixed"  # Mixed impact, varies by asset
    UNCLEAR = "unclear"  # Direction unclear or uncertain


@dataclass
class EventSource:
    """Information about the source of an event."""
    id: str
    type: SourceType
    name: str
    url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    reliability_score: float = 0.5  # 0-1 scale
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyEvent:
    """Represents a detected early event."""
    id: str
    title: str
    description: str
    category: EventCategory
    sources: List[EventSource] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assess_impact(self, 
                     assets: List[str],
                     magnitude: ImpactMagnitude = ImpactMagnitude.MODERATE,
                     direction: ImpactDirection = ImpactDirection.UNCLEAR,
                     timeframe: ImpactTimeframe = ImpactTimeframe.SHORT_TERM) -> Dict[str, Any]:
        """Assess the potential market impact of this event.
        
        Args:
            assets: List of assets to assess impact for
            magnitude: Expected magnitude of impact
            direction: Expected direction of impact
            timeframe: Expected timeframe for impact
            
        Returns:
            Impact assessment dictionary
        """
        impact = {
            "magnitude": magnitude,
            "direction": direction,
            "timeframe": timeframe,
            "assets": {}
        }
        
        # Assign impact to each asset
        # In a real implementation, this would be more sophisticated
        base_score = magnitude.value / 5.0  # Normalize to 0-1
        
        for asset in assets:
            # Default to the overall direction
            asset_direction = direction
            asset_score = base_score
            
            # Add to assets dictionary
            impact["assets"][asset] = {
                "direction": asset_direction,
                "score": asset_score,
                "confidence": self.confidence.value / 5.0  # Normalize to 0-1
            }
        
        # Store the impact assessment
        self.impact_assessment = impact
        
        return impact


@dataclass
class EventSignal:
    """Trading signal derived from detected early events."""
    id: str
    event_ids: List[str]
    title: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    confidence: float = 0.5  # 0-1 scale
    assets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 2  # 1-5 scale (5 being highest)
    metadata: Dict[str, Any] = field(default_factory=dict)