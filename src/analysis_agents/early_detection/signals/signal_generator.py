"""
Signal Generator for the Early Event Detection System.

This module converts detected early events into actionable trading signals.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta

from src.common.config import config
from src.common.logging import get_logger
from src.analysis_agents.early_detection.models import (
    EarlyEvent, EventSignal, EventCategory,
    ConfidenceLevel, ImpactMagnitude, ImpactDirection, ImpactTimeframe
)


class SignalGenerator:
    """Generates trading signals from detected early events."""
    
    def __init__(self):
        """Initialize the signal generator."""
        self.logger = get_logger("early_detection", "signal_generator")
        self.is_initialized = False
        
        # Configure signal generation
        self.confidence_threshold = config.get("early_detection.signals.confidence_threshold", 0.6)
        self.impact_threshold = config.get("early_detection.signals.impact_threshold", 0.5)
    
    async def initialize(self):
        """Initialize the signal generator."""
        self.logger.info("Initializing signal generator")
        self.is_initialized = True
        self.logger.info("Signal generator initialized")
    
    async def generate_signals(self, events: List[EarlyEvent], assets: List[str]) -> List[EventSignal]:
        """Generate trading signals from detected early events.
        
        Args:
            events: List of detected early events
            assets: List of assets to generate signals for
            
        Returns:
            List of generated signals
        """
        if not self.is_initialized:
            self.logger.warning("Signal generator not initialized")
            return []
        
        self.logger.info(f"Generating signals from {len(events)} events")
        signals = []
        
        # Group events by category
        events_by_category = {}
        for event in events:
            category = event.category
            if category not in events_by_category:
                events_by_category[category] = []
            events_by_category[category].append(event)
        
        # Process high-confidence events individually
        high_confidence_events = [
            event for event in events 
            if event.confidence.value >= ConfidenceLevel.HIGH.value
        ]
        
        for event in high_confidence_events:
            signal = await self._generate_signal_from_event(event, assets)
            if signal:
                signals.append(signal)
        
        # Process event clusters by category
        for category, category_events in events_by_category.items():
            if len(category_events) < 2:
                continue
            
            # Skip if all events in this category were already processed individually
            if all(event in high_confidence_events for event in category_events):
                continue
            
            # Process event cluster
            signal = await self._generate_signal_from_cluster(category, category_events, assets)
            if signal:
                signals.append(signal)
        
        self.logger.info(f"Generated {len(signals)} signals")
        return signals
    
    async def _generate_signal_from_event(self, event: EarlyEvent, assets: List[str]) -> Optional[EventSignal]:
        """Generate a signal from a single high-confidence event.
        
        Args:
            event: The event to generate a signal from
            assets: List of assets to generate signals for
            
        Returns:
            Generated signal or None
        """
        # Check if event has impact assessment
        if not event.impact_assessment:
            return None
        
        # Check if impact is significant enough
        magnitude = event.impact_assessment.get("magnitude")
        if not magnitude or magnitude.value < ImpactMagnitude.MODERATE.value:
            return None
        
        # Calculate signal confidence
        confidence = event.confidence.value / 5.0  # Normalize to 0-1
        if confidence < self.confidence_threshold:
            return None
        
        # Create signal ID
        signal_id = f"signal_{event.id}"
        
        # Determine signal expiration
        timeframe = event.impact_assessment.get("timeframe", ImpactTimeframe.SHORT_TERM)
        expires_at = None
        
        if timeframe == ImpactTimeframe.IMMEDIATE:
            expires_at = datetime.now() + timedelta(hours=4)
        elif timeframe == ImpactTimeframe.SHORT_TERM:
            expires_at = datetime.now() + timedelta(days=1)
        elif timeframe == ImpactTimeframe.MEDIUM_TERM:
            expires_at = datetime.now() + timedelta(days=3)
        
        # Generate asset-specific signals
        signal_assets = {}
        impact_assets = event.impact_assessment.get("assets", {})
        
        for asset in assets:
            if asset in impact_assets:
                asset_impact = impact_assets[asset]
                impact_score = asset_impact.get("score", 0.0)
                impact_direction = asset_impact.get("direction", ImpactDirection.UNCLEAR)
                asset_confidence = asset_impact.get("confidence", confidence)
                
                # Skip if impact is not significant enough
                if impact_score < self.impact_threshold:
                    continue
                
                # Determine signal direction and action
                signal_direction = "neutral"
                if impact_direction == ImpactDirection.POSITIVE:
                    signal_direction = "bullish"
                elif impact_direction == ImpactDirection.NEGATIVE:
                    signal_direction = "bearish"
                
                # Add to signal assets
                signal_assets[asset] = {
                    "direction": signal_direction,
                    "score": impact_score,
                    "confidence": asset_confidence,
                    "timeframe": timeframe.value
                }
        
        # Skip if no assets have significant impact
        if not signal_assets:
            return None
        
        # Determine signal priority based on event confidence and impact
        priority = min(5, max(1, int(confidence * 5) + int(magnitude.value / 2)))
        
        # Generate recommended actions
        recommended_actions = []
        for asset, asset_signal in signal_assets.items():
            direction = asset_signal["direction"]
            score = asset_signal["score"]
            
            if direction == "bullish" and score > 0.7:
                recommended_actions.append({
                    "asset": asset,
                    "action": "increase_position",
                    "size": "medium",
                    "reason": f"Positive early event impact expected on {asset}"
                })
            elif direction == "bearish" and score > 0.7:
                recommended_actions.append({
                    "asset": asset,
                    "action": "decrease_position",
                    "size": "medium",
                    "reason": f"Negative early event impact expected on {asset}"
                })
            elif direction == "bullish" and score > 0.5:
                recommended_actions.append({
                    "asset": asset,
                    "action": "increase_position",
                    "size": "small",
                    "reason": f"Moderately positive early event impact expected on {asset}"
                })
            elif direction == "bearish" and score > 0.5:
                recommended_actions.append({
                    "asset": asset,
                    "action": "decrease_position",
                    "size": "small",
                    "reason": f"Moderately negative early event impact expected on {asset}"
                })
        
        # Create signal
        signal = EventSignal(
            id=signal_id,
            event_ids=[event.id],
            title=event.title,
            description=event.description,
            created_at=datetime.now(),
            expires_at=expires_at,
            confidence=confidence,
            assets=signal_assets,
            recommended_actions=recommended_actions,
            priority=priority
        )
        
        return signal
    
    async def _generate_signal_from_cluster(self, category: EventCategory, 
                                         events: List[EarlyEvent], 
                                         assets: List[str]) -> Optional[EventSignal]:
        """Generate a signal from a cluster of related events.
        
        Args:
            category: The event category
            events: The related events
            assets: List of assets to generate signals for
            
        Returns:
            Generated signal or None
        """
        if len(events) < 2:
            return None
        
        # Calculate aggregate confidence
        confidences = [event.confidence.value / 5.0 for event in events]  # Normalize to 0-1
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        
        # Use higher of average and max, with a small boost for having multiple events
        confidence = min(1.0, max(avg_confidence, max_confidence) + (len(events) - 1) * 0.05)
        
        if confidence < self.confidence_threshold:
            return None
        
        # Find highest impact event
        impact_events = [event for event in events if event.impact_assessment]
        if not impact_events:
            return None
        
        impact_events.sort(
            key=lambda e: e.impact_assessment.get("magnitude", ImpactMagnitude.MINOR).value,
            reverse=True
        )
        
        primary_event = impact_events[0]
        
        # Create signal ID
        signal_id = f"cluster_signal_{category.value}_{uuid.uuid4().hex[:8]}"
        
        # Determine signal expiration
        timeframe = primary_event.impact_assessment.get("timeframe", ImpactTimeframe.SHORT_TERM)
        expires_at = None
        
        if timeframe == ImpactTimeframe.IMMEDIATE:
            expires_at = datetime.now() + timedelta(hours=6)  # Slightly longer for cluster
        elif timeframe == ImpactTimeframe.SHORT_TERM:
            expires_at = datetime.now() + timedelta(days=2)
        elif timeframe == ImpactTimeframe.MEDIUM_TERM:
            expires_at = datetime.now() + timedelta(days=5)
        
        # Aggregate impact across events
        aggregate_impact = {}
        
        for event in impact_events:
            impact_assets = event.impact_assessment.get("assets", {})
            
            for asset, asset_impact in impact_assets.items():
                if asset not in aggregate_impact:
                    aggregate_impact[asset] = {
                        "score_sum": 0.0,
                        "score_count": 0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0
                    }
                
                impact_score = asset_impact.get("score", 0.0)
                impact_direction = asset_impact.get("direction", ImpactDirection.UNCLEAR)
                
                aggregate_impact[asset]["score_sum"] += impact_score
                aggregate_impact[asset]["score_count"] += 1
                
                if impact_direction == ImpactDirection.POSITIVE:
                    aggregate_impact[asset]["positive_count"] += 1
                elif impact_direction == ImpactDirection.NEGATIVE:
                    aggregate_impact[asset]["negative_count"] += 1
                else:
                    aggregate_impact[asset]["neutral_count"] += 1
        
        # Generate asset-specific signals
        signal_assets = {}
        
        for asset, impact in aggregate_impact.items():
            if impact["score_count"] == 0:
                continue
            
            avg_score = impact["score_sum"] / impact["score_count"]
            
            # Skip if impact is not significant enough
            if avg_score < self.impact_threshold:
                continue
            
            # Determine direction based on counts
            signal_direction = "neutral"
            if impact["positive_count"] > impact["negative_count"] * 1.5:
                signal_direction = "bullish"
            elif impact["negative_count"] > impact["positive_count"] * 1.5:
                signal_direction = "bearish"
            
            # Add to signal assets
            signal_assets[asset] = {
                "direction": signal_direction,
                "score": avg_score,
                "confidence": confidence,
                "timeframe": timeframe.value
            }
        
        # Skip if no assets have significant impact
        if not signal_assets:
            return None
        
        # Determine signal priority based on confidence and event count
        priority = min(5, max(1, int(confidence * 3) + min(2, len(events) - 1)))
        
        # Create title and description
        title = f"Multiple {category.value.replace('_', ' ')} events detected"
        description = f"Cluster of {len(events)} related events suggesting potential market impact"
        
        # Collect all event IDs
        event_ids = [event.id for event in events]
        
        # Generate recommended actions
        recommended_actions = []
        for asset, asset_signal in signal_assets.items():
            direction = asset_signal["direction"]
            score = asset_signal["score"]
            
            if direction == "bullish" and score > 0.7:
                recommended_actions.append({
                    "asset": asset,
                    "action": "increase_position",
                    "size": "medium",
                    "reason": f"Multiple events suggest positive impact on {asset}"
                })
            elif direction == "bearish" and score > 0.7:
                recommended_actions.append({
                    "asset": asset,
                    "action": "decrease_position",
                    "size": "medium",
                    "reason": f"Multiple events suggest negative impact on {asset}"
                })
            elif direction == "bullish" and score > 0.5:
                recommended_actions.append({
                    "asset": asset,
                    "action": "increase_position",
                    "size": "small",
                    "reason": f"Multiple events suggest moderately positive impact on {asset}"
                })
            elif direction == "bearish" and score > 0.5:
                recommended_actions.append({
                    "asset": asset,
                    "action": "decrease_position",
                    "size": "small",
                    "reason": f"Multiple events suggest moderately negative impact on {asset}"
                })
        
        # Create signal
        signal = EventSignal(
            id=signal_id,
            event_ids=event_ids,
            title=title,
            description=description,
            created_at=datetime.now(),
            expires_at=expires_at,
            confidence=confidence,
            assets=signal_assets,
            recommended_actions=recommended_actions,
            priority=priority
        )
        
        return signal