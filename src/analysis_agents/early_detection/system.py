"""
Early Event Detection System main module.

This module implements the central system that coordinates data collection,
processing, analysis, and signal generation for early event detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus
from src.analysis_agents.early_detection.models import (
    EarlyEvent, EventSignal, EventCategory, SourceType,
    ConfidenceLevel, ImpactMagnitude, ImpactDirection, ImpactTimeframe
)
from src.analysis_agents.early_detection.data_collectors.collector_manager import DataCollectorManager
from src.analysis_agents.early_detection.processors.processor_manager import ProcessorManager
from src.analysis_agents.early_detection.analyzers.analyzer_manager import AnalyzerManager
from src.analysis_agents.early_detection.signals.signal_generator import SignalGenerator


class EarlyEventDetectionSystem:
    """Main system for early event detection.
    
    This class coordinates the various components of the early event detection system,
    including data collection, processing, analysis, and signal generation.
    """
    
    def __init__(self):
        """Initialize the early event detection system."""
        self.logger = get_logger("analysis_agents", "early_detection_system")
        
        # Configuration
        self.enabled = config.get("early_detection.enabled", True)
        self.update_interval = config.get("early_detection.update_interval", 3600)  # 1 hour
        self.assets = config.get("early_detection.assets", ["BTC", "ETH", "SOL", "XRP"])
        
        # Component managers
        self.collector_manager = DataCollectorManager()
        self.processor_manager = ProcessorManager()
        self.analyzer_manager = AnalyzerManager()
        self.signal_generator = SignalGenerator()
        
        # Data storage
        self.events: Dict[str, EarlyEvent] = {}
        self.signals: Dict[str, EventSignal] = {}
        
        # State tracking
        self.is_initialized = False
        self.is_running = False
        self.last_update = None
    
    async def initialize(self):
        """Initialize the early event detection system and its components."""
        if not self.enabled:
            self.logger.info("Early event detection system is disabled")
            return
        
        self.logger.info("Initializing early event detection system")
        
        try:
            # Initialize component managers
            await self.collector_manager.initialize()
            await self.processor_manager.initialize()
            await self.analyzer_manager.initialize()
            await self.signal_generator.initialize()
            
            self.is_initialized = True
            self.logger.info("Early event detection system initialized")
        except Exception as e:
            self.logger.error(f"Error initializing early event detection system: {e}")
            raise
    
    async def start(self):
        """Start the early event detection system."""
        if not self.enabled or not self.is_initialized:
            return
        
        if self.is_running:
            self.logger.warning("Early event detection system is already running")
            return
        
        self.logger.info("Starting early event detection system")
        self.is_running = True
        
        # Start background task
        asyncio.create_task(self._run_detection_loop())
    
    async def stop(self):
        """Stop the early event detection system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping early event detection system")
        self.is_running = False
    
    async def _run_detection_loop(self):
        """Run the main detection loop."""
        self.logger.info("Detection loop started")
        
        while self.is_running:
            try:
                self._detection_cycle()
                self.last_update = datetime.now()
                
                # Wait for the next update interval
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in detection cycle: {e}")
                await asyncio.sleep(60)  # Wait a bit before retrying
    
    async def _detection_cycle(self):
        """Run a single detection cycle."""
        self.logger.info("Starting detection cycle")
        
        # 1. Collect data from all sources
        collection_results = await self.collector_manager.collect_data()
        self.logger.info(f"Collected data from {len(collection_results)} sources")
        
        # 2. Process the collected data
        processed_data = await self.processor_manager.process_data(collection_results)
        self.logger.info(f"Processed {len(processed_data)} data items")
        
        # 3. Analyze the processed data to detect events
        detected_events = await self.analyzer_manager.analyze_data(processed_data)
        self.logger.info(f"Detected {len(detected_events)} events")
        
        # 4. Store the detected events
        for event in detected_events:
            self.events[event.id] = event
        
        # 5. Generate signals from the detected events
        new_signals = await self.signal_generator.generate_signals(detected_events, self.assets)
        self.logger.info(f"Generated {len(new_signals)} signals")
        
        # 6. Store and publish the signals
        for signal in new_signals:
            self.signals[signal.id] = signal
            await self._publish_signal(signal)
        
        self.logger.info("Detection cycle completed")
    
    async def _publish_signal(self, signal: EventSignal):
        """Publish a signal to the event bus.
        
        Args:
            signal: The signal to publish
        """
        # Convert signal to event payload
        payload = {
            "signal_id": signal.id,
            "title": signal.title,
            "description": signal.description,
            "confidence": signal.confidence,
            "assets": signal.assets,
            "priority": signal.priority,
            "created_at": signal.created_at.isoformat(),
            "expires_at": signal.expires_at.isoformat() if signal.expires_at else None,
            "recommended_actions": signal.recommended_actions
        }
        
        # Publish to event bus
        await event_bus.publish(
            event_type="EarlyEventSignal",
            source="early_detection_system",
            payload=payload
        )
    
    async def get_active_events(self) -> List[EarlyEvent]:
        """Get all active events.
        
        Returns:
            List of active events
        """
        # Consider events active if they were detected in the last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        return [
            event for event in self.events.values()
            if event.detected_at >= cutoff_time
        ]
    
    async def get_active_signals(self) -> List[EventSignal]:
        """Get all active signals.
        
        Returns:
            List of active signals
        """
        # Consider signals active if they haven't expired
        now = datetime.now()
        
        return [
            signal for signal in self.signals.values()
            if not signal.expires_at or signal.expires_at > now
        ]
    
    async def get_event_by_id(self, event_id: str) -> Optional[EarlyEvent]:
        """Get an event by its ID.
        
        Args:
            event_id: The ID of the event to get
            
        Returns:
            The event if found, otherwise None
        """
        return self.events.get(event_id)
    
    async def get_signal_by_id(self, signal_id: str) -> Optional[EventSignal]:
        """Get a signal by its ID.
        
        Args:
            signal_id: The ID of the signal to get
            
        Returns:
            The signal if found, otherwise None
        """
        return self.signals.get(signal_id)
    
    async def get_events_by_category(self, category: EventCategory) -> List[EarlyEvent]:
        """Get events by category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of events in the specified category
        """
        return [
            event for event in self.events.values()
            if event.category == category
        ]
    
    async def get_high_confidence_events(self, min_confidence: ConfidenceLevel = ConfidenceLevel.HIGH) -> List[EarlyEvent]:
        """Get high confidence events.
        
        Args:
            min_confidence: The minimum confidence level
            
        Returns:
            List of high confidence events
        """
        return [
            event for event in self.events.values()
            if event.confidence.value >= min_confidence.value
        ]
    
    async def get_high_impact_events(self, 
                                    min_impact: ImpactMagnitude = ImpactMagnitude.SIGNIFICANT, 
                                    asset: Optional[str] = None) -> List[EarlyEvent]:
        """Get high impact events.
        
        Args:
            min_impact: The minimum impact magnitude
            asset: Optional asset to filter by
            
        Returns:
            List of high impact events
        """
        high_impact_events = []
        
        for event in self.events.values():
            # Check if the event has an impact assessment
            if not event.impact_assessment:
                continue
            
            # Get the event magnitude
            magnitude = event.impact_assessment.get("magnitude")
            if not magnitude or magnitude.value < min_impact.value:
                continue
            
            # If asset is specified, check if it's affected
            if asset:
                assets_impact = event.impact_assessment.get("assets", {})
                if asset not in assets_impact:
                    continue
            
            high_impact_events.append(event)
        
        return high_impact_events
    
    async def generate_market_impact_report(self) -> Dict[str, Any]:
        """Generate a market impact report based on detected events.
        
        Returns:
            Dictionary with market impact assessment
        """
        # Get active events
        active_events = self.get_active_events()
        
        if not active_events:
            return {
                "timestamp": datetime.now().isoformat(),
                "assets": {},
                "summary": "No active events detected"
            }
        
        # Initialize asset impact dictionaries
        assets_impact = {}
        for asset in self.assets:
            assets_impact[asset] = {
                "positive_events": [],
                "negative_events": [],
                "mixed_events": [],
                "impact_score": 0.0,
                "confidence": 0.0
            }
        
        # Analyze events impact on each asset
        for event in active_events:
            if not event.impact_assessment or "assets" not in event.impact_assessment:
                continue
            
            for asset, impact in event.impact_assessment["assets"].items():
                if asset not in assets_impact:
                    continue
                
                direction = impact.get("direction", ImpactDirection.UNCLEAR)
                score = impact.get("score", 0.0)
                confidence = impact.get("confidence", 0.0)
                
                # Add to appropriate list
                event_summary = {
                    "id": event.id,
                    "title": event.title,
                    "category": event.category.value,
                    "score": score,
                    "confidence": confidence
                }
                
                if direction == ImpactDirection.POSITIVE:
                    assets_impact[asset]["positive_events"].append(event_summary)
                    assets_impact[asset]["impact_score"] += score * confidence
                elif direction == ImpactDirection.NEGATIVE:
                    assets_impact[asset]["negative_events"].append(event_summary)
                    assets_impact[asset]["impact_score"] -= score * confidence
                else:
                    assets_impact[asset]["mixed_events"].append(event_summary)
                
                # Update overall confidence
                assets_impact[asset]["confidence"] = max(assets_impact[asset]["confidence"], confidence)
        
        # Calculate overall market sentiment
        overall_sentiment = sum(asset_data["impact_score"] for asset_data in assets_impact.values()) / len(assets_impact)
        
        # Generate summary
        if overall_sentiment > 0.3:
            summary = "Early events suggest positive market developments"
        elif overall_sentiment < -0.3:
            summary = "Early events suggest negative market pressure"
        else:
            summary = "Early events suggest mixed or neutral market conditions"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "assets": assets_impact,
            "overall_sentiment": overall_sentiment,
            "summary": summary
        }