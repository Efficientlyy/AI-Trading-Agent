"""
Integration module for connecting the Early Event Detection System
with the main trading agent.

This module provides functionality to integrate early event detection
with the trading strategy and decision engine.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.analysis_agents.early_detection import EarlyEventDetectionSystem
from src.models.signals import Signal, SignalType
from src.models.analysis_events import SignalEvent


class EarlyEventIntegration:
    """Integration for Early Event Detection System.
    
    This class connects the early event detection system with the
    trading agent's strategy and decision engine.
    """
    
    def __init__(self):
        """Initialize the early event integration."""
        self.logger = get_logger("early_detection", "integration")
        self.detection_system = None
        self.is_initialized = False
        
        # Configuration
        self.enabled = config.get("early_detection.integration.enabled", True)
        self.signal_confidence_threshold = config.get("early_detection.integration.signal_confidence_threshold", 0.7)
    
    async def initialize(self):
        """Initialize the integration."""
        if not self.enabled:
            self.logger.info("Early event integration is disabled")
            return
        
        self.logger.info("Initializing early event integration")
        
        # Initialize the detection system
        self.detection_system = EarlyEventDetectionSystem()
        await self.detection_system.initialize()
        
        # Register event handlers
        await self._register_event_handlers()
        
        self.is_initialized = True
        self.logger.info("Early event integration initialized")
    
    async def start(self):
        """Start the early event integration."""
        if not self.enabled or not self.is_initialized:
            return
        
        self.logger.info("Starting early event integration")
        
        # Start the detection system
        await self.detection_system.start()
    
    async def stop(self):
        """Stop the early event integration."""
        if not self.enabled or not self.is_initialized:
            return
        
        self.logger.info("Stopping early event integration")
        
        # Stop the detection system
        await self.detection_system.stop()
    
    async def _register_event_handlers(self):
        """Register event handlers with the event bus."""
        # Register handler for early event signals
        await event_bus.subscribe("EarlyEventSignal", self._handle_early_event_signal)
    
    async def _handle_early_event_signal(self, event: Event):
        """Handle early event signals.
        
        Args:
            event: Event object containing the signal
        """
        if not event.payload:
            return
        
        self.logger.info(f"Handling early event signal: {event.payload.get('title')}")
        
        # Extract signal details
        signal_id = event.payload.get("signal_id")
        title = event.payload.get("title")
        description = event.payload.get("description")
        confidence = event.payload.get("confidence", 0.0)
        assets = event.payload.get("assets", {})
        
        # Skip signals with low confidence
        if confidence < self.signal_confidence_threshold:
            self.logger.debug(f"Skipping signal {signal_id} with low confidence: {confidence}")
            return
        
        # Generate trading signals for each affected asset
        for asset, asset_data in assets.items():
            direction = asset_data.get("direction", "neutral")
            score = asset_data.get("score", 0.0)
            
            # Skip neutral signals or low impact
            if direction == "neutral" or score < 0.4:
                continue
            
            # Convert to trading signal direction
            signal_direction = "buy" if direction == "bullish" else "sell" if direction == "bearish" else "neutral"
            
            # Create signal event
            await self._publish_signal_event(
                source="early_detection",
                symbol=asset,
                signal_type="entry" if score > 0.7 else "alert",
                direction=signal_direction,
                confidence=confidence,
                reason=title,
                metadata={
                    "early_event_signal_id": signal_id,
                    "description": description,
                    "score": score
                }
            )
    
    async def _publish_signal_event(self, 
                                   source: str, 
                                   symbol: str, 
                                   signal_type: str,
                                   direction: str,
                                   confidence: float,
                                   reason: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None):
        """Publish a signal event to the event bus.
        
        Args:
            source: Source of the signal
            symbol: Trading symbol
            signal_type: Type of signal (entry, exit, alert)
            direction: Direction of the signal (buy, sell, neutral)
            confidence: Confidence level (0-1)
            reason: Optional reason for the signal
            metadata: Optional additional metadata
        """
        # Create signal event
        signal_event = SignalEvent(
            source=source,
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            price=0.0,  # No specific price for early event signals
            timeframe="1d",  # Use daily timeframe for early event signals
            confidence=confidence,
            reason=reason,
            metadata=metadata
        )
        
        # Publish the signal event
        await event_bus.publish(
            event_type="SignalEvent",
            source=source,
            payload=signal_event.__dict__
        )
        
        self.logger.info(f"Published {direction} signal for {symbol} with confidence {confidence:.2f}")


# Helper function to initialize and start the integration
async def setup_early_event_integration():
    """Initialize and start the early event integration.
    
    Returns:
        The initialized integration object
    """
    integration = EarlyEventIntegration()
    await integration.initialize()
    await integration.start()
    return integration