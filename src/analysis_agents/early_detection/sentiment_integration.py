"""
Sentiment Integration for Real-Time Event Detection.

This module integrates the real-time event detection system with
the sentiment analysis system to improve event detection and analysis.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Union

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus
from src.common.monitoring import metrics
from src.analysis_agents.early_detection.models import (
    EarlyEvent, EventSource, EventCategory, SourceType,
    ConfidenceLevel, ImpactMagnitude, ImpactDirection, ImpactTimeframe
)
from src.analysis_agents.early_detection.realtime_detector import RealtimeEventDetector
from src.analysis_agents.sentiment.consensus_system import MultiModelConsensusAgent


class SentimentEventIntegration:
    """Integration between sentiment analysis and event detection.
    
    This class connects the sentiment analysis system with the
    real-time event detection system to share data and insights.
    """
    
    def __init__(self, 
                 event_detector: Optional[RealtimeEventDetector] = None, 
                 consensus_agent: Optional[MultiModelConsensusAgent] = None):
        """Initialize the sentiment event integration.
        
        Args:
            event_detector: Real-time event detector (optional)
            consensus_agent: Multi-model consensus agent (optional)
        """
        self.logger = get_logger("analysis_agents", "sentiment_event_integration")
        
        # Configuration
        self.enabled = config.get("early_detection.sentiment_integration.enabled", True)
        self.min_consensus_confidence = config.get(
            "early_detection.sentiment_integration.min_consensus_confidence", 0.85
        )
        self.min_disagreement_level = config.get(
            "early_detection.sentiment_integration.min_disagreement_level", 0.3
        )
        self.event_subscriptions = config.get(
            "early_detection.sentiment_integration.event_subscriptions", 
            ["sentiment_event", "EarlyEventSignal", "RealtimeEventDetected"]
        )
        
        # Components
        self.event_detector = event_detector
        self.consensus_agent = consensus_agent
        
        # State tracking
        self.is_initialized = False
    
    async def initialize(self, 
                         event_detector: Optional[RealtimeEventDetector] = None, 
                         consensus_agent: Optional[MultiModelConsensusAgent] = None):
        """Initialize the integration.
        
        Args:
            event_detector: Real-time event detector (optional)
            consensus_agent: Multi-model consensus agent (optional)
        """
        if not self.enabled:
            self.logger.info("Sentiment event integration is disabled")
            return
        
        self.logger.info("Initializing sentiment event integration")
        
        # Set components if provided
        if event_detector:
            self.event_detector = event_detector
        
        if consensus_agent:
            self.consensus_agent = consensus_agent
        
        # Subscribe to events
        self._subscribe_to_events()
        
        self.is_initialized = True
        self.logger.info("Sentiment event integration initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to events from both systems."""
        # Subscribe to sentiment events
        async def handle_sentiment_event(event):
            await self._process_sentiment_event(event)
        
        # Subscribe to early event signals
        async def handle_early_event_signal(event):
            await self._process_early_event_signal(event)
        
        # Subscribe to realtime detected events
        async def handle_realtime_event(event):
            await self._process_realtime_event(event)
        
        # Register handlers for configured event types
        if "sentiment_event" in self.event_subscriptions:
            await event_bus.subscribe("sentiment_event", handle_sentiment_event)
            self.logger.debug("Subscribed to sentiment events")
        
        if "EarlyEventSignal" in self.event_subscriptions:
            await event_bus.subscribe("EarlyEventSignal", handle_early_event_signal)
            self.logger.debug("Subscribed to early event signals")
        
        if "RealtimeEventDetected" in self.event_subscriptions:
            await event_bus.subscribe("RealtimeEventDetected", handle_realtime_event)
            self.logger.debug("Subscribed to realtime events")
    
    async def _process_sentiment_event(self, event: Dict[str, Any]):
        """Process a sentiment event.
        
        Args:
            event: Sentiment event data
        """
        if not self.event_detector:
            return
        
        try:
            # Check if this is a high confidence sentiment event
            if not event.get("confidence", 0) >= self.min_consensus_confidence:
                return
            
            # Extract key information
            symbol = event.get("symbol", "")
            direction = event.get("direction", "neutral")
            value = event.get("value", 0.5)
            signal_type = event.get("signal_type", "sentiment")
            details = event.get("details", {})
            sources = event.get("sources", [])
            
            # Check if this is a consensus event with disagreement
            is_consensus = signal_type == "consensus"
            has_disagreement = details.get("disagreement_level", 0) >= self.min_disagreement_level
            
            # If this is a consensus with disagreement, it could be a potential event
            if is_consensus and has_disagreement:
                # Generate text description for analysis
                text = self._generate_consensus_text(event)
                
                # Submit to real-time event detector
                source_name = "sentiment_consensus"
                source_type = "financial_data"
                
                # Additional metadata
                metadata = {
                    "disagreement_level": details.get("disagreement_level", 0),
                    "direction_counts": details.get("direction_counts", {}),
                    "source_types": details.get("source_types", []),
                    "models": details.get("models", []),
                    "event_type": details.get("event_type", "consensus")
                }
                
                # Process as potential event
                await self.event_detector.process_text(
                    text=text,
                    source_name=source_name,
                    source_type=source_type,
                    metadata=metadata
                )
                
                self.logger.debug(f"Submitted consensus sentiment disagreement for {symbol} to event detector")
                
            # Check if this is an extreme sentiment event
            is_extreme = event.get("is_extreme", False)
            if is_extreme and signal_type in ["sentiment", "aggregate"]:
                # Generate text description for extreme sentiment
                text = self._generate_extreme_sentiment_text(event)
                
                # Submit to real-time event detector
                source_name = f"extreme_{signal_type}_sentiment"
                source_type = "financial_data"
                
                # Process as potential event
                await self.event_detector.process_text(
                    text=text,
                    source_name=source_name,
                    source_type=source_type,
                    metadata=details
                )
                
                self.logger.debug(f"Submitted extreme sentiment for {symbol} to event detector")
                
            # Check if this is a divergence event
            is_divergence = signal_type == "divergence"
            if is_divergence:
                # Generate text description for divergence
                text = self._generate_divergence_text(event)
                
                # Submit to real-time event detector
                source_name = "sentiment_divergence"
                source_type = "financial_data"
                
                # Process as potential event
                await self.event_detector.process_text(
                    text=text,
                    source_name=source_name,
                    source_type=source_type,
                    metadata=details
                )
                
                self.logger.debug(f"Submitted sentiment divergence for {symbol} to event detector")
        
        except Exception as e:
            self.logger.error(f"Error processing sentiment event: {e}")
    
    async def _process_early_event_signal(self, event: Dict[str, Any]):
        """Process an early event signal.
        
        Args:
            event: Early event signal data
        """
        if not self.consensus_agent:
            return
        
        try:
            # Extract key information
            signal_id = event.get("payload", {}).get("signal_id", "")
            title = event.get("payload", {}).get("title", "")
            description = event.get("payload", {}).get("description", "")
            assets = event.get("payload", {}).get("assets", [])
            confidence = event.get("payload", {}).get("confidence", 0.7)
            
            # Skip low confidence signals
            if confidence < 0.6:
                return
            
            # Generate text for the signal
            text = f"{title}\n\n{description}"
            
            # For each asset, submit sentiment
            for asset in assets:
                # Map assets to symbols (simple mapping for now)
                symbol = self._map_asset_to_symbol(asset)
                if not symbol:
                    continue
                
                # Map signal to sentiment
                value, direction = self._map_signal_to_sentiment(
                    event.get("payload", {})
                )
                
                # Submit to consensus agent
                await self.consensus_agent.submit_sentiment(
                    symbol=symbol,
                    value=value,
                    direction=direction,
                    confidence=confidence,
                    source_type="early_event_signal",
                    model="event_detection",
                    metadata={
                        "signal_id": signal_id,
                        "title": title,
                        "description": description[:100] + "..." if len(description) > 100 else description
                    }
                )
                
                self.logger.debug(f"Submitted early event signal sentiment for {symbol} to consensus agent")
        
        except Exception as e:
            self.logger.error(f"Error processing early event signal: {e}")
    
    async def _process_realtime_event(self, event: Dict[str, Any]):
        """Process a realtime event.
        
        Args:
            event: Realtime event data
        """
        if not self.consensus_agent:
            return
        
        try:
            # Extract key information
            payload = event.get("payload", {})
            event_id = payload.get("event_id", "")
            title = payload.get("title", "")
            category = payload.get("category", "")
            confidence_level = payload.get("confidence", 3)  # Default medium
            
            # Convert confidence level to float (0-1)
            confidence = confidence_level / 5.0 if isinstance(confidence_level, int) else 0.6
            
            # Get impact assessment
            impact = payload.get("impact_assessment", {})
            if not impact:
                return
            
            # Process impact for each asset
            for asset_id, asset_impact in impact.get("assets", {}).items():
                # Map to symbol
                symbol = self._map_asset_to_symbol(asset_id)
                if not symbol:
                    continue
                
                # Extract impact details
                direction_enum = asset_impact.get("direction", "unclear")
                score = asset_impact.get("score", 0.5)
                asset_confidence = asset_impact.get("confidence", confidence)
                
                # Map direction to sentiment direction
                if direction_enum == "positive":
                    direction = "bullish"
                    value = 0.5 + (score / 2)  # 0.5-1.0
                elif direction_enum == "negative":
                    direction = "bearish"
                    value = 0.5 - (score / 2)  # 0.0-0.5
                else:
                    direction = "neutral"
                    value = 0.5
                
                # Submit to consensus agent
                await self.consensus_agent.submit_sentiment(
                    symbol=symbol,
                    value=value,
                    direction=direction,
                    confidence=asset_confidence,
                    source_type="realtime_event",
                    model="event_detection",
                    metadata={
                        "event_id": event_id,
                        "title": title,
                        "category": category,
                        "impact_score": score,
                        "impact_direction": direction_enum
                    }
                )
                
                self.logger.debug(f"Submitted realtime event impact for {symbol} to consensus agent")
        
        except Exception as e:
            self.logger.error(f"Error processing realtime event: {e}")
    
    def _generate_consensus_text(self, event: Dict[str, Any]) -> str:
        """Generate text for consensus disagreement.
        
        Args:
            event: Sentiment event data
            
        Returns:
            Text description
        """
        symbol = event.get("symbol", "")
        value = event.get("value", 0.5)
        direction = event.get("direction", "neutral")
        confidence = event.get("confidence", 0.7)
        details = event.get("details", {})
        
        # Get disagreement details
        disagreement = details.get("disagreement_level", 0)
        direction_counts = details.get("direction_counts", {})
        
        # Format direction counts
        direction_str = ", ".join([f"{count} {dir}" for dir, count in direction_counts.items()])
        
        return f"""
Market sentiment consensus disagreement detected for {symbol}.

Overall sentiment: {direction} ({value:.2f}) with {confidence:.2f} confidence
Disagreement level: {disagreement:.2f}
Direction breakdown: {direction_str}

This level of disagreement among sentiment sources and models may indicate a potential market-moving event or major shift in sentiment that has not yet fully propagated through the market.
"""
    
    def _generate_extreme_sentiment_text(self, event: Dict[str, Any]) -> str:
        """Generate text for extreme sentiment.
        
        Args:
            event: Sentiment event data
            
        Returns:
            Text description
        """
        symbol = event.get("symbol", "")
        value = event.get("value", 0.5)
        direction = event.get("direction", "neutral")
        confidence = event.get("confidence", 0.7)
        signal_type = event.get("signal_type", "sentiment")
        details = event.get("details", {})
        
        # Determine if this could be contrarian
        is_contrarian = value > 0.85 or value < 0.15
        contrarian_note = ""
        if is_contrarian:
            contrarian_note = "\n\nNote: This extreme sentiment level may be a contrarian indicator."
        
        # Get additional details
        source_types = details.get("source_types", [])
        sources_str = ", ".join(source_types) if source_types else "multiple sources"
        
        return f"""
Extreme {direction} sentiment detected for {symbol} from {sources_str}.

Sentiment value: {value:.2f} (extreme {direction})
Confidence: {confidence:.2f}
Signal type: {signal_type}

This extreme sentiment level could indicate a significant market event or potential market turning point.{contrarian_note}
"""
    
    def _generate_divergence_text(self, event: Dict[str, Any]) -> str:
        """Generate text for sentiment divergence.
        
        Args:
            event: Sentiment event data
            
        Returns:
            Text description
        """
        symbol = event.get("symbol", "")
        value = event.get("value", 0.5)
        direction = event.get("direction", "neutral")
        confidence = event.get("confidence", 0.7)
        details = event.get("details", {})
        
        # Get divergence details
        divergence_type = details.get("divergence_type", "unknown")
        price_change = details.get("price_change", 0)
        sentiment_direction = details.get("sentiment_direction", direction)
        price_direction = details.get("price_direction", "unknown")
        
        return f"""
Sentiment-price divergence detected for {symbol}.

Sentiment: {sentiment_direction} ({value:.2f})
Price movement: {price_direction} ({price_change:.2f}%)
Divergence type: {divergence_type}
Confidence: {confidence:.2f}

This divergence between sentiment and price action could indicate a potential market turning point or an opportunity for mean reversion.
"""
    
    def _map_asset_to_symbol(self, asset: str) -> str:
        """Map an asset ID to a trading symbol.
        
        Args:
            asset: Asset ID or name
            
        Returns:
            Trading symbol
        """
        # Simple mapping for common assets
        # In a real implementation, this would use a more sophisticated mapping system
        asset = asset.upper()
        
        if asset in ["BTC", "BITCOIN"]:
            return "BTC/USDT"
        elif asset in ["ETH", "ETHEREUM"]:
            return "ETH/USDT"
        elif asset in ["SOL", "SOLANA"]:
            return "SOL/USDT"
        elif asset in ["XRP", "RIPPLE"]:
            return "XRP/USDT"
        elif asset in ["BNB", "BINANCE"]:
            return "BNB/USDT"
        elif asset in ["ADA", "CARDANO"]:
            return "ADA/USDT"
        elif asset in ["DOT", "POLKADOT"]:
            return "DOT/USDT"
        elif asset in ["AVAX", "AVALANCHE"]:
            return "AVAX/USDT"
        elif "/" in asset:
            # Already in symbol format
            return asset
        else:
            # Default to asset/USDT
            return f"{asset}/USDT"
    
    def _map_signal_to_sentiment(self, signal: Dict[str, Any]) -> tuple:
        """Map an event signal to sentiment value and direction.
        
        Args:
            signal: Event signal data
            
        Returns:
            Tuple of (sentiment_value, direction)
        """
        # Check for impact assessment
        if "impact_assessment" in signal:
            impact = signal["impact_assessment"]
            direction_enum = impact.get("direction", "unclear")
            
            if direction_enum == "positive":
                return 0.75, "bullish"
            elif direction_enum == "negative":
                return 0.25, "bearish"
            else:
                return 0.5, "neutral"
        
        # Check recommended actions
        actions = signal.get("recommended_actions", [])
        if actions:
            action = actions[0]
            action_type = action.get("type", "monitor")
            
            if action_type in ["buy", "long", "accumulate"]:
                return 0.7, "bullish"
            elif action_type in ["sell", "short", "reduce"]:
                return 0.3, "bearish"
            else:
                return 0.5, "neutral"
        
        # Default to neutral
        return 0.5, "neutral"