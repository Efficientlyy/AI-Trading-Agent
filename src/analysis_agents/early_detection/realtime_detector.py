"""
Real-Time Event Detection System.

This module implements a real-time event detection system that uses
LLMs to identify and analyze potential market-moving events as they happen.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union

from src.common.config import config
from src.common.logging import get_logger
from src.common.monitoring import metrics
from src.common.events import event_bus
from src.analysis_agents.sentiment.llm_service import LLMService
from src.analysis_agents.early_detection.models import (
    EarlyEvent, EventSource, EventCategory, SourceType,
    ConfidenceLevel, ImpactMagnitude, ImpactDirection, ImpactTimeframe
)


class RealtimeEventDetector:
    """Real-time event detector using LLMs.
    
    This class monitors incoming text data from various sources and uses
    LLMs to detect potential market-moving events as they happen.
    """
    
    def __init__(self):
        """Initialize the real-time event detector."""
        self.logger = get_logger("analysis_agents", "realtime_detector")
        
        # Configuration
        self.enabled = config.get("early_detection.realtime.enabled", True)
        self.batch_size = config.get("early_detection.realtime.batch_size", 5)
        self.batch_interval = config.get("early_detection.realtime.batch_interval", 60)  # 1 minute
        self.min_confidence = config.get("early_detection.realtime.min_confidence", 0.7)
        self.min_severity = config.get("early_detection.realtime.min_severity", 5)  # 1-10 scale
        self.assets = config.get("early_detection.realtime.assets", ["BTC", "ETH", "SOL", "XRP"])
        
        # Source credibility scores
        self.source_credibility = config.get("early_detection.realtime.source_credibility", {
            "official": 0.9,
            "verified_news": 0.8,
            "financial_data": 0.85,
            "major_social": 0.7,
            "community": 0.5,
            "unverified": 0.3
        })
        
        # LLM service reference (will be set during initialization)
        self.llm_service = None
        
        # Pending text queue
        self.pending_texts = []
        
        # Cache of detected events to avoid duplicates
        self.detected_events = {}
        
        # Event de-duplication cache
        self.event_fingerprints = set()
        
        # Task reference for batch processing
        self.batch_task = None
        
        # Subscription tracking
        self.event_subscribers = []
        
        # State tracking
        self.is_running = False
    
    async def initialize(self):
        """Initialize the real-time event detector."""
        if not self.enabled:
            self.logger.info("Real-time event detector is disabled")
            return
        
        self.logger.info("Initializing real-time event detector")
        
        # Create and initialize LLM service
        self.llm_service = LLMService()
        await self.llm_service.initialize()
        
        # Subscribe to the event bus for new text data
        self._subscribe_to_text_sources()
        
        self.logger.info("Real-time event detector initialized")
    
    async def start(self):
        """Start the real-time event detector."""
        if not self.enabled or not self.llm_service:
            return
        
        if self.is_running:
            self.logger.warning("Real-time event detector is already running")
            return
        
        self.logger.info("Starting real-time event detector")
        self.is_running = True
        
        # Start the batch processing task
        self.batch_task = asyncio.create_task(self._process_batch_periodically())
    
    async def stop(self):
        """Stop the real-time event detector."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping real-time event detector")
        self.is_running = False
        
        # Cancel batch task
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        # Close LLM service
        if self.llm_service:
            await self.llm_service.close()
    
    async def process_text(self, 
                           text: str, 
                           source_name: str, 
                           source_type: str,
                           source_url: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Process a text for potential events.
        
        Args:
            text: The text content to analyze
            source_name: Name of the source
            source_type: Type of the source
            source_url: URL of the source (optional)
            metadata: Additional metadata about the source (optional)
        """
        if not self.enabled or not self.is_running:
            return
        
        # Add to pending texts
        self.pending_texts.append({
            "text": text,
            "source_name": source_name,
            "source_type": source_type,
            "source_url": source_url,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        })
        
        # If batch size reached, trigger processing
        if len(self.pending_texts) >= self.batch_size:
            asyncio.create_task(self._process_batch())
    
    async def _subscribe_to_text_sources(self):
        """Subscribe to various text sources through the event bus."""
        # Subscribe to social media text events
        async def handle_social_media_text(event):
            await self.process_text(
                text=event.get("text", ""),
                source_name=event.get("platform", "social_media"),
                source_type="social_media",
                source_url=event.get("url"),
                metadata={
                    "platform": event.get("platform"),
                    "user": event.get("user"),
                    "followers": event.get("followers"),
                    "verified": event.get("verified", False)
                }
            )
        
        # Subscribe to news text events
        async def handle_news_text(event):
            await self.process_text(
                text=event.get("text", ""),
                source_name=event.get("publisher", "news"),
                source_type="news",
                source_url=event.get("url"),
                metadata={
                    "publisher": event.get("publisher"),
                    "author": event.get("author"),
                    "title": event.get("title"),
                    "category": event.get("category")
                }
            )
        
        # Subscribe to official announcement events
        async def handle_official_announcement(event):
            await self.process_text(
                text=event.get("text", ""),
                source_name=event.get("organization", "official"),
                source_type="official",
                source_url=event.get("url"),
                metadata={
                    "organization": event.get("organization"),
                    "type": event.get("type"),
                    "title": event.get("title")
                }
            )
        
        # Register the handlers
        await event_bus.subscribe("social_media_text", handle_social_media_text)
        await event_bus.subscribe("news_text", handle_news_text)
        await event_bus.subscribe("official_announcement", handle_official_announcement)
        
        self.logger.info("Subscribed to text sources")
    
    async def _process_batch_periodically(self):
        """Process batches of text periodically."""
        self.logger.info("Batch processing task started")
        
        while self.is_running:
            try:
                # Check if we have pending texts
                if self.pending_texts:
                    self._process_batch()
                
                # Wait for the next interval
                await asyncio.sleep(self.batch_interval)
                
            except asyncio.CancelledError:
                self.logger.info("Batch processing task cancelled")
                raise
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(10)
    
    async def _process_batch(self):
        """Process a batch of pending texts."""
        if not self.pending_texts:
            return
        
        # Get the current batch
        batch = self.pending_texts[:self.batch_size]
        self.pending_texts = self.pending_texts[self.batch_size:]
        
        self.logger.info(f"Processing batch of {len(batch)} texts")
        
        # Extract the text content
        texts = [item["text"] for item in batch]
        
        try:
            # Use LLM service to detect potential events
            event_results = await self.llm_service.detect_market_event(texts)
            
            # Process each event result
            for i, result in enumerate(event_results):
                if result.get("is_market_event", False) and result.get("severity", 0) >= self.min_severity:
                    # This is a potential market event
                    source_data = batch[i]
                    
                    # Check event fingerprint to avoid duplicates
                    fingerprint = self._generate_event_fingerprint(result, source_data)
                    if fingerprint in self.event_fingerprints:
                        self.logger.debug(f"Skipping duplicate event: {result.get('event_type', 'unknown')}")
                        continue
                    
                    self.event_fingerprints.add(fingerprint)
                    
                    # Process the detected event
                    await self._process_detected_event(result, source_data)
        
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            # Return the texts to the queue for later processing
            self.pending_texts = batch + self.pending_texts
    
    def _generate_event_fingerprint(self, event_result: Dict[str, Any], source_data: Dict[str, Any]) -> str:
        """Generate a fingerprint for an event to detect duplicates.
        
        Args:
            event_result: The event detection result
            source_data: The source data
            
        Returns:
            Fingerprint string
        """
        # Use a combination of event type, affected assets, and key parts of the text
        event_type = event_result.get("event_type", "unknown")
        assets = "-".join(sorted(event_result.get("assets_affected", ["none"])))
        text_sample = source_data["text"][:100].lower()  # First 100 chars
        
        # Generate a fingerprint
        return f"{event_type}:{assets}:{hash(text_sample)}"
    
    async def _process_detected_event(self, event_result: Dict[str, Any], source_data: Dict[str, Any]):
        """Process a detected event.
        
        Args:
            event_result: The event detection result
            source_data: The source data
        """
        try:
            # Validate the event result
            if not self._validate_event_result(event_result):
                self.logger.warning(f"Invalid event result: {event_result}")
                return
            
            # Get event details
            event_type = event_result.get("event_type", "unknown")
            severity = event_result.get("severity", 0)
            credibility = event_result.get("credibility", 0.5)
            propagation = event_result.get("propagation_speed", "days")
            assets_affected = event_result.get("assets_affected", [])
            explanation = event_result.get("explanation", "")
            
            # Map to event category
            category = self._map_to_event_category(event_type)
            
            # Map to source type
            source_type_str = source_data.get("source_type", "other")
            try:
                source_type = SourceType(source_type_str)
            except ValueError:
                source_type = SourceType.OTHER
            
            # Map confidence and magnitude
            confidence = self._map_to_confidence_level(credibility)
            magnitude = self._map_to_impact_magnitude(severity)
            
            # Create source object
            source = EventSource(
                id=str(uuid.uuid4()),
                type=source_type,
                name=source_data.get("source_name", "unknown"),
                url=source_data.get("source_url"),
                timestamp=source_data.get("timestamp", datetime.now()),
                reliability_score=self._get_source_reliability(source_data),
                additional_info=source_data.get("metadata", {})
            )
            
            # Create event ID
            event_id = str(uuid.uuid4())
            
            # Create event title and description
            title = f"{event_type.title()} Event Detected: {self._generate_title(event_result, source_data)}"
            description = self._generate_description(event_result, source_data)
            
            # Create early event
            event = EarlyEvent(
                id=event_id,
                title=title,
                description=description,
                category=category,
                sources=[source],
                detected_at=datetime.now(),
                confidence=confidence,
                entities=self._extract_entities(event_result, source_data),
                keywords=self._extract_keywords(event_result, source_data),
                related_events=[],
                metadata={
                    "raw_detection": event_result,
                    "original_text": source_data.get("text", "")[:500],  # First 500 chars
                    "llm_model": event_result.get("_meta", {}).get("model", "unknown")
                }
            )
            
            # If significant, assess market impact
            if severity >= 7 or confidence.value >= 4:
                await self._assess_market_impact(event, event_result, source_data)
            
            # Record metric
            metrics.counter("realtime_event_detected", tags={
                "event_type": event_type,
                "severity": str(severity),
                "source_type": source_type_str
            })
            
            # Store the event
            self.detected_events[event_id] = event
            
            # Publish the event
            await self._publish_event(event)
            
            self.logger.info(f"Detected {category.value} event: {title[:50]}...",
                             severity=severity,
                             confidence=confidence.value,
                             source=source.name)
            
        except Exception as e:
            self.logger.error(f"Error processing detected event: {e}")
    
    def _validate_event_result(self, event_result: Dict[str, Any]) -> bool:
        """Validate an event detection result.
        
        Args:
            event_result: The event detection result
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not event_result.get("is_market_event", False):
            return False
        
        if "event_type" not in event_result:
            return False
        
        # Check confidence threshold
        credibility = event_result.get("credibility", 0.0)
        if credibility < self.min_confidence:
            return False
        
        # Check severity threshold
        severity = event_result.get("severity", 0)
        if severity < self.min_severity:
            return False
        
        return True
    
    def _map_to_event_category(self, event_type: str) -> EventCategory:
        """Map an event type to an event category.
        
        Args:
            event_type: The event type from the detection result
            
        Returns:
            Event category
        """
        event_type_lower = event_type.lower()
        
        # Map common event types to categories
        if any(term in event_type_lower for term in ["monetary", "fed", "central bank", "interest rate", "inflation"]):
            return EventCategory.MONETARY_POLICY
        elif any(term in event_type_lower for term in ["regulation", "sec", "legal", "law", "ban", "regulatory"]):
            return EventCategory.REGULATION
        elif any(term in event_type_lower for term in ["trade", "tariff", "sanction"]):
            return EventCategory.TRADE_WAR
        elif any(term in event_type_lower for term in ["war", "conflict", "political", "election", "government"]):
            return EventCategory.GEOPOLITICAL
        elif any(term in event_type_lower for term in ["economic", "gdp", "employment", "unemployment", "inflation"]):
            return EventCategory.ECONOMIC_DATA
        elif any(term in event_type_lower for term in ["corporate", "company", "acquisition", "merger", "earnings"]):
            return EventCategory.CORPORATE
        elif any(term in event_type_lower for term in ["market", "crash", "rally", "correction", "bubble"]):
            return EventCategory.MARKET
        elif any(term in event_type_lower for term in ["tech", "technology", "innovation", "upgrade", "launch"]):
            return EventCategory.TECHNOLOGY
        elif any(term in event_type_lower for term in ["social", "trend", "adoption", "sentiment"]):
            return EventCategory.SOCIAL
        else:
            return EventCategory.OTHER
    
    def _map_to_confidence_level(self, credibility: float) -> ConfidenceLevel:
        """Map a credibility score to a confidence level.
        
        Args:
            credibility: Credibility score (0-1)
            
        Returns:
            Confidence level
        """
        if credibility >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif credibility >= 0.75:
            return ConfidenceLevel.HIGH
        elif credibility >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif credibility >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _map_to_impact_magnitude(self, severity: int) -> ImpactMagnitude:
        """Map a severity score to an impact magnitude.
        
        Args:
            severity: Severity score (0-10)
            
        Returns:
            Impact magnitude
        """
        if severity >= 9:
            return ImpactMagnitude.CRITICAL
        elif severity >= 7:
            return ImpactMagnitude.SIGNIFICANT
        elif severity >= 5:
            return ImpactMagnitude.MODERATE
        elif severity >= 3:
            return ImpactMagnitude.MINOR
        else:
            return ImpactMagnitude.NEGLIGIBLE
    
    def _get_source_reliability(self, source_data: Dict[str, Any]) -> float:
        """Calculate the reliability score for a source.
        
        Args:
            source_data: Source data
            
        Returns:
            Reliability score (0-1)
        """
        source_type = source_data.get("source_type", "other")
        source_name = source_data.get("source_name", "unknown")
        metadata = source_data.get("metadata", {})
        
        # Base reliability on source type
        if source_type == "official":
            reliability = self.source_credibility.get("official", 0.9)
        elif source_type == "news":
            # Check if it's a verified news source
            if metadata.get("verified", False):
                reliability = self.source_credibility.get("verified_news", 0.8)
            else:
                reliability = self.source_credibility.get("unverified", 0.5)
        elif source_type == "social_media":
            # Check verification status
            if metadata.get("verified", False):
                reliability = self.source_credibility.get("major_social", 0.7)
            else:
                reliability = self.source_credibility.get("community", 0.5)
        else:
            reliability = self.source_credibility.get("unverified", 0.3)
        
        return reliability
    
    def _generate_title(self, event_result: Dict[str, Any], source_data: Dict[str, Any]) -> str:
        """Generate a title for the event.
        
        Args:
            event_result: Event detection result
            source_data: Source data
            
        Returns:
            Event title
        """
        event_type = event_result.get("event_type", "Unknown Event")
        assets = event_result.get("assets_affected", [])
        
        # If we have affected assets, include them in the title
        if assets:
            assets_str = ", ".join(assets[:3])
            if len(assets) > 3:
                assets_str += f" and {len(assets) - 3} more"
            
            return f"{event_type} affecting {assets_str}"
        
        # If event has explanation, use part of it
        explanation = event_result.get("explanation", "")
        if explanation:
            # Use first sentence or part of the explanation
            first_part = explanation.split(".")[0]
            if len(first_part) > 50:
                first_part = first_part[:50] + "..."
            
            return first_part
        
        # Fallback
        return f"{event_type} detected from {source_data.get('source_name', 'unknown source')}"
    
    def _generate_description(self, event_result: Dict[str, Any], source_data: Dict[str, Any]) -> str:
        """Generate a description for the event.
        
        Args:
            event_result: Event detection result
            source_data: Source data
            
        Returns:
            Event description
        """
        # Start with the explanation if available
        description = event_result.get("explanation", "")
        
        # Add severity and credibility information
        severity = event_result.get("severity", 0)
        credibility = event_result.get("credibility", 0.0)
        
        description += f"\n\nThis event has been assessed with a severity of {severity}/10 and a credibility score of {credibility:.2f}."
        
        # Add propagation speed
        propagation = event_result.get("propagation_speed", "days")
        description += f" Expected market impact timeframe: {propagation}."
        
        # Add affected assets if available
        assets = event_result.get("assets_affected", [])
        if assets:
            description += f"\n\nPotentially affected assets: {', '.join(assets)}."
        
        # Add source information
        source_name = source_data.get("source_name", "unknown")
        source_url = source_data.get("source_url", "")
        
        description += f"\n\nSource: {source_name}"
        if source_url:
            description += f" ({source_url})"
        
        return description
    
    def _extract_entities(self, event_result: Dict[str, Any], source_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from the event.
        
        Args:
            event_result: Event detection result
            source_data: Source data
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Add assets as entities
        for asset in event_result.get("assets_affected", []):
            entities.append({
                "type": "asset",
                "name": asset,
                "confidence": event_result.get("credibility", 0.5)
            })
        
        # Add source as entity
        entities.append({
            "type": "source",
            "name": source_data.get("source_name", "unknown"),
            "url": source_data.get("source_url", ""),
            "confidence": 1.0
        })
        
        # Additional entities could be extracted from NLP analysis
        
        return entities
    
    def _extract_keywords(self, event_result: Dict[str, Any], source_data: Dict[str, Any]) -> List[str]:
        """Extract keywords from the event.
        
        Args:
            event_result: Event detection result
            source_data: Source data
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Add event type as keyword
        event_type = event_result.get("event_type", "").lower()
        if event_type:
            keywords.append(event_type)
        
        # Add affected assets as keywords
        for asset in event_result.get("assets_affected", []):
            keywords.append(asset.lower())
        
        # Add propagation speed as keyword
        propagation = event_result.get("propagation_speed", "").lower()
        if propagation:
            keywords.append(propagation)
        
        # Add category as keyword
        category = self._map_to_event_category(event_type).value
        keywords.append(category)
        
        return keywords
    
    async def _assess_market_impact(self, event: EarlyEvent, event_result: Dict[str, Any], source_data: Dict[str, Any]):
        """Assess the market impact of an event.
        
        Args:
            event: The early event
            event_result: Event detection result
            source_data: Source data
        """
        if not self.llm_service:
            return
        
        try:
            # Prepare market context (in a real implementation, this would fetch actual market data)
            market_context = f"""
            Current market conditions:
            - BTC is trading around $100,000 with moderate volatility
            - ETH is trading around $5,000 with low volatility
            - General market sentiment has been positive in the last week
            - Overall crypto market cap is $3 trillion
            """
            
            # Generate event description
            event_description = f"{event.title}\n\n{event.description}"
            
            # Get impact assessment from LLM
            impact_result = await self.llm_service.assess_market_impact(
                event=event_description,
                market_context=market_context
            )
            
            if not impact_result:
                return
            
            # Extract impact direction and magnitude
            impact_direction_str = impact_result.get("primary_impact_direction", "neutral")
            impact_direction = self._map_to_impact_direction(impact_direction_str)
            
            impact_magnitude_float = impact_result.get("impact_magnitude", 0.5)
            impact_magnitude = self._map_magnitude_float_to_enum(impact_magnitude_float)
            
            # Map timeframe
            impact_timeframe_str = impact_result.get("estimated_duration", "days")
            impact_timeframe = self._map_to_impact_timeframe(impact_timeframe_str)
            
            # Get affected assets with their impacts
            affected_assets = impact_result.get("affected_assets", {})
            
            # Convert to the format expected by assess_impact
            converted_assets = list(self.assets)
            if affected_assets:
                # Add any assets in the impact assessment that aren't in our default list
                for asset in affected_assets:
                    if asset not in converted_assets:
                        converted_assets.append(asset)
            
            # Assess impact
            impact = event.assess_impact(
                assets=converted_assets,
                magnitude=impact_magnitude,
                direction=impact_direction,
                timeframe=impact_timeframe
            )
            
            # Update individual asset impacts if available
            for asset, asset_impact in affected_assets.items():
                if asset in impact["assets"]:
                    # Extract asset-specific information if available
                    asset_direction = asset_impact.get("direction", impact_direction_str)
                    asset_score = asset_impact.get("magnitude", impact_magnitude_float)
                    asset_confidence = asset_impact.get("confidence", impact_result.get("confidence", 0.7))
                    
                    # Update the asset impact
                    impact["assets"][asset] = {
                        "direction": self._map_to_impact_direction(asset_direction),
                        "score": asset_score,
                        "confidence": asset_confidence
                    }
            
            # Add reasoning
            impact["reasoning"] = impact_result.get("reasoning", "")
            
            # Add risk factors
            impact["risk_factors"] = impact_result.get("risk_factors", [])
            
            # Store the raw impact result for reference
            impact["raw_assessment"] = impact_result
            
        except Exception as e:
            self.logger.error(f"Error assessing market impact: {e}")
    
    def _map_to_impact_direction(self, direction: str) -> ImpactDirection:
        """Map a string direction to an impact direction enum.
        
        Args:
            direction: Direction string
            
        Returns:
            Impact direction enum
        """
        direction = direction.lower()
        
        if direction in ["positive", "bullish", "up", "upward"]:
            return ImpactDirection.POSITIVE
        elif direction in ["negative", "bearish", "down", "downward"]:
            return ImpactDirection.NEGATIVE
        elif direction in ["mixed", "varies", "variable"]:
            return ImpactDirection.MIXED
        else:
            return ImpactDirection.UNCLEAR
    
    def _map_to_impact_timeframe(self, timeframe: str) -> ImpactTimeframe:
        """Map a string timeframe to an impact timeframe enum.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Impact timeframe enum
        """
        timeframe = timeframe.lower()
        
        if timeframe in ["immediate", "hours", "minutes", "instant"]:
            return ImpactTimeframe.IMMEDIATE
        elif timeframe in ["short_term", "short-term", "days", "week"]:
            return ImpactTimeframe.SHORT_TERM
        elif timeframe in ["medium_term", "medium-term", "weeks", "month"]:
            return ImpactTimeframe.MEDIUM_TERM
        elif timeframe in ["long_term", "long-term", "months", "years"]:
            return ImpactTimeframe.LONG_TERM
        else:
            return ImpactTimeframe.SHORT_TERM  # Default
    
    def _map_magnitude_float_to_enum(self, magnitude: float) -> ImpactMagnitude:
        """Map a float magnitude (0-1) to an impact magnitude enum.
        
        Args:
            magnitude: Magnitude float (0-1)
            
        Returns:
            Impact magnitude enum
        """
        if magnitude >= 0.8:
            return ImpactMagnitude.CRITICAL
        elif magnitude >= 0.6:
            return ImpactMagnitude.SIGNIFICANT
        elif magnitude >= 0.4:
            return ImpactMagnitude.MODERATE
        elif magnitude >= 0.2:
            return ImpactMagnitude.MINOR
        else:
            return ImpactMagnitude.NEGLIGIBLE
    
    async def _publish_event(self, event: EarlyEvent):
        """Publish a detected event.
        
        Args:
            event: The early event to publish
        """
        # Convert to event payload
        payload = {
            "event_id": event.id,
            "title": event.title,
            "description": event.description,
            "category": event.category.value,
            "confidence": event.confidence.value,
            "detected_at": event.detected_at.isoformat(),
            "sources": [
                {
                    "id": source.id,
                    "type": source.type.value,
                    "name": source.name,
                    "url": source.url,
                    "reliability": source.reliability_score
                }
                for source in event.sources
            ],
            "entities": event.entities,
            "keywords": event.keywords,
            "impact_assessment": {
                "magnitude": event.impact_assessment.get("magnitude", ImpactMagnitude.MODERATE).value,
                "direction": event.impact_assessment.get("direction", ImpactDirection.UNCLEAR).value,
                "timeframe": event.impact_assessment.get("timeframe", ImpactTimeframe.SHORT_TERM).value,
                "assets": event.impact_assessment.get("assets", {})
            } if event.impact_assessment else None
        }
        
        # Publish to event bus
        await event_bus.publish(
            event_type="RealtimeEventDetected",
            source="realtime_event_detector",
            payload=payload
        )
    
    def get_event_by_id(self, event_id: str) -> Optional[EarlyEvent]:
        """Get an event by its ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            The event if found, None otherwise
        """
        return self.detected_events.get(event_id)
    
    def get_recent_events(self, hours: int = 24) -> List[EarlyEvent]:
        """Get recent events.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent events
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            event for event in self.detected_events.values()
            if event.detected_at >= cutoff_time
        ]
    
    def get_events_by_category(self, category: EventCategory) -> List[EarlyEvent]:
        """Get events by category.
        
        Args:
            category: Event category
            
        Returns:
            List of events in the specified category
        """
        return [
            event for event in self.detected_events.values()
            if event.category == category
        ]
    
    def get_high_impact_events(self, min_magnitude: ImpactMagnitude = ImpactMagnitude.SIGNIFICANT) -> List[EarlyEvent]:
        """Get high impact events.
        
        Args:
            min_magnitude: Minimum impact magnitude
            
        Returns:
            List of high impact events
        """
        return [
            event for event in self.detected_events.values()
            if event.impact_assessment and 
            event.impact_assessment.get("magnitude", ImpactMagnitude.NEGLIGIBLE).value >= min_magnitude.value
        ]