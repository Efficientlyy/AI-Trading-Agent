"""
Analyzer Manager for the Early Event Detection System.

This module manages the various analyzers used to detect early events from processed data.
"""

import asyncio
import hashlib
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.common.config import config
from src.common.logging import get_logger
from src.analysis_agents.early_detection.models import (
    EarlyEvent, EventCategory, SourceType,
    ConfidenceLevel, ImpactMagnitude, ImpactDirection, ImpactTimeframe
)


class BaseAnalyzer:
    """Base class for data analyzers."""
    
    def __init__(self, analyzer_name: str):
        """Initialize the analyzer.
        
        Args:
            analyzer_name: Name of the analyzer
        """
        self.name = analyzer_name
        self.logger = get_logger("early_detection", f"analyzer_{analyzer_name}")
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the analyzer."""
        self.is_initialized = True
    
    async def analyze(self, data: List[Dict[str, Any]]) -> List[EarlyEvent]:
        """Analyze the data to detect events.
        
        Args:
            data: Processed data to analyze
            
        Returns:
            List of detected early events
        """
        raise NotImplementedError("Subclasses must implement analyze()")


class SocialMediaAnalyzer(BaseAnalyzer):
    """Analyzer for social media data."""
    
    def __init__(self):
        """Initialize the social media analyzer."""
        super().__init__("social_media")
        
        # Configure detection thresholds
        self.influence_threshold = config.get("early_detection.social_media.influence_threshold", 0.7)
        self.spread_threshold = config.get("early_detection.social_media.spread_threshold", 0.6)
        self.sentiment_threshold = config.get("early_detection.social_media.sentiment_threshold", 0.3)
    
    async def initialize(self):
        """Initialize the social media analyzer."""
        await super().initialize()
        self.logger.info("Social media analyzer initialized")
    
    async def analyze(self, data: List[Dict[str, Any]]) -> List[EarlyEvent]:
        """Analyze social media data to detect early events.
        
        Args:
            data: Processed social media data
            
        Returns:
            List of detected early events
        """
        self.logger.info(f"Analyzing {len(data)} social media items")
        events = []
        
        # Group data by entity
        entity_items = self._group_by_entity(data)
        
        # Detect events from entity clusters
        for entity, items in entity_items.items():
            if len(items) < 2:
                continue
            
            # Skip if the items don't meet influence threshold
            avg_influence = sum(
                item.get("processed", {}).get("social_metrics", {}).get("influence_score", 0)
                for item in items
            ) / len(items)
            
            if avg_influence < self.influence_threshold:
                continue
            
            # Detect potential event
            event = self._detect_entity_event(entity, items)
            if event:
                events.append(event)
        
        # Group data by keyword
        keyword_items = self._group_by_keyword(data)
        
        # Detect events from keyword clusters
        for keyword, items in keyword_items.items():
            if len(items) < 3 or keyword in ['crypto', 'bitcoin', 'blockchain']:
                continue
            
            # Skip if the items don't meet spread threshold
            avg_spread = sum(
                item.get("processed", {}).get("social_metrics", {}).get("spread_potential", 0)
                for item in items
            ) / len(items)
            
            if avg_spread < self.spread_threshold:
                continue
            
            # Detect potential event
            event = self._detect_keyword_event(keyword, items)
            if event:
                events.append(event)
        
        self.logger.info(f"Detected {len(events)} events from social media")
        return events
    
    def _group_by_entity(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group social media items by entity.
        
        Args:
            data: Processed social media data
            
        Returns:
            Dictionary mapping entity names to lists of items
        """
        entity_items = {}
        
        for item in data:
            processed = item.get("processed", {})
            entities = processed.get("entities", [])
            
            for entity in entities:
                entity_text = entity.get("text", "").lower()
                entity_type = entity.get("type", "")
                
                # Skip common or short entities
                if len(entity_text) < 4 or entity_text in ["the", "this", "that", "these", "those"]:
                    continue
                
                # Create entity key
                entity_key = f"{entity_type}:{entity_text}"
                
                if entity_key not in entity_items:
                    entity_items[entity_key] = []
                
                entity_items[entity_key].append(item)
        
        return entity_items
    
    def _group_by_keyword(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group social media items by keyword.
        
        Args:
            data: Processed social media data
            
        Returns:
            Dictionary mapping keywords to lists of items
        """
        keyword_items = {}
        
        for item in data:
            processed = item.get("processed", {})
            keywords = processed.get("keywords", [])
            
            for keyword in keywords:
                keyword_text = keyword.lower()
                
                # Skip common or short keywords
                if len(keyword_text) < 4 or keyword_text in ["the", "this", "that", "these", "those"]:
                    continue
                
                if keyword_text not in keyword_items:
                    keyword_items[keyword_text] = []
                
                keyword_items[keyword_text].append(item)
        
        return keyword_items
    
    def _detect_entity_event(self, entity_key: str, items: List[Dict[str, Any]]) -> Optional[EarlyEvent]:
        """Detect an event based on entity mentions.
        
        Args:
            entity_key: Entity key (type:text)
            items: Social media items mentioning the entity
            
        Returns:
            Detected event or None
        """
        # Parse entity key
        parts = entity_key.split(":", 1)
        if len(parts) != 2:
            return None
        
        entity_type, entity_text = parts
        
        # Calculate average sentiment
        sentiments = []
        for item in items:
            processed = item.get("processed", {})
            sentiment = processed.get("sentiment")
            if sentiment is not None:
                sentiments.append(sentiment)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
        
        # Check if sentiment deviation is significant
        sentiment_deviation = abs(avg_sentiment - 0.5)
        if sentiment_deviation < self.sentiment_threshold:
            return None
        
        # Determine sentiment direction
        sentiment_direction = "positive" if avg_sentiment > 0.5 else "negative"
        
        # Create event ID
        event_id = f"social_entity_{hashlib.md5(entity_key.encode()).hexdigest()[:8]}"
        
        # Determine event category based on entity type
        category = EventCategory.OTHER
        if entity_type == "ORGANIZATION":
            category = EventCategory.CORPORATE
        elif entity_type == "PERSON":
            if any("CEO" in item.get("content", "") for item in items):
                category = EventCategory.CORPORATE
            else:
                category = EventCategory.SOCIAL
        elif entity_type == "LOCATION":
            category = EventCategory.GEOPOLITICAL
        
        # Create event sources
        sources = []
        for item in items:
            source = item.get("source")
            if source and isinstance(source, dict):
                sources.append(source)
        
        # Create event
        event = EarlyEvent(
            id=event_id,
            title=f"Social media buzz about {entity_text.title()}",
            description=f"Significant {sentiment_direction} sentiment detected in social media posts about {entity_text.title()}",
            category=category,
            sources=sources[:5],  # Limit to 5 sources
            confidence=ConfidenceLevel.MEDIUM,
            entities=[{"text": entity_text, "type": entity_type}],
            keywords=[kw for item in items for kw in item.get("processed", {}).get("keywords", [])[:3]]
        )
        
        # Assess impact
        event.assess_impact(
            assets=["BTC", "ETH", "SOL", "XRP"],  # Default assets
            magnitude=ImpactMagnitude.MODERATE,
            direction=ImpactDirection.POSITIVE if avg_sentiment > 0.7 else 
                     ImpactDirection.NEGATIVE if avg_sentiment < 0.3 else
                     ImpactDirection.MIXED,
            timeframe=ImpactTimeframe.SHORT_TERM
        )
        
        return event
    
    def _detect_keyword_event(self, keyword: str, items: List[Dict[str, Any]]) -> Optional[EarlyEvent]:
        """Detect an event based on keyword mentions.
        
        Args:
            keyword: The keyword
            items: Social media items mentioning the keyword
            
        Returns:
            Detected event or None
        """
        # Calculate engagement metrics
        engagement_scores = [
            item.get("processed", {}).get("social_metrics", {}).get("engagement_level", 0)
            for item in items
        ]
        
        avg_engagement = sum(engagement_scores) / len(engagement_scores)
        if avg_engagement < 0.4:  # Minimum engagement threshold
            return None
        
        # Create event ID
        event_id = f"social_keyword_{hashlib.md5(keyword.encode()).hexdigest()[:8]}"
        
        # Determine category based on keyword
        category = EventCategory.OTHER
        if any(term in keyword for term in ["regulation", "law", "ban", "approve"]):
            category = EventCategory.REGULATION
        elif any(term in keyword for term in ["fed", "central", "rate", "inflation"]):
            category = EventCategory.MONETARY_POLICY
        elif any(term in keyword for term in ["war", "conflict", "sanction", "tension"]):
            category = EventCategory.GEOPOLITICAL
        elif any(term in keyword for term in ["technology", "upgrade", "protocol", "launch"]):
            category = EventCategory.TECHNOLOGY
        
        # Create event sources
        sources = []
        for item in items[:5]:  # Limit to 5 sources
            source = item.get("source")
            if source and isinstance(source, dict):
                sources.append(source)
        
        # Create event
        event = EarlyEvent(
            id=event_id,
            title=f"Trending topic: {keyword}",
            description=f"High engagement detected for content related to '{keyword}' across social media platforms",
            category=category,
            sources=sources,
            confidence=ConfidenceLevel.MEDIUM,
            entities=[],
            keywords=[keyword] + [kw for item in items for kw in item.get("processed", {}).get("keywords", [])[:2] if kw != keyword]
        )
        
        # Assess impact
        event.assess_impact(
            assets=["BTC", "ETH", "SOL", "XRP"],  # Default assets
            magnitude=ImpactMagnitude.MODERATE,
            direction=ImpactDirection.MIXED,  # Default to mixed for keywords
            timeframe=ImpactTimeframe.SHORT_TERM
        )
        
        return event


class NewsAnalyzer(BaseAnalyzer):
    """Analyzer for news and official source data."""
    
    def __init__(self):
        """Initialize the news analyzer."""
        super().__init__("news")
        
        # Configure detection thresholds
        self.entity_relevance_threshold = config.get("early_detection.news.entity_relevance_threshold", 0.6)
        self.keyword_relevance_threshold = config.get("early_detection.news.keyword_relevance_threshold", 0.7)
        self.sentiment_threshold = config.get("early_detection.news.sentiment_threshold", 0.3)
    
    async def initialize(self):
        """Initialize the news analyzer."""
        await super().initialize()
        self.logger.info("News analyzer initialized")
    
    async def analyze(self, data: List[Dict[str, Any]]) -> List[EarlyEvent]:
        """Analyze news data to detect early events.
        
        Args:
            data: Processed news data
            
        Returns:
            List of detected early events
        """
        self.logger.info(f"Analyzing {len(data)} news items")
        events = []
        
        # Group by source reliability
        high_reliability_items = []
        for item in data:
            source = item.get("source")
            if not source or not isinstance(source, dict):
                continue
            
            reliability = source.get("reliability_score", 0)
            if reliability >= 0.8:  # High reliability threshold
                high_reliability_items.append(item)
        
        # Analyze high reliability items individually
        for item in high_reliability_items:
            event = self._detect_news_event(item)
            if event:
                events.append(event)
        
        # Group by entity for cross-source corroboration
        entity_items = self._group_by_entity(data)
        
        # Detect events from entity clusters
        for entity, items in entity_items.items():
            if len(items) < 2:
                continue
            
            # Skip if already detected as individual event
            individual_event_ids = [e.id for e in events if entity in [ent.get("text") for ent in e.entities]]
            if individual_event_ids:
                continue
            
            # Detect potential event
            event = self._detect_entity_news_event(entity, items)
            if event:
                events.append(event)
        
        self.logger.info(f"Detected {len(events)} events from news")
        return events
    
    def _detect_news_event(self, item: Dict[str, Any]) -> Optional[EarlyEvent]:
        """Detect an event from a single high-reliability news item.
        
        Args:
            item: News item
            
        Returns:
            Detected event or None
        """
        # Get source information
        source = item.get("source")
        if not source or not isinstance(source, dict):
            return None
        
        # Check if this is an official source (higher importance)
        is_official = source.get("type") == SourceType.OFFICIAL
        
        # Get processed data
        processed = item.get("processed", {})
        
        # Get sentiment
        sentiment = processed.get("sentiment")
        if sentiment is None:
            sentiment = 0.5
        
        # Check if sentiment is significant
        sentiment_deviation = abs(sentiment - 0.5)
        if sentiment_deviation < self.sentiment_threshold and not is_official:
            return None
        
        # Get title and content
        title = item.get("title", "")
        content = item.get("content", "")
        
        if not title or not content:
            return None
        
        # Create event ID
        event_id = f"news_{hashlib.md5(title.encode()).hexdigest()[:8]}"
        
        # Get entities and keywords
        entities = processed.get("entities", [])
        keywords = processed.get("keywords", [])
        
        # Determine category based on content
        category = self._determine_news_category(title, content, keywords)
        
        # Determine confidence level
        confidence = ConfidenceLevel.HIGH if is_official else ConfidenceLevel.MEDIUM
        
        # Create event
        event = EarlyEvent(
            id=event_id,
            title=title,
            description=content[:200] + "..." if len(content) > 200 else content,
            category=category,
            sources=[source],
            confidence=confidence,
            entities=entities[:5],  # Limit to 5 entities
            keywords=keywords[:5]   # Limit to 5 keywords
        )
        
        # Assess impact
        impact_magnitude = ImpactMagnitude.SIGNIFICANT if is_official else ImpactMagnitude.MODERATE
        impact_direction = ImpactDirection.POSITIVE if sentiment > 0.7 else ImpactDirection.NEGATIVE if sentiment < 0.3 else ImpactDirection.MIXED
        
        event.assess_impact(
            assets=["BTC", "ETH", "SOL", "XRP"],  # Default assets
            magnitude=impact_magnitude,
            direction=impact_direction,
            timeframe=ImpactTimeframe.MEDIUM_TERM if is_official else ImpactTimeframe.SHORT_TERM
        )
        
        return event
    
    def _determine_news_category(self, title: str, content: str, keywords: List[str]) -> EventCategory:
        """Determine the event category based on news content.
        
        Args:
            title: News title
            content: News content
            keywords: Extracted keywords
            
        Returns:
            Event category
        """
        # Combine text for analysis
        text = (title + " " + content).lower()
        keyword_text = " ".join(keywords).lower()
        
        # Check for monetary policy indicators
        if any(term in text for term in ["fed", "federal reserve", "central bank", "interest rate", "inflation", 
                                        "monetary policy", "rate hike", "rate cut"]):
            return EventCategory.MONETARY_POLICY
        
        # Check for regulatory indicators
        if any(term in text for term in ["regulation", "regulatory", "sec", "cftc", "law", "legal", 
                                        "compliance", "approve", "ban", "forbid"]):
            return EventCategory.REGULATION
        
        # Check for trade war indicators
        if any(term in text for term in ["trade war", "tariff", "trade tension", "trade conflict", 
                                        "trade agreement", "trade deal"]):
            return EventCategory.TRADE_WAR
        
        # Check for geopolitical indicators
        if any(term in text for term in ["war", "conflict", "military", "sanction", "diplomatic", 
                                        "international", "political", "election", "government"]):
            return EventCategory.GEOPOLITICAL
        
        # Check for economic indicators
        if any(term in text for term in ["gdp", "economy", "economic", "growth", "recession", 
                                        "unemployment", "jobs", "economic data"]):
            return EventCategory.ECONOMIC_DATA
        
        # Check for corporate indicators
        if any(term in text for term in ["company", "corporation", "ceo", "earnings", "profit", 
                                        "loss", "stock", "shares", "acquisition", "merger"]):
            return EventCategory.CORPORATE
        
        # Check for technology indicators
        if any(term in text for term in ["technology", "tech", "innovation", "blockchain", "protocol", 
                                        "upgrade", "development", "launch", "release"]):
            return EventCategory.TECHNOLOGY
        
        # Default to OTHER
        return EventCategory.OTHER
    
    def _group_by_entity(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group news items by entity.
        
        Args:
            data: Processed news data
            
        Returns:
            Dictionary mapping entity names to lists of items
        """
        entity_items = {}
        
        for item in data:
            processed = item.get("processed", {})
            entities = processed.get("entities", [])
            
            for entity in entities:
                entity_text = entity.get("text", "").lower()
                
                # Skip common or short entities
                if len(entity_text) < 4 or entity_text in ["the", "this", "that", "these", "those"]:
                    continue
                
                if entity_text not in entity_items:
                    entity_items[entity_text] = []
                
                entity_items[entity_text].append(item)
        
        return entity_items
    
    def _detect_entity_news_event(self, entity: str, items: List[Dict[str, Any]]) -> Optional[EarlyEvent]:
        """Detect an event based on entity mentions across news sources.
        
        Args:
            entity: Entity text
            items: News items mentioning the entity
            
        Returns:
            Detected event or None
        """
        if len(items) < 2:
            return None
        
        # Calculate average sentiment
        sentiments = []
        for item in items:
            processed = item.get("processed", {})
            sentiment = processed.get("sentiment")
            if sentiment is not None:
                sentiments.append(sentiment)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
        
        # Check if sentiment deviation is significant
        sentiment_deviation = abs(avg_sentiment - 0.5)
        if sentiment_deviation < self.sentiment_threshold and len(items) < 3:
            return None
        
        # Create event ID
        event_id = f"news_entity_{hashlib.md5(entity.encode()).hexdigest()[:8]}"
        
        # Get sources
        sources = []
        for item in items[:5]:  # Limit to 5 sources
            source = item.get("source")
            if source and isinstance(source, dict):
                sources.append(source)
        
        # Get representative title
        # Use title from item with highest source reliability
        sorted_items = sorted(items, 
                            key=lambda x: x.get("source", {}).get("reliability_score", 0) 
                            if isinstance(x.get("source"), dict) else 0, 
                            reverse=True)
        title = sorted_items[0].get("title", f"News about {entity}")
        
        # Get keywords
        all_keywords = []
        for item in items:
            processed = item.get("processed", {})
            keywords = processed.get("keywords", [])
            all_keywords.extend(keywords)
        
        # Count keyword occurrences
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Get top keywords
        top_keywords = [k for k, v in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # Get entities
        all_entities = []
        for item in items:
            processed = item.get("processed", {})
            entities = processed.get("entities", [])
            all_entities.extend(entities)
        
        # Determine category based on content
        # Combine all titles and content for better category determination
        combined_text = " ".join([item.get("title", "") + " " + item.get("content", "") for item in items])
        category = self._determine_news_category(combined_text, "", top_keywords)
        
        # Determine confidence level based on number of sources and their reliability
        avg_reliability = sum(source.get("reliability_score", 0) for source in sources) / len(sources) if sources else 0.5
        
        confidence = ConfidenceLevel.HIGH if avg_reliability > 0.8 and len(items) >= 3 else \
                    ConfidenceLevel.MEDIUM if avg_reliability > 0.6 or len(items) >= 3 else \
                    ConfidenceLevel.LOW
        
        # Create event
        event = EarlyEvent(
            id=event_id,
            title=title,
            description=f"Multiple sources reporting on {entity} with significant attention",
            category=category,
            sources=sources,
            confidence=confidence,
            entities=[{"text": entity, "type": "ENTITY"}] + [e for e in all_entities if e.get("text").lower() != entity][:4],
            keywords=top_keywords
        )
        
        # Assess impact
        impact_magnitude = ImpactMagnitude.SIGNIFICANT if confidence == ConfidenceLevel.HIGH else \
                          ImpactMagnitude.MODERATE if confidence == ConfidenceLevel.MEDIUM else \
                          ImpactMagnitude.MINOR
        
        impact_direction = ImpactDirection.POSITIVE if avg_sentiment > 0.7 else \
                          ImpactDirection.NEGATIVE if avg_sentiment < 0.3 else \
                          ImpactDirection.MIXED
        
        event.assess_impact(
            assets=["BTC", "ETH", "SOL", "XRP"],  # Default assets
            magnitude=impact_magnitude,
            direction=impact_direction,
            timeframe=ImpactTimeframe.MEDIUM_TERM if category in [EventCategory.MONETARY_POLICY, EventCategory.REGULATION] else \
                     ImpactTimeframe.SHORT_TERM
        )
        
        return event


class FinancialDataAnalyzer(BaseAnalyzer):
    """Analyzer for financial market data."""
    
    def __init__(self):
        """Initialize the financial data analyzer."""
        super().__init__("financial_data")
        
        # Configure detection thresholds
        self.anomaly_threshold = config.get("early_detection.financial_data.anomaly_threshold", 0.7)
    
    async def initialize(self):
        """Initialize the financial data analyzer."""
        await super().initialize()
        self.logger.info("Financial data analyzer initialized")
    
    async def analyze(self, data: List[Dict[str, Any]]) -> List[EarlyEvent]:
        """Analyze financial data to detect early events.
        
        Args:
            data: Processed financial data
            
        Returns:
            List of detected early events
        """
        self.logger.info(f"Analyzing {len(data)} financial data items")
        events = []
        
        # Process each data item
        for item in data:
            # Check for anomalies
            processed = item.get("processed", {})
            anomalies = processed.get("anomalies", {})
            
            for asset, anomaly_data in anomalies.items():
                if not anomaly_data.get("is_anomaly", False):
                    continue
                
                # Check anomaly score against threshold
                anomaly_score = anomaly_data.get("score", 0)
                if anomaly_score < self.anomaly_threshold:
                    continue
                
                # Create event for this anomaly
                event = self._create_anomaly_event(asset, anomaly_data, item)
                if event:
                    events.append(event)
        
        self.logger.info(f"Detected {len(events)} events from financial data")
        return events
    
    def _create_anomaly_event(self, asset: str, anomaly_data: Dict[str, Any], item: Dict[str, Any]) -> Optional[EarlyEvent]:
        """Create an event from a detected anomaly.
        
        Args:
            asset: Asset symbol
            anomaly_data: Anomaly detection results
            item: Original data item
            
        Returns:
            Created event or None
        """
        # Get source and data type
        source = item.get("source")
        if not source or not isinstance(source, dict):
            return None
        
        data_type = item.get("data_type", "unknown")
        
        # Create event ID
        event_id = f"financial_{data_type}_{asset}_{int(datetime.now().timestamp())}"
        
        # Determine category based on data type
        category = EventCategory.MARKET
        
        # Create description
        description = anomaly_data.get("description", f"Unusual {data_type} activity detected for {asset}")
        
        # Create event
        event = EarlyEvent(
            id=event_id,
            title=f"Unusual {data_type} activity for {asset}",
            description=description,
            category=category,
            sources=[source],
            confidence=ConfidenceLevel.HIGH,  # Financial data anomalies are typically reliable
            entities=[{"text": asset, "type": "ASSET"}],
            keywords=[data_type, "anomaly", "unusual", "activity", asset.lower()]
        )
        
        # Determine impact details based on data type
        impact_magnitude = ImpactMagnitude.SIGNIFICANT if anomaly_data.get("score", 0) > 0.8 else ImpactMagnitude.MODERATE
        
        # For volume and options, the impact direction is typically positive
        # For other data types, it's unclear without more context
        impact_direction = ImpactDirection.POSITIVE if data_type in ["volume", "options"] else ImpactDirection.UNCLEAR
        
        # Assess impact (focused on the specific asset)
        impact_assets = [asset]
        
        # Include other assets (with reduced impact) based on correlation
        other_assets = [a for a in ["BTC", "ETH", "SOL", "XRP"] if a != asset]
        impact_assets.extend(other_assets)
        
        event.assess_impact(
            assets=impact_assets,
            magnitude=impact_magnitude,
            direction=impact_direction,
            timeframe=ImpactTimeframe.IMMEDIATE
        )
        
        # Customize impact for each asset
        impact = event.impact_assessment
        for other_asset in other_assets:
            impact["assets"][other_asset]["score"] *= 0.5  # Reduce impact for other assets
        
        return event


class CrossSourceAnalyzer(BaseAnalyzer):
    """Analyzer for cross-source correlation and verification."""
    
    def __init__(self):
        """Initialize the cross-source analyzer."""
        super().__init__("cross_source")
        
        # Configure detection thresholds
        self.correlation_threshold = config.get("early_detection.cross_source.correlation_threshold", 0.6)
    
    async def initialize(self):
        """Initialize the cross-source analyzer."""
        await super().initialize()
        self.logger.info("Cross-source analyzer initialized")
    
    async def analyze(self, data: List[Dict[str, Any]], events: List[EarlyEvent]) -> List[EarlyEvent]:
        """Analyze data and events across sources to find correlations.
        
        Args:
            data: All processed data
            events: Events detected by other analyzers
            
        Returns:
            List of new events or enhanced existing events
        """
        self.logger.info("Analyzing cross-source correlations")
        
        # Group events by category
        events_by_category = {}
        for event in events:
            category = event.category
            if category not in events_by_category:
                events_by_category[category] = []
            events_by_category[category].append(event)
        
        # Look for corroborating evidence across sources
        for category, category_events in events_by_category.items():
            if len(category_events) < 2:
                continue
            
            # Check for related events
            for i, event1 in enumerate(category_events):
                for event2 in category_events[i+1:]:
                    # Check if events are related
                    correlation = self._calculate_event_correlation(event1, event2)
                    if correlation >= self.correlation_threshold:
                        # Link events together
                        event1.related_events.append(event2.id)
                        event2.related_events.append(event1.id)
                        
                        # Enhance confidence for corroborated events
                        if event1.confidence.value < ConfidenceLevel.HIGH.value:
                            event1.confidence = ConfidenceLevel(min(event1.confidence.value + 1, ConfidenceLevel.VERY_HIGH.value))
                        
                        if event2.confidence.value < ConfidenceLevel.HIGH.value:
                            event2.confidence = ConfidenceLevel(min(event2.confidence.value + 1, ConfidenceLevel.VERY_HIGH.value))
        
        # Return the updated events list (no new events created)
        return events
    
    def _calculate_event_correlation(self, event1: EarlyEvent, event2: EarlyEvent) -> float:
        """Calculate correlation between two events.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Correlation score (0-1)
        """
        # Initial correlation score
        correlation = 0.0
        
        # Same category gives base correlation
        if event1.category == event2.category:
            correlation += 0.3
        
        # Check keyword overlap
        event1_keywords = set(k.lower() for k in event1.keywords)
        event2_keywords = set(k.lower() for k in event2.keywords)
        
        keyword_overlap = event1_keywords.intersection(event2_keywords)
        if keyword_overlap:
            correlation += min(0.4, len(keyword_overlap) * 0.1)
        
        # Check entity overlap
        event1_entities = set(e.get("text", "").lower() for e in event1.entities)
        event2_entities = set(e.get("text", "").lower() for e in event2.entities)
        
        entity_overlap = event1_entities.intersection(event2_entities)
        if entity_overlap:
            correlation += min(0.4, len(entity_overlap) * 0.1)
        
        # Check temporal closeness
        time_diff = abs((event1.detected_at - event2.detected_at).total_seconds())
        if time_diff < 3600:  # Within an hour
            correlation += 0.2
        elif time_diff < 86400:  # Within a day
            correlation += 0.1
        
        # Cap correlation at 1.0
        return min(1.0, correlation)


class AnalyzerManager:
    """Manager for data analyzers."""
    
    def __init__(self):
        """Initialize the analyzer manager."""
        self.logger = get_logger("early_detection", "analyzer_manager")
        self.analyzers = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the analyzer manager and all analyzers."""
        self.logger.info("Initializing analyzer manager")
        
        # Initialize analyzers
        self.analyzers = {
            "social_media": SocialMediaAnalyzer(),
            "news": NewsAnalyzer(),
            "financial_data": FinancialDataAnalyzer(),
            "cross_source": CrossSourceAnalyzer()
        }
        
        # Initialize each analyzer
        for analyzer_name, analyzer in self.analyzers.items():
            try:
                await analyzer.initialize()
                self.logger.info(f"Initialized {analyzer_name} analyzer")
            except Exception as e:
                self.logger.error(f"Error initializing {analyzer_name} analyzer: {e}")
        
        self.is_initialized = True
        self.logger.info("Analyzer manager initialized")
    
    async def analyze_data(self, data: List[Dict[str, Any]]) -> List[EarlyEvent]:
        """Analyze processed data to detect early events.
        
        Args:
            data: Processed data to analyze
            
        Returns:
            List of detected early events
        """
        if not self.is_initialized:
            self.logger.warning("Analyzer manager not initialized")
            return []
        
        self.logger.info(f"Analyzing {len(data)} data items")
        
        # Group data by source type
        data_by_source = {}
        for item in data:
            source = item.get("source")
            if not source or not isinstance(source, dict):
                continue
            
            source_type = source.get("type")
            if not source_type:
                continue
            
            if source_type not in data_by_source:
                data_by_source[source_type] = []
            
            data_by_source[source_type].append(item)
        
        # Create analysis tasks
        tasks = []
        
        # Analyze social media data
        if SourceType.SOCIAL_MEDIA in data_by_source and "social_media" in self.analyzers:
            social_media_data = data_by_source[SourceType.SOCIAL_MEDIA]
            tasks.append(asyncio.create_task(
                self.analyzers["social_media"].analyze(social_media_data)
            ))
        
        # Analyze news and official data
        news_data = []
        if SourceType.NEWS in data_by_source:
            news_data.extend(data_by_source[SourceType.NEWS])
        if SourceType.OFFICIAL in data_by_source:
            news_data.extend(data_by_source[SourceType.OFFICIAL])
        
        if news_data and "news" in self.analyzers:
            tasks.append(asyncio.create_task(
                self.analyzers["news"].analyze(news_data)
            ))
        
        # Analyze financial data
        if SourceType.FINANCIAL_DATA in data_by_source and "financial_data" in self.analyzers:
            financial_data = data_by_source[SourceType.FINANCIAL_DATA]
            tasks.append(asyncio.create_task(
                self.analyzers["financial_data"].analyze(financial_data)
            ))
        
        # Wait for all tasks to complete
        events = []
        for task in tasks:
            try:
                task_events = await task
                events.extend(task_events)
            except Exception as e:
                self.logger.error(f"Error in analysis task: {e}")
        
        # Perform cross-source analysis if multiple events were detected
        if events and len(events) > 1 and "cross_source" in self.analyzers:
            try:
                events = await self.analyzers["cross_source"].analyze(data, events)
            except Exception as e:
                self.logger.error(f"Error in cross-source analysis: {e}")
        
        self.logger.info(f"Detected {len(events)} events")
        return events