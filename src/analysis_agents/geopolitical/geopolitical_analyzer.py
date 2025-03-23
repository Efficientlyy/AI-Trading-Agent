"""
Geopolitical Event Analyzer

This module tracks and analyzes geopolitical events that may impact financial markets,
with a particular focus on their potential effects on cryptocurrency markets.
"""

import asyncio
import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

from src.common.config import config
from src.common.logging import get_logger


class EventType(Enum):
    """Types of geopolitical events."""
    CONFLICT = "conflict"                # Wars, military actions
    ELECTION = "election"                # Elections, political transitions
    POLICY = "policy"                    # Major policy announcements
    DIPLOMATIC = "diplomatic"            # Diplomatic relations, treaties
    ECONOMIC = "economic"                # Trade agreements, sanctions
    NATURAL_DISASTER = "natural_disaster"  # Earthquakes, hurricanes, etc.
    SOCIAL_UNREST = "social_unrest"      # Protests, riots, civil unrest
    TERRORISM = "terrorism"              # Terrorist attacks
    OTHER = "other"                      # Other events


class EventSeverity(Enum):
    """Severity levels for geopolitical events."""
    CRITICAL = 5    # Major war, global crisis
    SEVERE = 4      # Regional conflict, severe economic sanctions
    SIGNIFICANT = 3 # National elections, major policy changes
    MODERATE = 2    # Diplomatic tensions, minor policy shifts
    MINOR = 1       # Local issues, routine political developments


class MarketImpactType(Enum):
    """Types of market impacts from geopolitical events."""
    RISK_AVERSION = "risk_aversion"        # Flight to safety
    CURRENCY_DEVALUATION = "currency_devaluation"  # Currency losing value
    INFLATION = "inflation"                # Price increases
    SUPPLY_DISRUPTION = "supply_disruption"  # Disruptions to supply chains
    REGULATORY = "regulatory"             # Regulatory changes
    CAPITAL_FLIGHT = "capital_flight"      # Money leaving a market
    MARKET_UNCERTAINTY = "market_uncertainty"  # Increased volatility
    ECONOMIC_SLOWDOWN = "economic_slowdown"  # Reduced economic activity


@dataclass
class GeopoliticalEvent:
    """Represents a geopolitical event with market impact analysis."""
    id: str
    title: str
    description: str
    event_type: EventType
    countries: List[str]
    regions: List[str]
    start_date: datetime.datetime
    end_date: Optional[datetime.datetime] = None
    ongoing: bool = True
    severity: EventSeverity = EventSeverity.MODERATE
    confidence: float = 0.7  # 0-1 scale
    sources: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)
    market_impacts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    impact_duration: str = "short_term"  # short_term, medium_term, long_term
    impact_types: List[MarketImpactType] = field(default_factory=list)
    updates: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)


class GeopoliticalAnalyzer:
    """Analyzes geopolitical events and their market impacts."""
    
    def __init__(self):
        """Initialize the geopolitical analyzer."""
        self.logger = get_logger("analysis_agents", "geopolitical_analyzer")
        self.events: Dict[str, GeopoliticalEvent] = {}
        self.countries_of_interest = config.get("geopolitical.countries_of_interest", [
            "USA", "China", "Russia", "EU", "Japan", "India", "UK", "Brazil", 
            "South Korea", "Iran", "Israel", "Saudi Arabia", "Turkey", "UAE"
        ])
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Mapping of countries to their economic/political importance for crypto
        self.country_importance = {
            "USA": 0.9,         # High regulatory and economic influence
            "China": 0.85,      # Major mining presence and market
            "EU": 0.8,          # Large market and regulatory influence
            "Russia": 0.7,      # Significant mining and usage
            "Japan": 0.7,       # Large trading volume and regulatory clarity
            "South Korea": 0.65,  # Major trading volume
            "UK": 0.6,          # Financial center and regulatory influence
            "India": 0.6,       # Large potential market
            "Singapore": 0.55,  # Crypto hub
            "Switzerland": 0.5, # Banking and crypto-friendly
            "Germany": 0.5,     # Large economy in EU
            "Canada": 0.45,     # Mining and adoption
            "Brazil": 0.4,      # Large economy, growing adoption
            "UAE": 0.4,         # Growing crypto hub
        }
        
        # Default impact for unlisted countries
        self.default_country_importance = 0.3
        
        # Historical patterns of geopolitical events and their market impacts
        self.historical_patterns = [
            {
                "pattern": "military_conflict",
                "keywords": ["war", "invasion", "military", "troops", "missile", "attack", "bomb"],
                "typical_impacts": {
                    "crypto": {
                        "short_term": -0.4,  # Initially negative
                        "medium_term": -0.2,
                        "long_term": 0.1     # Sometimes positive long-term (capital flight)
                    },
                    "stocks": {
                        "short_term": -0.5,
                        "medium_term": -0.3,
                        "long_term": -0.1
                    },
                    "gold": {
                        "short_term": 0.4,
                        "medium_term": 0.3,
                        "long_term": 0.2
                    },
                    "usd": {
                        "short_term": 0.3,
                        "medium_term": 0.2,
                        "long_term": 0.1
                    }
                },
                "impact_types": [
                    MarketImpactType.RISK_AVERSION,
                    MarketImpactType.MARKET_UNCERTAINTY
                ]
            },
            {
                "pattern": "sanctions",
                "keywords": ["sanction", "embargo", "restrict", "ban", "blacklist"],
                "typical_impacts": {
                    "crypto": {
                        "short_term": 0.3,  # Often positive (evasion)
                        "medium_term": 0.2,
                        "long_term": 0.1
                    },
                    "stocks": {
                        "short_term": -0.3,
                        "medium_term": -0.2,
                        "long_term": -0.1
                    },
                    "commodities": {
                        "short_term": 0.4,
                        "medium_term": 0.3,
                        "long_term": 0.2
                    }
                },
                "impact_types": [
                    MarketImpactType.CURRENCY_DEVALUATION,
                    MarketImpactType.CAPITAL_FLIGHT,
                    MarketImpactType.SUPPLY_DISRUPTION
                ]
            },
            {
                "pattern": "election",
                "keywords": ["election", "vote", "ballot", "campaign", "candidate", "president", "prime minister"],
                "typical_impacts": {
                    "crypto": {
                        "short_term": -0.1,  # Slight negative (uncertainty)
                        "medium_term": 0.0,
                        "long_term": 0.1     # Depends on outcome
                    },
                    "stocks": {
                        "short_term": -0.2,
                        "medium_term": 0.0,
                        "long_term": 0.1
                    },
                    "local_currency": {
                        "short_term": -0.2,
                        "medium_term": 0.0,
                        "long_term": 0.1
                    }
                },
                "impact_types": [
                    MarketImpactType.MARKET_UNCERTAINTY,
                    MarketImpactType.REGULATORY
                ]
            },
            {
                "pattern": "central_bank_policy",
                "keywords": ["central bank", "interest rate", "monetary policy", "federal reserve", "ecb", "bank of japan"],
                "typical_impacts": {
                    "crypto": {
                        "short_term": -0.3,  # Negative (tightening)
                        "medium_term": -0.2,
                        "long_term": 0.0
                    },
                    "stocks": {
                        "short_term": -0.4,
                        "medium_term": -0.3,
                        "long_term": -0.1
                    },
                    "bonds": {
                        "short_term": -0.3,
                        "medium_term": -0.1,
                        "long_term": 0.0
                    }
                },
                "impact_types": [
                    MarketImpactType.MARKET_UNCERTAINTY,
                    MarketImpactType.ECONOMIC_SLOWDOWN
                ]
            },
            {
                "pattern": "trade_agreement",
                "keywords": ["trade agreement", "trade deal", "free trade", "treaty", "partnership"],
                "typical_impacts": {
                    "crypto": {
                        "short_term": 0.1,
                        "medium_term": 0.2,
                        "long_term": 0.2
                    },
                    "stocks": {
                        "short_term": 0.3,
                        "medium_term": 0.4,
                        "long_term": 0.3
                    },
                    "local_currencies": {
                        "short_term": 0.2,
                        "medium_term": 0.3,
                        "long_term": 0.2
                    }
                },
                "impact_types": [
                    MarketImpactType.ECONOMIC_SLOWDOWN
                ]
            }
        ]
    
    async def initialize(self):
        """Initialize the geopolitical analyzer."""
        self.logger.info("Initializing geopolitical analyzer")
        await self.load_events()
        
        # Check if we need to load default events
        if not self.events:
            await self.load_default_events()
        
        self.logger.info(f"Loaded {len(self.events)} geopolitical events")
    
    async def load_events(self):
        """Load geopolitical events from storage."""
        events_file = os.path.join(self.data_dir, "geopolitical_events.json")
        
        if not os.path.exists(events_file):
            return
        
        try:
            with open(events_file, 'r') as f:
                events_data = json.load(f)
            
            for event_data in events_data:
                try:
                    # Convert strings to enums
                    event_type = EventType(event_data.get("event_type", "other"))
                    severity = EventSeverity(event_data.get("severity", 2))
                    
                    # Convert string dates to datetime
                    start_date = datetime.datetime.fromisoformat(event_data.get("start_date"))
                    end_date = None
                    if event_data.get("end_date"):
                        end_date = datetime.datetime.fromisoformat(event_data["end_date"])
                    
                    last_updated = datetime.datetime.fromisoformat(event_data.get("last_updated", datetime.datetime.now().isoformat()))
                    
                    # Convert impact types
                    impact_types = [MarketImpactType(it) for it in event_data.get("impact_types", [])]
                    
                    # Create event object
                    event = GeopoliticalEvent(
                        id=event_data["id"],
                        title=event_data["title"],
                        description=event_data["description"],
                        event_type=event_type,
                        countries=event_data["countries"],
                        regions=event_data["regions"],
                        start_date=start_date,
                        end_date=end_date,
                        ongoing=event_data.get("ongoing", True),
                        severity=severity,
                        confidence=event_data.get("confidence", 0.7),
                        sources=event_data.get("sources", []),
                        related_events=event_data.get("related_events", []),
                        market_impacts=event_data.get("market_impacts", {}),
                        impact_duration=event_data.get("impact_duration", "short_term"),
                        impact_types=impact_types,
                        updates=event_data.get("updates", []),
                        last_updated=last_updated
                    )
                    
                    self.events[event.id] = event
                except Exception as e:
                    self.logger.error(f"Error loading event: {e}")
        except Exception as e:
            self.logger.error(f"Error loading events: {e}")
    
    async def save_events(self):
        """Save geopolitical events to storage."""
        events_file = os.path.join(self.data_dir, "geopolitical_events.json")
        
        try:
            events_data = []
            
            for event_id, event in self.events.items():
                # Convert enums to strings
                event_dict = {
                    "id": event.id,
                    "title": event.title,
                    "description": event.description,
                    "event_type": event.event_type.value,
                    "countries": event.countries,
                    "regions": event.regions,
                    "start_date": event.start_date.isoformat(),
                    "end_date": event.end_date.isoformat() if event.end_date else None,
                    "ongoing": event.ongoing,
                    "severity": event.severity.value,
                    "confidence": event.confidence,
                    "sources": event.sources,
                    "related_events": event.related_events,
                    "market_impacts": event.market_impacts,
                    "impact_duration": event.impact_duration,
                    "impact_types": [it.value for it in event.impact_types],
                    "updates": event.updates,
                    "last_updated": event.last_updated.isoformat()
                }
                
                events_data.append(event_dict)
            
            with open(events_file, 'w') as f:
                json.dump(events_data, f, indent=2)
                
            self.logger.info(f"Saved {len(self.events)} geopolitical events")
        except Exception as e:
            self.logger.error(f"Error saving events: {e}")
    
    async def load_default_events(self):
        """Load default geopolitical events (for testing/demo)."""
        self.logger.info("Loading default geopolitical events")
        
        # Sample events based on real-world situations
        default_events = [
            {
                "id": "us_china_trade_tensions_2023",
                "title": "US-China Trade Tensions",
                "description": "Ongoing trade tensions between the United States and China, with tariffs and technology restrictions.",
                "event_type": EventType.ECONOMIC,
                "countries": ["USA", "China"],
                "regions": ["North America", "Asia"],
                "start_date": datetime.datetime(2023, 1, 1),
                "ongoing": True,
                "severity": EventSeverity.SIGNIFICANT,
                "confidence": 0.9,
                "impact_types": [MarketImpactType.SUPPLY_DISRUPTION, MarketImpactType.MARKET_UNCERTAINTY],
                "impact_duration": "long_term"
            },
            {
                "id": "russia_ukraine_conflict",
                "title": "Russia-Ukraine Conflict",
                "description": "Military conflict between Russia and Ukraine with global economic repercussions.",
                "event_type": EventType.CONFLICT,
                "countries": ["Russia", "Ukraine"],
                "regions": ["Eastern Europe"],
                "start_date": datetime.datetime(2022, 2, 24),
                "ongoing": True,
                "severity": EventType.CRITICAL,
                "confidence": 1.0,
                "impact_types": [
                    MarketImpactType.SUPPLY_DISRUPTION, 
                    MarketImpactType.CURRENCY_DEVALUATION,
                    MarketImpactType.INFLATION,
                    MarketImpactType.RISK_AVERSION
                ],
                "impact_duration": "long_term"
            },
            {
                "id": "fed_interest_rate_hikes",
                "title": "Federal Reserve Interest Rate Hikes",
                "description": "The US Federal Reserve's cycle of interest rate increases to combat inflation.",
                "event_type": EventType.POLICY,
                "countries": ["USA"],
                "regions": ["North America", "Global"],
                "start_date": datetime.datetime(2022, 3, 16),
                "ongoing": True,
                "severity": EventSeverity.SIGNIFICANT,
                "confidence": 1.0,
                "impact_types": [
                    MarketImpactType.MARKET_UNCERTAINTY,
                    MarketImpactType.ECONOMIC_SLOWDOWN
                ],
                "impact_duration": "medium_term"
            },
            {
                "id": "middle_east_conflict_2023",
                "title": "Middle East Regional Conflict",
                "description": "Escalating conflicts in the Middle East with implications for oil prices and global stability.",
                "event_type": EventType.CONFLICT,
                "countries": ["Israel", "Palestine", "Iran", "Lebanon"],
                "regions": ["Middle East"],
                "start_date": datetime.datetime(2023, 10, 7),
                "ongoing": True,
                "severity": EventSeverity.SEVERE,
                "confidence": 0.95,
                "impact_types": [
                    MarketImpactType.RISK_AVERSION,
                    MarketImpactType.SUPPLY_DISRUPTION
                ],
                "impact_duration": "medium_term"
            },
            {
                "id": "global_supply_chain_disruptions",
                "title": "Global Supply Chain Disruptions",
                "description": "Ongoing disruptions to global supply chains affecting multiple industries.",
                "event_type": EventType.ECONOMIC,
                "countries": ["Global"],
                "regions": ["Global"],
                "start_date": datetime.datetime(2021, 1, 1),
                "ongoing": True,
                "severity": EventSeverity.SIGNIFICANT,
                "confidence": 0.85,
                "impact_types": [
                    MarketImpactType.SUPPLY_DISRUPTION,
                    MarketImpactType.INFLATION
                ],
                "impact_duration": "medium_term"
            },
            {
                "id": "crypto_regulatory_developments",
                "title": "Global Cryptocurrency Regulatory Developments",
                "description": "Evolving regulatory frameworks for cryptocurrencies across major economies.",
                "event_type": EventType.POLICY,
                "countries": ["USA", "EU", "China", "Japan", "UK"],
                "regions": ["Global"],
                "start_date": datetime.datetime(2022, 1, 1),
                "ongoing": True,
                "severity": EventSeverity.SIGNIFICANT,
                "confidence": 0.8,
                "impact_types": [
                    MarketImpactType.REGULATORY,
                    MarketImpactType.MARKET_UNCERTAINTY
                ],
                "impact_duration": "long_term"
            }
        ]
        
        # Create event objects
        for event_data in default_events:
            event_id = event_data.pop("id")
            self.events[event_id] = GeopoliticalEvent(id=event_id, **event_data)
        
        # Calculate market impacts for default events
        for event_id, event in self.events.items():
            event.market_impacts = await self._calculate_market_impacts(event)
        
        # Save default events
        await self.save_events()
    
    async def add_event(self, event_data: Dict[str, Any]) -> str:
        """Add a new geopolitical event.
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            Event ID
        """
        # Generate ID if not provided
        if "id" not in event_data:
            import hashlib
            import time
            event_id = hashlib.md5(f"{event_data.get('title', '')}-{time.time()}".encode()).hexdigest()[:12]
        else:
            event_id = event_data["id"]
        
        # Set defaults
        if "start_date" not in event_data:
            event_data["start_date"] = datetime.datetime.now()
        
        # Create event object
        event = GeopoliticalEvent(
            id=event_id,
            title=event_data.get("title", "Unknown Event"),
            description=event_data.get("description", ""),
            event_type=event_data.get("event_type", EventType.OTHER),
            countries=event_data.get("countries", []),
            regions=event_data.get("regions", []),
            start_date=event_data.get("start_date"),
            end_date=event_data.get("end_date"),
            ongoing=event_data.get("ongoing", True),
            severity=event_data.get("severity", EventSeverity.MODERATE),
            confidence=event_data.get("confidence", 0.7),
            sources=event_data.get("sources", []),
            related_events=event_data.get("related_events", []),
            market_impacts=event_data.get("market_impacts", {}),
            impact_duration=event_data.get("impact_duration", "short_term"),
            impact_types=event_data.get("impact_types", []),
            updates=event_data.get("updates", []),
            last_updated=datetime.datetime.now()
        )
        
        # Calculate market impacts if not provided
        if not event.market_impacts:
            event.market_impacts = await self._calculate_market_impacts(event)
        
        # Add the event
        self.events[event_id] = event
        
        # Save changes
        await self.save_events()
        
        self.logger.info(f"Added geopolitical event: {event.title} (ID: {event_id})")
        
        return event_id
    
    async def update_event(self, event_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing geopolitical event.
        
        Args:
            event_id: Event ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful
        """
        if event_id not in self.events:
            self.logger.error(f"Event not found: {event_id}")
            return False
        
        event = self.events[event_id]
        
        # Add update to history
        update_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "changes": updates
        }
        event.updates.append(update_entry)
        
        # Update fields
        for field, value in updates.items():
            if hasattr(event, field):
                setattr(event, field, value)
        
        # Update last_updated timestamp
        event.last_updated = datetime.datetime.now()
        
        # Recalculate market impacts if relevant fields changed
        impact_relevant_fields = ["event_type", "countries", "severity", "ongoing"]
        if any(field in updates for field in impact_relevant_fields):
            event.market_impacts = await self._calculate_market_impacts(event)
        
        # Save changes
        await self.save_events()
        
        self.logger.info(f"Updated geopolitical event: {event.title} (ID: {event_id})")
        
        return True
    
    async def get_event(self, event_id: str) -> Optional[GeopoliticalEvent]:
        """Get a geopolitical event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            GeopoliticalEvent or None if not found
        """
        return self.events.get(event_id)
    
    async def get_active_events(self) -> List[GeopoliticalEvent]:
        """Get all active (ongoing) geopolitical events.
        
        Returns:
            List of active events
        """
        return [event for event in self.events.values() if event.ongoing]
    
    async def get_events_by_country(self, country: str) -> List[GeopoliticalEvent]:
        """Get geopolitical events related to a specific country.
        
        Args:
            country: Country name
            
        Returns:
            List of related events
        """
        return [event for event in self.events.values() if country in event.countries]
    
    async def get_events_by_type(self, event_type: EventType) -> List[GeopoliticalEvent]:
        """Get geopolitical events of a specific type.
        
        Args:
            event_type: Event type
            
        Returns:
            List of events of that type
        """
        return [event for event in self.events.values() if event.event_type == event_type]
    
    async def get_high_impact_events(self, threshold: int = 3) -> List[GeopoliticalEvent]:
        """Get high-impact geopolitical events.
        
        Args:
            threshold: Minimum severity level (default: SIGNIFICANT)
            
        Returns:
            List of high-impact events
        """
        return [event for event in self.events.values() if event.severity.value >= threshold and event.ongoing]
    
    async def _calculate_market_impacts(self, event: GeopoliticalEvent) -> Dict[str, Dict[str, float]]:
        """Calculate market impacts for an event.
        
        Args:
            event: Geopolitical event
            
        Returns:
            Dictionary mapping asset classes to impact values
        """
        impacts = {
            "crypto": {
                "BTC": 0.0,
                "ETH": 0.0,
                "overall": 0.0
            },
            "stocks": {
                "US": 0.0,
                "GLOBAL": 0.0,
                "overall": 0.0
            },
            "commodities": {
                "GOLD": 0.0,
                "OIL": 0.0,
                "overall": 0.0
            },
            "forex": {
                "USD": 0.0,
                "EUR": 0.0,
                "overall": 0.0
            }
        }
        
        # Base impact based on severity
        base_impact = 0.0
        if event.severity == EventSeverity.CRITICAL:
            base_impact = 0.8
        elif event.severity == EventSeverity.SEVERE:
            base_impact = 0.6
        elif event.severity == EventSeverity.SIGNIFICANT:
            base_impact = 0.4
        elif event.severity == EventSeverity.MODERATE:
            base_impact = 0.2
        elif event.severity == EventSeverity.MINOR:
            base_impact = 0.1
        
        # Adjust based on countries involved
        country_weight = 0.0
        for country in event.countries:
            country_weight = max(country_weight, self.country_importance.get(country, self.default_country_importance))
        
        # If global event, set high country weight
        if "Global" in event.countries or "Global" in event.regions:
            country_weight = 0.9
        
        # Apply country weight
        weighted_impact = base_impact * country_weight
        
        # Apply specific impact patterns based on event type
        if event.event_type == EventType.CONFLICT:
            # Conflicts: negative for risk assets, positive for safe havens
            impacts["crypto"]["overall"] = -weighted_impact
            impacts["crypto"]["BTC"] = -weighted_impact * 0.8  # BTC often less affected
            impacts["crypto"]["ETH"] = -weighted_impact * 1.1  # ETH often more affected
            
            impacts["stocks"]["overall"] = -weighted_impact
            impacts["stocks"]["US"] = -weighted_impact * 0.7  # US often less affected
            impacts["stocks"]["GLOBAL"] = -weighted_impact
            
            impacts["commodities"]["GOLD"] = weighted_impact  # Safe haven
            impacts["commodities"]["OIL"] = weighted_impact * 0.5  # Often disrupted but price increase
            impacts["commodities"]["overall"] = weighted_impact * 0.7
            
            impacts["forex"]["USD"] = weighted_impact * 0.5  # Safe haven
            impacts["forex"]["EUR"] = -weighted_impact * 0.3
            impacts["forex"]["overall"] = weighted_impact * 0.1
            
        elif event.event_type == EventType.POLICY:
            # Policy: depends on specifics, generally negative for crypto if restrictive
            impacts["crypto"]["overall"] = -weighted_impact * 0.7
            impacts["crypto"]["BTC"] = -weighted_impact * 0.6
            impacts["crypto"]["ETH"] = -weighted_impact * 0.8
            
            impacts["stocks"]["overall"] = -weighted_impact * 0.5
            impacts["stocks"]["US"] = -weighted_impact * 0.5
            impacts["stocks"]["GLOBAL"] = -weighted_impact * 0.4
            
            impacts["commodities"]["overall"] = -weighted_impact * 0.3
            impacts["commodities"]["GOLD"] = weighted_impact * 0.4  # Sometimes safe haven
            impacts["commodities"]["OIL"] = -weighted_impact * 0.2
            
            impacts["forex"]["overall"] = weighted_impact * 0.1
            impacts["forex"]["USD"] = weighted_impact * 0.3
            impacts["forex"]["EUR"] = -weighted_impact * 0.1
            
        elif event.event_type == EventType.ECONOMIC:
            # Economic: can be positive or negative, depends on nature
            if "sanction" in event.title.lower() or "tariff" in event.title.lower():
                # Sanctions and tariffs: often negative
                impacts["crypto"]["overall"] = weighted_impact * 0.3  # Sometimes positive (evasion)
                impacts["crypto"]["BTC"] = weighted_impact * 0.4
                impacts["crypto"]["ETH"] = weighted_impact * 0.2
                
                impacts["stocks"]["overall"] = -weighted_impact * 0.6
                impacts["stocks"]["US"] = -weighted_impact * 0.5
                impacts["stocks"]["GLOBAL"] = -weighted_impact * 0.7
                
                impacts["commodities"]["overall"] = weighted_impact * 0.4
                impacts["commodities"]["GOLD"] = weighted_impact * 0.3
                impacts["commodities"]["OIL"] = weighted_impact * 0.5
                
                impacts["forex"]["overall"] = -weighted_impact * 0.2
                impacts["forex"]["USD"] = weighted_impact * 0.2
                impacts["forex"]["EUR"] = -weighted_impact * 0.4
            else:
                # Other economic events: generally neutral to slight negative
                impacts["crypto"]["overall"] = -weighted_impact * 0.2
                impacts["crypto"]["BTC"] = -weighted_impact * 0.1
                impacts["crypto"]["ETH"] = -weighted_impact * 0.3
                
                impacts["stocks"]["overall"] = -weighted_impact * 0.3
                impacts["stocks"]["US"] = -weighted_impact * 0.3
                impacts["stocks"]["GLOBAL"] = -weighted_impact * 0.3
                
                impacts["commodities"]["overall"] = -weighted_impact * 0.1
                impacts["commodities"]["GOLD"] = weighted_impact * 0.1
                impacts["commodities"]["OIL"] = -weighted_impact * 0.2
                
                impacts["forex"]["overall"] = 0.0
                impacts["forex"]["USD"] = weighted_impact * 0.1
                impacts["forex"]["EUR"] = -weighted_impact * 0.1
                
        elif event.event_type == EventType.NATURAL_DISASTER:
            # Natural disasters: generally negative for affected regions
            impacts["crypto"]["overall"] = -weighted_impact * 0.1  # Limited impact
            impacts["crypto"]["BTC"] = -weighted_impact * 0.1
            impacts["crypto"]["ETH"] = -weighted_impact * 0.1
            
            impacts["stocks"]["overall"] = -weighted_impact * 0.4
            impacts["stocks"]["US"] = -weighted_impact * 0.3
            impacts["stocks"]["GLOBAL"] = -weighted_impact * 0.2
            
            impacts["commodities"]["overall"] = weighted_impact * 0.3
            impacts["commodities"]["GOLD"] = weighted_impact * 0.1
            impacts["commodities"]["OIL"] = weighted_impact * 0.4  # If oil infrastructure affected
            
            impacts["forex"]["overall"] = -weighted_impact * 0.2
            impacts["forex"]["USD"] = weighted_impact * 0.1
            impacts["forex"]["EUR"] = -weighted_impact * 0.1
            
        else:
            # Default moderate impacts
            impacts["crypto"]["overall"] = -weighted_impact * 0.3
            impacts["crypto"]["BTC"] = -weighted_impact * 0.2
            impacts["crypto"]["ETH"] = -weighted_impact * 0.3
            
            impacts["stocks"]["overall"] = -weighted_impact * 0.3
            impacts["stocks"]["US"] = -weighted_impact * 0.2
            impacts["stocks"]["GLOBAL"] = -weighted_impact * 0.3
            
            impacts["commodities"]["overall"] = weighted_impact * 0.2
            impacts["commodities"]["GOLD"] = weighted_impact * 0.3
            impacts["commodities"]["OIL"] = weighted_impact * 0.1
            
            impacts["forex"]["overall"] = 0.0
            impacts["forex"]["USD"] = weighted_impact * 0.1
            impacts["forex"]["EUR"] = -weighted_impact * 0.1
        
        # Apply duration adjustment
        if event.impact_duration == "short_term":
            duration_factor = 1.0  # No adjustment for short-term
        elif event.impact_duration == "medium_term":
            duration_factor = 0.8  # Reduced impact for medium-term
        else:  # long_term
            duration_factor = 0.6  # Further reduced for long-term
        
        # Apply duration factor
        for asset_class in impacts:
            for asset in impacts[asset_class]:
                impacts[asset_class][asset] *= duration_factor
        
        return impacts
    
    async def get_market_signals(self) -> Dict[str, Dict[str, Any]]:
        """Generate market signals based on geopolitical events.
        
        Returns:
            Dictionary mapping assets to their signals
        """
        signals = {}
        
        # Initialize signals for all asset classes
        assets = {
            "crypto": ["BTC", "ETH", "overall"],
            "stocks": ["US", "GLOBAL", "overall"],
            "commodities": ["GOLD", "OIL", "overall"],
            "forex": ["USD", "EUR", "overall"]
        }
        
        for asset_class, class_assets in assets.items():
            for asset in class_assets:
                asset_key = f"{asset_class}.{asset}"
                signals[asset_key] = {
                    "value": 0.0,
                    "direction": "neutral",
                    "confidence": 0.0,
                    "drivers": []
                }
        
        # Get active events
        active_events = await self.get_active_events()
        
        # Skip if no active events
        if not active_events:
            return signals
        
        # Calculate combined impact
        for event in active_events:
            # Skip low confidence events
            if event.confidence < 0.5:
                continue
                
            event_weight = event.confidence * event.severity.value / 5.0
            
            # Add impact from this event
            for asset_class in assets:
                if asset_class in event.market_impacts:
                    for asset in assets[asset_class]:
                        if asset in event.market_impacts[asset_class]:
                            asset_key = f"{asset_class}.{asset}"
                            impact = event.market_impacts[asset_class][asset]
                            
                            # Accumulate weighted impact
                            current_value = signals[asset_key]["value"]
                            current_confidence = signals[asset_key]["confidence"]
                            
                            # Weighted average
                            if current_confidence + event_weight > 0:
                                signals[asset_key]["value"] = (
                                    (current_value * current_confidence) + 
                                    (impact * event_weight)
                                ) / (current_confidence + event_weight)
                                
                                signals[asset_key]["confidence"] = current_confidence + event_weight
                            
                            # Add to drivers if significant impact
                            if abs(impact) >= 0.2:
                                driver = {
                                    "event_id": event.id,
                                    "title": event.title,
                                    "impact": impact,
                                    "confidence": event.confidence,
                                    "severity": event.severity.value
                                }
                                signals[asset_key]["drivers"].append(driver)
        
        # Set directions based on values
        for asset_key, signal in signals.items():
            value = signal["value"]
            
            if value >= 0.15:
                signal["direction"] = "bullish"
            elif value <= -0.15:
                signal["direction"] = "bearish"
            else:
                signal["direction"] = "neutral"
                
            # Cap confidence at 1.0
            signal["confidence"] = min(1.0, signal["confidence"])
            
            # Sort drivers by impact
            signal["drivers"] = sorted(
                signal["drivers"], 
                key=lambda x: abs(x["impact"]), 
                reverse=True
            )[:3]  # Keep top 3 drivers
        
        return signals
    
    async def get_geopolitical_summary(self) -> Dict[str, Any]:
        """Generate a summary of current geopolitical situation.
        
        Returns:
            Dictionary with geopolitical summary
        """
        # Get active events
        active_events = await self.get_active_events()
        
        # Skip if no active events
        if not active_events:
            return {
                "overall_risk": "low",
                "major_events": [],
                "regions_of_concern": [],
                "market_outlook": "neutral",
                "recommendation": "No significant geopolitical events to monitor."
            }
        
        # Sort events by severity
        sorted_events = sorted(active_events, key=lambda e: e.severity.value, reverse=True)
        
        # Calculate overall risk level
        risk_score = sum(event.severity.value * event.confidence for event in active_events) / len(active_events)
        
        if risk_score >= 4.0:
            overall_risk = "extreme"
        elif risk_score >= 3.0:
            overall_risk = "high"
        elif risk_score >= 2.0:
            overall_risk = "moderate"
        else:
            overall_risk = "low"
        
        # Identify regions of concern
        regions_count = {}
        for event in active_events:
            for region in event.regions:
                regions_count[region] = regions_count.get(region, 0) + event.severity.value
        
        regions_of_concern = sorted(
            [(region, count) for region, count in regions_count.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 regions
        
        # Determine market outlook
        market_signals = await self.get_market_signals()
        crypto_impact = market_signals.get("crypto.overall", {}).get("value", 0.0)
        
        if crypto_impact >= 0.3:
            market_outlook = "strongly_positive"
        elif crypto_impact >= 0.1:
            market_outlook = "positive"
        elif crypto_impact <= -0.3:
            market_outlook = "strongly_negative"
        elif crypto_impact <= -0.1:
            market_outlook = "negative"
        else:
            market_outlook = "neutral"
        
        # Generate recommendation
        if market_outlook in ["strongly_positive", "positive"]:
            recommendation = "Consider increasing cryptocurrency exposure despite geopolitical risks."
        elif market_outlook in ["strongly_negative", "negative"]:
            recommendation = "Consider reducing cryptocurrency exposure due to geopolitical risks."
        else:
            recommendation = "Monitor geopolitical situation closely before making significant portfolio changes."
        
        # Format major events
        major_events = []
        for event in sorted_events[:5]:  # Top 5 events
            major_events.append({
                "id": event.id,
                "title": event.title,
                "severity": event.severity.value,
                "type": event.event_type.value,
                "countries": event.countries,
                "crypto_impact": event.market_impacts.get("crypto", {}).get("overall", 0.0)
            })
        
        return {
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "major_events": major_events,
            "regions_of_concern": [region for region, count in regions_of_concern],
            "market_outlook": market_outlook,
            "recommendation": recommendation
        }


# Helper function for using the geopolitical analyzer

async def analyze_geopolitical_situation() -> Dict[str, Any]:
    """Analyze current geopolitical situation and its market impacts.
    
    Returns:
        Dictionary with analysis results
    """
    # Initialize geopolitical analyzer
    analyzer = GeopoliticalAnalyzer()
    await analyzer.initialize()
    
    # Get market signals
    market_signals = await analyzer.get_market_signals()
    
    # Get geopolitical summary
    geopolitical_summary = await analyzer.get_geopolitical_summary()
    
    # Get high impact events
    high_impact_events = await analyzer.get_high_impact_events()
    
    # Format high impact events
    formatted_events = []
    for event in high_impact_events:
        formatted_events.append({
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "event_type": event.event_type.value,
            "countries": event.countries,
            "severity": event.severity.value,
            "confidence": event.confidence,
            "crypto_impact": event.market_impacts.get("crypto", {}).get("overall", 0.0)
        })
    
    # Compile results
    results = {
        "market_signals": market_signals,
        "summary": geopolitical_summary,
        "high_impact_events": formatted_events,
        "analysis_timestamp": datetime.datetime.now().isoformat()
    }
    
    return results


# Example usage
async def main():
    """Run a geopolitical analysis demo."""
    logging.basicConfig(level=logging.INFO)
    
    print("Running geopolitical analysis demo...")
    
    results = await analyze_geopolitical_situation()
    
    print(f"Analysis completed at {results['analysis_timestamp']}")
    print(f"Overall geopolitical risk: {results['summary']['overall_risk']}")
    print(f"Market outlook: {results['summary']['market_outlook']}")
    print(f"Recommendation: {results['summary']['recommendation']}")
    
    print("\nHigh impact events:")
    for i, event in enumerate(results['high_impact_events']):
        print(f"{i+1}. {event['title']} (Severity: {event['severity']})")
    
    print("\nMarket signals:")
    crypto_signals = {k: v for k, v in results['market_signals'].items() if k.startswith("crypto")}
    for asset, signal in crypto_signals.items():
        print(f"{asset}: {signal['direction'].upper()} (value: {signal['value']:.2f}, confidence: {signal['confidence']:.2f})")
    
    print("\nDemo completed")


if __name__ == "__main__":
    asyncio.run(main())