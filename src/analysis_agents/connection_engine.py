"""
Connection Engine for Global Event Analysis

This module links together information from different sources (news, geopolitical events,
economic data, etc.) to provide a comprehensive understanding of market-moving events.
"""

import asyncio
import datetime
import hashlib
import logging
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from src.common.config import config
from src.common.logging import get_logger
from src.analysis_agents.news.news_analyzer import NewsAnalyzer, NewsItem
from src.analysis_agents.geopolitical.geopolitical_analyzer import GeopoliticalAnalyzer, GeopoliticalEvent


class ConnectionType(Enum):
    """Types of connections between events/news items."""
    CAUSAL = "causal"  # Event A causes Event B
    CORRELATED = "correlated"  # Events are correlated
    CONTRADICTORY = "contradictory"  # Events contradict each other
    REINFORCING = "reinforcing"  # Events reinforce each other
    SEQUENTIAL = "sequential"  # Events are part of a sequence
    THEMATIC = "thematic"  # Events share themes/topics


@dataclass
class Connection:
    """Represents a connection between two items (news, events, etc.)."""
    id: str
    source_id: str
    source_type: str
    target_id: str
    target_type: str
    connection_type: ConnectionType
    strength: float  # 0-1 scale
    description: str
    confidence: float
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    market_implications: Dict[str, float] = field(default_factory=dict)


class ConnectionEngine:
    """Engine for connecting and analyzing relationships between different data sources."""
    
    def __init__(self):
        """Initialize the connection engine."""
        self.logger = get_logger("analysis_agents", "connection_engine")
        self.connections: Dict[str, Connection] = {}
        self.news_analyzer = None
        self.geo_analyzer = None
        
        # Graph for network analysis
        self.graph = nx.DiGraph()
        
        # Thematic keywords to help identify connections
        self.themes = {
            "monetary_policy": [
                "interest rate", "fed", "federal reserve", "central bank", "rate hike",
                "inflation", "tightening", "easing", "monetary policy", "fomc", "powell"
            ],
            "regulation": [
                "regulation", "regulatory", "sec", "cftc", "compliance", "law", "legal",
                "framework", "guidelines", "enforcement", "restrictions", "ban"
            ],
            "trade_tensions": [
                "trade war", "tariff", "trade tension", "trade talk", "trade agreement",
                "trade deal", "trade negotiation", "import duty", "export", "protectionism"
            ],
            "geopolitical_conflict": [
                "war", "military", "attack", "invasion", "missile", "troops", "conflict",
                "tension", "crisis", "security", "defense", "sanction", "terrorism"
            ],
            "technology": [
                "ai", "artificial intelligence", "blockchain", "crypto", "innovation",
                "digital", "technological", "advancement", "development", "disruption"
            ],
            "economic_indicators": [
                "gdp", "growth", "recession", "employment", "unemployment", "jobs",
                "economic data", "consumer", "retail", "manufacturing", "services"
            ],
            "energy_commodities": [
                "oil", "gas", "energy", "opec", "petroleum", "crude", "renewable",
                "solar", "wind", "production", "reserve", "supply", "demand"
            ]
        }
        
        # Connection patterns
        self.connection_patterns = [
            {
                "source_theme": "monetary_policy",
                "target_theme": "crypto",
                "connection_type": ConnectionType.CAUSAL,
                "typical_strength": 0.7,
                "description": "Monetary policy decisions typically impact crypto markets with a slight delay",
                "market_implications": {
                    "BTC": -0.6,  # Tightening monetary policy is typically negative for crypto
                    "stocks": -0.5
                }
            },
            {
                "source_theme": "regulation",
                "target_theme": "crypto",
                "connection_type": ConnectionType.CAUSAL,
                "typical_strength": 0.8,
                "description": "Regulatory developments directly impact crypto markets",
                "market_implications": {
                    "BTC": -0.7,  # Regulation news is typically negative initially
                    "crypto_exchanges": -0.8
                }
            },
            {
                "source_theme": "geopolitical_conflict",
                "target_theme": "economic_indicators",
                "connection_type": ConnectionType.CAUSAL,
                "typical_strength": 0.6,
                "description": "Geopolitical conflicts often lead to economic disruptions",
                "market_implications": {
                    "stocks": -0.5,
                    "commodities": 0.6,
                    "BTC": -0.3  # Mixed effect on crypto
                }
            },
            {
                "source_theme": "energy_commodities",
                "target_theme": "economic_indicators",
                "connection_type": ConnectionType.CAUSAL,
                "typical_strength": 0.6,
                "description": "Energy price movements impact broader economic indicators",
                "market_implications": {
                    "stocks": -0.4,
                    "inflation": 0.7
                }
            }
        ]
    
    async def initialize(self):
        """Initialize the connection engine."""
        self.logger.info("Initializing connection engine")
        
        # Initialize news analyzer
        self.news_analyzer = NewsAnalyzer()
        await self.news_analyzer.initialize()
        
        # Initialize geopolitical analyzer
        self.geo_analyzer = GeopoliticalAnalyzer()
        await self.geo_analyzer.initialize()
        
        self.logger.info("Connection engine initialized")
    
    async def analyze_connections(self):
        """Analyze connections between different data sources."""
        self.logger.info("Analyzing connections between data sources")
        
        # Reset graph
        self.graph = nx.DiGraph()
        
        # Fetch recent news
        await self.news_analyzer.fetch_news(days_back=7)
        news_items = self.news_analyzer.news_items
        
        # Fetch geopolitical events
        geo_events = await self.geo_analyzer.get_active_events()
        
        # Add nodes to graph
        for news_id, news_item in news_items.items():
            self.graph.add_node(
                news_id,
                type="news",
                title=news_item.title,
                timestamp=news_item.published_at,
                importance=news_item.importance_score
            )
        
        for event in geo_events:
            self.graph.add_node(
                event.id,
                type="geopolitical",
                title=event.title,
                timestamp=event.start_date,
                importance=event.severity.value / 5.0  # Normalize to 0-1
            )
        
        # Find connections between news items
        await self._connect_news_items(news_items)
        
        # Find connections between geopolitical events
        await self._connect_geo_events(geo_events)
        
        # Find connections between news and geopolitical events
        await self._connect_news_and_geo(news_items, geo_events)
        
        # Identify key nodes and patterns in the network
        self._analyze_network()
        
        self.logger.info(f"Connection analysis complete. Found {len(self.connections)} connections.")
        
        return self.connections
    
    async def _connect_news_items(self, news_items: Dict[str, NewsItem]):
        """Find connections between news items.
        
        Args:
            news_items: Dictionary of news items
        """
        self.logger.debug("Finding connections between news items")
        
        # Process existing connections from news analyzer
        for connection in self.news_analyzer.connections:
            conn_id = f"news_{connection.source_id}_{connection.target_id}"
            
            # Create connection object
            conn = Connection(
                id=conn_id,
                source_id=connection.source_id,
                source_type="news",
                target_id=connection.target_id,
                target_type="news",
                connection_type=self._map_relation_type(connection.relation_type),
                strength=connection.strength,
                description=connection.explanation,
                confidence=0.7,  # Default confidence
                created_at=datetime.datetime.now(),
                market_implications={}  # Will be filled in later
            )
            
            # Add to connections dictionary
            self.connections[conn_id] = conn
            
            # Add edge to graph
            self.graph.add_edge(
                connection.source_id,
                connection.target_id,
                id=conn_id,
                type=conn.connection_type.value,
                strength=conn.strength
            )
        
        # Done - news analyzer already found connections between news items
    
    async def _connect_geo_events(self, geo_events: List[GeopoliticalEvent]):
        """Find connections between geopolitical events.
        
        Args:
            geo_events: List of geopolitical events
        """
        self.logger.debug("Finding connections between geopolitical events")
        
        # Compare each pair of events
        for i, event1 in enumerate(geo_events):
            for event2 in geo_events[i+1:]:
                # Skip if already connected via related_events
                if event2.id in event1.related_events:
                    continue
                
                # Check for connections based on countries/regions
                countries_overlap = set(event1.countries).intersection(set(event2.countries))
                regions_overlap = set(event1.regions).intersection(set(event2.regions))
                
                if not countries_overlap and not regions_overlap:
                    continue
                
                # Determine connection type and strength
                conn_type = ConnectionType.CORRELATED  # Default
                strength = 0.0
                
                # Add base strength from overlap
                strength += len(countries_overlap) * 0.2
                strength += len(regions_overlap) * 0.1
                
                # Check for causal relationship (based on timing)
                time_diff = (event2.start_date - event1.start_date).total_seconds()
                if 0 < time_diff < 7 * 24 * 60 * 60:  # Within a week
                    if event1.event_type == event2.event_type:
                        # Same type, likely sequential
                        conn_type = ConnectionType.SEQUENTIAL
                        strength += 0.3
                    else:
                        # Different types, might be causal
                        conn_type = ConnectionType.CAUSAL
                        strength += 0.4
                
                # Minimum strength threshold
                if strength < 0.3:
                    continue
                
                # Cap strength at 1.0
                strength = min(1.0, strength)
                
                # Create connection ID
                conn_id = f"geo_{event1.id}_{event2.id}"
                
                # Create description
                description = f"Connected via {len(countries_overlap)} common countries and {len(regions_overlap)} common regions."
                if conn_type == ConnectionType.CAUSAL:
                    description += f" {event1.title} may have influenced {event2.title}."
                elif conn_type == ConnectionType.SEQUENTIAL:
                    description += f" Both events represent a sequence of {event1.event_type.value} developments."
                
                # Create connection object
                conn = Connection(
                    id=conn_id,
                    source_id=event1.id,
                    source_type="geopolitical",
                    target_id=event2.id,
                    target_type="geopolitical",
                    connection_type=conn_type,
                    strength=strength,
                    description=description,
                    confidence=0.6,  # Default confidence
                    created_at=datetime.datetime.now(),
                    market_implications={}  # Will be filled in later
                )
                
                # Add to connections dictionary
                self.connections[conn_id] = conn
                
                # Add edge to graph
                self.graph.add_edge(
                    event1.id,
                    event2.id,
                    id=conn_id,
                    type=conn_type.value,
                    strength=strength
                )
    
    async def _connect_news_and_geo(self, news_items: Dict[str, NewsItem], geo_events: List[GeopoliticalEvent]):
        """Find connections between news items and geopolitical events.
        
        Args:
            news_items: Dictionary of news items
            geo_events: List of geopolitical events
        """
        self.logger.debug("Finding connections between news items and geopolitical events")
        
        # For each news item, check for connections to geopolitical events
        for news_id, news_item in news_items.items():
            news_text = f"{news_item.title} {news_item.content}"
            news_text_lower = news_text.lower()
            
            for event in geo_events:
                # Check for mentions of the event
                event_mentioned = False
                connection_strength = 0.0
                
                # Check for country mentions
                countries_mentioned = []
                for country in event.countries:
                    if country.lower() in news_text_lower:
                        countries_mentioned.append(country)
                        event_mentioned = True
                        connection_strength += 0.1
                
                # Check for key terms from event title/description
                event_terms = set(event.title.lower().split() + event.description.lower().split())
                event_terms = {term for term in event_terms if len(term) > 3}  # Filter short words
                
                term_matches = []
                for term in event_terms:
                    if term in news_text_lower:
                        term_matches.append(term)
                        event_mentioned = True
                        connection_strength += 0.05
                
                # Limit strength contribution from terms
                connection_strength = min(connection_strength, 0.5)
                
                # Check timing - news should be after event start date or close to it
                time_diff = (news_item.published_at - event.start_date).total_seconds()
                if time_diff > 0:  # News after event
                    timing_factor = min(1.0, 7 * 24 * 60 * 60 / max(time_diff, 3600))  # Decay over time
                    connection_strength *= timing_factor
                elif abs(time_diff) < 24 * 60 * 60:  # News within a day before event
                    connection_strength *= 0.5  # Reduced strength for news before event
                else:
                    connection_strength = 0  # No connection for much older news
                
                # Minimum strength threshold
                if connection_strength < 0.2 or not event_mentioned:
                    continue
                
                # Determine connection type
                if time_diff > 0:  # News after event
                    conn_type = ConnectionType.CAUSAL  # Event caused news
                else:
                    conn_type = ConnectionType.THEMATIC  # Thematic connection
                
                # Create connection ID
                conn_id = f"news_geo_{news_id}_{event.id}"
                
                # Create description
                description = f"News mentions {len(countries_mentioned)} countries and {len(term_matches)} key terms related to the event."
                if conn_type == ConnectionType.CAUSAL:
                    description += f" The news appears to be reporting on consequences of the {event.title} event."
                
                # Create connection object
                conn = Connection(
                    id=conn_id,
                    source_id=event.id if conn_type == ConnectionType.CAUSAL else news_id,
                    source_type="geopolitical" if conn_type == ConnectionType.CAUSAL else "news",
                    target_id=news_id if conn_type == ConnectionType.CAUSAL else event.id,
                    target_type="news" if conn_type == ConnectionType.CAUSAL else "geopolitical",
                    connection_type=conn_type,
                    strength=connection_strength,
                    description=description,
                    confidence=0.7 * connection_strength,  # Confidence proportional to strength
                    created_at=datetime.datetime.now(),
                    market_implications={}  # Will be filled in later
                )
                
                # Add to connections dictionary
                self.connections[conn_id] = conn
                
                # Add edge to graph
                self.graph.add_edge(
                    conn.source_id,
                    conn.target_id,
                    id=conn_id,
                    type=conn_type.value,
                    strength=connection_strength
                )
    
    async def _analyze_network(self):
        """Analyze the network of connections to identify patterns and key nodes."""
        self.logger.debug("Analyzing connection network")
        
        # Skip if the graph is empty
        if len(self.graph) == 0:
            return
        
        try:
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            try:
                betweenness_centrality = nx.betweenness_centrality(self.graph)
            except:
                betweenness_centrality = {node: 0.0 for node in self.graph.nodes()}
                
            try:
                # For small graphs, eigenvector centrality might not converge
                eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except:
                eigenvector_centrality = {node: 0.0 for node in self.graph.nodes()}
                
            # Calculate PageRank
            pagerank = nx.pagerank(self.graph)
            
            # Identify key nodes
            key_nodes = []
            for node in self.graph.nodes():
                importance = degree_centrality.get(node, 0) * 0.3 + \
                             betweenness_centrality.get(node, 0) * 0.3 + \
                             eigenvector_centrality.get(node, 0) * 0.2 + \
                             pagerank.get(node, 0) * 0.2
                
                node_data = self.graph.nodes[node]
                
                key_nodes.append({
                    "id": node,
                    "type": node_data.get("type", "unknown"),
                    "title": node_data.get("title", ""),
                    "importance": node_data.get("importance", 0.0),
                    "network_centrality": importance
                })
            
            # Sort by combined importance
            key_nodes.sort(key=lambda x: x["importance"] * 0.6 + x["network_centrality"] * 0.4, reverse=True)
            
            # Get top key nodes
            top_key_nodes = key_nodes[:min(5, len(key_nodes))]
            
            self.logger.info(f"Identified {len(top_key_nodes)} key nodes in the network")
            
            # Identify connection clusters
            try:
                # Try to find communities
                communities = list(nx.community.greedy_modularity_communities(self.graph.to_undirected()))
                
                # Extract key themes for each community
                community_themes = []
                for i, community in enumerate(communities):
                    # Get nodes in this community
                    community_nodes = list(community)
                    
                    # Extract titles
                    titles = []
                    for node in community_nodes:
                        node_data = self.graph.nodes[node]
                        if "title" in node_data:
                            titles.append(node_data["title"])
                    
                    # Create theme summary
                    theme = {
                        "id": f"community_{i}",
                        "size": len(community_nodes),
                        "nodes": community_nodes[:10],  # First 10 nodes
                        "key_terms": self._extract_key_terms(titles),
                        "primary_type": self._get_primary_type(community_nodes)
                    }
                    
                    community_themes.append(theme)
                
                self.logger.info(f"Identified {len(community_themes)} thematic clusters in the network")
            
            except Exception as e:
                self.logger.error(f"Error finding communities: {e}")
                community_themes = []
            
            # Store analysis results for later use
            self.network_analysis = {
                "key_nodes": top_key_nodes,
                "community_themes": community_themes
            }
        
        except Exception as e:
            self.logger.error(f"Error in network analysis: {e}")
            self.network_analysis = {
                "key_nodes": [],
                "community_themes": []
            }
    
    def _extract_key_terms(self, texts: List[str]) -> List[str]:
        """Extract key terms from a list of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of key terms
        """
        if not texts:
            return []
            
        # Combine texts
        combined_text = " ".join(texts).lower()
        
        # Count theme matches
        theme_matches = {}
        for theme, keywords in self.themes.items():
            matches = sum(1 for keyword in keywords if keyword in combined_text)
            if matches > 0:
                theme_matches[theme] = matches
        
        # Get top themes
        sorted_themes = sorted(theme_matches.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:3]]
    
    def _get_primary_type(self, nodes: List[str]) -> str:
        """Get the primary node type in a list of nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            Primary node type
        """
        type_counts = {}
        for node in nodes:
            node_data = self.graph.nodes[node]
            node_type = node_data.get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        # Get most common type
        if not type_counts:
            return "unknown"
            
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    def _map_relation_type(self, relation_type: Any) -> ConnectionType:
        """Map relation type from news analyzer to connection type.
        
        Args:
            relation_type: Relation type from news analyzer
            
        Returns:
            ConnectionType enum
        """
        # Map from news analyzer's EventRelation to ConnectionType
        type_name = getattr(relation_type, "value", str(relation_type))
        
        if type_name == "causal":
            return ConnectionType.CAUSAL
        elif type_name == "related":
            return ConnectionType.CORRELATED
        elif type_name == "contradictory":
            return ConnectionType.CONTRADICTORY
        elif type_name == "reinforcing":
            return ConnectionType.REINFORCING
        else:
            return ConnectionType.THEMATIC
    
    async def get_key_events(self) -> List[Dict[str, Any]]:
        """Get key events based on connection analysis.
        
        Returns:
            List of key events with their connections
        """
        # Check if we have network analysis
        if not hasattr(self, "network_analysis"):
            self.analyze_connections()
        
        if not hasattr(self, "network_analysis") or not self.network_analysis["key_nodes"]:
            return []
        
        # Get key nodes from network analysis
        key_nodes = self.network_analysis["key_nodes"]
        
        # Format result
        key_events = []
        for node in key_nodes:
            node_id = node["id"]
            node_type = node["type"]
            
            # Get connections
            connections = []
            for conn_id, conn in self.connections.items():
                if conn.source_id == node_id or conn.target_id == node_id:
                    other_id = conn.target_id if conn.source_id == node_id else conn.source_id
                    other_type = conn.target_type if conn.source_id == node_id else conn.source_type
                    
                    try:
                        other_title = ""
                        if other_type == "news" and other_id in self.news_analyzer.news_items:
                            other_title = self.news_analyzer.news_items[other_id].title
                        elif other_type == "geopolitical":
                            for event in await self.geo_analyzer.get_active_events():
                                if event.id == other_id:
                                    other_title = event.title
                                    break
                        
                        connections.append({
                            "id": conn_id,
                            "connection_type": conn.connection_type.value,
                            "strength": conn.strength,
                            "direction": "outgoing" if conn.source_id == node_id else "incoming",
                            "other_id": other_id,
                            "other_type": other_type,
                            "other_title": other_title
                        })
                    except Exception as e:
                        self.logger.error(f"Error processing connection {conn_id}: {e}")
            
            # Add event with connections
            key_events.append({
                "id": node_id,
                "type": node_type,
                "title": node["title"],
                "importance": node["importance"],
                "network_centrality": node["network_centrality"],
                "connections": sorted(connections, key=lambda x: x["strength"], reverse=True)
            })
        
        return key_events
    
    async def get_market_impact_chains(self) -> List[Dict[str, Any]]:
        """Identify causal chains of events that may impact markets.
        
        Returns:
            List of causal chains with market impact analysis
        """
        # We need a directed graph with causal connections
        causal_graph = nx.DiGraph()
        
        # Add only causal connections
        for conn_id, conn in self.connections.items():
            if conn.connection_type == ConnectionType.CAUSAL:
                causal_graph.add_edge(
                    conn.source_id,
                    conn.target_id,
                    id=conn_id,
                    strength=conn.strength
                )
        
        # Find all simple paths of length 1-3
        impact_chains = []
        
        # Start from geopolitical events
        for event in await self.geo_analyzer.get_active_events():
            if event.id not in causal_graph:
                continue
                
            # Find all paths starting from this event
            for length in range(1, 4):  # Paths of length 1-3
                try:
                    # Get all paths starting from this node with given length
                    paths = []
                    for target in causal_graph.nodes():
                        if target != event.id:
                            try:
                                for path in nx.all_simple_paths(causal_graph, event.id, target, cutoff=length):
                                    if len(path) - 1 == length:  # -1 because path includes start node
                                        paths.append(path)
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                continue
                    
                    # Process each path
                    for path in paths:
                        # Calculate combined strength
                        combined_strength = 1.0
                        path_edges = []
                        
                        for i in range(len(path) - 1):
                            source, target = path[i], path[i+1]
                            edge_data = causal_graph.get_edge_data(source, target)
                            combined_strength *= edge_data["strength"]
                            path_edges.append({
                                "source": source,
                                "target": target,
                                "strength": edge_data["strength"],
                                "id": edge_data["id"]
                            })
                        
                        # Get node information
                        path_nodes = []
                        for node_id in path:
                            node_type = None
                            node_title = None
                            if node_id in self.graph.nodes:
                                node_data = self.graph.nodes[node_id]
                                node_type = node_data.get("type", "unknown")
                                node_title = node_data.get("title", "")
                            
                            path_nodes.append({
                                "id": node_id,
                                "type": node_type,
                                "title": node_title
                            })
                        
                        # Calculate market impact
                        market_impact = {
                            "crypto": 0.0,
                            "stocks": 0.0,
                            "commodities": 0.0,
                            "forex": 0.0
                        }
                        
                        # Start with the impact of the geopolitical event
                        if event.market_impacts:
                            for market, impacts in event.market_impacts.items():
                                if "overall" in impacts:
                                    market_impact[market] = impacts["overall"]
                        
                        # Add chain ID
                        chain_id = f"chain_{'_'.join(path)}"
                        
                        # Add to impact chains
                        impact_chains.append({
                            "id": chain_id,
                            "nodes": path_nodes,
                            "edges": path_edges,
                            "combined_strength": combined_strength,
                            "length": length,
                            "market_impact": market_impact
                        })
                
                except Exception as e:
                    self.logger.error(f"Error finding paths of length {length} from {event.id}: {e}")
        
        # Sort by combined strength
        impact_chains.sort(key=lambda x: x["combined_strength"], reverse=True)
        
        return impact_chains[:10]  # Return top 10 chains
    
    async def get_geopolitical_market_insights(self) -> Dict[str, Any]:
        """Get market insights based on geopolitical events and news.
        
        Returns:
            Dictionary with market insights
        """
        # First run connection analysis if not done already
        if not self.connections:
            self.analyze_connections()
        
        # Get geopolitical summary
        geo_summary = await self.geo_analyzer.get_geopolitical_summary()
        
        # Get key events
        key_events = self.get_key_events()
        
        # Get market impact chains
        impact_chains = self.get_market_impact_chains()
        
        # Get community themes if available
        community_themes = []
        if hasattr(self, "network_analysis") and "community_themes" in self.network_analysis:
            community_themes = self.network_analysis["community_themes"]
        
        # Combine insights
        insights = {
            "geopolitical_summary": geo_summary,
            "key_events": key_events,
            "impact_chains": impact_chains,
            "thematic_clusters": community_themes,
            "market_implications": {
                "crypto": {
                    "short_term": self._calculate_crypto_impact(impact_chains, "short_term"),
                    "medium_term": self._calculate_crypto_impact(impact_chains, "medium_term")
                }
            },
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
        
        return insights
    
    def _calculate_crypto_impact(self, impact_chains: List[Dict[str, Any]], timeframe: str) -> Dict[str, Any]:
        """Calculate crypto market impact based on impact chains.
        
        Args:
            impact_chains: List of impact chains
            timeframe: Time frame for impact (short_term, medium_term)
            
        Returns:
            Dictionary with crypto market impact insights
        """
        if not impact_chains:
            return {
                "direction": "neutral",
                "strength": 0.0,
                "confidence": 0.0,
                "key_drivers": []
            }
        
        # Default time discount factors
        time_factors = {
            "short_term": 1.0,
            "medium_term": 0.7
        }
        
        # Set default if unknown timeframe
        if timeframe not in time_factors:
            timeframe = "short_term"
        
        # Get time discount factor
        time_factor = time_factors[timeframe]
        
        # Calculate weighted impact
        total_strength = 0.0
        weighted_impact = 0.0
        key_drivers = []
        
        for chain in impact_chains:
            chain_strength = chain["combined_strength"]
            crypto_impact = chain["market_impact"].get("crypto", 0.0)
            
            weighted_impact += crypto_impact * chain_strength * time_factor
            total_strength += chain_strength
            
            # Add to key drivers if significant
            if abs(crypto_impact) >= 0.3 and chain_strength >= 0.4:
                start_node = chain["nodes"][0]
                end_node = chain["nodes"][-1]
                
                driver = {
                    "chain_id": chain["id"],
                    "source": f"{start_node['title']}",
                    "target": f"{end_node['title']}",
                    "impact": crypto_impact,
                    "strength": chain_strength
                }
                key_drivers.append(driver)
        
        # Calculate overall impact
        if total_strength > 0:
            overall_impact = weighted_impact / total_strength
        else:
            overall_impact = 0.0
        
        # Determine direction and strength
        if overall_impact >= 0.2:
            direction = "bullish"
            strength = min(1.0, overall_impact * 5)  # Scale up for clearer signal
        elif overall_impact <= -0.2:
            direction = "bearish"
            strength = min(1.0, abs(overall_impact) * 5)  # Scale up for clearer signal
        else:
            direction = "neutral"
            strength = abs(overall_impact) * 5
        
        # Calculate confidence
        confidence = min(0.9, total_strength / len(impact_chains))  # Cap at 0.9
        
        # Sort key drivers by absolute impact
        key_drivers.sort(key=lambda x: abs(x["impact"]) * x["strength"], reverse=True)
        
        return {
            "direction": direction,
            "strength": strength,
            "value": overall_impact,
            "confidence": confidence,
            "key_drivers": key_drivers[:3]  # Top 3 drivers
        }


# Helper function for using the connection engine

async def analyze_global_connections() -> Dict[str, Any]:
    """Analyze global connections between news, events, and other data sources.
    
    Returns:
        Dictionary with analysis results
    """
    # Initialize connection engine
    engine = ConnectionEngine()
    engine.initialize()
    
    try:
        # Analyze connections
        engine.analyze_connections()
        
        # Get market insights
        insights = engine.get_geopolitical_market_insights()
        
        return insights
    except Exception as e:
        logging.error(f"Error in global connection analysis: {e}")
        return {
            "error": str(e),
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }


# Example usage
async def main():
    """Run a connection analysis demo."""
    logging.basicConfig(level=logging.INFO)
    
    print("Running global connection analysis demo...")
    
    results = await analyze_global_connections()
    
    print(f"Analysis completed at {results['analysis_timestamp']}")
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print("\nGeopolitical Summary:")
    print(f"Overall risk: {results['geopolitical_summary']['overall_risk']}")
    print(f"Market outlook: {results['geopolitical_summary']['market_outlook']}")
    
    print("\nKey Events:")
    for i, event in enumerate(results['key_events'][:3]):
        print(f"{i+1}. {event['title']} (Importance: {event['importance']:.2f})")
    
    print("\nCrypto Market Implications:")
    crypto_impact = results['market_implications']['crypto']['short_term']
    print(f"Direction: {crypto_impact['direction']}")
    print(f"Strength: {crypto_impact['strength']:.2f}")
    print(f"Confidence: {crypto_impact['confidence']:.2f}")
    
    print("\nKey Drivers:")
    for i, driver in enumerate(crypto_impact['key_drivers']):
        impact_direction = "positive" if driver['impact'] > 0 else "negative"
        print(f"{i+1}. {driver['source']} -> {driver['target']} ({impact_direction} impact, strength: {driver['strength']:.2f})")
    
    print("\nDemo completed")


if __name__ == "__main__":
    asyncio.run(main())