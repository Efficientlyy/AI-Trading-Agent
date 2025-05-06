"""
Agent Activity Tracker

This module tracks the activity of various agents in the system and provides
real-time updates on their status, processing times, and interactions.
"""

import time
import uuid
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from ..common import logger

class AgentActivityTracker:
    """
    Tracks the activity of agents in the system and provides real-time updates.
    """
    
    def __init__(self):
        """Initialize the agent activity tracker."""
        self.agents = {}
        self.data_sources = {}
        self.interactions = []
        self.max_interactions = 100  # Maximum number of interactions to store
        self.session_agents = {}  # Maps session_id to list of agent_ids
        
    def register_agent(self, session_id: str, name: str, agent_type: str) -> str:
        """
        Register a new agent with the tracker.
        
        Args:
            session_id: The ID of the session the agent belongs to
            name: The name of the agent
            agent_type: The type of agent (e.g., 'strategy', 'data', 'execution')
            
        Returns:
            The ID of the registered agent
        """
        agent_id = str(uuid.uuid4())
        
        self.agents[agent_id] = {
            'id': agent_id,
            'name': name,
            'type': agent_type,
            'status': 'idle',
            'lastActivity': datetime.now().isoformat(),
            'processingTime': 0,
            'dataVolume': 0,
            'signalStrength': 0,
            'connections': [],
            'session_id': session_id
        }
        
        # Add to session mapping
        if session_id not in self.session_agents:
            self.session_agents[session_id] = []
        
        self.session_agents[session_id].append(agent_id)
        
        logger.info(f"Registered agent {name} ({agent_id}) for session {session_id}")
        return agent_id
    
    def register_data_source(self, session_id: str, name: str, source_type: str, symbols: List[str]) -> str:
        """
        Register a new data source with the tracker.
        
        Args:
            session_id: The ID of the session the data source belongs to
            name: The name of the data source
            source_type: The type of data source (e.g., 'api', 'file', 'stream')
            symbols: List of symbols the data source provides
            
        Returns:
            The ID of the registered data source
        """
        source_id = str(uuid.uuid4())
        
        self.data_sources[source_id] = {
            'id': source_id,
            'name': name,
            'type': source_type,
            'status': 'idle',
            'lastUpdated': datetime.now().isoformat(),
            'symbols': symbols,
            'session_id': session_id
        }
        
        logger.info(f"Registered data source {name} ({source_id}) for session {session_id}")
        return source_id
    
    def connect_agent_to_source(self, agent_id: str, source_id: str) -> bool:
        """
        Connect an agent to a data source.
        
        Args:
            agent_id: The ID of the agent
            source_id: The ID of the data source
            
        Returns:
            True if the connection was successful, False otherwise
        """
        if agent_id not in self.agents or source_id not in self.data_sources:
            return False
        
        if source_id not in self.agents[agent_id]['connections']:
            self.agents[agent_id]['connections'].append(source_id)
            
        return True
    
    def update_agent_status(self, agent_id: str, status: str, 
                           processing_time: Optional[float] = None,
                           data_volume: Optional[int] = None,
                           signal_strength: Optional[float] = None) -> bool:
        """
        Update the status of an agent.
        
        Args:
            agent_id: The ID of the agent
            status: The new status ('active', 'idle', 'error')
            processing_time: Optional processing time in milliseconds
            data_volume: Optional data volume processed
            signal_strength: Optional signal strength (0-1)
            
        Returns:
            True if the update was successful, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        self.agents[agent_id]['status'] = status
        self.agents[agent_id]['lastActivity'] = datetime.now().isoformat()
        
        if processing_time is not None:
            self.agents[agent_id]['processingTime'] = processing_time
            
        if data_volume is not None:
            self.agents[agent_id]['dataVolume'] = data_volume
            
        if signal_strength is not None:
            self.agents[agent_id]['signalStrength'] = signal_strength
            
        return True
    
    def update_data_source_status(self, source_id: str, status: str) -> bool:
        """
        Update the status of a data source.
        
        Args:
            source_id: The ID of the data source
            status: The new status ('active', 'idle', 'error')
            
        Returns:
            True if the update was successful, False otherwise
        """
        if source_id not in self.data_sources:
            return False
        
        self.data_sources[source_id]['status'] = status
        self.data_sources[source_id]['lastUpdated'] = datetime.now().isoformat()
        
        return True
    
    def record_interaction(self, from_id: str, to_id: str, interaction_type: str, data: Any = None) -> bool:
        """
        Record an interaction between an agent and a data source or another agent.
        
        Args:
            from_id: The ID of the source entity
            to_id: The ID of the target entity
            interaction_type: The type of interaction (e.g., 'data_request', 'signal')
            data: Optional data associated with the interaction
            
        Returns:
            True if the interaction was recorded, False otherwise
        """
        # Verify that both entities exist
        from_entity = self.agents.get(from_id) or self.data_sources.get(from_id)
        to_entity = self.agents.get(to_id) or self.data_sources.get(to_id)
        
        if not from_entity or not to_entity:
            return False
        
        # Create interaction record
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'from': from_entity.get('name', 'Unknown'),
            'to': to_entity.get('name', 'Unknown'),
            'type': interaction_type
        }
        
        if data is not None:
            interaction['data'] = data
            
        # Add to interactions list, maintaining max size
        self.interactions.append(interaction)
        if len(self.interactions) > self.max_interactions:
            self.interactions = self.interactions[-self.max_interactions:]
            
        return True
    
    def get_session_activity_data(self, session_id: str) -> Dict[str, Any]:
        """
        Get all activity data for a specific session.
        
        Args:
            session_id: The ID of the session
            
        Returns:
            Dictionary containing agents, data sources, and interactions for the session
        """
        # Get agent IDs for this session
        agent_ids = self.session_agents.get(session_id, [])
        
        # Filter agents by session
        session_agents = [
            self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents
        ]
        
        # Filter data sources by session
        session_data_sources = [
            source for source_id, source in self.data_sources.items()
            if source.get('session_id') == session_id
        ]
        
        # Filter interactions involving these agents
        agent_names = [agent['name'] for agent in session_agents]
        data_source_names = [source['name'] for source in session_data_sources]
        entity_names = agent_names + data_source_names
        
        session_interactions = [
            interaction for interaction in self.interactions
            if interaction['from'] in entity_names or interaction['to'] in entity_names
        ]
        
        return {
            'agents': session_agents,
            'dataSources': session_data_sources,
            'interactions': session_interactions
        }
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Remove all agents and data sources associated with a session.
        
        Args:
            session_id: The ID of the session to clean up
            
        Returns:
            True if cleanup was successful
        """
        if session_id not in self.session_agents:
            return False
        
        # Get agent IDs for this session
        agent_ids = self.session_agents.get(session_id, [])
        
        # Remove agents
        for agent_id in agent_ids:
            if agent_id in self.agents:
                del self.agents[agent_id]
        
        # Remove data sources
        data_source_ids = [
            source_id for source_id, source in self.data_sources.items()
            if source.get('session_id') == session_id
        ]
        
        for source_id in data_source_ids:
            if source_id in self.data_sources:
                del self.data_sources[source_id]
        
        # Remove session from mapping
        del self.session_agents[session_id]
        
        # Note: We don't remove interactions as they might be useful for history
        
        logger.info(f"Cleaned up session {session_id}: removed {len(agent_ids)} agents and {len(data_source_ids)} data sources")
        return True


# Create singleton instance
agent_activity_tracker = AgentActivityTracker()
