"""
Agent Activity Integration

This module integrates the agent activity tracker with the trading orchestrator
to provide real-time visualization of agent activities during paper trading.
"""

import asyncio
from typing import Dict, Any, Optional
from ..common import logger
from ..common.event_emitter import global_event_emitter
from .agent_activity_tracker import agent_activity_tracker

class AgentActivityIntegration:
    """
    Integrates agent activity tracking with the trading orchestrator.
    """
    
    def __init__(self, session_id: str):
        """Initialize the agent activity integration."""
        self.session_id = session_id
        self.agent_ids = {}  # Maps agent names to their IDs
        self.data_source_ids = {}  # Maps data source names to their IDs
        self.initialized = False
        
    def initialize_tracking(self, orchestrator: Any) -> bool:
        """
        Initialize tracking for a trading orchestrator.
        
        Args:
            orchestrator: The trading orchestrator to track
            
        Returns:
            True if initialization was successful
        """
        if self.initialized:
            return True
            
        try:
            # Register data manager
            if hasattr(orchestrator, 'data_manager'):
                data_manager = orchestrator.data_manager
                provider_info = data_manager.get_provider_info() if hasattr(data_manager, 'get_provider_info') else {}
                provider_name = provider_info.get('provider_name', 'Unknown Data Provider')
                provider_type = provider_info.get('provider_type', 'api')
                symbols = provider_info.get('symbols', [])
                
                source_id = agent_activity_tracker.register_data_source(
                    self.session_id,
                    provider_name,
                    provider_type,
                    symbols
                )
                self.data_source_ids[provider_name] = source_id
                
            # Register strategy manager
            if hasattr(orchestrator, 'strategy_manager'):
                strategy_manager = orchestrator.strategy_manager
                strategy_name = strategy_manager.name if hasattr(strategy_manager, 'name') else 'Strategy Manager'
                
                agent_id = agent_activity_tracker.register_agent(
                    self.session_id,
                    strategy_name,
                    'strategy'
                )
                self.agent_ids[strategy_name] = agent_id
                
                # Connect to data source if available
                if provider_name in self.data_source_ids:
                    agent_activity_tracker.connect_agent_to_source(
                        agent_id,
                        self.data_source_ids[provider_name]
                    )
                
                # Register individual strategies if available
                if hasattr(strategy_manager, 'strategies'):
                    for strategy_key, strategy in strategy_manager.strategies.items():
                        strategy_name = strategy.name if hasattr(strategy, 'name') else f"Strategy {strategy_key}"
                        
                        strategy_id = agent_activity_tracker.register_agent(
                            self.session_id,
                            strategy_name,
                            'strategy'
                        )
                        self.agent_ids[strategy_name] = strategy_id
                        
                        # Connect to strategy manager
                        agent_activity_tracker.connect_agent_to_source(
                            strategy_id,
                            self.agent_ids[strategy_manager.name]
                        )
            
            # Register risk manager
            if hasattr(orchestrator, 'risk_manager'):
                risk_manager = orchestrator.risk_manager
                risk_name = risk_manager.name if hasattr(risk_manager, 'name') else 'Risk Manager'
                
                agent_id = agent_activity_tracker.register_agent(
                    self.session_id,
                    risk_name,
                    'risk'
                )
                self.agent_ids[risk_name] = agent_id
                
                # Connect to strategy manager if available
                if hasattr(orchestrator, 'strategy_manager'):
                    strategy_manager = orchestrator.strategy_manager
                    strategy_name = strategy_manager.name if hasattr(strategy_manager, 'name') else 'Strategy Manager'
                    
                    if strategy_name in self.agent_ids:
                        agent_activity_tracker.connect_agent_to_source(
                            agent_id,
                            self.agent_ids[strategy_name]
                        )
            
            # Register portfolio manager
            if hasattr(orchestrator, 'portfolio_manager'):
                portfolio_manager = orchestrator.portfolio_manager
                portfolio_name = portfolio_manager.name if hasattr(portfolio_manager, 'name') else 'Portfolio Manager'
                
                agent_id = agent_activity_tracker.register_agent(
                    self.session_id,
                    portfolio_name,
                    'portfolio'
                )
                self.agent_ids[portfolio_name] = agent_id
                
                # Connect to risk manager if available
                if hasattr(orchestrator, 'risk_manager'):
                    risk_manager = orchestrator.risk_manager
                    risk_name = risk_manager.name if hasattr(risk_manager, 'name') else 'Risk Manager'
                    
                    if risk_name in self.agent_ids:
                        agent_activity_tracker.connect_agent_to_source(
                            agent_id,
                            self.agent_ids[risk_name]
                        )
            
            # Register execution handler
            if hasattr(orchestrator, 'execution_handler'):
                execution_handler = orchestrator.execution_handler
                execution_name = execution_handler.name if hasattr(execution_handler, 'name') else 'Execution Handler'
                
                agent_id = agent_activity_tracker.register_agent(
                    self.session_id,
                    execution_name,
                    'execution'
                )
                self.agent_ids[execution_name] = agent_id
                
                # Connect to portfolio manager if available
                if hasattr(orchestrator, 'portfolio_manager'):
                    portfolio_manager = orchestrator.portfolio_manager
                    portfolio_name = portfolio_manager.name if hasattr(portfolio_manager, 'name') else 'Portfolio Manager'
                    
                    if portfolio_name in self.agent_ids:
                        agent_activity_tracker.connect_agent_to_source(
                            agent_id,
                            self.agent_ids[portfolio_name]
                        )
            
            self.initialized = True
            logger.info(f"Initialized agent activity tracking for session {self.session_id}")
            
            # Emit initial activity update
            asyncio.create_task(self._emit_activity_update())
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent activity tracking: {e}")
            return False
    
    def update_agent_status(self, agent_name: str, status: str, 
                           processing_time: Optional[float] = None,
                           data_volume: Optional[int] = None,
                           signal_strength: Optional[float] = None) -> bool:
        """
        Update the status of an agent.
        
        Args:
            agent_name: The name of the agent
            status: The new status ('active', 'idle', 'error')
            processing_time: Optional processing time in milliseconds
            data_volume: Optional data volume processed
            signal_strength: Optional signal strength (0-1)
            
        Returns:
            True if the update was successful
        """
        if not self.initialized or agent_name not in self.agent_ids:
            return False
            
        success = agent_activity_tracker.update_agent_status(
            self.agent_ids[agent_name],
            status,
            processing_time,
            data_volume,
            signal_strength
        )
        
        if success:
            # Emit activity update
            asyncio.create_task(self._emit_activity_update())
            
        return success
    
    def update_data_source_status(self, source_name: str, status: str) -> bool:
        """
        Update the status of a data source.
        
        Args:
            source_name: The name of the data source
            status: The new status ('active', 'idle', 'error')
            
        Returns:
            True if the update was successful
        """
        if not self.initialized or source_name not in self.data_source_ids:
            return False
            
        success = agent_activity_tracker.update_data_source_status(
            self.data_source_ids[source_name],
            status
        )
        
        if success:
            # Emit activity update
            asyncio.create_task(self._emit_activity_update())
            
        return success
    
    def record_interaction(self, from_name: str, to_name: str, interaction_type: str, data: Any = None) -> bool:
        """
        Record an interaction between agents or data sources.
        
        Args:
            from_name: The name of the source entity
            to_name: The name of the target entity
            interaction_type: The type of interaction
            data: Optional data associated with the interaction
            
        Returns:
            True if the interaction was recorded
        """
        if not self.initialized:
            return False
            
        from_id = self.agent_ids.get(from_name) or self.data_source_ids.get(from_name)
        to_id = self.agent_ids.get(to_name) or self.data_source_ids.get(to_name)
        
        if not from_id or not to_id:
            return False
            
        success = agent_activity_tracker.record_interaction(
            from_id,
            to_id,
            interaction_type,
            data
        )
        
        if success:
            # Emit activity update
            asyncio.create_task(self._emit_activity_update())
            
        return success
    
    async def _emit_activity_update(self) -> None:
        """Emit an agent activity update event."""
        await global_event_emitter.emit_async('agent_activity_update', {
            'session_id': self.session_id
        })
    
    def cleanup(self) -> bool:
        """
        Clean up tracking for this session.
        
        Returns:
            True if cleanup was successful
        """
        if not self.initialized:
            return True
            
        success = agent_activity_tracker.cleanup_session(self.session_id)
        self.initialized = False
        self.agent_ids = {}
        self.data_source_ids = {}
        
        logger.info(f"Cleaned up agent activity tracking for session {self.session_id}")
        return success
