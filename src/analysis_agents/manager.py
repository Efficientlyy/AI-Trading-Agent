"""Analysis manager for the AI Crypto Trading System.

This module defines the AnalysisManager class, which manages all analysis agents.
"""

import asyncio
import gc
from typing import Dict, List, Optional, Type

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.data_collection.service import DataCollectionService
from src.models.events import ErrorEvent, SystemStatusEvent

from src.analysis_agents.base_agent import AnalysisAgent


class AnalysisManager(Component):
    """Manager for market data analysis.
    
    This component manages the lifecycle of all analysis agents and coordinates
    the analysis of market data across different strategies and timeframes.
    """
    
    def __init__(self):
        """Initialize the analysis manager."""
        super().__init__("analysis_manager")
        self.logger = get_logger("analysis_agents", "manager")
        self.agents: Dict[str, AnalysisAgent] = {}
        self.enabled = config.get("analysis_agents.enabled", True)
        self.data_collection_service: Optional[DataCollectionService] = None
    
    async def _initialize(self) -> None:
        """Initialize the analysis manager."""
        if not self.enabled:
            self.logger.info("Analysis manager is disabled")
            return
        
        self.logger.info("Initializing analysis manager")
        
        # Get reference to the data collection service
        try:
            # Find the running application instance (similar approach as in base_agent)
            from src.main import Application
            
            # Find the running application instance via gc
            app_instance = None
            for obj in gc.get_objects():
                if isinstance(obj, Application) and hasattr(obj, "data_collection_service"):
                    app_instance = obj
                    break
            
            if app_instance and app_instance.data_collection_service:
                self.data_collection_service = app_instance.data_collection_service
                self.logger.debug("Found data collection service from application instance")
            else:
                self.logger.error("Failed to get reference to data collection service")
                await self.publish_error(
                    "initialization_error",
                    "Failed to get reference to data collection service"
                )
                return
        except ImportError as e:
            self.logger.error("Failed to import main module", error=str(e))
            await self.publish_error(
                "initialization_error",
                f"Failed to import main module: {str(e)}"
            )
            return
        except Exception as e:
            self.logger.error("Error getting data collection service", error=str(e))
            await self.publish_error(
                "initialization_error",
                f"Error getting data collection service: {str(e)}"
            )
            return
        
        # Load all enabled analysis agents from configuration
        enabled_agents = config.get("analysis_agents.enabled_agents", [])
        if not enabled_agents:
            self.logger.warning("No analysis agents enabled in configuration")
            return
        
        # Initialize each agent
        for agent_id in enabled_agents:
            try:
                # Get the agent class
                agent_class = await self._get_agent_class(agent_id)
                if not agent_class:
                    self.logger.error("Failed to load agent class", agent=agent_id)
                    continue
                
                # Create and initialize the agent
                agent = agent_class(agent_id)
                agent.data_collection_service = self.data_collection_service
                self.agents[agent_id] = agent
                
                # Initialize the agent
                success = agent.initialize()
                if not success:
                    self.logger.error("Failed to initialize analysis agent", agent=agent_id)
                    continue
                
                self.logger.info("Initialized analysis agent", agent=agent_id)
                
            except Exception as e:
                self.logger.exception("Error initializing analysis agent", 
                                     agent=agent_id, error=str(e))
                await self.publish_error(
                    "initialization_error",
                    f"Error initializing analysis agent {agent_id}: {str(e)}",
                    {"agent": agent_id, "error": str(e)}
                )
        
        if not self.agents:
            self.logger.warning("No analysis agents were initialized")
        else:
            self.logger.info("Analysis manager initialized", 
                           agent_count=len(self.agents))
    
    async def _start(self) -> None:
        """Start the analysis manager."""
        if not self.enabled:
            return
        
        self.logger.info("Starting analysis manager")
        
        # Start all agents
        for agent_id, agent in self.agents.items():
            try:
                success = agent.start()
                if not success:
                    self.logger.error("Failed to start analysis agent", agent=agent_id)
                    continue
                
                self.logger.info("Started analysis agent", agent=agent_id)
                
            except Exception as e:
                self.logger.exception("Error starting analysis agent", 
                                     agent=agent_id, error=str(e))
                await self.publish_error(
                    "start_error",
                    f"Error starting analysis agent {agent_id}: {str(e)}",
                    {"agent": agent_id, "error": str(e)}
                )
        
        self.logger.info("Analysis manager started")
        await self.publish_status("Analysis manager started")
    
    async def _stop(self) -> None:
        """Stop the analysis manager."""
        if not self.enabled:
            return
        
        self.logger.info("Stopping analysis manager")
        
        # Stop all agents in reverse order
        for agent_id, agent in reversed(list(self.agents.items())):
            try:
                success = agent.stop()
                if not success:
                    self.logger.error("Failed to stop analysis agent", agent=agent_id)
                    continue
                
                self.logger.info("Stopped analysis agent", agent=agent_id)
                
            except Exception as e:
                self.logger.exception("Error stopping analysis agent", 
                                     agent=agent_id, error=str(e))
        
        self.logger.info("Analysis manager stopped")
        await self.publish_status("Analysis manager stopped")
    
    async def _get_agent_class(self, agent_id: str) -> Optional[Type[AnalysisAgent]]:
        """Get the agent class for an agent ID.
        
        Args:
            agent_id: The agent identifier
            
        Returns:
            Optional[Type[AnalysisAgent]]: The agent class, or None if not found
        """
        import importlib
        
        # Get the agent class name from configuration
        agent_class_name = config.get(f"analysis_agents.{agent_id}.class_name")
        if not agent_class_name:
            self.logger.error("No agent class specified for agent", agent=agent_id)
            return None
        
        # Get the module name from configuration or use the default
        module_name = config.get(
            f"analysis_agents.{agent_id}.module",
            f"src.analysis_agents.{agent_id.lower()}"
        )
        
        # Try to import the agent module
        try:
            module = importlib.import_module(module_name)
            
            # Get the agent class from the module
            agent_class = getattr(module, agent_class_name)
            
            # Verify that the class is a subclass of AnalysisAgent
            if not issubclass(agent_class, AnalysisAgent):
                self.logger.error("Agent class is not a subclass of AnalysisAgent", 
                                agent=agent_id, class_name=agent_class_name)
                return None
            
            return agent_class
            
        except ImportError:
            self.logger.error("Failed to import agent module", 
                            agent=agent_id, module=module_name)
            return None
        except AttributeError:
            self.logger.error("Agent class not found in module", 
                            agent=agent_id, class_name=agent_class_name)
            return None
        except Exception as e:
            self.logger.exception("Error loading agent class", 
                                agent=agent_id, error=str(e))
            return None
    
    async def publish_status(self, message: str, details: Optional[Dict] = None) -> None:
        """Publish a status event.
        
        Args:
            message: The status message
            details: Optional details about the status
        """
        await self.publish_event(SystemStatusEvent(
            source=self.name,
            status="info",
            message=message,
            details=details or {}
        ))
    
    async def publish_error(self, error_type: str, error_message: str, 
                          error_details: Optional[Dict] = None) -> None:
        """Publish an error event.
        
        Args:
            error_type: The type of error
            error_message: The error message
            error_details: Optional details about the error
        """
        await self.publish_event(ErrorEvent(
            source=self.name,
            error_type=error_type,
            error_message=error_message,
            error_details=error_details or {}
        )) 