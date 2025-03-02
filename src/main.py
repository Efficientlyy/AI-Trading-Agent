"""Main application entry point for the AI Crypto Trading System."""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

import structlog

from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger, system_logger
from src.data_collection.service import DataCollectionService
from src.analysis_agents.manager import AnalysisManager
from src.strategy.manager import StrategyManager
from src.portfolio.manager import PortfolioManager
from src.execution.service import ExecutionService
from src.models.events import (
    CandleDataEvent, ErrorEvent, MarketDataEvent, MarketRegimeEvent,
    OrderBookEvent, PatternEvent, SentimentEvent, SignalEvent,
    SymbolListEvent, SystemStatusEvent, TechnicalIndicatorEvent,
    TradeDataEvent, OrderEvent
)

# Configure logger
logger = get_logger("system", "main")


class Application:
    """Main application class that manages the lifecycle of all components."""
    
    def __init__(self):
        """Initialize the application."""
        self.running = False
        self.components = []
        self.shutdown_event = asyncio.Event()
        
        # Initialize service instances
        self.data_collection_service = None
        self.analysis_manager = None
        self.strategy_manager = None
        self.portfolio_manager = None
        self.execution_service = None
    
    async def start(self):
        """Start the application and all its components."""
        logger.info("Starting AI Crypto Trading System")
        
        # Register signal handlers for graceful shutdown
        self.register_signal_handlers()
        
        try:
            # Load all component configurations
            self.load_component_configs()
            
            # Register event types
            self.register_event_types()
            
            # Start the event bus
            await event_bus.start()
            
            # Start all enabled components
            await self.start_components()
            
            # Application started successfully
            self.running = True
            logger.info("AI Crypto Trading System started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.exception("Error starting application", error=str(e))
            raise
    
    def load_component_configs(self):
        """Load all component configuration files."""
        config_dir = Path("config")
        config_files = list(config_dir.glob("*.yaml"))
        
        logger.info("Loading component configurations", files=len(config_files))
        
        for config_file in config_files:
            if config_file.name != "system.yaml":  # System config already loaded
                try:
                    config.load_config_file(config_file)
                    logger.debug("Loaded configuration file", file=config_file.name)
                except Exception as e:
                    logger.error("Failed to load configuration file", file=config_file.name, error=str(e))
    
    def register_event_types(self):
        """Register all event types used by the system."""
        # Register base events
        event_bus.register_event_type("Event")
        event_bus.register_event_type("ErrorEvent")
        event_bus.register_event_type("SystemStatusEvent")
        
        # Register market data events
        event_bus.register_event_type("MarketDataEvent")
        event_bus.register_event_type("CandleDataEvent")
        event_bus.register_event_type("TradeDataEvent")
        event_bus.register_event_type("OrderBookEvent")
        event_bus.register_event_type("SymbolListEvent")
        
        # Register analysis events
        event_bus.register_event_type("TechnicalIndicatorEvent")
        event_bus.register_event_type("PatternEvent")
        event_bus.register_event_type("SentimentEvent")
        event_bus.register_event_type("MarketRegimeEvent")
        
        # Register decision events
        event_bus.register_event_type("SignalEvent")
        
        # Register execution events
        event_bus.register_event_type("OrderEvent")
        
        logger.debug("Registered standard event types")
    
    async def start_components(self):
        """Start all enabled components based on configuration."""
        # Initialize and start data collection service if enabled
        if config.get("data_collection.enabled", True):
            try:
                logger.info("Initializing data collection service")
                self.data_collection_service = DataCollectionService()
                success = await self.data_collection_service.initialize()
                
                if success:
                    logger.info("Starting data collection service")
                    await self.data_collection_service.start()
                    self.components.append(self.data_collection_service)
                    logger.info("Data collection service started successfully")
                else:
                    logger.error("Failed to initialize data collection service")
            except Exception as e:
                logger.exception("Error starting data collection service", error=str(e))
        else:
            logger.info("Data collection service is disabled in configuration")

        # Initialize and start analysis manager if enabled
        if config.get("analysis_agents.enabled", True):
            try:
                logger.info("Initializing analysis manager")
                self.analysis_manager = AnalysisManager()
                success = await self.analysis_manager.initialize()
                
                if success:
                    logger.info("Starting analysis manager")
                    await self.analysis_manager.start()
                    self.components.append(self.analysis_manager)
                    logger.info("Analysis manager started successfully")
                else:
                    logger.error("Failed to initialize analysis manager")
            except Exception as e:
                logger.exception("Error starting analysis manager", error=str(e))
        else:
            logger.info("Analysis manager is disabled in configuration")
        
        # Initialize and start strategy manager if enabled
        if config.get("strategies.enabled", True):
            try:
                logger.info("Initializing strategy manager")
                self.strategy_manager = StrategyManager()
                success = await self.strategy_manager.initialize()
                
                if success:
                    logger.info("Starting strategy manager")
                    await self.strategy_manager.start()
                    self.components.append(self.strategy_manager)
                    logger.info("Strategy manager started successfully")
                else:
                    logger.error("Failed to initialize strategy manager")
            except Exception as e:
                logger.exception("Error starting strategy manager", error=str(e))
        else:
            logger.info("Strategy manager is disabled in configuration")
            
        # Initialize and start portfolio manager if enabled
        if config.get("portfolio.enabled", True):
            try:
                logger.info("Initializing portfolio manager")
                self.portfolio_manager = PortfolioManager()
                success = await self.portfolio_manager.initialize()
                
                if success:
                    logger.info("Starting portfolio manager")
                    await self.portfolio_manager.start()
                    self.components.append(self.portfolio_manager)
                    logger.info("Portfolio manager started successfully")
                else:
                    logger.error("Failed to initialize portfolio manager")
            except Exception as e:
                logger.exception("Error starting portfolio manager", error=str(e))
        else:
            logger.info("Portfolio manager is disabled in configuration")
        
        # Initialize and start execution service if enabled
        if config.get("execution.enabled", True):
            try:
                logger.info("Initializing execution service")
                self.execution_service = ExecutionService()
                success = await self.execution_service.initialize()
                
                if success:
                    logger.info("Starting execution service")
                    await self.execution_service.start()
                    self.components.append(self.execution_service)
                    logger.info("Execution service started successfully")
                else:
                    logger.error("Failed to initialize execution service")
            except Exception as e:
                logger.exception("Error starting execution service", error=str(e))
        else:
            logger.info("Execution service is disabled in configuration")
        
        # Add other component initialization here as they are implemented
    
    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        # Register for SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(self.shutdown())
            )
        
        logger.debug("Registered signal handlers")
    
    async def shutdown(self):
        """Shut down the application gracefully."""
        if not self.running:
            return
        
        logger.info("Shutting down AI Crypto Trading System")
        self.running = False
        
        # Shut down components in reverse order
        for component in reversed(self.components):
            try:
                if hasattr(component, "stop"):
                    if asyncio.iscoroutinefunction(component.stop):
                        await component.stop()
                    else:
                        component.stop()
            except Exception as e:
                logger.error("Error stopping component", component=component.__class__.__name__, error=str(e))
        
        # Stop the event bus
        await event_bus.stop()
        
        # Set shutdown event to release the main task
        self.shutdown_event.set()
        
        logger.info("AI Crypto Trading System shutdown complete")

    def get_component(self, component_id: str) -> Optional[Any]:
        """Get a component by its ID.
        
        Args:
            component_id: The component identifier
            
        Returns:
            The component if found, None otherwise
        """
        if component_id == "data_collection" and self.data_collection_service:
            return self.data_collection_service
        elif component_id == "analysis_manager" and self.analysis_manager:
            return self.analysis_manager
        elif component_id == "strategy_manager" and self.strategy_manager:
            return self.strategy_manager
        elif component_id == "portfolio" and self.portfolio_manager:
            return self.portfolio_manager
        elif component_id == "execution" and self.execution_service:
            return self.execution_service
        
        # Search through all components
        for component in self.components:
            if hasattr(component, "name") and component.name == component_id:
                return component
        
        return None


async def main():
    """Main application entry point."""
    app = Application()
    await app.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        system_logger.info("Application stopped by user")
    except Exception as e:
        system_logger.exception("Unhandled exception", error=str(e))
        sys.exit(1) 