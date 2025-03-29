"""Strategy manager for the AI Crypto Trading System.

This module defines the StrategyManager class, which manages all trading strategies.
"""

import asyncio
import gc
from typing import Dict, List, Optional, Type

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.models.events import ErrorEvent, SystemStatusEvent
from src.strategy.base_strategy import Strategy


class StrategyManager(Component):
    """Manager for trading strategies.
    
    This component manages the lifecycle of all trading strategies and coordinates
    the generation of trading signals across different strategies.
    """
    
    def __init__(self):
        """Initialize the strategy manager."""
        super().__init__("strategy_manager")
        self.logger = get_logger("strategy", "manager")
        self.strategies: Dict[str, Strategy] = {}
        self.enabled = config.get("strategies.enabled", True)
    
    async def _initialize(self) -> None:
        """Initialize the strategy manager."""
        if not self.enabled:
            self.logger.info("Strategy manager is disabled")
            return
        
        self.logger.info("Initializing strategy manager")
        
        # Load all enabled strategies from configuration
        enabled_strategies = config.get("strategies.enabled_strategies", [])
        if not enabled_strategies:
            self.logger.warning("No strategies enabled in configuration")
            return
        
        # Initialize each strategy
        for strategy_id in enabled_strategies:
            try:
                # Get the strategy class
                strategy_class = await self._get_strategy_class(strategy_id)
                if not strategy_class:
                    self.logger.error("Failed to load strategy class", strategy=strategy_id)
                    continue
                
                # Create and initialize the strategy
                strategy = strategy_class(strategy_id)
                self.strategies[strategy_id] = strategy
                
                # Initialize the strategy
                success = strategy.initialize()
                if not success:
                    self.logger.error("Failed to initialize strategy", strategy=strategy_id)
                    continue
                
                self.logger.info("Initialized strategy", strategy=strategy_id)
                
            except Exception as e:
                self.logger.exception("Error initializing strategy", 
                                     strategy=strategy_id, error=str(e))
                await self.publish_error(
                    "initialization_error",
                    f"Error initializing strategy {strategy_id}: {str(e)}",
                    {"strategy": strategy_id, "error": str(e)}
                )
        
        if not self.strategies:
            self.logger.warning("No strategies were initialized")
        else:
            self.logger.info("Strategy manager initialized", 
                           strategy_count=len(self.strategies))
    
    async def _start(self) -> None:
        """Start the strategy manager."""
        if not self.enabled:
            return
        
        self.logger.info("Starting strategy manager")
        
        # Start all strategies
        for strategy_id, strategy in self.strategies.items():
            try:
                success = strategy.start()
                if not success:
                    self.logger.error("Failed to start strategy", strategy=strategy_id)
                    continue
                
                self.logger.info("Started strategy", strategy=strategy_id)
                
            except Exception as e:
                self.logger.exception("Error starting strategy", 
                                     strategy=strategy_id, error=str(e))
                await self.publish_error(
                    "start_error",
                    f"Error starting strategy {strategy_id}: {str(e)}",
                    {"strategy": strategy_id, "error": str(e)}
                )
        
        self.logger.info("Strategy manager started")
        await self.publish_status("Strategy manager started")
    
    async def _stop(self) -> None:
        """Stop the strategy manager."""
        if not self.enabled:
            return
        
        self.logger.info("Stopping strategy manager")
        
        # Stop all strategies in reverse order
        for strategy_id, strategy in reversed(list(self.strategies.items())):
            try:
                success = strategy.stop()
                if not success:
                    self.logger.error("Failed to stop strategy", strategy=strategy_id)
                    continue
                
                self.logger.info("Stopped strategy", strategy=strategy_id)
                
            except Exception as e:
                self.logger.exception("Error stopping strategy", 
                                     strategy=strategy_id, error=str(e))
        
        self.logger.info("Strategy manager stopped")
        await self.publish_status("Strategy manager stopped")
    
    async def _get_strategy_class(self, strategy_id: str) -> Optional[Type[Strategy]]:
        """Get the strategy class for a strategy ID.
        
        Args:
            strategy_id: The strategy identifier
            
        Returns:
            Optional[Type[Strategy]]: The strategy class, or None if not found
        """
        import importlib
        
        # Get the strategy class name from configuration
        strategy_class_name = config.get(f"strategies.{strategy_id}.class_name")
        if not strategy_class_name:
            self.logger.error("No strategy class specified for strategy", strategy=strategy_id)
            return None
        
        # Get the module name from configuration or use the default
        module_name = config.get(
            f"strategies.{strategy_id}.module",
            f"src.strategy.{strategy_id.lower()}"
        )
        
        # Try to import the strategy module
        try:
            module = importlib.import_module(module_name)
            
            # Get the strategy class from the module
            strategy_class = getattr(module, strategy_class_name)
            
            # Verify that the class is a subclass of Strategy
            if not issubclass(strategy_class, Strategy):
                self.logger.error("Strategy class is not a subclass of Strategy", 
                                strategy=strategy_id, class_name=strategy_class_name)
                return None
            
            return strategy_class
            
        except ImportError:
            self.logger.error("Failed to import strategy module", 
                            strategy=strategy_id, module=module_name)
            return None
        except AttributeError:
            self.logger.error("Strategy class not found in module", 
                            strategy=strategy_id, class_name=strategy_class_name)
            return None
        except Exception as e:
            self.logger.exception("Error loading strategy class", 
                                strategy=strategy_id, error=str(e))
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