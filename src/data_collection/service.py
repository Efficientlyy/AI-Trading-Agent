"""Data collection service for the AI Crypto Trading System.

This module defines the DataCollectionService class, which manages the collection
of market data from various sources, including cryptocurrency exchanges.
"""

import asyncio
import importlib
from typing import Dict, List, Optional, Set, Type

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.data_collection.exchange_connector import ExchangeConnector
from src.data_collection.persistence.storage_manager import StorageManager
from src.models.events import ErrorEvent, SystemStatusEvent, SymbolListEvent
from src.models.market_data import TimeFrame


class DataCollectionService(Component):
    """Service for collecting market data from various sources.
    
    This service manages exchange connectors and other data sources,
    coordinating the collection of market data for the system.
    """
    
    def __init__(self):
        """Initialize the data collection service."""
        super().__init__("data_collection")
        self.logger = get_logger("data_collection", "service")
        self.exchange_connectors: Dict[str, ExchangeConnector] = {}
        self.enabled_exchanges: Set[str] = set()
        self.storage_manager = StorageManager()
    
    async def _initialize(self) -> None:
        """Initialize the data collection service.
        
        Implementation of the abstract method from Component class.
        """
        # This is already handled in the non-underscore version
        pass
    
    async def _start(self) -> None:
        """Start the data collection service.
        
        Implementation of the abstract method from Component class.
        """
        # This is already handled in the non-underscore version
        pass
    
    async def _stop(self) -> None:
        """Stop the data collection service.
        
        Implementation of the abstract method from Component class.
        """
        # This is already handled in the non-underscore version
        pass
    
    async def initialize(self) -> bool:
        """Initialize the data collection service.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing data collection service")
        
        # Initialize storage manager
        success = await self.storage_manager.initialize()
        if not success:
            self.logger.error("Failed to initialize storage manager")
            return False
        
        # Load enabled exchanges from configuration
        self.enabled_exchanges = set(config.get("data_collection.enabled_exchanges", []))
        
        if not self.enabled_exchanges:
            self.logger.warning("No exchanges enabled in configuration")
            await self.publish_status("No exchanges enabled in configuration")
            await super().initialize()
            return True
        
        self.logger.info("Enabled exchanges", exchanges=list(self.enabled_exchanges))
        
        # Initialize exchange connectors
        for exchange_id in self.enabled_exchanges:
            try:
                # Get exchange connector class
                connector_class = await self._get_exchange_connector_class(exchange_id)
                if not connector_class:
                    self.logger.error("Failed to load exchange connector", exchange=exchange_id)
                    continue
                
                # Create and initialize exchange connector
                connector = connector_class(exchange_id)
                self.exchange_connectors[exchange_id] = connector
                
                # Initialize the connector
                success = await connector.initialize()
                if not success:
                    self.logger.error("Failed to initialize exchange connector", exchange=exchange_id)
                    await self.publish_error(
                        "initialization_error",
                        f"Failed to initialize exchange connector for {exchange_id}",
                        {"exchange": exchange_id}
                    )
                    continue
                
                self.logger.info("Initialized exchange connector", exchange=exchange_id)
                
            except Exception as e:
                self.logger.exception("Error initializing exchange connector", 
                                     exchange=exchange_id, error=str(e))
                await self.publish_error(
                    "initialization_error",
                    f"Error initializing exchange connector for {exchange_id}: {str(e)}",
                    {"exchange": exchange_id, "error": str(e)}
                )
        
        if not self.exchange_connectors:
            self.logger.error("Failed to initialize any exchange connectors")
            await self.publish_status("Failed to initialize any exchange connectors")
            return False
        
        self.logger.info("Data collection service initialized", 
                       connector_count=len(self.exchange_connectors))
        
        await super().initialize()
        return True
    
    async def start(self) -> bool:
        """Start the data collection service.
        
        Returns:
            bool: True if start was successful, False otherwise
        """
        self.logger.info("Starting data collection service")
        
        # Start the storage manager
        success = await self.storage_manager.start()
        if not success:
            self.logger.error("Failed to start storage manager")
            return False
        
        # Start all exchange connectors
        for exchange_id, connector in self.exchange_connectors.items():
            try:
                success = await connector.start()
                if not success:
                    self.logger.error("Failed to start exchange connector", exchange=exchange_id)
                    await self.publish_error(
                        "start_error",
                        f"Failed to start exchange connector for {exchange_id}",
                        {"exchange": exchange_id}
                    )
                    continue
                
                self.logger.info("Started exchange connector", exchange=exchange_id)
                
            except Exception as e:
                self.logger.exception("Error starting exchange connector", 
                                     exchange=exchange_id, error=str(e))
                await self.publish_error(
                    "start_error",
                    f"Error starting exchange connector for {exchange_id}: {str(e)}",
                    {"exchange": exchange_id, "error": str(e)}
                )
        
        # Register event handlers
        event_bus.subscribe("SymbolListEvent", self._handle_symbol_list_event)
        
        self.logger.info("Data collection service started")
        await self.publish_status("Data collection service started")
        
        await super().start()
        return True
    
    async def stop(self) -> bool:
        """Stop the data collection service.
        
        Returns:
            bool: True if stop was successful, False otherwise
        """
        self.logger.info("Stopping data collection service")
        
        # Unregister event handlers
        event_bus.unsubscribe("SymbolListEvent", self._handle_symbol_list_event)
        
        # Stop all exchange connectors in reverse order
        for exchange_id, connector in reversed(list(self.exchange_connectors.items())):
            try:
                success = await connector.stop()
                if not success:
                    self.logger.error("Failed to stop exchange connector", exchange=exchange_id)
                    continue
                
                self.logger.info("Stopped exchange connector", exchange=exchange_id)
                
            except Exception as e:
                self.logger.exception("Error stopping exchange connector", 
                                     exchange=exchange_id, error=str(e))
        
        # Stop the storage manager
        try:
            await self.storage_manager.stop()
            self.logger.info("Stopped storage manager")
        except Exception as e:
            self.logger.error("Error stopping storage manager", error=str(e))
        
        self.logger.info("Data collection service stopped")
        await self.publish_status("Data collection service stopped")
        
        await super().stop()
        return True
    
    async def subscribe_candles(self, exchange: str, symbol: str, timeframe: TimeFrame) -> bool:
        """Subscribe to candle data for a symbol and timeframe.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if exchange not in self.exchange_connectors:
            self.logger.error("Cannot subscribe to unknown exchange", 
                            exchange=exchange, symbol=symbol, timeframe=timeframe.value)
            return False
        
        connector = self.exchange_connectors[exchange]
        return await connector.subscribe_candles(symbol, timeframe)
    
    async def subscribe_orderbook(self, exchange: str, symbol: str) -> bool:
        """Subscribe to order book data for a symbol.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if exchange not in self.exchange_connectors:
            self.logger.error("Cannot subscribe to unknown exchange", 
                            exchange=exchange, symbol=symbol)
            return False
        
        connector = self.exchange_connectors[exchange]
        return await connector.subscribe_orderbook(symbol)
    
    async def subscribe_trades(self, exchange: str, symbol: str) -> bool:
        """Subscribe to trade data for a symbol.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if exchange not in self.exchange_connectors:
            self.logger.error("Cannot subscribe to unknown exchange", 
                            exchange=exchange, symbol=symbol)
            return False
        
        connector = self.exchange_connectors[exchange]
        return await connector.subscribe_trades(symbol)
    
    async def unsubscribe_candles(self, exchange: str, symbol: str, timeframe: TimeFrame) -> bool:
        """Unsubscribe from candle data for a symbol and timeframe.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if exchange not in self.exchange_connectors:
            return True
        
        connector = self.exchange_connectors[exchange]
        return await connector.unsubscribe_candles(symbol, timeframe)
    
    async def unsubscribe_orderbook(self, exchange: str, symbol: str) -> bool:
        """Unsubscribe from order book data for a symbol.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if exchange not in self.exchange_connectors:
            return True
        
        connector = self.exchange_connectors[exchange]
        return await connector.unsubscribe_orderbook(symbol)
    
    async def unsubscribe_trades(self, exchange: str, symbol: str) -> bool:
        """Unsubscribe from trade data for a symbol.
        
        Args:
            exchange: The exchange identifier
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if exchange not in self.exchange_connectors:
            return True
        
        connector = self.exchange_connectors[exchange]
        return await connector.unsubscribe_trades(symbol)
    
    async def _get_exchange_connector_class(self, exchange_id: str) -> Optional[Type[ExchangeConnector]]:
        """Get the exchange connector class for an exchange.
        
        Args:
            exchange_id: The exchange identifier
            
        Returns:
            Optional[Type[ExchangeConnector]]: The exchange connector class, or None if not found
        """
        # Get the connector class name from configuration
        connector_class_name = config.get(f"exchanges.{exchange_id}.connector_class")
        if not connector_class_name:
            self.logger.error("No connector class specified for exchange", exchange=exchange_id)
            return None
        
        # Try to import the connector module
        try:
            module_name = f"src.data_collection.connectors.{exchange_id.lower()}"
            module = importlib.import_module(module_name)
            
            # Get the connector class from the module
            connector_class = getattr(module, connector_class_name)
            
            # Verify that the class is a subclass of ExchangeConnector
            if not issubclass(connector_class, ExchangeConnector):
                self.logger.error("Connector class is not a subclass of ExchangeConnector", 
                                exchange=exchange_id, class_name=connector_class_name)
                return None
            
            return connector_class
            
        except ImportError:
            self.logger.error("Failed to import connector module", 
                            exchange=exchange_id, module=module_name)
            return None
        except AttributeError:
            self.logger.error("Connector class not found in module", 
                            exchange=exchange_id, class_name=connector_class_name)
            return None
        except Exception as e:
            self.logger.exception("Error loading connector class", 
                                exchange=exchange_id, error=str(e))
            return None
    
    async def _handle_symbol_list_event(self, event: 'SymbolListEvent') -> None:
        """Handle a symbol list event from an exchange connector.
        
        Args:
            event: The symbol list event
        """
        self.logger.debug("Received symbol list event", 
                        exchange=event.exchange, symbol_count=len(event.symbols))
        
        # Process the symbol list (e.g., update database, notify other components)
        # This is a placeholder for future implementation
        pass
    
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