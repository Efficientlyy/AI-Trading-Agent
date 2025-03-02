"""Base analysis agent for the AI Crypto Trading System.

This module defines the base class for all analysis agents in the system.
"""

import asyncio
import gc
import importlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.data_collection.service import DataCollectionService
from src.models.events import (
    CandleDataEvent, ErrorEvent, PatternEvent, SystemStatusEvent, 
    TechnicalIndicatorEvent
)
from src.models.market_data import CandleData, TimeFrame


class AnalysisAgent(Component, ABC):
    """Base class for all analysis agents.
    
    Analysis agents are responsible for processing market data and generating
    signals, indicators, and patterns for the trading system.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the analysis agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(f"analysis_agent_{agent_id}")
        self.agent_id = agent_id
        self.logger = get_logger("analysis_agents", agent_id)
        self.data_collection_service: Optional[DataCollectionService] = None
        
        # Configuration
        self.enabled = config.get(f"analysis_agents.{agent_id}.enabled", True)
        self.symbols: Set[str] = set(config.get(f"analysis_agents.{agent_id}.symbols", []))
        self.exchanges: Set[str] = set(config.get(f"analysis_agents.{agent_id}.exchanges", []))
        self.timeframes: Set[TimeFrame] = set()
        
        # Parse timeframes from config
        timeframe_strs = config.get(f"analysis_agents.{agent_id}.timeframes", [])
        for tf_str in timeframe_strs:
            try:
                self.timeframes.add(TimeFrame(tf_str))
            except ValueError:
                self.logger.error("Invalid timeframe in configuration", timeframe=tf_str)
        
        # Analysis intervals
        self.analysis_interval = config.get(
            f"analysis_agents.{agent_id}.analysis_interval", 60
        )  # seconds
        self.last_analysis_time: Dict[Tuple[str, str, TimeFrame], datetime] = {}
        self.analysis_task = None
    
    async def _initialize(self) -> None:
        """Initialize the analysis agent."""
        if not self.enabled:
            self.logger.info("Analysis agent is disabled")
            return
        
        self.logger.info("Initializing analysis agent", 
                        symbols=list(self.symbols),
                        exchanges=list(self.exchanges),
                        timeframes=[tf.value for tf in self.timeframes])
        
        # Get reference to the data collection service
        if not hasattr(self, "data_collection_service") or self.data_collection_service is None:
            try:
                # The data collection service should be provided by the manager that creates this agent
                # but we'll try to find it via the application if not set
                from src.main import Application
                
                # Find the running application instance (singleton approach)
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
    
    async def _start(self) -> None:
        """Start the analysis agent."""
        if not self.enabled:
            return
        
        self.logger.info("Starting analysis agent")
        
        # Subscribe to relevant events
        event_bus.subscribe("CandleDataEvent", self._handle_candle_event)
        
        # Start periodic analysis task
        self.analysis_task = self.create_task(self._run_analysis_periodically())
    
    async def _stop(self) -> None:
        """Stop the analysis agent."""
        if not self.enabled:
            return
        
        self.logger.info("Stopping analysis agent")
        
        # Unsubscribe from events
        event_bus.unsubscribe("CandleDataEvent", self._handle_candle_event)
        
        # Cancel the analysis task
        if self.analysis_task and not self.analysis_task.done():
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
    
    async def _handle_candle_event(self, event: CandleDataEvent) -> None:
        """Handle a candle data event.
        
        Args:
            event: The candle data event
        """
        if not self.enabled:
            return
        
        # Check if this candle is relevant to this agent
        candle = event.candle
        if (not self.symbols or candle.symbol in self.symbols) and \
           (not self.exchanges or candle.exchange in self.exchanges) and \
           (not self.timeframes or candle.timeframe in self.timeframes):
            
            # Process the candle data
            try:
                await self.process_candle(candle)
            except Exception as e:
                self.logger.error("Error processing candle", 
                               symbol=candle.symbol,
                               exchange=candle.exchange,
                               timeframe=candle.timeframe.value,
                               error=str(e))
    
    async def _run_analysis_periodically(self) -> None:
        """Run analysis periodically based on the configured interval."""
        try:
            while True:
                await asyncio.sleep(self.analysis_interval)
                
                if not self.data_collection_service:
                    continue
                
                # Run analysis for each configured symbol, exchange, and timeframe
                for exchange in self.exchanges if self.exchanges else ["*"]:
                    for symbol in self.symbols if self.symbols else ["*"]:
                        for timeframe in self.timeframes if self.timeframes else [TimeFrame("1h")]:
                            key = (exchange, symbol, timeframe)
                            
                            # Skip if we've analyzed this recently
                            now = datetime.utcnow()
                            if key in self.last_analysis_time:
                                elapsed = (now - self.last_analysis_time[key]).total_seconds()
                                if elapsed < self.analysis_interval:
                                    continue
                            
                            # Skip wildcard combinations - these are just for event handling
                            if exchange == "*" or symbol == "*":
                                continue
                            
                            try:
                                # Fetch recent candles for analysis
                                end_time = now
                                start_time = end_time - timedelta(days=30)  # Get up to 30 days of data
                                
                                candles = await self.data_collection_service.storage_manager.get_candles(
                                    symbol=symbol,
                                    exchange=exchange,
                                    timeframe=timeframe,
                                    start_time=start_time,
                                    end_time=end_time
                                )
                                
                                if candles:
                                    # Run analysis on the candles
                                    await self.analyze_market_data(
                                        symbol=symbol,
                                        exchange=exchange,
                                        timeframe=timeframe,
                                        candles=candles
                                    )
                                    
                                    # Update last analysis time
                                    self.last_analysis_time[key] = now
                            
                            except Exception as e:
                                self.logger.error("Error running periodic analysis", 
                                               symbol=symbol,
                                               exchange=exchange,
                                               timeframe=timeframe.value,
                                               error=str(e))
                
        except asyncio.CancelledError:
            self.logger.debug("Periodic analysis task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in periodic analysis task", error=str(e))
    
    @abstractmethod
    async def process_candle(self, candle: CandleData) -> None:
        """Process a new candle data event.
        
        This method is called whenever a new candle is received that is relevant
        to this agent. It should process the candle and update the agent's internal
        state, but not necessarily generate output events.
        
        Args:
            candle: The candle data to process
        """
        pass
    
    @abstractmethod
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data for a symbol, exchange, and timeframe.
        
        This method is called periodically to analyze market data and generate
        output events such as indicators, patterns, or signals.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        pass
    
    async def publish_indicator(
        self,
        symbol: str,
        timeframe: TimeFrame,
        indicator_name: str,
        values: Union[Dict[datetime, float], Dict[datetime, Dict[str, float]], List[Dict]],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Publish a technical indicator event.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
            indicator_name: The name of the indicator (e.g., "RSI", "MACD")
            values: The indicator values, either as a simple dict mapping timestamps to values,
                  a dict mapping timestamps to a dict of named values, or a list of dicts
            timestamp: Optional timestamp for the event (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        await self.publish_event(TechnicalIndicatorEvent(
            source=self.name,
            symbol=symbol,
            timeframe=timeframe,
            indicator_name=indicator_name,
            values=values,
            timestamp=timestamp
        ))
    
    async def publish_pattern(
        self,
        symbol: str,
        timeframe: TimeFrame,
        pattern_name: str,
        confidence: float,
        target_price: Optional[float] = None,
        invalidation_price: Optional[float] = None,
        completion_timeframe: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Publish a pattern event.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
            pattern_name: The name of the pattern (e.g., "Double Top", "Bull Flag")
            confidence: The confidence score for the pattern (0.0 to 1.0)
            target_price: Optional price target for the pattern
            invalidation_price: Optional price level that would invalidate the pattern
            completion_timeframe: Optional timeframe for the pattern to complete
            timestamp: Optional timestamp for the event (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        await self.publish_event(PatternEvent(
            source=self.name,
            symbol=symbol,
            timeframe=timeframe,
            pattern_name=pattern_name,
            confidence=confidence,
            target_price=target_price,
            invalidation_price=invalidation_price,
            completion_timeframe=completion_timeframe,
            timestamp=timestamp
        ))
    
    async def publish_error(
        self, 
        error_type: str, 
        error_message: str, 
        error_details: Optional[Dict] = None
    ) -> None:
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
    
    async def publish_status(
        self, 
        message: str, 
        details: Optional[Dict] = None
    ) -> None:
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