"""
Technical Analysis Agent Integration with Trading Orchestrator

This module provides the integration layer between the Technical Analysis Agent
and the Trading Orchestrator, handling signal routing and data flow.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import pandas as pd
import numpy as np
from datetime import datetime
import time
import asyncio
import threading
from queue import Queue

from ..agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ..config.data_source_config import get_data_source_config, DataSourceConfig
from ..data.data_source_factory import get_data_source_factory
from ..common.utils import get_logger
from ..common.event_bus import EventBus
from ..common.signal_types import Signal, SignalType, SignalConfidence, SignalDirection

# Setup logging
logger = get_logger(__name__)

class TechnicalAgentOrchestrator:
    """
    Integration layer between Technical Analysis Agent and Trading Orchestrator.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the Technical Agent Orchestrator.
        
        Args:
            config: Configuration dictionary
            event_bus: Event bus for system-wide communication
        """
        self.config = config or {}
        self.event_bus = event_bus or EventBus()
        
        # Create Technical Analysis Agent
        self.ta_agent = AdvancedTechnicalAnalysisAgent()
        
        # Signal queues for output routing
        self.signal_queues = {
            'decision': Queue(),
            'visualization': Queue(),
            'logging': Queue()
        }
        
        # Track registered consumers
        self.signal_consumers = {
            'decision': [],
            'visualization': [],
            'logging': []
        }
        
        # Agent status
        self.status = {
            'running': False,
            'last_update': None,
            'symbols_monitored': [],
            'timeframes_monitored': [],
            'data_source': 'mock',
            'signal_count': 0,
            'error_count': 0
        }
        
        # Start signal processing threads
        self._start_signal_processors()
        
        # Register for data source toggle events
        self.event_bus.subscribe('data_source_toggled', self._handle_data_source_toggle)
    
    def _start_signal_processors(self):
        """Start the signal processing threads."""
        for queue_name in self.signal_queues:
            thread = threading.Thread(
                target=self._process_signal_queue,
                args=(queue_name,),
                daemon=True
            )
            thread.start()
    
    def _process_signal_queue(self, queue_name: str):
        """
        Process signals from a specific queue.
        
        Args:
            queue_name: Name of the queue to process
        """
        queue = self.signal_queues[queue_name]
        
        while True:
            try:
                signal = queue.get()
                
                # Process the signal
                for consumer in self.signal_consumers[queue_name]:
                    try:
                        consumer(signal)
                    except Exception as e:
                        logger.error(f"Error in {queue_name} consumer: {str(e)}")
                
                queue.task_done()
            except Exception as e:
                logger.error(f"Error processing {queue_name} queue: {str(e)}")
                time.sleep(1)  # Avoid tight loop on error
    
    def register_consumer(self, queue_name: str, consumer: Callable[[Signal], None]):
        """
        Register a consumer for a specific signal queue.
        
        Args:
            queue_name: Name of the queue to consume from
            consumer: Callback function to process signals
        """
        if queue_name not in self.signal_queues:
            raise ValueError(f"Invalid queue name: {queue_name}")
        
        self.signal_consumers[queue_name].append(consumer)
        logger.info(f"Registered consumer for {queue_name} queue")
    
    def start(self, symbols: List[str], timeframes: List[str]):
        """
        Start the Technical Analysis Agent.
        
        Args:
            symbols: List of symbols to monitor
            timeframes: List of timeframes to monitor
        """
        if self.status['running']:
            logger.warning("Technical Analysis Agent already running")
            return
        
        try:
            # Update status
            self.status['running'] = True
            self.status['last_update'] = datetime.now()
            self.status['symbols_monitored'] = symbols
            self.status['timeframes_monitored'] = timeframes
            self.status['data_source'] = 'mock' if get_data_source_config().use_mock_data else 'real'
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(symbols, timeframes),
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info(f"Started Technical Analysis Agent monitoring {len(symbols)} symbols on {len(timeframes)} timeframes")
        except Exception as e:
            self.status['running'] = False
            self.status['error_count'] += 1
            logger.error(f"Failed to start Technical Analysis Agent: {str(e)}")
            raise
    
    def stop(self):
        """Stop the Technical Analysis Agent."""
        if not self.status['running']:
            logger.warning("Technical Analysis Agent not running")
            return
        
        self.status['running'] = False
        logger.info("Stopping Technical Analysis Agent")
    
    def _monitoring_loop(self, symbols: List[str], timeframes: List[str]):
        """
        Monitoring loop for Technical Analysis Agent.
        
        Args:
            symbols: List of symbols to monitor
            timeframes: List of timeframes to monitor
        """
        while self.status['running']:
            try:
                # Update status
                self.status['last_update'] = datetime.now()
                
                # Process each symbol and timeframe
                for symbol in symbols:
                    for timeframe in timeframes:
                        self._process_symbol_timeframe(symbol, timeframe)
                
                # Sleep to avoid excessive processing
                time.sleep(10)  # 10 seconds between cycles
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                self.status['error_count'] += 1
                time.sleep(30)  # Longer sleep on error
    
    def _process_symbol_timeframe(self, symbol: str, timeframe: str):
        """
        Process a specific symbol and timeframe.
        
        Args:
            symbol: Symbol to process
            timeframe: Timeframe to process
        """
        try:
            # Get data from the appropriate source
            data_source_factory = get_data_source_factory()
            data_provider = data_source_factory.get_data_provider()
            
            if get_data_source_config().use_mock_data:
                market_data = data_provider.generate_data(symbols=[symbol], timeframes=[timeframe], periods=100)
            else:
                market_data = data_provider.get_historical_data(symbols=[symbol], timeframes=[timeframe], periods=100)
            
            # Run technical analysis
            analysis_results = self.ta_agent.analyze_market_data(symbol, timeframe, market_data)
            
            # Process signals
            if 'signals' in analysis_results:
                for signal in analysis_results['signals']:
                    self._route_signal(signal)
                    self.status['signal_count'] += 1
        except Exception as e:
            logger.error(f"Error processing {symbol} on {timeframe}: {str(e)}")
            self.status['error_count'] += 1
    
    def _route_signal(self, signal: Dict[str, Any]):
        """
        Route a signal to the appropriate queues.
        
        Args:
            signal: Signal to route
        """
        # Create a structured signal object
        signal_obj = Signal(
            symbol=signal.get('symbol', ''),
            timeframe=signal.get('timeframe', ''),
            timestamp=signal.get('timestamp', datetime.now()),
            signal_type=SignalType(signal.get('type', 'unknown')),
            direction=SignalDirection(signal.get('direction', 'neutral')),
            confidence=SignalConfidence(signal.get('confidence', 0.5)),
            source='technical_analysis',
            metadata=signal.get('metadata', {})
        )
        
        # Route to decision queue for trading decisions
        self.signal_queues['decision'].put(signal_obj)
        
        # Route to visualization queue for UI updates
        self.signal_queues['visualization'].put(signal_obj)
        
        # Route to logging queue for record keeping
        self.signal_queues['logging'].put(signal_obj)
    
    def _handle_data_source_toggle(self, event_data: Dict[str, Any]):
        """
        Handle data source toggle event.
        
        Args:
            event_data: Event data with toggle information
        """
        is_mock = event_data.get('is_mock', True)
        self.status['data_source'] = 'mock' if is_mock else 'real'
        
        logger.info(f"Technical Analysis Agent data source toggled to {'mock' if is_mock else 'real'}")
        
        # Propagate change to the Technical Analysis Agent
        self.ta_agent.toggle_data_source()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the Technical Analysis Agent.
        
        Returns:
            Status dictionary
        """
        return {
            'agent_id': id(self.ta_agent),
            'running': self.status['running'],
            'last_update': self.status['last_update'],
            'symbols_monitored': self.status['symbols_monitored'],
            'timeframes_monitored': self.status['timeframes_monitored'],
            'data_source': self.status['data_source'],
            'signal_count': self.status['signal_count'],
            'error_count': self.status['error_count'],
            'queue_sizes': {
                name: queue.qsize() for name, queue in self.signal_queues.items()
            }
        }

def register_with_orchestrator(orchestrator: Any, symbols: List[str], timeframes: List[str]) -> str:
    """
    Register the Technical Analysis Agent with the Trading Orchestrator.
    
    Args:
        orchestrator: Trading Orchestrator instance
        symbols: List of symbols to monitor
        timeframes: List of timeframes to monitor
        
    Returns:
        Agent ID
    """
    # Create a Technical Agent Orchestrator
    ta_orchestrator = TechnicalAgentOrchestrator(
        event_bus=orchestrator.event_bus if hasattr(orchestrator, 'event_bus') else None
    )
    
    # Register the TA Agent with the main orchestrator
    agent_id = orchestrator.register_agent(
        name="AdvancedTechnicalAnalysisAgent",
        agent_type="analysis",
        description="Advanced Technical Analysis Agent with pattern recognition and regime awareness",
        status_callback=ta_orchestrator.get_status,
        start_callback=lambda: ta_orchestrator.start(symbols, timeframes),
        stop_callback=ta_orchestrator.stop
    )
    
    # Register signal consumers from the main orchestrator
    if hasattr(orchestrator, 'process_analysis_signal'):
        ta_orchestrator.register_consumer('decision', orchestrator.process_analysis_signal)
    
    if hasattr(orchestrator, 'update_visualization'):
        ta_orchestrator.register_consumer('visualization', orchestrator.update_visualization)
    
    logger.info(f"Registered Technical Analysis Agent with orchestrator, agent_id: {agent_id}")
    
    return agent_id
