"""
Integration tests for Technical Analysis Orchestration and Monitoring.

These tests verify that the orchestration and monitoring components work correctly
with the Technical Analysis Agent.
"""

import unittest
import json
import os
import sys
import asyncio
import threading
import time
from unittest.mock import patch, MagicMock, ANY
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adjust Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ai_trading_agent.orchestration.ta_agent_integration import TechnicalAgentOrchestrator
from ai_trading_agent.monitoring.ta_agent_monitor import TAAgentMonitor, setup_production_monitoring
from ai_trading_agent.common.event_bus import EventBus, Event
from ai_trading_agent.common.signal_types import Signal, SignalType, SignalDirection, SignalConfidence
from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.config.data_source_config import DataSourceConfig, get_data_source_config
from ai_trading_agent.data.data_source_factory import DataSourceFactory

class TestTechnicalAnalysisOrchestration(unittest.TestCase):
    """Test the orchestration of Technical Analysis components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock TA agent
        self.mock_ta_agent = MagicMock(spec=AdvancedTechnicalAnalysisAgent)
        self.mock_ta_agent.analyze_market_data.return_value = {
            'signals': [
                {
                    'symbol': 'BTC/USD',
                    'timeframe': '1h',
                    'timestamp': datetime.now(),
                    'type': 'pattern',
                    'direction': 'bullish',
                    'confidence': 0.85,
                    'metadata': {'pattern': 'morning_star'}
                }
            ]
        }
        self.mock_ta_agent.get_status.return_value = {
            'running': True,
            'error_count': 0
        }
        self.mock_ta_agent.get_statistics.return_value = {
            'operation_count': 10,
            'error_count': 0,
            'average_latency_ms': 50,
            'signal_count': 5,
            'pattern_count': 3
        }
        
        # Create mock data provider
        self.mock_data_provider = MagicMock()
        self.mock_data_provider.generate_data.return_value = self._generate_mock_market_data()
        self.mock_data_provider.get_historical_data.return_value = self._generate_mock_market_data()
        
        self.mock_factory = MagicMock()
        self.mock_factory.get_data_provider.return_value = self.mock_data_provider
        
        # Set up patchers
        self.factory_patcher = patch('ai_trading_agent.orchestration.ta_agent_integration.get_data_source_factory',
                                    return_value=self.mock_factory)
        self.mock_factory_func = self.factory_patcher.start()
        
        self.config_patcher = patch('ai_trading_agent.orchestration.ta_agent_integration.get_data_source_config')
        self.mock_config_func = self.config_patcher.start()
        self.mock_config = MagicMock(spec=DataSourceConfig)
        self.mock_config.use_mock_data = True
        self.mock_config_func.return_value = self.mock_config
        
        # Create event bus
        self.event_bus = EventBus()
        
        # Store received signals
        self.received_signals = []
        
    def tearDown(self):
        """Clean up after tests."""
        self.factory_patcher.stop()
        self.config_patcher.stop()
        
    def _generate_mock_market_data(self):
        """Generate mock market data for testing."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2025-01-01', periods=100)
        data = {
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(101, 5, 100),
            'volume': np.random.normal(1000000, 500000, 100),
        }
        
        df = pd.DataFrame(data, index=dates)
        return {'BTC/USD': {'1h': df}}
    
    def test_orchestrator_initialization(self):
        """Test that the orchestrator initializes correctly."""
        # Create orchestrator with our mocks
        orchestrator = TechnicalAgentOrchestrator(event_bus=self.event_bus)
        
        # Verify initial state
        self.assertFalse(orchestrator.status['running'])
        self.assertEqual(orchestrator.status['symbols_monitored'], [])
        self.assertEqual(orchestrator.status['timeframes_monitored'], [])
        self.assertEqual(orchestrator.status['data_source'], 'mock')
        self.assertEqual(orchestrator.status['signal_count'], 0)
        self.assertEqual(orchestrator.status['error_count'], 0)
    
    def test_orchestrator_start_stop(self):
        """Test starting and stopping the orchestrator."""
        # Create orchestrator with our mocks and custom TA agent
        with patch('ai_trading_agent.orchestration.ta_agent_integration.AdvancedTechnicalAnalysisAgent', 
                  return_value=self.mock_ta_agent):
            orchestrator = TechnicalAgentOrchestrator(event_bus=self.event_bus)
            
            # Start orchestrator
            symbols = ['BTC/USD', 'ETH/USD']
            timeframes = ['1h', '4h']
            orchestrator.start(symbols, timeframes)
            
            # Verify started state
            self.assertTrue(orchestrator.status['running'])
            self.assertEqual(orchestrator.status['symbols_monitored'], symbols)
            self.assertEqual(orchestrator.status['timeframes_monitored'], timeframes)
            
            # Let the monitoring loop run for a moment
            time.sleep(0.5)
            
            # Stop orchestrator
            orchestrator.stop()
            
            # Verify stopped state
            self.assertFalse(orchestrator.status['running'])
    
    def test_signal_routing(self):
        """Test that signals are routed correctly."""
        # Create orchestrator with our mocks and custom TA agent
        with patch('ai_trading_agent.orchestration.ta_agent_integration.AdvancedTechnicalAnalysisAgent', 
                  return_value=self.mock_ta_agent):
            orchestrator = TechnicalAgentOrchestrator(event_bus=self.event_bus)
            
            # Create signal consumers
            def decision_consumer(signal):
                self.received_signals.append(('decision', signal))
            
            def visualization_consumer(signal):
                self.received_signals.append(('visualization', signal))
            
            # Register consumers
            orchestrator.register_consumer('decision', decision_consumer)
            orchestrator.register_consumer('visualization', visualization_consumer)
            
            # Create and route a test signal
            test_signal = {
                'symbol': 'BTC/USD',
                'timeframe': '1h',
                'timestamp': datetime.now(),
                'type': 'pattern',
                'direction': 'bullish',
                'confidence': 0.9,
                'metadata': {'pattern': 'three_white_soldiers'}
            }
            
            orchestrator._route_signal(test_signal)
            
            # Wait for signal processing
            time.sleep(0.5)
            
            # Verify signals were routed to both queues
            self.assertEqual(len(self.received_signals), 2)
            queue_names = [signal[0] for signal in self.received_signals]
            self.assertIn('decision', queue_names)
            self.assertIn('visualization', queue_names)
            
            # Verify signal content
            for _, signal in self.received_signals:
                self.assertEqual(signal.symbol, 'BTC/USD')
                self.assertEqual(signal.timeframe, '1h')
                self.assertEqual(signal.signal_type, 'pattern')
                self.assertEqual(signal.direction, 'bullish')
                self.assertEqual(signal.confidence, 0.9)
                self.assertEqual(signal.metadata['pattern'], 'three_white_soldiers')
    
    def test_data_source_toggle_handling(self):
        """Test handling of data source toggle events."""
        # Create orchestrator with our mocks and custom TA agent
        with patch('ai_trading_agent.orchestration.ta_agent_integration.AdvancedTechnicalAnalysisAgent', 
                  return_value=self.mock_ta_agent):
            orchestrator = TechnicalAgentOrchestrator(event_bus=self.event_bus)
            
            # Initial state is mock
            self.assertEqual(orchestrator.status['data_source'], 'mock')
            
            # Publish data source toggle event (to real data)
            self.event_bus.publish('data_source_toggled', {'is_mock': False})
            
            # Wait for event processing
            time.sleep(0.5)
            
            # Verify data source was updated
            self.assertEqual(orchestrator.status['data_source'], 'real')
            
            # Verify TA agent toggle method was called
            self.mock_ta_agent.toggle_data_source.assert_called_once()
    
    def test_monitoring_initialization(self):
        """Test that the monitoring system initializes correctly."""
        # Create monitor with our mock TA agent
        monitor = TAAgentMonitor(ta_agent=self.mock_ta_agent)
        
        # Verify initial state
        self.assertFalse(monitor.running)
        self.assertIsNotNone(monitor.metrics['operation_count'])
        self.assertIsNotNone(monitor.metrics['error_count'])
        self.assertIsNotNone(monitor.metrics['average_latency'])
        self.assertIsNotNone(monitor.metrics['signal_count'])
        self.assertIsNotNone(monitor.metrics['pattern_count'])
        self.assertIsNotNone(monitor.metrics['memory_usage'])
    
    def test_monitoring_health_checks(self):
        """Test that the monitoring system performs health checks correctly."""
        # Create monitor with our mock TA agent
        monitor = TAAgentMonitor(ta_agent=self.mock_ta_agent)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Wait for health checks to run
        time.sleep(0.5)
        
        # Verify agent health was checked
        self.mock_ta_agent.get_status.assert_called()
        
        # Agent should be healthy
        self.assertEqual(monitor.health_status['agent'].status, monitor.health_status['agent'].HEALTHY)
        
        # Stop monitoring
        monitor.stop_monitoring()
    
    def test_alert_handling(self):
        """Test that alerts are properly handled."""
        # Create monitor with our mock TA agent
        monitor = TAAgentMonitor(ta_agent=self.mock_ta_agent)
        
        # Create alert handler
        alert_received = []
        def test_alert_handler(alert):
            alert_received.append(alert)
        
        # Register alert handler
        monitor.register_alert_handler(test_alert_handler)
        
        # Manually trigger an alert by manipulating the health status
        monitor.health_status['agent'].update(monitor.health_status['agent'].UNHEALTHY, "Test alert")
        
        # Process alert
        test_alert = {
            'type': 'health',
            'severity': 'high',
            'component': 'agent',
            'message': 'Agent health check failed: Test alert',
            'timestamp': datetime.now().isoformat()
        }
        monitor._handle_alert(test_alert)
        
        # Verify alert was received by handler
        self.assertEqual(len(alert_received), 1)
        self.assertEqual(alert_received[0]['type'], 'health')
        self.assertEqual(alert_received[0]['severity'], 'high')
        self.assertEqual(alert_received[0]['component'], 'agent')
    
    def test_metrics_collection(self):
        """Test that metrics are properly collected."""
        # Create monitor with our mock TA agent
        monitor = TAAgentMonitor(ta_agent=self.mock_ta_agent)
        
        # Manually collect metrics
        monitor._collect_metrics()
        
        # Verify metrics were collected from the agent
        self.mock_ta_agent.get_statistics.assert_called_once()
        
        # Check that metrics were updated
        self.assertEqual(monitor.metrics['operation_count'].get_latest(), 10)
        self.assertEqual(monitor.metrics['error_count'].get_latest(), 0)
        self.assertEqual(monitor.metrics['average_latency'].get_latest(), 50)
        self.assertEqual(monitor.metrics['signal_count'].get_latest(), 5)
        self.assertEqual(monitor.metrics['pattern_count'].get_latest(), 3)
        self.assertIsNotNone(monitor.metrics['memory_usage'].get_latest())
    
    def test_status_reporting(self):
        """Test that status reporting works correctly."""
        # Create monitor with our mock TA agent
        monitor = TAAgentMonitor(ta_agent=self.mock_ta_agent)
        
        # Manually collect metrics
        monitor._collect_metrics()
        
        # Get monitoring status
        status = monitor.get_monitoring_status()
        
        # Verify status contains expected sections
        self.assertIn('timestamp', status)
        self.assertIn('metrics', status)
        self.assertIn('health', status)
        self.assertIn('alerts', status)
        self.assertIn('config', status)
        
        # Get health summary
        summary = monitor.get_health_summary()
        
        # Verify summary contains expected fields
        self.assertIn('timestamp', summary)
        self.assertIn('overall_status', summary)
        self.assertIn('component_status', summary)
        self.assertIn('alert_count', summary)
        self.assertIn('status_counts', summary)
        
        # All components should be healthy by default
        self.assertEqual(summary['overall_status'], monitor.health_status['agent'].HEALTHY)


if __name__ == "__main__":
    unittest.main()
