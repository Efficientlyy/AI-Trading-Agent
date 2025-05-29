"""
End-to-End tests for Technical Analysis Agent.

These tests verify the complete Technical Analysis Agent functionality
with all components integrated and working together.
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock
import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
from fastapi.testclient import TestClient

# Adjust Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ai_trading_agent.api.main import app
from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.orchestration.ta_agent_integration import TechnicalAgentOrchestrator
from ai_trading_agent.monitoring.ta_agent_monitor import TAAgentMonitor
from ai_trading_agent.config.data_source_config import DataSourceConfig, get_data_source_config
from ai_trading_agent.common.event_bus import EventBus

# Create test client
client = TestClient(app)

class TestTechnicalAnalysisAgentE2E(unittest.TestCase):
    """End-to-End tests for Technical Analysis Agent functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock data source config
        self.data_source_config = DataSourceConfig()
        self.data_source_config.use_mock_data = True
        
        # Create patches
        self.config_patcher = patch('ai_trading_agent.api.technical_analysis_api.get_data_source_config', 
                                   return_value=self.data_source_config)
        self.mock_config = self.config_patcher.start()
        
        # Create event bus
        self.event_bus = EventBus()
        
        # Create TA agent
        self.ta_agent = AdvancedTechnicalAnalysisAgent()
        
        # Create orchestrator
        self.orchestrator = TechnicalAgentOrchestrator(event_bus=self.event_bus)
        
        # Create monitor
        self.monitor = TAAgentMonitor(ta_agent=self.ta_agent)
        
        # Set up symbols and timeframes for testing
        self.test_symbols = ['BTC/USD', 'ETH/USD']
        self.test_timeframes = ['1h', '4h']
        
    def tearDown(self):
        """Clean up after tests."""
        self.config_patcher.stop()
        
    def test_complete_ta_workflow(self):
        """Test the complete Technical Analysis workflow from data to signals."""
        # 1. Start the orchestrator
        self.orchestrator.start(self.test_symbols, self.test_timeframes)
        
        # 2. Check that the orchestrator is running
        status = self.orchestrator.get_status()
        self.assertTrue(status['running'])
        self.assertEqual(status['symbols_monitored'], self.test_symbols)
        self.assertEqual(status['timeframes_monitored'], self.test_timeframes)
        
        # 3. Simulate running for a short period
        time.sleep(1)
        
        # 4. Toggle data source (simulating UI toggle)
        self.event_bus.publish('data_source_toggled', {'is_mock': False})
        
        # 5. Verify data source was toggled in orchestrator
        time.sleep(0.5)  # Wait for event processing
        status = self.orchestrator.get_status()
        self.assertEqual(status['data_source'], 'real')
        
        # 6. Toggle back to mock
        self.event_bus.publish('data_source_toggled', {'is_mock': True})
        
        # 7. Verify data source was toggled back
        time.sleep(0.5)  # Wait for event processing
        status = self.orchestrator.get_status()
        self.assertEqual(status['data_source'], 'mock')
        
        # 8. Stop the orchestrator
        self.orchestrator.stop()
        
        # 9. Verify it stopped
        status = self.orchestrator.get_status()
        self.assertFalse(status['running'])
    
    def test_api_integration(self):
        """Test the Technical Analysis API endpoints with mock data."""
        # 1. Set to mock data
        self.data_source_config.use_mock_data = True
        
        # 2. Test indicators endpoint
        response = client.get("/api/technical-analysis/indicators?symbol=BTC/USD&timeframe=1h")
        self.assertEqual(response.status_code, 200)
        indicators = response.json()
        self.assertIsInstance(indicators, list)
        
        # 3. Test patterns endpoint
        response = client.get("/api/technical-analysis/patterns?symbol=BTC/USD&timeframe=1h")
        self.assertEqual(response.status_code, 200)
        patterns = response.json()
        self.assertIsInstance(patterns, list)
        
        # 4. Test advanced patterns endpoint
        response = client.get("/api/technical-analysis/advanced-patterns?symbol=BTC/USD&timeframe=1h")
        self.assertEqual(response.status_code, 200)
        advanced_patterns = response.json()
        self.assertIsInstance(advanced_patterns, list)
        
        # 5. Test full analysis endpoint
        response = client.get("/api/technical-analysis/analysis?symbol=BTC/USD&timeframe=1h")
        self.assertEqual(response.status_code, 200)
        analysis = response.json()
        self.assertEqual(analysis['data_source'], 'mock')
        self.assertIn('indicators', analysis)
        self.assertIn('patterns', analysis)
        
        # 6. Test performance metrics endpoint
        response = client.get("/api/technical-analysis/metrics")
        self.assertEqual(response.status_code, 200)
        metrics = response.json()
        self.assertIn('api_calls', metrics)
        
        # 7. Test health status endpoint
        response = client.get("/api/technical-analysis/monitoring/health")
        self.assertEqual(response.status_code, 200)
        health = response.json()
        self.assertIn('status', health)
        
        # 8. Test orchestrator status endpoint
        response = client.get("/api/technical-analysis/orchestrator/status")
        self.assertEqual(response.status_code, 200)
        status = response.json()
        self.assertIn('status', status)
    
    def test_monitoring_integration(self):
        """Test the monitoring system integration."""
        # 1. Start monitoring
        self.monitor.start_monitoring()
        
        # 2. Wait for initial health checks
        time.sleep(0.5)
        
        # 3. Get monitoring status
        status = self.monitor.get_monitoring_status()
        
        # 4. Check key components
        self.assertIn('metrics', status)
        self.assertIn('health', status)
        
        # 5. Check health summary
        summary = self.monitor.get_health_summary()
        self.assertIn('overall_status', summary)
        self.assertIn('component_status', summary)
        
        # 6. Stop monitoring
        self.monitor.stop_monitoring()
    
    def test_event_bus_integration(self):
        """Test the event bus integration."""
        # Create test event handler
        received_events = []
        def test_handler(event):
            received_events.append(event)
        
        # Subscribe to test events
        self.event_bus.subscribe('test_event', test_handler)
        
        # Publish a test event
        self.event_bus.publish('test_event', {'test_data': 'value'})
        
        # Wait for event processing
        time.sleep(0.5)
        
        # Check event was received
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].event_type, 'test_event')
        self.assertEqual(received_events[0].data['test_data'], 'value')
    
    def test_data_source_toggle_with_pattern_detection(self):
        """Test that pattern detection works with different data sources."""
        # Create test data
        test_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            'high': [105, 106, 107, 108, 109, 110, 109, 108, 107, 106],
            'low': [95, 96, 97, 98, 99, 100, 99, 98, 97, 96],
            'close': [101, 102, 103, 104, 105, 104, 103, 102, 101, 100]
        })
        
        # Test with mock data
        self.data_source_config.use_mock_data = True
        
        # Mock data provider to return our test data
        with patch.object(self.ta_agent, 'get_market_data', return_value={'BTC/USD': {'1h': test_data}}):
            # Detect patterns
            patterns_mock = self.ta_agent.detect_patterns(test_data)
            
            # Switch to real data
            self.data_source_config.use_mock_data = False
            
            # Detect patterns again
            patterns_real = self.ta_agent.detect_patterns(test_data)
            
            # Patterns should be detected regardless of data source
            self.assertIsNotNone(patterns_mock)
            self.assertIsNotNone(patterns_real)
            
            # The pattern detection logic should work the same
            self.assertEqual(len(patterns_mock), len(patterns_real))
            
    def test_signal_routing_with_different_data_sources(self):
        """Test that signals are properly routed with different data sources."""
        # Set up signal consumers
        mock_decision_consumer = MagicMock()
        mock_visualization_consumer = MagicMock()
        
        # Register consumers
        self.orchestrator.register_consumer('decision', mock_decision_consumer)
        self.orchestrator.register_consumer('visualization', mock_visualization_consumer)
        
        # Create test signals for mock and real data
        mock_signal = {
            'symbol': 'BTC/USD',
            'timeframe': '1h',
            'timestamp': datetime.now(),
            'type': 'pattern',
            'direction': 'bullish',
            'confidence': 0.9,
            'metadata': {'pattern': 'three_white_soldiers', 'data_source': 'mock'}
        }
        
        real_signal = {
            'symbol': 'BTC/USD',
            'timeframe': '1h',
            'timestamp': datetime.now(),
            'type': 'pattern',
            'direction': 'bearish',
            'confidence': 0.8,
            'metadata': {'pattern': 'three_black_crows', 'data_source': 'real'}
        }
        
        # Test with mock data
        self.data_source_config.use_mock_data = True
        self.orchestrator._route_signal(mock_signal)
        
        # Wait for signal processing
        time.sleep(0.5)
        
        # Verify mock signal was routed
        mock_decision_consumer.assert_called_once()
        mock_visualization_consumer.assert_called_once()
        
        # Reset mocks
        mock_decision_consumer.reset_mock()
        mock_visualization_consumer.reset_mock()
        
        # Test with real data
        self.data_source_config.use_mock_data = False
        self.orchestrator._route_signal(real_signal)
        
        # Wait for signal processing
        time.sleep(0.5)
        
        # Verify real signal was routed
        mock_decision_consumer.assert_called_once()
        mock_visualization_consumer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
