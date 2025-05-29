"""
Integration tests for Technical Analysis with Mock/Real data toggle functionality.

These tests verify that the technical analysis components properly interact
with the data source toggle system.
"""

import unittest
import json
import os
import sys
from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock

# Adjust Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ai_trading_agent.api.main import app
from ai_trading_agent.config.data_source_config import DataSourceConfig, get_data_source_config
from ai_trading_agent.data.data_source_factory import DataSourceFactory

# Create test client
client = TestClient(app)

class TestTechnicalAnalysisIntegration(unittest.TestCase):
    """Test the integration of Technical Analysis with Mock/Real data toggle."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset data source config to mock data for tests
        self.original_config = get_data_source_config()
        self.data_source_config = DataSourceConfig()
        self.data_source_config.use_mock_data = True
        
        # Create patches for data source configuration
        self.config_patcher = patch('ai_trading_agent.api.technical_analysis_api.get_data_source_config', 
                                    return_value=self.data_source_config)
        self.mock_config = self.config_patcher.start()
        
        # Create mock data source factory
        self.mock_data_provider = MagicMock()
        self.mock_data_provider.generate_data.return_value = self._generate_mock_market_data()
        self.mock_data_provider.get_historical_data.return_value = self._generate_mock_market_data()
        
        self.mock_factory = MagicMock()
        self.mock_factory.get_data_provider.return_value = self.mock_data_provider
        
        self.factory_patcher = patch('ai_trading_agent.api.technical_analysis_api.get_data_source_factory',
                                    return_value=self.mock_factory)
        self.mock_factory_func = self.factory_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.config_patcher.stop()
        self.factory_patcher.stop()
        
    def _generate_mock_market_data(self):
        """Generate mock market data for testing."""
        import pandas as pd
        import numpy as np
        
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
    
    def test_technical_analysis_api_mock_data(self):
        """Test that technical analysis API properly uses mock data."""
        # Set config to use mock data
        self.data_source_config.use_mock_data = True
        
        # Test indicators endpoint
        response = client.get("/api/technical-analysis/indicators?symbol=BTC/USD&timeframe=1h")
        assert response.status_code == 200
        indicators = response.json()
        
        # Verify mock data provider was called
        self.mock_data_provider.generate_data.assert_called_once()
        self.mock_data_provider.get_historical_data.assert_not_called()
        self.mock_data_provider.reset_mock()
        
        # Test patterns endpoint
        response = client.get("/api/technical-analysis/patterns?symbol=BTC/USD&timeframe=1h")
        assert response.status_code == 200
        patterns = response.json()
        
        # Verify mock data provider was called
        self.mock_data_provider.generate_data.assert_called_once()
        self.mock_data_provider.get_historical_data.assert_not_called()
        self.mock_data_provider.reset_mock()
        
        # Test full analysis endpoint
        response = client.get("/api/technical-analysis/analysis?symbol=BTC/USD&timeframe=1h")
        assert response.status_code == 200
        analysis = response.json()
        
        # Check that data source is correctly reported
        assert analysis["data_source"] == "mock"
        
    def test_technical_analysis_api_real_data(self):
        """Test that technical analysis API properly uses real data."""
        # Set config to use real data
        self.data_source_config.use_mock_data = False
        
        # Test indicators endpoint
        response = client.get("/api/technical-analysis/indicators?symbol=BTC/USD&timeframe=1h")
        assert response.status_code == 200
        indicators = response.json()
        
        # Verify real data provider was called
        self.mock_data_provider.get_historical_data.assert_called_once()
        self.mock_data_provider.generate_data.assert_not_called()
        self.mock_data_provider.reset_mock()
        
        # Test patterns endpoint
        response = client.get("/api/technical-analysis/patterns?symbol=BTC/USD&timeframe=1h")
        assert response.status_code == 200
        patterns = response.json()
        
        # Verify real data provider was called
        self.mock_data_provider.get_historical_data.assert_called_once()
        self.mock_data_provider.generate_data.assert_not_called()
        self.mock_data_provider.reset_mock()
        
        # Test full analysis endpoint
        response = client.get("/api/technical-analysis/analysis?symbol=BTC/USD&timeframe=1h")
        assert response.status_code == 200
        analysis = response.json()
        
        # Check that data source is correctly reported
        assert analysis["data_source"] == "real"
        
    def test_data_source_toggle_affects_technical_analysis(self):
        """Test that toggling data source affects technical analysis results."""
        # Start with mock data
        self.data_source_config.use_mock_data = True
        
        # Get analysis with mock data
        response = client.get("/api/technical-analysis/analysis?symbol=BTC/USD&timeframe=1h")
        mock_analysis = response.json()
        assert mock_analysis["data_source"] == "mock"
        
        # Toggle to real data
        self.data_source_config.use_mock_data = False
        
        # Get analysis with real data
        response = client.get("/api/technical-analysis/analysis?symbol=BTC/USD&timeframe=1h")
        real_analysis = response.json()
        assert real_analysis["data_source"] == "real"
        
        # Verify both were called correctly
        assert self.mock_data_provider.generate_data.call_count == 1
        assert self.mock_data_provider.get_historical_data.call_count == 1


if __name__ == "__main__":
    unittest.main()
