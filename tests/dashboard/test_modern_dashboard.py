"""
Tests for the Modern Dashboard Data Service and Theme System.

This module specifically tests the DataService class for caching and
data source management, as well as the theme switching functionality.
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
import json

from src.dashboard.modern_dashboard import (
    DataService, DataSource, MockDataGenerator, SystemState, 
    get_theme_settings, save_theme_settings, toggle_theme
)


@pytest.fixture
def mock_data_generator():
    """Create a mock data generator with controlled output."""
    mock_gen = mock.MagicMock(spec=MockDataGenerator)
    
    # Configure mock data returns
    mock_gen.generate_system_health.return_value = {
        "status": "healthy",
        "state": SystemState.RUNNING,
        "cpu_usage": 32.5,
        "memory_usage": 45.6,
        "disk_usage": 28.9,
        "uptime": "3 days, 12:45:18",
        "last_updated": datetime.now().isoformat()
    }
    
    mock_gen.generate_trading_performance.return_value = {
        "total_pnl": 12534.25,
        "daily_pnl": 523.87,
        "win_rate": 62.5,
        "sharpe_ratio": 1.82,
        "drawdown": 8.3,
        "trades_today": 28,
        "last_updated": datetime.now().isoformat()
    }
    
    mock_gen.generate_market_regime_data.return_value = {
        "current_regime": "bullish",
        "regime_probability": 0.85,
        "regime_duration": "14 days",
        "previous_regime": "neutral",
        "confidence_score": 0.92,
        "last_updated": datetime.now().isoformat()
    }
    
    return mock_gen


@pytest.fixture
def data_service(mock_data_generator):
    """Create a data service with mocked dependencies."""
    with mock.patch("src.dashboard.modern_dashboard.MockDataGenerator", 
                   return_value=mock_data_generator):
        service = DataService(data_source=DataSource.MOCK)
        yield service


@pytest.fixture
def mock_local_storage():
    """Create a mock localStorage implementation."""
    class MockLocalStorage:
        def __init__(self):
            self.items = {}
            
        def getItem(self, key):
            return self.items.get(key)
            
        def setItem(self, key, value):
            self.items[key] = value
            
        def removeItem(self, key):
            if key in self.items:
                del self.items[key]
    
    return MockLocalStorage()


class TestDataService:
    """Tests for the data service component."""
    
    def test_init_default_state(self):
        """Test the default state of data service."""
        service = DataService()
        assert service.data_source == DataSource.MOCK
        assert isinstance(service.mock_data, MockDataGenerator)
        assert service.cache == {}
        
        # Verify cache expiry configuration
        assert 'system_health' in service.cache_expiry
        assert service.cache_expiry['system_health'] == 5  # 5 seconds
        assert service.cache_expiry['trading_performance'] == 30  # 30 seconds
        assert service.cache_expiry['market_regime'] == 60  # 60 seconds
    
    def test_set_data_source(self, data_service):
        """Test switching data sources."""
        # Initial state
        assert data_service.data_source == DataSource.MOCK
        
        # Switch to REAL
        data_service.set_data_source(DataSource.REAL)
        assert data_service.data_source == DataSource.REAL
        assert data_service.cache == {}  # Cache should be cleared
        
        # Switch back to MOCK
        data_service.set_data_source(DataSource.MOCK)
        assert data_service.data_source == DataSource.MOCK
        assert data_service.cache == {}  # Cache should be cleared again
    
    def test_get_data_mock_source(self, data_service, mock_data_generator):
        """Test getting data from mock source."""
        # Get system health data
        health_data = data_service.get_data('system_health')
        
        # Verify mock was called
        mock_data_generator.generate_system_health.assert_called_once()
        
        # Verify data was cached
        assert 'system_health' in data_service.cache
        assert len(data_service.cache['system_health']) == 2  # (timestamp, data)
        assert data_service.cache['system_health'][1] == health_data
        assert health_data['status'] == 'healthy'
        assert health_data['state'] == SystemState.RUNNING
        
        # Get trading performance data
        perf_data = data_service.get_data('trading_performance')
        
        # Verify mock was called
        mock_data_generator.generate_trading_performance.assert_called_once()
        
        # Verify data was cached
        assert 'trading_performance' in data_service.cache
        assert data_service.cache['trading_performance'][1] == perf_data
        assert perf_data['total_pnl'] == 12534.25
        
        # Get market regime data
        regime_data = data_service.get_data('market_regime')
        
        # Verify mock was called
        mock_data_generator.generate_market_regime_data.assert_called_once()
        
        # Verify data was cached
        assert 'market_regime' in data_service.cache
        assert data_service.cache['market_regime'][1] == regime_data
        assert regime_data['current_regime'] == 'bullish'
    
    def test_get_data_caching(self, data_service, mock_data_generator):
        """Test data caching functionality."""
        # First call should hit the generator
        data_service.get_data('system_health')
        assert mock_data_generator.generate_system_health.call_count == 1
        
        # Second call should use cache
        data_service.get_data('system_health')
        assert mock_data_generator.generate_system_health.call_count == 1  # Still 1
        
        # Force refresh should bypass cache
        data_service.get_data('system_health', force_refresh=True)
        assert mock_data_generator.generate_system_health.call_count == 2
    
    def test_cache_expiry(self, data_service, mock_data_generator):
        """Test cache expiry functionality."""
        # Get data to populate cache
        data_service.get_data('system_health')
        
        # Manually expire the cache by setting timestamp to past
        old_time = datetime.now() - timedelta(seconds=10)
        data_service.cache['system_health'] = (old_time, data_service.cache['system_health'][1])
        
        # Next call should refresh since system_health expires after 5 seconds
        data_service.get_data('system_health')
        assert mock_data_generator.generate_system_health.call_count == 2

    def test_get_real_data_fallback(self, data_service):
        """Test fallback to mock data when real data fails."""
        # Configure data service to use REAL data
        data_service.set_data_source(DataSource.REAL)
        
        # Mock the _get_real_data method to raise an exception
        with mock.patch.object(data_service, '_get_real_data', side_effect=Exception("API Error")):
            # Mock the _get_mock_data method for verification
            with mock.patch.object(data_service, '_get_mock_data') as mock_get_mock:
                # Call get_data, which should try real data, fail, and fall back to mock
                data_service.get_data('system_health')
                
                # Verify fallback occurred
                mock_get_mock.assert_called_once_with('system_health')
    
    def test_real_data_source(self, data_service):
        """Test fetching data from real sources when available."""
        # Configure data service to use REAL data
        data_service.set_data_source(DataSource.REAL)
        
        # Mock SystemMonitor for real data
        with mock.patch('src.dashboard.modern_dashboard.SystemMonitor') as mock_monitor:
            mock_monitor.get_system_health.return_value = {
                "status": "healthy",
                "state": SystemState.RUNNING,
                "cpu_usage": 25.0,
                "memory_usage": 40.2,
                "uptime": "2 days, 5:45:12"
            }
            
            # Get real system health data
            health_data = data_service.get_data('system_health')
            
            # Verify SystemMonitor was called
            mock_monitor.get_system_health.assert_called_once()
            
            # Verify correct data was returned
            assert health_data['status'] == 'healthy'
            assert health_data['cpu_usage'] == 25.0


class TestThemeSystem:
    """Tests for the theme system."""
    
    def test_get_theme_settings_default(self, mock_local_storage):
        """Test getting default theme settings."""
        # Mock localStorage being empty
        with mock.patch('src.dashboard.modern_dashboard.localStorage', mock_local_storage):
            settings = get_theme_settings()
            
            # Default should be light theme
            assert settings['theme'] == 'light'
            assert 'fontSize' in settings
            assert 'chartColors' in settings
    
    def test_save_theme_settings(self, mock_local_storage):
        """Test saving theme settings."""
        # Create test settings
        test_settings = {
            'theme': 'dark',
            'fontSize': 'large',
            'chartColors': 'vibrant'
        }
        
        # Mock localStorage
        with mock.patch('src.dashboard.modern_dashboard.localStorage', mock_local_storage):
            # Save settings
            save_theme_settings(test_settings)
            
            # Verify localStorage was updated
            stored_json = mock_local_storage.getItem('dashboard_theme_settings')
            assert stored_json is not None
            
            # Parse stored JSON
            stored_settings = json.loads(stored_json)
            assert stored_settings['theme'] == 'dark'
            assert stored_settings['fontSize'] == 'large'
            assert stored_settings['chartColors'] == 'vibrant'
    
    def test_toggle_theme(self, mock_local_storage):
        """Test theme toggling functionality."""
        # Set initial theme to light
        initial_settings = {
            'theme': 'light',
            'fontSize': 'medium',
            'chartColors': 'standard'
        }
        mock_local_storage.setItem('dashboard_theme_settings', json.dumps(initial_settings))
        
        # Mock DOM elements
        mock_body = mock.MagicMock()
        mock_icons = mock.MagicMock()
        
        with mock.patch('src.dashboard.modern_dashboard.localStorage', mock_local_storage):
            with mock.patch('src.dashboard.modern_dashboard.document.body', mock_body):
                with mock.patch('src.dashboard.modern_dashboard.document.querySelectorAll', return_value=mock_icons):
                    # Toggle theme from light to dark
                    toggle_theme()
                    
                    # Verify localStorage was updated
                    stored_json = mock_local_storage.getItem('dashboard_theme_settings')
                    stored_settings = json.loads(stored_json)
                    assert stored_settings['theme'] == 'dark'
                    
                    # Verify DOM was updated
                    mock_body.classList.remove.assert_called_with('light-theme')
                    mock_body.classList.add.assert_called_with('dark-theme')
                    
                    # Toggle again to switch back to light
                    toggle_theme()
                    
                    # Verify localStorage was updated again
                    stored_json = mock_local_storage.getItem('dashboard_theme_settings')
                    stored_settings = json.loads(stored_json)
                    assert stored_settings['theme'] == 'light'