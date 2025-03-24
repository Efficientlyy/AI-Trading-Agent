"""
Tests for the Modern Dashboard implementation.

This module tests the dashboard components, data services, 
WebSocket functionality, and user authentication.
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta

from src.dashboard.modern_dashboard import (
    DataService, DataSource, MockDataGenerator, 
    login_required, role_required, UserRole
)


@pytest.fixture
def mock_data_generator():
    """Create a mock data generator with controlled output."""
    mock_gen = mock.MagicMock(spec=MockDataGenerator)
    
    # Configure mock data returns
    mock_gen.generate_system_health.return_value = {
        "status": "healthy",
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
    
    return mock_gen


@pytest.fixture
def data_service(mock_data_generator):
    """Create a data service with mocked dependencies."""
    with mock.patch("src.dashboard.modern_dashboard.MockDataGenerator", 
                   return_value=mock_data_generator):
        service = DataService(data_source=DataSource.MOCK)
        yield service


class TestDataService:
    """Tests for the data service component."""
    
    def test_init_default_state(self):
        """Test the default state of data service."""
        service = DataService()
        assert service.data_source == DataSource.MOCK
        assert isinstance(service.mock_data, MockDataGenerator)
        assert service.cache == {}
    
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
        
        # Get trading performance data
        perf_data = data_service.get_data('trading_performance')
        
        # Verify mock was called
        mock_data_generator.generate_trading_performance.assert_called_once()
        
        # Verify data was cached
        assert 'trading_performance' in data_service.cache
        assert data_service.cache['trading_performance'][1] == perf_data
    
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


class TestAuthDecorators:
    """Tests for authentication decorators."""
    
    def test_login_required(self):
        """Test login_required decorator."""
        # Create a mock function to decorate
        mock_view_func = mock.MagicMock()
        mock_view_func.__name__ = 'mock_view_func'
        
        # Apply decorator
        decorated = login_required(mock_view_func)
        
        # Test with no session
        with mock.patch('src.dashboard.modern_dashboard.session', {}):
            with mock.patch('src.dashboard.modern_dashboard.redirect') as mock_redirect:
                with mock.patch('src.dashboard.modern_dashboard.url_for') as mock_url_for:
                    mock_url_for.return_value = '/login'
                    decorated()
                    mock_redirect.assert_called_once()
                    mock_view_func.assert_not_called()
        
        # Test with session containing user_id
        with mock.patch('src.dashboard.modern_dashboard.session', {'user_id': '123'}):
            decorated()
            mock_view_func.assert_called_once()
    
    def test_role_required(self):
        """Test role_required decorator."""
        # Create a mock function to decorate
        mock_view_func = mock.MagicMock()
        mock_view_func.__name__ = 'mock_view_func'
        
        # Apply decorator
        decorated = role_required([UserRole.ADMIN])(mock_view_func)
        
        # Test with missing role
        with mock.patch('src.dashboard.modern_dashboard.session', {'user_id': '123'}):
            with mock.patch('src.dashboard.modern_dashboard.flash') as mock_flash:
                with mock.patch('src.dashboard.modern_dashboard.redirect') as mock_redirect:
                    with mock.patch('src.dashboard.modern_dashboard.url_for') as mock_url_for:
                        mock_url_for.return_value = '/dashboard'
                        decorated()
                        mock_flash.assert_called_once()
                        mock_redirect.assert_called_once()
                        mock_view_func.assert_not_called()
        
        # Test with insufficient role
        with mock.patch('src.dashboard.modern_dashboard.session', 
                       {'user_id': '123', 'user_role': UserRole.VIEWER}):
            with mock.patch('src.dashboard.modern_dashboard.flash') as mock_flash:
                with mock.patch('src.dashboard.modern_dashboard.redirect') as mock_redirect:
                    decorated()
                    mock_flash.assert_called_once()
                    mock_redirect.assert_called_once()
                    mock_view_func.assert_not_called()
        
        # Test with sufficient role
        with mock.patch('src.dashboard.modern_dashboard.session', 
                       {'user_id': '123', 'user_role': UserRole.ADMIN}):
            decorated()
            mock_view_func.assert_called_once()