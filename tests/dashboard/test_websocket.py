"""
Tests for the WebSocket functionality in the Modern Dashboard.

This module tests the WebSocket event handlers and data emissions.
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
import json

# Import the module under test
from src.dashboard.modern_dashboard import (
    DataService, DataSource, MockDataGenerator, 
    setup_socketio_events, send_data_update
)


@pytest.fixture
def mock_socketio():
    """Mock for the SocketIO instance."""
    socketio_mock = mock.MagicMock()
    
    # Mock emit method
    socketio_mock.emit = mock.MagicMock()
    
    return socketio_mock


@pytest.fixture
def mock_data_service():
    """Mock for the DataService."""
    data_service = mock.MagicMock(spec=DataService)
    
    # Configure mock data returns
    data_service.get_data.side_effect = lambda data_type, force_refresh=False: {
        'system_health': {
            "status": "healthy",
            "cpu_usage": 32.5,
            "memory_usage": 45.6,
            "disk_usage": 28.9,
            "uptime": "3 days, 12:45:18",
            "last_updated": datetime.now().isoformat()
        },
        'trading_performance': {
            "total_pnl": 12534.25,
            "daily_pnl": 523.87,
            "win_rate": 62.5,
            "sharpe_ratio": 1.82,
            "drawdown": 8.3,
            "trades_today": 28,
            "last_updated": datetime.now().isoformat()
        },
        'current_positions': [
            {"symbol": "BTC-USD", "size": 0.5, "entry_price": 37500.25, "current_price": 38200.75},
            {"symbol": "ETH-USD", "size": 5.2, "entry_price": 2100.50, "current_price": 2205.25}
        ],
        'system_alerts': [
            {"level": "info", "message": "Trading session started", "timestamp": datetime.now().isoformat()},
            {"level": "warning", "message": "API rate limit at 80%", "timestamp": datetime.now().isoformat()}
        ]
    }[data_type]
    
    return data_service


class TestWebSocketFunctionality:
    """Tests for the WebSocket functionality."""
    
    def test_send_data_update(self, mock_socketio, mock_data_service):
        """Test the send_data_update function."""
        # Call send_data_update for system health
        with mock.patch('src.dashboard.modern_dashboard.data_service', mock_data_service):
            with mock.patch('src.dashboard.modern_dashboard.socketio', mock_socketio):
                send_data_update('system_health')
                
                # Verify socketio.emit was called correctly
                mock_socketio.emit.assert_called_once()
                args, kwargs = mock_socketio.emit.call_args
                
                # Verify the event name
                assert args[0] == 'update_system_health'
                
                # Verify data was passed
                assert 'status' in args[1]
                assert 'cpu_usage' in args[1]
                assert 'memory_usage' in args[1]
                
    def test_send_data_update_multiple_types(self, mock_socketio, mock_data_service):
        """Test send_data_update with multiple data types."""
        with mock.patch('src.dashboard.modern_dashboard.data_service', mock_data_service):
            with mock.patch('src.dashboard.modern_dashboard.socketio', mock_socketio):
                # Reset call count
                mock_socketio.emit.reset_mock()
                
                # Call for trading performance
                send_data_update('trading_performance')
                
                # Verify first emit
                mock_socketio.emit.assert_called_once()
                args, kwargs = mock_socketio.emit.call_args
                assert args[0] == 'update_trading_performance'
                assert 'total_pnl' in args[1]
                
                # Reset and call for current positions
                mock_socketio.emit.reset_mock()
                send_data_update('current_positions')
                
                # Verify second emit
                mock_socketio.emit.assert_called_once()
                args, kwargs = mock_socketio.emit.call_args
                assert args[0] == 'update_current_positions'
                assert isinstance(args[1], list)
                assert len(args[1]) == 2
                
    def test_setup_socketio_events(self, mock_socketio, mock_data_service):
        """Test the setup_socketio_events function."""
        # Store the event handlers registered
        event_handlers = {}
        
        def mock_on(event, **kwargs):
            # Capture the handler function
            def decorator(f):
                event_handlers[event] = f
                return f
            return decorator
        
        # Replace socketio.on with our mock
        mock_socketio.on = mock_on
        
        # Call setup_socketio_events
        with mock.patch('src.dashboard.modern_dashboard.socketio', mock_socketio):
            with mock.patch('src.dashboard.modern_dashboard.data_service', mock_data_service):
                # We need to patch the actual function to capture its reference
                with mock.patch('src.dashboard.modern_dashboard.setup_socketio_events') as mock_setup:
                    # Extract the original setup function from the module's namespace
                    original_setup = setup_socketio_events
                    # Call it to register handlers
                    original_setup(mock_socketio)
        
        # Verify at least some of the expected event handlers were registered
        assert 'connect' in event_handlers
        assert 'disconnect' in event_handlers
        assert 'request_data' in event_handlers
        
        # Test the request_data handler if it exists
        if 'request_data' in event_handlers:
            handler = event_handlers['request_data']
            
            # Mock the necessary context
            with mock.patch('src.dashboard.modern_dashboard.data_service', mock_data_service):
                with mock.patch('src.dashboard.modern_dashboard.socketio', mock_socketio):
                    # Call the handler
                    mock_socketio.emit.reset_mock()
                    handler({'data_type': 'system_health'})
                    
                    # Verify emit was called
                    mock_socketio.emit.assert_called_once()
                    args, kwargs = mock_socketio.emit.call_args
                    assert args[0] == 'update_system_health'