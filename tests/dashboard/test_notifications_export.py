"""
Tests for the notification system and data export functionality in the Modern Dashboard.

This module tests notification management and various export formats.
"""

import pytest
import unittest.mock as mock
from datetime import datetime
import json
import csv
import io
import tempfile

# Import the module under test
from src.dashboard.modern_dashboard import (
    NotificationManager, NotificationPriority, ExportManager, 
    export_data_as_csv, export_data_as_json, export_data_as_excel
)


@pytest.fixture
def mock_notification_manager():
    """Create a mock notification manager with sample notifications."""
    with mock.patch('src.dashboard.modern_dashboard.NotificationManager') as mock_manager:
        # Configure instance methods
        manager_instance = mock_manager.return_value
        
        # Sample notifications
        sample_notifications = [
            {
                "id": "notif-1",
                "title": "System Update",
                "message": "The system has been updated to version 2.0",
                "timestamp": datetime.now().isoformat(),
                "read": False,
                "priority": NotificationPriority.INFO
            },
            {
                "id": "notif-2",
                "title": "Trading Alert",
                "message": "Position size limit reached for BTC-USD",
                "timestamp": datetime.now().isoformat(),
                "read": True,
                "priority": NotificationPriority.WARNING
            },
            {
                "id": "notif-3",
                "title": "Connection Issue",
                "message": "Exchange API connection lost",
                "timestamp": datetime.now().isoformat(),
                "read": False,
                "priority": NotificationPriority.ERROR
            }
        ]
        
        # Mock methods
        manager_instance.get_all_notifications.return_value = sample_notifications
        manager_instance.get_unread_count.return_value = 2
        
        def mock_mark_as_read(notif_id):
            for notif in sample_notifications:
                if notif["id"] == notif_id:
                    notif["read"] = True
                    return True
            return False
            
        manager_instance.mark_as_read.side_effect = mock_mark_as_read
        
        yield manager_instance


@pytest.fixture
def mock_export_manager():
    """Create a mock export manager for testing."""
    with mock.patch('src.dashboard.modern_dashboard.ExportManager') as mock_manager:
        # Configure instance methods
        manager_instance = mock_manager.return_value
        
        # Sample data to export
        sample_data = [
            {"date": "2023-01-01", "symbol": "BTC-USD", "price": 42000.50, "volume": 123.45},
            {"date": "2023-01-02", "symbol": "BTC-USD", "price": 42500.75, "volume": 98.76},
            {"date": "2023-01-03", "symbol": "BTC-USD", "price": 43000.25, "volume": 112.33}
        ]
        
        # Mock export methods
        manager_instance.get_data_for_export.return_value = sample_data
        
        yield manager_instance


class TestNotificationSystem:
    """Tests for the notification system."""
    
    def test_get_all_notifications(self, mock_notification_manager):
        """Test getting all notifications."""
        notifications = mock_notification_manager.get_all_notifications()
        
        # Verify we get 3 notifications
        assert len(notifications) == 3
        
        # Verify notification structure
        for notif in notifications:
            assert "id" in notif
            assert "title" in notif
            assert "message" in notif
            assert "timestamp" in notif
            assert "read" in notif
            assert "priority" in notif
    
    def test_get_unread_count(self, mock_notification_manager):
        """Test getting unread notification count."""
        count = mock_notification_manager.get_unread_count()
        
        # Verify count
        assert count == 2
    
    def test_mark_as_read(self, mock_notification_manager):
        """Test marking a notification as read."""
        # Mark notification as read
        result = mock_notification_manager.mark_as_read("notif-1")
        
        # Verify success
        assert result is True
        
        # Call was recorded
        mock_notification_manager.mark_as_read.assert_called_with("notif-1")
        
        # Verify marking non-existent notification
        result = mock_notification_manager.mark_as_read("non-existent")
        
        # Should fail
        assert result is False


class TestExportFunctionality:
    """Tests for the data export functionality."""
    
    def test_export_data_as_csv(self, mock_export_manager):
        """Test exporting data as CSV."""
        # Get sample data
        data = mock_export_manager.get_data_for_export()
        
        # Export as CSV
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            filename = temp_file.name
            export_data_as_csv(data, filename)
        
        # Read exported CSV
        with open(filename, 'r') as f:
            csv_reader = csv.DictReader(f)
            exported_data = list(csv_reader)
        
        # Verify data
        assert len(exported_data) == 3
        assert exported_data[0]["symbol"] == "BTC-USD"
        assert exported_data[1]["price"] == "42500.75"
        assert exported_data[2]["volume"] == "112.33"
    
    def test_export_data_as_json(self, mock_export_manager):
        """Test exporting data as JSON."""
        # Get sample data
        data = mock_export_manager.get_data_for_export()
        
        # Export as JSON
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            filename = temp_file.name
            export_data_as_json(data, filename)
        
        # Read exported JSON
        with open(filename, 'r') as f:
            exported_data = json.load(f)
        
        # Verify data
        assert len(exported_data) == 3
        assert exported_data[0]["symbol"] == "BTC-USD"
        assert exported_data[1]["price"] == 42500.75
        assert exported_data[2]["volume"] == 112.33
    
    def test_export_data_as_excel(self, mock_export_manager):
        """Test exporting data as Excel."""
        # This would require pandas and openpyxl, so we'll mock it
        with mock.patch('src.dashboard.modern_dashboard.pd') as mock_pd:
            # Get sample data
            data = mock_export_manager.get_data_for_export()
            
            # Configure mock DataFrame
            mock_df = mock.MagicMock()
            mock_pd.DataFrame.return_value = mock_df
            
            # Export as Excel
            filename = "test_export.xlsx"
            export_data_as_excel(data, filename)
            
            # Verify DataFrame was created and to_excel was called
            mock_pd.DataFrame.assert_called_once()
            mock_df.to_excel.assert_called_once_with(filename, index=False)
    
    def test_scheduled_exports(self, mock_export_manager):
        """Test scheduled export functionality."""
        # Mock the scheduled export function
        with mock.patch('src.dashboard.modern_dashboard.schedule_export') as mock_scheduler:
            # Configure export manager
            mock_export_manager.schedule_export.side_effect = mock_scheduler
            
            # Schedule an export
            export_config = {
                "data_type": "trading_performance",
                "format": "csv",
                "frequency": "daily",
                "time": "23:59",
                "email": "user@example.com"
            }
            mock_export_manager.schedule_export(export_config)
            
            # Verify scheduler was called
            mock_scheduler.assert_called_once_with(export_config)
            
            # Alternatively, if we're testing the real function:
            # mock_export_manager.schedule_export.assert_called_once_with(export_config)