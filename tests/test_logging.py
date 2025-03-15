"""Tests for the logging system.

This module tests the functionality of the logging system, particularly
the proper handling of datetime objects in structured logs.
"""

import json
import logging
import unittest
from datetime import datetime, timezone
from io import StringIO
from unittest.mock import patch

import structlog
from structlog.stdlib import BoundLogger

from src.common.datetime_utils import utc_now
from src.common.logging import (
    configure_logging,
    get_logger, 
    _process_datetime_objects, 
    _process_value,
    _add_timestamp,
    _mask_sensitive_data,
    _process_sensitive_value,
    LogOperation
)


class TestLogging(unittest.TestCase):
    """Test the logging system functionality."""

    def test_process_datetime_objects(self):
        """Test that datetime objects are properly converted to ISO-8601 strings."""
        # Create a datetime object
        dt = datetime(2025, 3, 1, 12, 30, 45, tzinfo=timezone.utc)
        
        # Test simple dictionary
        event_dict = {"event": "test", "timestamp": dt}
        processed = _process_datetime_objects(None, None, event_dict)
        self.assertEqual(processed["timestamp"], "2025-03-01T12:30:45+00:00")
        
        # Test nested dictionary
        event_dict = {"event": "test", "data": {"timestamp": dt}}
        processed = _process_datetime_objects(None, None, event_dict)
        self.assertEqual(processed["data"]["timestamp"], "2025-03-01T12:30:45+00:00")
        
        # Test list
        event_dict = {"event": "test", "timestamps": [dt, dt]}
        processed = _process_datetime_objects(None, None, event_dict)
        self.assertEqual(processed["timestamps"][0], "2025-03-01T12:30:45+00:00")
        self.assertEqual(processed["timestamps"][1], "2025-03-01T12:30:45+00:00")
    
    def test_process_value(self):
        """Test the recursive value processing function."""
        # Create a datetime object
        dt = datetime(2025, 3, 1, 12, 30, 45, tzinfo=timezone.utc)
        
        # Test simple datetime
        self.assertEqual(_process_value(dt), "2025-03-01T12:30:45+00:00")
        
        # Test dictionary
        self.assertEqual(
            _process_value({"dt": dt}),
            {"dt": "2025-03-01T12:30:45+00:00"}
        )
        
        # Test list
        self.assertEqual(
            _process_value([dt, dt]),
            ["2025-03-01T12:30:45+00:00", "2025-03-01T12:30:45+00:00"]
        )
        
        # Test complex nested structure
        complex_structure = {
            "dt": dt,
            "nested": {
                "dt": dt,
                "list": [dt, {"more": dt}]
            }
        }
        expected = {
            "dt": "2025-03-01T12:30:45+00:00",
            "nested": {
                "dt": "2025-03-01T12:30:45+00:00",
                "list": [
                    "2025-03-01T12:30:45+00:00",
                    {"more": "2025-03-01T12:30:45+00:00"}
                ]
            }
        }
        self.assertEqual(_process_value(complex_structure), expected)
    
    def test_logger_with_datetime(self):
        """Test that the logger properly processes datetime objects."""
        # Create a datetime object
        dt = datetime(2025, 3, 1, 12, 30, 45, tzinfo=timezone.utc)
        
        # Create an event dictionary with a datetime
        event_dict = {"event": "test", "dt": dt, "nested": {"dt": dt}}
        
        # Process the event dictionary
        processed = _process_datetime_objects(None, None, event_dict)
        
        # Verify datetime was converted to ISO-8601 string
        self.assertEqual(processed["dt"], "2025-03-01T12:30:45+00:00")
        self.assertEqual(processed["nested"]["dt"], "2025-03-01T12:30:45+00:00")
    
    def test_add_timestamp(self):
        """Test the add_timestamp processor."""
        event_dict = {"event": "test"}
        processed = _add_timestamp(None, None, event_dict)
        
        # Verify a timestamp was added
        self.assertIn("timestamp", processed)
        
        # Verify the timestamp is an ISO-8601 string
        timestamp = processed["timestamp"]
        self.assertIsInstance(timestamp, str)
        
        # Very basic ISO-8601 format check
        self.assertRegex(timestamp, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    
    def test_get_logger(self):
        """Test the get_logger function."""
        # Get a logger for a component
        logger = get_logger("test_component")
        
        # Check that it's a BoundLogger
        self.assertIsInstance(logger, BoundLogger)
        
        # Check that the component was bound correctly
        self.assertEqual(logger._context.get("component"), "test_component")
        
        # Get a logger with a subcomponent
        logger = get_logger("test_component", "test_subcomponent")
        
        # Check that the combined component was bound correctly
        self.assertEqual(logger._context.get("component"), "test_component.test_subcomponent")
    
    def test_mask_sensitive_data(self):
        """Test the sensitive data masking functionality."""
        # Simple test
        event_dict = {"password": "secret123", "api_key": "abc123xyz", "normal_field": "visible"}
        masked_fields = {"password", "api_key"}
        
        processed = _mask_sensitive_data(event_dict, masked_fields)
        
        # Check that sensitive fields are masked
        self.assertEqual(processed["password"], "*********")  # Length preserved
        self.assertEqual(processed["api_key"], "*********")  # Length preserved
        self.assertEqual(processed["normal_field"], "visible")  # Not masked
        
        # Test with nested structure
        nested_dict = {
            "user": {
                "name": "John",
                "password": "verysecret",
                "credentials": {
                    "api_key": "abcdef123456"
                }
            },
            "data": [
                {"token": "secret_token"},
                {"public": "public_value"}
            ]
        }
        
        masked_fields = {"password", "api_key", "token"}
        processed = _mask_sensitive_data(nested_dict, masked_fields)
        
        # Check nested masking
        self.assertEqual(processed["user"]["password"], "**********")
        self.assertEqual(processed["user"]["credentials"]["api_key"], "************")
        self.assertEqual(processed["data"][0]["token"], "************")
        self.assertEqual(processed["data"][1]["public"], "public_value")
    
    def test_log_operation(self):
        """Test the LogOperation context manager."""
        logger = get_logger("test")
        
        # Capture log output
        with patch.object(logger, 'info') as mock_info:
            # Use the context manager
            with LogOperation(logger, "test_operation", level="info", extra="value"):
                # Do some work
                pass
            
            # Check start message was logged
            mock_info.assert_any_call("Starting test_operation", extra="value")
            
            # Check completion message was logged with duration
            for call_args in mock_info.call_args_list:
                args, kwargs = call_args
                if args[0] == "Completed test_operation":
                    self.assertIn("duration_ms", kwargs)
                    self.assertIsInstance(kwargs["duration_ms"], float)
                    self.assertEqual(kwargs["extra"], "value")
                    break
            else:
                self.fail("Completion log message not found")


if __name__ == "__main__":
    unittest.main()
