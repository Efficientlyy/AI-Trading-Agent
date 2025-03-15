"""Tests for datetime utilities."""

import datetime
import unittest
from unittest import mock

from src.common.datetime_utils import days_between, format_iso, parse_iso, utc_now


class TestDatetimeUtils(unittest.TestCase):
    """Tests for datetime utilities."""

    def test_utc_now(self):
        """Test utc_now returns a timezone-aware datetime with UTC timezone."""
        now = utc_now()
        self.assertIsInstance(now, datetime.datetime)
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.tzinfo, datetime.timezone.utc)

    def test_format_iso(self):
        """Test format_iso correctly formats a datetime as ISO 8601."""
        dt = datetime.datetime(2025, 3, 1, 12, 30, 45, tzinfo=datetime.timezone.utc)
        iso_str = format_iso(dt)
        self.assertEqual(iso_str, "2025-03-01T12:30:45+00:00")

    def test_format_iso_defaults_to_now(self):
        """Test format_iso uses current time when no datetime is provided."""
        mock_now = datetime.datetime(2025, 3, 1, 12, 30, 45, tzinfo=datetime.timezone.utc)
        
        with mock.patch('src.common.datetime_utils.utc_now', return_value=mock_now):
            iso_str = format_iso()
            self.assertEqual(iso_str, "2025-03-01T12:30:45+00:00")

    def test_parse_iso(self):
        """Test parse_iso correctly parses an ISO 8601 string."""
        iso_str = "2025-03-01T12:30:45+00:00"
        dt = parse_iso(iso_str)
        self.assertEqual(dt, datetime.datetime(2025, 3, 1, 12, 30, 45, tzinfo=datetime.timezone.utc))

    def test_parse_iso_adds_utc_to_naive_datetime(self):
        """Test parse_iso adds UTC timezone to naive datetime strings."""
        iso_str = "2025-03-01T12:30:45"  # No timezone info
        dt = parse_iso(iso_str)
        self.assertEqual(dt.tzinfo, datetime.timezone.utc)
        self.assertEqual(dt, datetime.datetime(2025, 3, 1, 12, 30, 45, tzinfo=datetime.timezone.utc))

    def test_days_between(self):
        """Test days_between calculates correct number of days."""
        dt1 = datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc)
        dt2 = datetime.datetime(2025, 3, 10, tzinfo=datetime.timezone.utc)
        
        self.assertEqual(days_between(dt1, dt2), 9)
        # Order shouldn't matter
        self.assertEqual(days_between(dt2, dt1), 9)

    def test_days_between_defaults_to_now(self):
        """Test days_between uses current time when dt2 is not provided."""
        dt1 = datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc)
        mock_now = datetime.datetime(2025, 3, 10, tzinfo=datetime.timezone.utc)
        
        with mock.patch('src.common.datetime_utils.utc_now', return_value=mock_now):
            self.assertEqual(days_between(dt1), 9)
