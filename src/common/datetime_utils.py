"""Datetime utilities for the trading system.

This module provides common datetime utilities to ensure consistent handling
of dates and times throughout the application.
"""

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Get the current datetime in UTC.
    
    This is a replacement for the deprecated datetime.utcnow() function.
    
    Returns:
        datetime: The current datetime in UTC timezone
    """
    return datetime.now(timezone.utc)


def format_iso(dt: Optional[datetime] = None) -> str:
    """Format a datetime object as ISO 8601 string.
    
    Args:
        dt: The datetime to format (default: current UTC time)
    
    Returns:
        str: ISO 8601 formatted datetime string
    """
    if dt is None:
        dt = utc_now()
    return dt.isoformat()


def parse_iso(iso_str: str) -> datetime:
    """Parse an ISO 8601 string into a datetime object.
    
    Args:
        iso_str: The ISO 8601 string to parse
    
    Returns:
        datetime: The parsed datetime object with UTC timezone
    """
    dt = datetime.fromisoformat(iso_str)
    # Add UTC timezone if the datetime is naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def days_between(dt1: datetime, dt2: Optional[datetime] = None) -> int:
    """Calculate the number of days between two datetimes.
    
    Args:
        dt1: The first datetime
        dt2: The second datetime (default: current UTC time)
    
    Returns:
        int: The number of days between the two datetimes
    """
    if dt2 is None:
        dt2 = utc_now()
    
    # Ensure both datetimes have UTC timezone
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=timezone.utc)
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=timezone.utc)
        
    return abs((dt2 - dt1).days)
