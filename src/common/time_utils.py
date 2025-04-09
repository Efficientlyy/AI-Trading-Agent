"""
Time utilities for consistent UTC-naive timestamp handling and comparison.
"""

from datetime import datetime, timezone, timedelta
import pandas as pd


def to_utc_naive(dt) -> datetime:
    """
    Convert a datetime or pandas Timestamp to naive UTC datetime.
    
    Args:
        dt: datetime.datetime or pd.Timestamp
    
    Returns:
        datetime.datetime (naive, UTC)
    """
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is not None:
            dt = dt.tz_convert('UTC').replace(tzinfo=None)
        else:
            dt = dt.to_pydatetime()
    elif isinstance(dt, datetime):
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        raise TypeError(f"Unsupported type for to_utc_naive: {type(dt)}")
    return dt


def compare_timestamps(dt1, dt2, tolerance=timedelta(microseconds=1)) -> bool:
    """
    Compare two timestamps with a tolerance, after converting to naive UTC.
    
    Args:
        dt1, dt2: datetime.datetime or pd.Timestamp
        tolerance: allowed difference (default 1 microsecond)
    
    Returns:
        True if timestamps are within tolerance, False otherwise
    """
    dt1_naive = to_utc_naive(dt1)
    dt2_naive = to_utc_naive(dt2)
    return abs(dt1_naive - dt2_naive) <= tolerance
