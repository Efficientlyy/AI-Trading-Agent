"""
Dateutil compatibility module for Python 3.13

This module provides compatibility patches for the dateutil.tz module
which is required by pandas but has compatibility issues with Python 3.13.
"""

import sys
import datetime
import re
from types import ModuleType
from typing import Dict, List, Optional, Union, Any, Tuple, Callable


# Simplified compatibility layer for dateutil.tz
class tzutc(datetime.tzinfo):
    """UTC implementation for dateutil compatibility."""
    
    _instance = None
    
    def __new__(cls):
        """Implement the singleton pattern for tzutc."""
        if cls._instance is None:
            cls._instance = super(tzutc, cls).__new__(cls)
        return cls._instance
    
    def __repr__(self):
        """Return string representation."""
        return "tzutc()"
    
    def utcoffset(self, dt):
        """Return UTC offset (always 0)."""
        return datetime.timedelta(0)
    
    def dst(self, dt):
        """Return daylight savings time offset (always 0)."""
        return datetime.timedelta(0)
    
    def tzname(self, dt):
        """Return timezone name."""
        return "UTC"


class tzoffset(datetime.tzinfo):
    """Fixed offset in minutes east from UTC."""
    
    def __init__(self, name, offset):
        """Initialize with timezone name and offset in seconds."""
        self._name = name
        if isinstance(offset, datetime.timedelta):
            self._offset = offset
        else:
            self._offset = datetime.timedelta(seconds=offset)
    
    def __repr__(self):
        """Return string representation."""
        return f"tzoffset({self._name!r}, {self._offset.total_seconds()!r})"
    
    def utcoffset(self, dt):
        """Return UTC offset."""
        return self._offset
    
    def dst(self, dt):
        """Return daylight savings time offset (always 0)."""
        return datetime.timedelta(0)
    
    def tzname(self, dt):
        """Return timezone name."""
        return self._name


class tzlocal(datetime.tzinfo):
    """Local timezone implementation for dateutil compatibility."""
    
    def __init__(self):
        """Initialize local timezone."""
        self._std_offset = datetime.datetime.now().astimezone().utcoffset()
        self._dst_offset = datetime.datetime.now().dst() or datetime.timedelta(0)
    
    def __repr__(self):
        """Return string representation."""
        return "tzlocal()"
    
    def utcoffset(self, dt):
        """Return UTC offset for local timezone."""
        return self._std_offset
    
    def dst(self, dt):
        """Return daylight savings time offset."""
        return self._dst_offset
    
    def tzname(self, dt):
        """Return timezone name."""
        return datetime.datetime.now().astimezone().tzname()


class tzfile(datetime.tzinfo):
    """Timezone file implementation for dateutil compatibility."""
    
    def __init__(self, fileobj, filename=None):
        """Initialize from a file."""
        self._filename = filename
        self._std_offset = datetime.timedelta(0)
        self._dst_offset = datetime.timedelta(0)
        self._name = "tzfile"
    
    def __repr__(self):
        """Return string representation."""
        return f"tzfile({self._filename})"
    
    def utcoffset(self, dt):
        """Return UTC offset."""
        return self._std_offset
    
    def dst(self, dt):
        """Return daylight savings time offset."""
        return self._dst_offset
    
    def tzname(self, dt):
        """Return timezone name."""
        return self._name


def gettz(name: Optional[str] = None) -> Optional[Union[tzutc, tzlocal]]:
    """Get timezone by name."""
    if name is None or name == "":
        return tzlocal()
    elif name.lower() == "utc" or name.lower() == "gmt":
        return tzutc()
    else:
        # For simplicity, return local timezone for all other names
        return tzlocal()


# Simplified compatibility layer for dateutil.parser
class ParserInfo:
    """
    Mock implementation of ParserInfo from dateutil.parser.
    """
    def __init__(self, dayfirst=False, yearfirst=False):
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst

# Create a default parser instance for compatibility
DEFAULTPARSER = ParserInfo()

def parse(timestr, parserinfo=None, **kwargs):
    """
    Parse a string into a datetime object.
    A simplified version for compatibility that handles common formats.
    """
    # Handle common ISO formats
    if not timestr:
        raise ValueError("timestr cannot be empty")
    
    # Try to use datetime's own parser for ISO format first
    try:
        return datetime.datetime.fromisoformat(timestr)
    except ValueError:
        pass
    
    # Handle common formats with regex
    # ISO 8601: YYYY-MM-DDTHH:MM:SS.sssZ
    iso_pattern = r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(?:Z|([+-])(\d{2}):(\d{2}))?"
    match = re.match(iso_pattern, timestr)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups()[:6])
        microsecond = int((match.group(7) or "0").ljust(6, "0")[:6])
        
        dt = datetime.datetime(year, month, day, hour, minute, second, microsecond)
        
        # Handle timezone
        if match.group(8):  # Has timezone offset
            sign = 1 if match.group(8) == "+" else -1
            offset_hour, offset_minute = int(match.group(9)), int(match.group(10))
            offset = datetime.timedelta(hours=sign * offset_hour, minutes=sign * offset_minute)
            dt = dt - offset  # Convert to UTC
            
        return dt
    
    # Handle common date formats: YYYY-MM-DD
    date_pattern = r"(\d{4})-(\d{1,2})-(\d{1,2})$"
    match = re.match(date_pattern, timestr)
    if match:
        year, month, day = map(int, match.groups())
        return datetime.datetime(year, month, day)
    
    # Handle American date format: MM/DD/YYYY
    us_date_pattern = r"(\d{1,2})/(\d{1,2})/(\d{4})$"
    match = re.match(us_date_pattern, timestr)
    if match:
        month, day, year = map(int, match.groups())
        return datetime.datetime(year, month, day)
    
    # Fallback to current time if parsing fails
    raise ValueError(f"Could not parse date string: {timestr}")


def apply_dateutil_patches():
    """Apply all dateutil compatibility patches for pandas."""
    
    if sys.version_info < (3, 13):
        # No need to apply patches for older Python versions
        print("Running on Python < 3.13, no patches needed")
        return
    
    print("Applying dateutil.tz compatibility patches for Python 3.13...")
    
    # Simple method: create the modules directly
    if "dateutil" not in sys.modules:
        dateutil = ModuleType("dateutil")
        dateutil.__path__ = []
        sys.modules["dateutil"] = dateutil
    else:
        dateutil = sys.modules["dateutil"]
    
    # Create tz module if not exists
    if "dateutil.tz" not in sys.modules:
        tz = ModuleType("dateutil.tz")
        sys.modules["dateutil.tz"] = tz
        if hasattr(dateutil, "tz"):
            # Don't replace existing module
            pass
        else:
            dateutil.tz = tz
    else:
        tz = sys.modules["dateutil.tz"]
    
    # Add our timezone classes to the module
    tz.tzutc = tzutc
    tz.tzlocal = tzlocal
    tz.tzfile = tzfile
    tz.tzoffset = tzoffset
    tz.gettz = gettz
    
    # Create parser module if not exists
    if "dateutil.parser" not in sys.modules:
        parser = ModuleType("dateutil.parser")
        sys.modules["dateutil.parser"] = parser
        if hasattr(dateutil, "parser"):
            # Don't replace existing module
            pass
        else:
            dateutil.parser = parser
    else:
        parser = sys.modules["dateutil.parser"]
    
    # Add our parser functions to the module
    parser.parse = parse
    parser.ParserInfo = ParserInfo
    parser.DEFAULTPARSER = DEFAULTPARSER
    
    print("✓ Successfully applied dateutil.tz compatibility patches")
    return True


# When this module is imported directly, apply the patches
if __name__ != "__main__":
    # Don't apply patches when just defining the module
    pass
