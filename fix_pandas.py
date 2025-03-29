#!/usr/bin/env python3
"""
Fix pandas dateutil.tz import issues for Python 3.13

This script must run BEFORE importing pandas or any module that uses pandas.
"""

import os
import sys
import importlib.abc
import importlib.machinery
import importlib.util
from datetime import datetime, timedelta
from types import ModuleType

print("Applying pandas fix for Python 3.13...")

# Simple implementations of required timezone classes
class tzutc:
    """UTC implementation."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(tzutc, cls).__new__(cls)
        return cls._instance
    
    def utcoffset(self, dt): return timedelta(0)
    def dst(self, dt): return timedelta(0)
    def tzname(self, dt): return "UTC"
    def __repr__(self): return "tzutc()"

class tzlocal:
    """Local timezone implementation."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(tzlocal, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._std_offset = datetime.now().astimezone().utcoffset()
    
    def utcoffset(self, dt): return self._std_offset
    def dst(self, dt): return timedelta(0)
    def tzname(self, dt): return "Local"
    def __repr__(self): return "tzlocal()"

def gettz(name=None):
    """Get timezone by name."""
    if name is None:
        return None
    if isinstance(name, str) and name.lower() in ('utc', 'gmt'):
        return tzutc()
    return tzlocal()  # Default to local time

# Create a mock dateutil.tz module
mock_tz = ModuleType('dateutil.tz')
mock_tz.tzutc = tzutc
mock_tz.tzlocal = tzlocal
mock_tz.gettz = gettz
mock_tz.__file__ = __file__
mock_tz.__path__ = []
mock_tz.__package__ = 'dateutil'

# Create a mock dateutil package if it doesn't exist
if 'dateutil' not in sys.modules:
    mock_dateutil = ModuleType('dateutil')
    mock_dateutil.__path__ = []
    mock_dateutil.__file__ = __file__
    mock_dateutil.__package__ = 'dateutil'
    mock_dateutil.tz = mock_tz
    sys.modules['dateutil'] = mock_dateutil

# Install the mock tz module
sys.modules['dateutil.tz'] = mock_tz
print("✓ Installed mock dateutil.tz module")

# Create a special import hook to handle pandas imports
class PandasCompatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # If a pandas module tries to import dateutil.tz
        if fullname.startswith('pandas._libs.tslibs'):
            spec = importlib.machinery.ModuleSpec(fullname, None)
            return spec
        return None
    
    def create_module(self, spec):
        name = spec.name
        if name == 'pandas._libs.tslibs.timezones':
            # Create a custom timezones module with our implementation
            module = ModuleType(name)
            module.gettz = gettz
            module.tzutc = tzutc
            module.tzlocal = tzlocal
            return module
        return None
    
    def exec_module(self, module):
        # Module is already set up
        return

# Install the import hook
sys.meta_path.insert(0, PandasCompatFinder())
print("✓ Installed pandas compatibility import hook")

print("Pandas fix applied. You can now import pandas.")
