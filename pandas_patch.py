"""
Direct patch for pandas to work with Python 3.13
Specifically targeting the dateutil.tz import issues
"""

import sys
import datetime
import importlib

class tzutc:
    """UTC implementation for dateutil compatibility."""
    def utcoffset(self, dt): return datetime.timedelta(0)
    def dst(self, dt): return datetime.timedelta(0)
    def tzname(self, dt): return "UTC"
    def __repr__(self): return "tzutc()"

class tzlocal:
    """Local timezone implementation for dateutil compatibility."""
    def __init__(self):
        self._offset = datetime.datetime.now().astimezone().utcoffset()
    def utcoffset(self, dt): return self._offset
    def dst(self, dt): return datetime.timedelta(0)
    def tzname(self, dt): return "tzlocal()"
    def __repr__(self): return "tzlocal()"

def gettz(name=None):
    """Get timezone by name - simplified implementation."""
    return tzutc()  # Always return UTC for simplicity

# This is the direct fix - monkey patch the specific pandas module
def apply_patch():
    """Directly patch pandas internals to fix dateutil.tz import issues."""
    try:
        # First make sure dateutil.tz exists in sys.modules
        if 'dateutil.tz' not in sys.modules:
            # Create and inject mock module
            class MockTZ:
                tzutc = tzutc
                tzlocal = tzlocal
                gettz = gettz
            sys.modules['dateutil.tz'] = MockTZ()
            print("✓ Created mock dateutil.tz module")
        
        # Now patch pandas._libs.tslibs.timezones which actually imports gettz
        if 'pandas._libs.tslibs.timezones' in sys.modules:
            pandas_tz = sys.modules['pandas._libs.tslibs.timezones']
            # Add our gettz implementation to the module
            pandas_tz.gettz = gettz
            print("✓ Patched pandas._libs.tslibs.timezones with mock gettz")
        
        # Patch pandas conversion module
        if 'pandas._libs.tslibs.conversion' in sys.modules:
            pandas_conv = sys.modules['pandas._libs.tslibs.conversion']
            # If needed, add timezone methods
            if not hasattr(pandas_conv, 'UTC'):
                pandas_conv.UTC = tzutc()
                print("✓ Added UTC timezone to pandas conversion module")
                
        return True
    except Exception as e:
        print(f"✗ Failed to patch pandas: {e}")
        return False

if __name__ == "__main__":
    # Can be run directly to apply the patch
    success = apply_patch()
    print(f"Pandas patch {'successfully applied' if success else 'failed'}")
