"""
Simplified configuration wrapper to get the modern dashboard running.
"""

class Config:
    """Simple configuration class to fix import issues."""
    
    @staticmethod
    def get(key, default=None):
        """Get a configuration value."""
        return default
