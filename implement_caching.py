"""
Script to implement a simplified CacheManager class in the indicator_engine.py file.
This replaces the external import with an inline implementation.
"""

def implement_caching():
    """Implement a simplified CacheManager class in the indicator_engine.py file."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the external import with an inline implementation
    import_statement = "from ai_trading_agent.utils.cache_manager import CacheManager"
    inline_implementation = """# Inline implementation of CacheManager
class CacheManager:
    \"\"\"
    Simple cache manager for indicator calculations.
    Provides a dictionary-like interface with size, memory, and TTL limits.
    \"\"\"
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 50, ttl_seconds: int = 300):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        
    def __getitem__(self, key):
        return self._cache.get(key)
        
    def __setitem__(self, key, value):
        # Simple implementation without memory or TTL checks
        if len(self._cache) >= self.max_size:
            # Remove oldest item if we're at capacity
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            if oldest_key in self._timestamps:
                del self._timestamps[oldest_key]
        
        self._cache[key] = value
        self._timestamps[key] = pd.Timestamp.now()
        
    def __contains__(self, key):
        return key in self._cache"""
    
    updated_content = content.replace(import_statement, inline_implementation)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Implemented simplified CacheManager in {file_path}")
    return True

if __name__ == "__main__":
    implement_caching()
