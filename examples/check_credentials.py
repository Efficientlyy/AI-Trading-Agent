#\!/usr/bin/env python
"""
Check API credentials script

This script checks if the required API credentials are available
in the environment variables and reports their status.
"""

import os
import sys
import logging
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import dotenv
try:
    import dotenv
    dotenv.load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_env_var(name: str) -> bool:
    """Check if an environment variable exists and is not empty.
    
    Args:
        name: Name of the environment variable
        
    Returns:
        True if variable exists and is not empty
    """
    value = os.environ.get(name)
    return value is not None and value \!= ""

def check_api_credentials(api_name: str, required_vars: List[str]) -> Dict[str, bool]:
    """Check credentials for a specific API.
    
    Args:
        api_name: API name for display purposes
        required_vars: List of required environment variables
        
    Returns:
        Dictionary mapping variable names to their availability status
    """
    results = {}
    for var in required_vars:
        results[var] = check_env_var(var)
    
    return results

def main():
    """Check API credentials status."""
    print("\n===== API Credentials Status =====\n")
    
    if not DOTENV_AVAILABLE:
        print("Warning: dotenv package not available. Install with: pip install python-dotenv")
        print("Environment variables will only be loaded from the system environment.\n")
    
    # Check Twitter API credentials
    twitter_vars = ["TWITTER_API_KEY", "TWITTER_API_SECRET", 
                   "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET"]
    twitter_results = check_api_credentials("Twitter", twitter_vars)
    
    print("Twitter API:")
    all_twitter_available = all(twitter_results.values())
    for var, available in twitter_results.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {var}: {status}")
    
    if all_twitter_available:
        print("  Status: Ready to use real Twitter data")
    else:
        print("  Status: Will use mock Twitter data")
    
    # Check Reddit API credentials
    reddit_vars = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
    reddit_results = check_api_credentials("Reddit", reddit_vars)
    
    print("\nReddit API:")
    all_reddit_available = all(reddit_results.values())
    for var, available in reddit_results.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {var}: {status}")
    
    if all_reddit_available:
        print("  Status: Ready to use real Reddit data")
    else:
        print("  Status: Will use mock Reddit data")
    
    # Check Binance API credentials
    binance_vars = ["BINANCE_API_KEY", "BINANCE_API_SECRET"]
    binance_results = check_api_credentials("Binance", binance_vars)
    
    print("\nBinance API:")
    all_binance_available = all(binance_results.values())
    for var, available in binance_results.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {var}: {status}")
    
    if all_binance_available:
        print("  Status: Ready to use real Binance data")
    else:
        print("  Status: Will use public API (rate limited)")
    
    print("\n===== Overall Status =====\n")
    
    if all_twitter_available and all_reddit_available and all_binance_available:
        print("✅ All credentials are available\! The system can use real data.")
    else:
        print("⚠️ Some credentials are missing. The system will use mock data where needed.")
    
    print("\nTo add missing credentials:")
    print("1. Edit the .env file in the project root")
    print("2. Add the missing variables with their values")
    print("3. Restart your application")
    
    print("\nFor more information, see TWITTER_SETUP.md and README.md")

if __name__ == "__main__":
    main()
