"""
Quick test to verify the Rust extension can be imported and used.
This script tests the core functionality of the ai_trading_agent_rs module.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Print Python path for debugging
print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Try to import the Rust module
print("\nAttempting to import ai_trading_agent_rs...")
try:
    import ai_trading_agent_rs
    print(f"SUCCESS: Imported ai_trading_agent_rs (version: {getattr(ai_trading_agent_rs, '__version__', 'unknown')})")
    print(f"Module location: {ai_trading_agent_rs.__file__}")
    
    # List available functions
    print("\nAvailable functions:")
    rust_funcs = [attr for attr in dir(ai_trading_agent_rs) 
                 if not attr.startswith('_') and 
                 callable(getattr(ai_trading_agent_rs, attr, None))]
    for func in rust_funcs:
        print(f"  {func}")
    
    # Test lag features
    print("\nTesting lag features...")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    lags = [1, 2]
    result = ai_trading_agent_rs.create_lag_features_rs(data, lags)
    print(f"create_lag_features_rs({data}, {lags}) = {result}")
    
    # Test bollinger bands
    print("\nTesting bollinger bands...")
    window = 2
    num_std = 2.0
    result = ai_trading_agent_rs.create_bollinger_bands_rs(data, window, num_std)
    print(f"create_bollinger_bands_rs({data}, {window}, {num_std}) = {result}")
    
    # Test RSI
    print("\nTesting RSI...")
    period = 2
    result = ai_trading_agent_rs.create_rsi_features_rs(data, period)
    print(f"create_rsi_features_rs({data}, {period}) = {result}")
    
    print("\nAll tests passed successfully!")
    
except ImportError as e:
    print(f"ERROR: Failed to import ai_trading_agent_rs: {e}")
    
    # Check for the compiled module
    print("\nChecking for compiled module...")
    venv_site_packages = None
    for p in sys.path:
        if "site-packages" in p:
            venv_site_packages = p
            break
    
    if venv_site_packages:
        print(f"Site-packages directory: {venv_site_packages}")
        module_files = list(Path(venv_site_packages).glob("ai_trading_agent_rs*"))
        if module_files:
            print("Found module files:")
            for f in module_files:
                print(f"  {f}")
        else:
            print("No module files found in site-packages.")
    
except Exception as e:
    print(f"ERROR: Unexpected error: {e}")
