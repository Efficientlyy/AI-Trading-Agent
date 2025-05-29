"""
This script creates a Python wrapper module that imports the ai_trading_agent_rs Rust functions
and re-exports them with the correct names for backwards compatibility.
"""

import sys
import importlib.util
import os

# Create a new module that wraps the Rust functions
rust_wrapper_code = """
# Auto-generated Rust wrapper module for backward compatibility

# Import the real Rust functions
try:
    import ai_trading_agent_rs
    print("Successfully imported ai_trading_agent_rs in wrapper")
    _rust_module = ai_trading_agent_rs
except ImportError:
    print("Could not import ai_trading_agent_rs, falling back to empty module")
    _rust_module = None

# Re-export the functions with both the original and compatible names
if _rust_module is not None:
    # Function mapping (new name -> original name)
    function_mappings = {
        'create_lag_features': 'create_lag_features_rs',
        'create_ema_features': 'create_ema_features_rs',
        'create_rsi_features': 'create_rsi_features_rs',
        'create_bollinger_bands': 'create_bollinger_bands_rs',
        'create_diff_features': 'create_diff_features_rs',
        'create_pct_change_features': 'create_pct_change_features_rs',
        'create_rolling_window_features': 'create_rolling_window_features_rs'
    }
    
    # Import all functions from the original module
    for orig_name in dir(_rust_module):
        if not orig_name.startswith('_'):
            globals()[orig_name] = getattr(_rust_module, orig_name)
    
    # Create wrapper functions for backward compatibility
    for compat_name, orig_name in function_mappings.items():
        if hasattr(_rust_module, orig_name):
            globals()[compat_name] = getattr(_rust_module, orig_name)
            print(f"Created compatibility mapping: {compat_name} -> {orig_name}")
else:
    print("Rust module not available, not creating function mappings")
"""

# Write the wrapper module to the site-packages directory or a local directory
wrapper_filename = "rust_lag_features.py"

# Try writing to local directory first
try:
    with open(wrapper_filename, "w") as f:
        f.write(rust_wrapper_code)
    print(f"Successfully created wrapper module at {os.path.abspath(wrapper_filename)}")
except Exception as e:
    print(f"Error creating wrapper module: {e}")

print("\nTesting wrapper module:")
try:
    import rust_lag_features
    print("Successfully imported rust_lag_features wrapper")
    for func in ['create_lag_features', 'create_lag_features_rs']:
        if hasattr(rust_lag_features, func):
            print(f"Function '{func}' is available in the wrapper")
        else:
            print(f"Function '{func}' is NOT available in the wrapper")
except Exception as e:
    print(f"Error testing wrapper module: {e}")
