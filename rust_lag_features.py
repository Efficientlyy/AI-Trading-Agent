
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
