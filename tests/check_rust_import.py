"""
Simple script to verify Rust module imports and availability of key functions
"""
import sys

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("\nAttempting to import ai_trading_agent_rs...")

try:
    import ai_trading_agent_rs
    print(f"SUCCESS: Imported ai_trading_agent_rs")
    
    # Get module location and version
    print(f"Module location: {getattr(ai_trading_agent_rs, '__file__', 'unknown')}")
    print(f"Module version: {getattr(ai_trading_agent_rs, '__version__', 'unknown')}")
    
    # Check for required functions
    required_functions = [
        'create_lag_features_rs',
        'create_bollinger_bands_rs',
        'create_rsi_features_rs',
        'create_ema_features_rs'
    ]
    
    print("\nChecking for required functions:")
    for func_name in required_functions:
        if hasattr(ai_trading_agent_rs, func_name):
            print(f"  ✓ {func_name} - Available")
        else:
            print(f"  ✗ {func_name} - Not found")
    
    # List all available functions
    print("\nAll available functions:")
    for name in dir(ai_trading_agent_rs):
        if not name.startswith('_') and callable(getattr(ai_trading_agent_rs, name)):
            print(f"  - {name}")
    
    # Test a simple function call
    print("\nTesting create_lag_features_rs with sample data:")
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    test_lags = [1, 2]
    
    if hasattr(ai_trading_agent_rs, 'create_lag_features_rs'):
        try:
            result = ai_trading_agent_rs.create_lag_features_rs(test_data, test_lags)
            print(f"Result: {result}")
            print("Function call successful!")
        except Exception as e:
            print(f"Function call failed with error: {e}")
    else:
        print("Cannot test function as it doesn't exist")

except ImportError as e:
    print(f"FAILED: Could not import ai_trading_agent_rs: {e}")
    
    # Try the fallback name
    print("\nAttempting to import rust_lag_features...")
    try:
        import rust_lag_features
        print(f"SUCCESS: Imported rust_lag_features")
        print(f"Module location: {getattr(rust_lag_features, '__file__', 'unknown')}")
    except ImportError as e2:
        print(f"FAILED: Could not import rust_lag_features: {e2}")
        
    # Check in site-packages
    import site
    print("\nSite packages directories:")
    for path in site.getsitepackages():
        print(f"  {path}")
