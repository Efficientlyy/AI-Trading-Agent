import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    import ai_trading_agent_rs
    print(f"\nSuccessfully imported ai_trading_agent_rs")
    print(f"Module location: {ai_trading_agent_rs.__file__}")
    print(f"Available functions: {[f for f in dir(ai_trading_agent_rs) if not f.startswith('_')]}")
    
    if hasattr(ai_trading_agent_rs, 'create_lag_features_rs'):
        print("create_lag_features_rs is available")
    else:
        print("create_lag_features_rs is NOT available")
except ImportError as e:
    print(f"Failed to import ai_trading_agent_rs: {e}")

try:
    import rust_lag_features
    print(f"\nSuccessfully imported rust_lag_features")
    print(f"Module location: {rust_lag_features.__file__}")
    print(f"Available functions: {[f for f in dir(rust_lag_features) if not f.startswith('_')]}")
    
    if hasattr(rust_lag_features, 'create_lag_features_rs'):
        print("create_lag_features_rs is available")
    else:
        print("create_lag_features_rs is NOT available")
except ImportError as e:
    print(f"Failed to import rust_lag_features: {e}")
