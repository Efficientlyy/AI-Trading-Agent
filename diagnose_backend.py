"""
Diagnostic script to identify what's causing the backend to crash on startup
"""
import os
import sys
import traceback
import importlib

def test_module_import(module_name):
    """Test importing a module and print detailed error if it fails."""
    print(f"Testing import: {module_name}")
    try:
        module = importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"✗ Error importing {module_name}: {e}")
        traceback.print_exc()
        return False

def test_app_creation():
    """Test creating the FastAPI app object."""
    print("\nTesting FastAPI app creation...")
    try:
        # Use a separate scope to prevent module-level errors from stopping further tests
        scope_locals = {}
        exec("from ai_trading_agent.api.main import app", scope_locals)
        app = scope_locals.get("app")
        print(f"✓ Successfully created FastAPI app")
        return True
    except Exception as e:
        print(f"✗ Error creating FastAPI app: {e}")
        traceback.print_exc()
        return False

# List of critical modules to test in order
modules_to_test = [
    "fastapi",
    "uvicorn",
    "ai_trading_agent.config",
    "ai_trading_agent.common",
    "ai_trading_agent.api.paper_trading_api",
    "ai_trading_agent.api.routers.agent", 
    "ai_trading_agent.api.websocket_api",
    "ai_trading_agent.data_acquisition.mexc_connector",
    "ai_trading_agent.api.mexc_websocket"
]

def main():
    print("=== Backend Diagnostic Tool ===")
    
    # Test each module import
    for module in modules_to_test:
        test_module_import(module)
        print()
    
    # Test app creation specifically
    test_app_creation()
    
    print("\nDiagnostic completed.")

if __name__ == "__main__":
    main()