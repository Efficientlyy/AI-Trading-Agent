# minimal_import_test.py
import sys
import os
import traceback

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print(f"PYTHONPATH includes: {project_root}")
print("Attempting minimal import from trading_engine.models...")

try:
    # Try importing a class defined in models.py
    from ai_trading_agent.trading_engine.models import Order 
    print("SUCCESS: Imported 'Order' from ai_trading_agent.trading_engine.models")
except ImportError as e:
    print(f"IMPORT FAILED: {e}")
    print("--- Traceback --- ")
    traceback.print_exc()
    print("-----------------")
except Exception as e:
    print(f"OTHER ERROR DURING IMPORT: {e}")
    print("--- Traceback --- ")
    traceback.print_exc()
    print("-----------------")

print("Minimal import test finished.")
