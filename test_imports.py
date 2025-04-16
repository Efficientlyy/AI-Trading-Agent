#!/usr/bin/env python
# test_imports.py

"""
This script tests imports in a specific order to isolate the issue.
"""

import sys
import os
import traceback

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print(f"Python path: {sys.path}")
print("Testing imports in a specific order...")

# First, try importing enums directly
try:
    print("\n1. Importing enums...")
    from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus, PositionSide
    print("✓ Successfully imported enums")
except Exception as e:
    print(f"✗ Failed to import enums: {e}")
    traceback.print_exc()

# Then try importing models
try:
    print("\n2. Importing models...")
    from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio
    print("✓ Successfully imported models")
except Exception as e:
    print(f"✗ Failed to import models: {e}")
    traceback.print_exc()

# Then try importing order_manager
try:
    print("\n3. Importing order_manager...")
    from ai_trading_agent.trading_engine.order_manager import OrderManager
    print("✓ Successfully imported OrderManager")
except Exception as e:
    print(f"✗ Failed to import OrderManager: {e}")
    traceback.print_exc()

# Then try importing backtester components
try:
    print("\n4. Importing backtester components...")
    from ai_trading_agent.backtesting.performance_metrics import calculate_metrics
    print("✓ Successfully imported performance_metrics")
except Exception as e:
    print(f"✗ Failed to import performance_metrics: {e}")
    traceback.print_exc()

try:
    print("\n5. Importing Backtester...")
    from ai_trading_agent.backtesting.backtester import Backtester
    print("✓ Successfully imported Backtester")
except Exception as e:
    print(f"✗ Failed to import Backtester: {e}")
    traceback.print_exc()

# Finally, try importing the entire backtesting package
try:
    print("\n6. Importing backtesting package...")
    import ai_trading_agent.backtesting
    print("✓ Successfully imported backtesting package")
except Exception as e:
    print(f"✗ Failed to import backtesting package: {e}")
    traceback.print_exc()

print("\nTest completed.")
