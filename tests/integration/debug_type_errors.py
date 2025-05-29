"""
Type Error Debug Script

This script helps identify and debug type annotation errors in the Phase 5 components
without requiring all dependencies to be installed.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def debug_coordination_module():
    """Debug type errors in the coordination module"""
    try:
        print("Testing coordination components...")
        from ai_trading_agent.coordination.strategy_coordinator import StrategyCoordinator
        from ai_trading_agent.coordination.performance_attribution import PerformanceAttributor
        
        print("Successfully imported coordination components")
        
        # Simple test initialization
        try:
            coord_config = {
                "strategies": ["Strategy1", "Strategy2"],
                "lookback_periods": 20,
                "conflict_resolution_method": "performance_weighted",
                "capital_allocation_method": "dynamic"
            }
            
            coordinator = StrategyCoordinator(coord_config)
            print("Successfully initialized StrategyCoordinator")
            
            attr_config = {
                "strategies": ["Strategy1", "Strategy2"],
                "metrics": ["returns", "sharpe_ratio", "max_drawdown"],
                "output_path": "./test_output/attribution"
            }
            
            attributor = PerformanceAttributor(attr_config)
            print("Successfully initialized PerformanceAttributor")
            
        except Exception as e:
            print(f"Error initializing coordination components: {str(e)}")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"Failed to import coordination modules: {str(e)}")
        traceback.print_exc()

def examine_type_annotations():
    """Examine problematic type annotations in the codebase"""
    try:
        print("\nExamining type annotations:")
        import typing
        
        print("Looking for Dict instantiations...")
        # List files to check
        modules_to_check = [
            'ai_trading_agent.coordination.strategy_coordinator',
            'ai_trading_agent.coordination.performance_attribution',
            'ai_trading_agent.ml.reinforcement_learning',
            'ai_trading_agent.ml.feature_engineering'
        ]
        
        for module_name in modules_to_check:
            try:
                print(f"\nExamining {module_name}...")
                # Import the module
                module = __import__(module_name, fromlist=['*'])
                
                # Look for all classes
                for name in dir(module):
                    item = getattr(module, name)
                    if isinstance(item, type):
                        print(f"  Found class: {name}")
                        
                        # Examine class methods
                        for method_name in dir(item):
                            if method_name.startswith('_') or not callable(getattr(item, method_name)):
                                continue
                            
                            method = getattr(item, method_name)
                            if hasattr(method, '__annotations__'):
                                annotations = getattr(method, '__annotations__')
                                if annotations:
                                    print(f"    Method {method_name} has annotations: {annotations}")
            
            except ImportError as e:
                print(f"  Error importing {module_name}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error examining type annotations: {str(e)}")
        traceback.print_exc()

def test_rich_signal_creation():
    """Test creating RichSignal instances with proper Dict handling"""
    try:
        print("\nTesting RichSignal creation:")
        
        # Create a simple RichSignal class to test
        class RichSignal:
            def __init__(self, action, quantity, price, metadata=None):
                self.action = action
                self.quantity = quantity
                self.price = price
                self.metadata = metadata or dict()
        
        # Test creating signals with different metadata approaches
        print("Creating signal with empty metadata...")
        signal1 = RichSignal(action=1, quantity=100, price=100.0)
        print(f"  Success: {signal1.metadata}")
        
        print("Creating signal with dict literal...")
        signal2 = RichSignal(action=1, quantity=100, price=100.0, metadata=dict(confidence=0.8))
        print(f"  Success: {signal2.metadata}")
        
        print("Creating signal with dict constructor...")
        signal3 = RichSignal(action=1, quantity=100, price=100.0, metadata=dict(strategy="test", confidence=0.9))
        print(f"  Success: {signal3.metadata}")
        
        return True
        
    except Exception as e:
        print(f"Error testing RichSignal creation: {str(e)}")
        traceback.print_exc()
        return False
        
def fix_common_type_errors():
    """
    Print guidance on how to fix common type annotation errors
    """
    print("\n=== GUIDANCE FOR FIXING TYPE ERRORS ===")
    print("1. Replace Dict instantiations:")
    print("   - BAD:  metadata = metadata or {}")
    print("   - GOOD: metadata = metadata or dict()")
    print()
    print("2. Replace Dict type annotations:")
    print("   - BAD:  def function(params: Dict[str, Any])")
    print("   - GOOD: def function(params: dict)")
    print("   - ALSO GOOD: from typing import Dict; def function(params: Dict[str, Any])")
    print()
    print("3. Fix signal creation:")
    print("   - BAD:  RichSignal(action, quantity, price, metadata={'key': value})")
    print("   - GOOD: RichSignal(action, quantity, price, metadata=dict(key=value))")
    print()
    print("4. When copying metadata:")
    print("   - BAD:  metadata = signal.metadata.copy() if signal.metadata else {}")
    print("   - GOOD: metadata = signal.metadata.copy() if signal.metadata else dict()")
    print()
    print("Remember: Always use dict() instead of {} when creating dictionaries that might")
    print("be used with type annotations.")
    
if __name__ == "__main__":
    print("=== PHASE 5 TYPE ERROR DEBUGGING ===")
    debug_coordination_module()
    examine_type_annotations()
    test_rich_signal_creation()
    fix_common_type_errors()
