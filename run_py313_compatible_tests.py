"""
Python 3.13 Compatible Test Runner

This script runs tests in a way that's compatible with Python 3.13,
bypassing common collection issues by directly importing and running test modules.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
import traceback

def import_module_from_path(module_path):
    """Import a module from its file path."""
    module_name = os.path.basename(module_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {module_path}: {e}")
        traceback.print_exc()
        return None

def is_test_class(obj):
    """Check if an object is a test class."""
    return (isinstance(obj, type) and 
            obj.__name__.startswith('Test') and 
            hasattr(obj, '__module__'))

def is_test_function(obj, name):
    """Check if an object is a test function."""
    return callable(obj) and name.startswith('test_')

def run_test_file(file_path):
    """Run all tests in a file."""
    print(f"\n===== Running tests in {file_path} =====")
    
    module = import_module_from_path(file_path)
    if not module:
        print(f"Failed to import {file_path}")
        return False
    
    # Find all test classes and functions
    test_suite = unittest.TestSuite()
    
    for name in dir(module):
        obj = getattr(module, name)
        
        if is_test_class(obj):
            # Add all test methods from the class
            for method_name in dir(obj):
                if method_name.startswith('test_'):
                    test_case = obj(method_name)
                    test_suite.addTest(test_case)
            print(f"Added test class: {name}")
        
        elif is_test_function(obj, name):
            # Add standalone test functions
            test_case = unittest.FunctionTestCase(obj)
            test_suite.addTest(test_case)
            print(f"Added test function: {name}")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def run_tests_in_directory(directory, pattern="test_*.py"):
    """Run all test files in a directory."""
    print(f"\n===== Running tests in directory {directory} =====")
    
    # Find all test files
    test_files = list(Path(directory).glob(pattern))
    if not test_files:
        print(f"No test files found in {directory}")
        return True
    
    # Run each test file
    success = True
    for file_path in test_files:
        if not run_test_file(file_path):
            success = False
    
    return success

def main():
    """Run the Python 3.13 compatible test runner."""
    print("Python 3.13 Compatible Test Runner")
    print("=" * 40)
    
    # Add the project root to the Python path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)
    
    # Set environment variables for testing
    os.environ["ENVIRONMENT"] = "testing"
    
    # Accept command-line arguments for directories or files
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        # Default to sentiment analysis tests
        targets = [
            os.path.join(base_dir, "tests", "analysis_agents", "sentiment", "test_sentiment_validator.py")
        ]
    
    # Run the specified tests
    success = True
    for target in targets:
        if os.path.isfile(target):
            if not run_test_file(target):
                success = False
        elif os.path.isdir(target):
            if not run_tests_in_directory(target):
                success = False
        else:
            print(f"Target not found: {target}")
            success = False
    
    # Exit with appropriate status code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
