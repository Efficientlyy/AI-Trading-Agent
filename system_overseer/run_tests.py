#!/usr/bin/env python
"""
System Test Runner for System Overseer

This script runs the integration and system tests for the System Overseer
and reports the results.
"""

import os
import sys
import time
import logging
import unittest
import argparse
from typing import List, Dict, Any, Optional

# Add parent directory to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_overseer.tests.runner")

# Import test modules
from system_overseer.tests.integration_tests import (
    TestModuleRegistry,
    TestConfigRegistry,
    TestEventBus,
    TestPluginManager,
    TestDialogueManager,
    TestSystemIntegration
)


def run_tests(test_classes: List[type], verbose: bool = False) -> Dict[str, Any]:
    """Run tests and return results.
    
    Args:
        test_classes: List of test classes to run
        verbose: Whether to print verbose output
        
    Returns:
        dict: Test results
    """
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    
    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Collect results
    results = {
        "total": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful(),
        "duration": end_time - start_time
    }
    
    # Add details
    if result.failures:
        results["failure_details"] = [
            {
                "test": str(test),
                "message": message
            }
            for test, message in result.failures
        ]
    
    if result.errors:
        results["error_details"] = [
            {
                "test": str(test),
                "message": message
            }
            for test, message in result.errors
        ]
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print test results.
    
    Args:
        results: Test results
    """
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {results['total']} tests run in {results['duration']:.2f} seconds")
    print(f"  Successes: {results['total'] - results['failures'] - results['errors'] - results['skipped']}")
    print(f"  Failures: {results['failures']}")
    print(f"  Errors: {results['errors']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Overall: {'SUCCESS' if results['success'] else 'FAILURE'}")
    print("=" * 80)
    
    # Print failure details
    if results.get("failure_details"):
        print("\nFAILURES:")
        for i, detail in enumerate(results["failure_details"], 1):
            print(f"\n{i}. {detail['test']}")
            print("-" * 40)
            print(detail["message"])
    
    # Print error details
    if results.get("error_details"):
        print("\nERRORS:")
        for i, detail in enumerate(results["error_details"], 1):
            print(f"\n{i}. {detail['test']}")
            print("-" * 40)
            print(detail["message"])
    
    print("\n" + "=" * 80)


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run System Overseer tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--module", "-m", help="Run specific test module")
    args = parser.parse_args()
    
    # Select test classes
    if args.module:
        # Run specific test module
        test_classes = []
        module_name = args.module.lower()
        
        if "module" in module_name or "registry" in module_name:
            test_classes.append(TestModuleRegistry)
        if "config" in module_name:
            test_classes.append(TestConfigRegistry)
        if "event" in module_name:
            test_classes.append(TestEventBus)
        if "plugin" in module_name:
            test_classes.append(TestPluginManager)
        if "dialogue" in module_name:
            test_classes.append(TestDialogueManager)
        if "system" in module_name or "integration" in module_name:
            test_classes.append(TestSystemIntegration)
        
        if not test_classes:
            print(f"No test module matches '{args.module}'")
            return 1
    else:
        # Run all tests
        test_classes = [
            TestModuleRegistry,
            TestConfigRegistry,
            TestEventBus,
            TestPluginManager,
            TestDialogueManager,
            TestSystemIntegration
        ]
    
    # Run tests
    results = run_tests(test_classes, args.verbose)
    
    # Print results
    print_results(results)
    
    # Return exit code
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
