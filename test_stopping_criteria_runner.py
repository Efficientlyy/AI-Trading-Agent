#!/usr/bin/env python3
"""
Simplified test runner for stopping criteria tests.

This script runs the unit tests for the automatic stopping criteria feature
in a way that's compatible with the current project structure.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Make sure environment is set properly for testing
os.environ["ENVIRONMENT"] = "testing"

# Import Python 3.13 compatibility patches if needed
if sys.version_info >= (3, 13):
    from py313_compatibility_patch import apply_mock_modules
    apply_mock_modules()

def run_tests():
    """Run the stopping criteria tests."""
    # Discover all tests related to stopping criteria
    test_loader = unittest.TestLoader()
    
    test_suite = unittest.TestSuite()
    
    # Add specific test modules
    stopping_criteria_tests = test_loader.discover(
        os.path.join('tests', 'analysis_agents', 'sentiment', 'continuous_improvement'),
        pattern='test_stopping_criteria.py'
    )
    test_suite.addTests(stopping_criteria_tests)
    
    improvement_manager_tests = test_loader.discover(
        os.path.join('tests', 'analysis_agents', 'sentiment', 'continuous_improvement'),
        pattern='test_improvement_manager_stopping.py'
    )
    test_suite.addTests(improvement_manager_tests)
    
    bayesian_viz_tests = test_loader.discover(
        os.path.join('tests', 'dashboard'),
        pattern='test_bayesian_visualizations.py'
    )
    test_suite.addTests(bayesian_viz_tests)
    
    integration_tests = test_loader.discover(
        os.path.join('tests', 'integration'),
        pattern='test_stopping_criteria_integration.py'
    )
    test_suite.addTests(integration_tests)
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status (0 for success, 1 for failure)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    print("Running Stopping Criteria Tests...")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Running tests from: {project_root}")
    print("=" * 60)
    
    exit_code = run_tests()
    
    print("=" * 60)
    print("Test Summary:")
    if exit_code == 0:
        print("✅ All stopping criteria tests passed!")
        print("\nThe automatic stopping criteria feature has been fully verified!")
        print("The implementation ensures experiments stop at the right time based on")
        print("statistical thresholds, ensuring efficient resource use while maintaining")
        print("decision quality.")
    else:
        print("❌ Some tests failed. Please check the output above for details.")
    
    # Set exit code (used by CI systems)
    sys.exit(exit_code)
