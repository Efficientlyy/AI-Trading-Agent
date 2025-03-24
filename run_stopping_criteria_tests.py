#!/usr/bin/env python3
"""
Script to run the stopping criteria tests for the continuous improvement system.

This script provides an easy way to run the tests for the automatic stopping criteria
implementation added to the continuous improvement system.
"""

import os
import sys
import unittest
import asyncio

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test modules
from tests.analysis_agents.sentiment.continuous_improvement.test_stopping_criteria import (
    TestStoppingCriteria, TestStoppingCriteriaManager
)
from tests.dashboard.test_bayesian_visualizations import TestBayesianVisualizations
from tests.integration.test_stopping_criteria_integration import TestStoppingCriteriaIntegration
from tests.analysis_agents.sentiment.continuous_improvement.test_improvement_manager_stopping import (
    TestImprovementManagerStopping
)


def run_tests():
    """Run the stopping criteria test suite."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestStoppingCriteria))
    suite.addTest(unittest.makeSuite(TestStoppingCriteriaManager))
    suite.addTest(unittest.makeSuite(TestBayesianVisualizations))
    suite.addTest(unittest.makeSuite(TestImprovementManagerStopping))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


async def run_async_tests():
    """Run the asynchronous integration tests separately."""
    print("\n=== Running Asynchronous Integration Tests ===\n")
    
    # Create test case and run tests manually
    test_case = TestStoppingCriteriaIntegration()
    methods = [method for method in dir(test_case) if method.startswith('test_')]
    
    success = True
    for method in methods:
        test_method = getattr(test_case, method)
        print(f"Running {method}...")
        try:
            # Set up before test
            if hasattr(test_case, 'setUp'):
                await test_case.setUp()
            
            # Run test
            await test_method()
            print(f" {method} PASSED")
            
        except Exception as e:
            print(f" {method} FAILED: {str(e)}")
            success = False
            
        finally:
            # Clean up after test
            if hasattr(test_case, 'tearDown'):
                await test_case.tearDown()
    
    return success


if __name__ == "__main__":
    print("Running stopping criteria tests...")
    
    # Run regular tests
    print("\n=== Running Standard Tests ===\n")
    standard_tests_success = run_tests()
    
    # Run async tests
    if sys.version_info >= (3, 7):
        # Python 3.7+ simplified async usage
        async_tests_success = asyncio.run(run_async_tests())
    else:
        # Older Python versions
        loop = asyncio.get_event_loop()
        async_tests_success = loop.run_until_complete(run_async_tests())
    
    # Overall success
    if standard_tests_success and async_tests_success:
        print("\n All tests PASSED!")
        sys.exit(0)
    else:
        print("\n Some tests FAILED!")
        sys.exit(1)