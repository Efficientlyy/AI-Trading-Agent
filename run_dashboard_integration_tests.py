"""
Dashboard Integration Test Runner

This script runs integration tests for dashboard components to ensure they work correctly
together and meet performance requirements.
"""

import os
import sys
import pytest

def main():
    """Run dashboard integration tests."""
    print("Running Dashboard Integration Tests...")
    
    # Set up environment variables
    os.environ["DASHBOARD_TEST_ENV"] = "true"
    
    # Run tests
    test_path = "tests/dashboard/test_visualization_components.py"
    result = pytest.main(["-v", test_path])
    
    if result == 0:
        print("\nAll dashboard integration tests passed!")
    else:
        print("\nSome dashboard integration tests failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()