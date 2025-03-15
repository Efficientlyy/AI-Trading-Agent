"""
Run all logging system tests.

This script sets up the environment and runs all tests for the logging system.
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_environment():
    """Set up the test environment."""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["PYTHONPATH"] = "."
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_DIR"] = "./logs"


def generate_test_data():
    """Generate test data for the tests."""
    print("Generating test data...")
    try:
        subprocess.run(
            [sys.executable, "tests/common/generate_test_logs.py", "--duration", "1"],
            check=True
        )
        print("Test data generated successfully.")
    except subprocess.CalledProcessError:
        print("Error generating test data.")
        return False
    
    return True


def run_tests():
    """Run the logging system tests."""
    print("\nRunning logging system tests...")
    
    test_files = [
        "tests/common/test_log_query.py",
        "tests/common/test_log_replay.py",
        "tests/common/test_health_monitoring.py",
        "tests/common/test_integration.py"
    ]
    
    # Run each test file
    results = []
    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        try:
            result = subprocess.run(
                ["pytest", "-v", test_file],
                capture_output=True,
                text=True,
                check=False
            )
            
            print(result.stdout)
            
            if result.returncode != 0:
                print(f"Tests in {test_file} failed.")
                print(result.stderr)
                results.append(False)
            else:
                print(f"Tests in {test_file} passed.")
                results.append(True)
                
        except Exception as e:
            print(f"Error running {test_file}: {e}")
            results.append(False)
    
    # Print summary
    print("\n=== Test Summary ===")
    for i, test_file in enumerate(test_files):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_file}: {status}")
    
    return all(results)


def main():
    """Main function."""
    print("=== Logging System Test Runner ===")
    
    # Set up environment
    setup_environment()
    
    # Generate test data
    if not generate_test_data():
        print("Failed to generate test data. Exiting.")
        return 1
    
    # Run tests
    if run_tests():
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
