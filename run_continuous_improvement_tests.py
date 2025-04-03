#!/usr/bin/env python3
"""
Run comprehensive tests for the Continuous Improvement System.

This script runs a full suite of tests for the Continuous Improvement System,
including unit tests, integration tests, and end-to-end tests.
"""

import os
import sys
import argparse
import subprocess
import time


def main():
    """Run the continuous improvement system tests."""
    parser = argparse.ArgumentParser(description='Run continuous improvement system tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    parser.add_argument('--coverage', '-c', action='store_true', help='Generate coverage report')
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 80)
    print("Running Continuous Improvement System Tests")
    print("=" * 80)

    # Base command for pytest
    cmd = [
        "pytest",
        "tests/analysis_agents/sentiment/test_continuous_improvement.py",
        "tests/dashboard/components/test_continuous_improvement_dashboard.py",
        "tests/examples/test_continuous_improvement_demo.py",
        "-v" if args.verbose else "",
    ]

    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=src/analysis_agents/sentiment/continuous_improvement",
            "--cov=src/dashboard/components/continuous_improvement_dashboard.py",
            "--cov=examples/continuous_improvement_demo.py",
            "--cov-report=term",
            "--cov-report=html:reports/continuous_improvement_coverage"
        ])

    # Filter out empty strings
    cmd = [arg for arg in cmd if arg]

    # Run the tests
    try:
        subprocess.run(cmd, check=True)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"All continuous improvement system tests completed in {elapsed_time:.2f} seconds")
        
        if args.coverage:
            print(f"Coverage report generated in reports/continuous_improvement_coverage")
        
        print("=" * 80)
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Tests failed with exit code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())