#!/usr/bin/env python3
"""
Dashboard Test Runner

This script runs the tests for the modern dashboard implementation.

Usage:
    python run_dashboard_tests.py [options]

Options:
    --unit         Run only unit tests
    --integration  Run only integration tests
    --frontend     Run only frontend tests
    --api          Run only API tests
    --auth         Run only authentication tests
    --websocket    Run only WebSocket tests
    --all          Run all tests (default)
    --coverage     Generate code coverage report
    --verbose      Show verbose output
"""

import os
import sys
import subprocess
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dashboard Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--frontend", action="store_true", help="Run only frontend tests")
    parser.add_argument("--api", action="store_true", help="Run only API tests")
    parser.add_argument("--auth", action="store_true", help="Run only authentication tests")
    parser.add_argument("--websocket", action="store_true", help="Run only WebSocket tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--coverage", action="store_true", help="Generate code coverage report")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    
    return parser.parse_args()


def run_tests(args):
    """Run the specified tests."""
    # Base command
    cmd = ["pytest", "tests/dashboard"]
    
    # Add options based on args
    if args.unit:
        cmd.append("-m unit")
    elif args.integration:
        cmd.append("-m integration")
    elif args.frontend:
        cmd.append("-m frontend")
    elif args.api:
        cmd.append("-m api")
    elif args.auth:
        cmd.append("-m auth")
    elif args.websocket:
        cmd.append("-m websocket")
    # If no specific test type is specified, run all tests
    
    # Add coverage if requested
    if args.coverage:
        cmd.append("--cov=src.dashboard")
        cmd.append("--cov-report=term")
        cmd.append("--cov-report=html:reports/dashboard_coverage")
    
    # Add verbosity if requested
    if args.verbose:
        cmd.append("-v")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(" ".join(cmd), shell=True)
    
    return result.returncode


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set default if no test type is specified
    if not any([args.unit, args.integration, args.frontend, args.api, 
                args.auth, args.websocket, args.all]):
        args.all = True
    
    # Run tests
    exit_code = run_tests(args)
    
    # Report results
    if exit_code == 0:
        print("\nAll dashboard tests passed! 🎉")
        
        # Show coverage report path if generated
        if args.coverage:
            print("\nCoverage report generated at: reports/dashboard_coverage/index.html")
    else:
        print("\nSome tests failed. Please check the output above for details.")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())