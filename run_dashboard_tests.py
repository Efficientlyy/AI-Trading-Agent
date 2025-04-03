#!/usr/bin/env python3
"""
Dashboard Test Runner

This script runs the tests for the modern dashboard implementation.

Usage:
    python run_dashboard_tests.py [options]

Options:
    --data-service      Run tests for DataService and theme system
    --websocket         Run WebSocket tests for real-time updates
    --flask-routes      Run tests for Flask routes and authentication
    --ui-components     Run tests for UI components (notifications, guided tour)
    --performance       Run tests for performance optimizations
    --export            Run tests for data export functionality
    --all               Run all tests (default)
    --file=<filename>   Run tests from a specific file
    --coverage          Generate code coverage report
    --verbose           Show verbose output
"""

import os
import sys
import subprocess
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dashboard Test Runner")
    parser.add_argument("--data-service", action="store_true", help="Run tests for DataService and theme system")
    parser.add_argument("--websocket", action="store_true", help="Run WebSocket tests for real-time updates")
    parser.add_argument("--flask-routes", action="store_true", help="Run tests for Flask routes and authentication")
    parser.add_argument("--ui-components", action="store_true", help="Run tests for UI components")
    parser.add_argument("--performance", action="store_true", help="Run tests for performance optimizations")
    parser.add_argument("--export", action="store_true", help="Run tests for data export functionality")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--file", help="Run tests from a specific file")
    parser.add_argument("--coverage", action="store_true", help="Generate code coverage report")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    
    return parser.parse_args()


def get_test_path(test_type):
    """Get the path to the test file for a given test type."""
    tests = {
        "data-service": "test_modern_dashboard.py",
        "websocket": "test_websocket.py",
        "flask-routes": "test_flask_routes.py",
        "ui-components": "test_frontend_utils.py",
        "performance": "test_frontend_utils.py::TestPerformanceOptimizations",
        "export": "test_notifications_export.py"
    }
    
    return tests.get(test_type, "")


def run_tests(args):
    """Run the specified tests."""
    # Base command
    cmd = ["pytest"]
    
    # Add specific test files or patterns based on args
    if args.file:
        # If a specific file is requested, run only that file
        cmd.append(f"tests/dashboard/{args.file}")
    elif any([args.data_service, args.websocket, args.flask_routes, 
              args.ui_components, args.performance, args.export]):
        # Add specific test files based on requested test types
        test_paths = []
        
        if args.data_service:
            test_paths.append(f"tests/dashboard/{get_test_path('data-service')}")
        
        if args.websocket:
            test_paths.append(f"tests/dashboard/{get_test_path('websocket')}")
        
        if args.flask_routes:
            test_paths.append(f"tests/dashboard/{get_test_path('flask-routes')}")
        
        if args.ui_components:
            test_paths.append(f"tests/dashboard/{get_test_path('ui-components')}")
        
        if args.performance:
            test_paths.append(f"tests/dashboard/{get_test_path('performance')}")
        
        if args.export:
            test_paths.append(f"tests/dashboard/{get_test_path('export')}")
        
        cmd.extend(test_paths)
    else:
        # Run all dashboard tests
        cmd.append("tests/dashboard/")
    
    # Add coverage if requested
    if args.coverage:
        cmd.append("--cov=src.dashboard")
        cmd.append("--cov-report=term")
        cmd.append("--cov-report=html:reports/dashboard_coverage")
    
    # Add verbosity if requested
    if args.verbose:
        cmd.append("-v")
    
    # Run the command
    command_str = " ".join(cmd)
    print(f"Running command: {command_str}")
    result = subprocess.run(command_str, shell=True)
    
    return result.returncode


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set default if no test type is specified
    if not any([args.data_service, args.websocket, args.flask_routes, 
                args.ui_components, args.performance, args.export, 
                args.all, args.file]):
        args.all = True
    
    # Create reports directory if it doesn't exist and coverage is requested
    if args.coverage and not os.path.exists("reports/dashboard_coverage"):
        os.makedirs("reports/dashboard_coverage", exist_ok=True)
    
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