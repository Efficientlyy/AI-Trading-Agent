#!/usr/bin/env python
"""
Master Script for Running All Sentiment Analysis Tests

This script runs all the sentiment analysis tests in sequence and generates a comprehensive report.
"""

import asyncio
import os
import sys
import subprocess
import logging
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_output/sentiment_tests.log")
    ]
)
logger = logging.getLogger("sentiment_tests")

class SentimentTestRunner:
    """Runner for all sentiment tests."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Create output directory
        os.makedirs("test_output", exist_ok=True)
    
    def run_test_script(self, script_name, timeout=600):
        """Run a test script and return the result.
        
        Args:
            script_name: Name of the script to run
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, output)
        """
        logger.info(f"Running test script: {script_name}")
        
        try:
            # Run the script
            start_time = time.time()
            process = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            
            # Check result
            success = process.returncode == 0
            
            # Format output
            output = process.stdout
            
            logger.info(f"Test script {script_name} completed with status: {'SUCCESS' if success else 'FAILURE'}")
            
            # Return result
            return {
                "script": script_name,
                "success": success,
                "duration": end_time - start_time,
                "output": output,
                "error": process.stderr if not success else None
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Test script {script_name} timed out after {timeout} seconds")
            return {
                "script": script_name,
                "success": False,
                "duration": timeout,
                "output": None,
                "error": f"Timed out after {timeout} seconds"
            }
            
        except Exception as e:
            logger.error(f"Error running test script {script_name}: {str(e)}")
            return {
                "script": script_name,
                "success": False,
                "duration": 0,
                "output": None,
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all test scripts."""
        logger.info("Starting all sentiment analysis tests")
        
        # List of test scripts to run
        test_scripts = [
            "test_sentiment_system.py",
            "test_sentiment_backtest.py",
            "test_sentiment_dashboard.py"
        ]
        
        # Run each test script
        for script in test_scripts:
            # Run the script
            result = self.run_test_script(script)
            
            # Store the result
            self.results["tests"][script] = result
        
        # Generate summary report
        self.generate_report()
        
        # Return overall success
        all_success = all(result["success"] for result in self.results["tests"].values())
        return all_success
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        logger.info("Generating comprehensive test report")
        
        # Calculate overall metrics
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for result in self.results["tests"].values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        # Add summary to results
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0
        }
        
        # Save results to file
        report_path = "test_output/sentiment_tests_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Comprehensive test report saved to {report_path}")
        
        # Generate HTML report
        self.generate_html_report()
    
    def generate_html_report(self):
        """Generate an HTML report from the test results."""
        report_path = "test_output/sentiment_tests_report.html"
        
        # HTML template
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentiment Analysis Tests Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .summary { margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
                .test { margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .success { border-left: 5px solid #4CAF50; }
                .failure { border-left: 5px solid #F44336; }
                .test-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
                .test-name { font-weight: bold; }
                .test-status { padding: 2px 8px; border-radius: 3px; color: white; }
                .test-status.success { background-color: #4CAF50; }
                .test-status.failure { background-color: #F44336; }
                .test-details { margin-top: 10px; }
                pre { background-color: #f9f9f9; padding: 10px; overflow-x: auto; }
                .error { color: #F44336; }
            </style>
        </head>
        <body>
            <h1>Sentiment Analysis Tests Report</h1>
            <div class="timestamp">Generated: {timestamp}</div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {total_tests}</p>
                <p>Passed Tests: {passed_tests}</p>
                <p>Failed Tests: {failed_tests}</p>
                <p>Success Rate: {success_rate:.2%}</p>
            </div>
            
            <h2>Test Results</h2>
            {test_results}
        </body>
        </html>
        """
        
        # Generate test results HTML
        test_results_html = ""
        for script, result in self.results["tests"].items():
            success = result["success"]
            status_class = "success" if success else "failure"
            status_text = "SUCCESS" if success else "FAILURE"
            
            test_html = f"""
            <div class="test {status_class}">
                <div class="test-header">
                    <div class="test-name">{script}</div>
                    <div class="test-status {status_class}">{status_text}</div>
                </div>
                <div>Duration: {result["duration"]:.2f} seconds</div>
                <div class="test-details">
            """
            
            if result["output"]:
                test_html += f"""
                    <h3>Output:</h3>
                    <pre>{result["output"]}</pre>
                """
            
            if result["error"] and not success:
                test_html += f"""
                    <h3>Error:</h3>
                    <pre class="error">{result["error"]}</pre>
                """
            
            test_html += """
                </div>
            </div>
            """
            
            test_results_html += test_html
        
        # Format HTML
        summary = self.results["summary"]
        html = html.format(
            timestamp=self.results["timestamp"],
            total_tests=summary["total_tests"],
            passed_tests=summary["passed_tests"],
            failed_tests=summary["failed_tests"],
            success_rate=summary["success_rate"],
            test_results=test_results_html
        )
        
        # Save HTML report
        with open(report_path, "w") as f:
            f.write(html)
            
        logger.info(f"HTML report saved to {report_path}")


def main():
    """Run all sentiment tests."""
    # Display banner
    print("=" * 80)
    print("SENTIMENT ANALYSIS COMPREHENSIVE TESTS".center(80))
    print("=" * 80)
    
    # Create and run test runner
    runner = SentimentTestRunner()
    all_success = runner.run_all_tests()
    
    # Display final status
    print("\n" + "=" * 80)
    print(f"FINAL STATUS: {'SUCCESS' if all_success else 'FAILURE'}".center(80))
    print("=" * 80)
    
    print("\nTest reports saved to test_output/ directory")
    print("  - JSON report: test_output/sentiment_tests_report.json")
    print("  - HTML report: test_output/sentiment_tests_report.html")
    print("  - Log file: test_output/sentiment_tests.log")
    
    # Return with appropriate exit code
    return 0 if all_success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        sys.exit(1)