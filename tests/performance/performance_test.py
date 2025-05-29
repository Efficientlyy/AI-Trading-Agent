#!/usr/bin/env python
"""
Performance testing framework for AI Trading Agent.

This module tests the performance of various components of the
trading system under normal operating conditions.
"""

import time
import json
import logging
import argparse
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceTest:
    """Base class for performance tests."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 test_duration: int = 300,
                 report_path: str = "./performance_report.html"):
        """
        Initialize the performance test.
        
        Args:
            base_url: Base URL of the API
            test_duration: Test duration in seconds
            report_path: Path to save the HTML report
        """
        self.base_url = base_url
        self.test_duration = test_duration
        self.report_path = report_path
        self.results = {}
        self.start_time = None
        
    def setup(self):
        """Set up the test."""
        logger.info(f"Setting up performance test: {self.__class__.__name__}")
        
        # Check if the API is accessible
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            logger.info("API is accessible")
        except requests.exceptions.RequestException as e:
            logger.error(f"API is not accessible: {e}")
            raise
        
        self.start_time = time.time()
        
    def run(self):
        """Run the performance test."""
        self.setup()
        
        logger.info(f"Running performance test for {self.test_duration} seconds")
        
        try:
            while time.time() - self.start_time < self.test_duration:
                self.execute_test_iteration()
                
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.exception(f"Error during test: {e}")
        finally:
            self.analyze_results()
            self.generate_report()
    
    def execute_test_iteration(self):
        """Execute a single test iteration."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def analyze_results(self):
        """Analyze the test results."""
        logger.info("Analyzing test results")
        
        for endpoint, metrics in self.results.items():
            response_times = metrics.get("response_times", [])
            
            if response_times:
                metrics["min"] = min(response_times)
                metrics["max"] = max(response_times)
                metrics["avg"] = statistics.mean(response_times)
                metrics["median"] = statistics.median(response_times)
                metrics["p95"] = np.percentile(response_times, 95)
                metrics["p99"] = np.percentile(response_times, 99)
                
                logger.info(f"Endpoint: {endpoint}")
                logger.info(f"  Requests: {metrics.get('requests', 0)}")
                logger.info(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
                logger.info(f"  Avg response time: {metrics['avg']:.2f} ms")
                logger.info(f"  95th percentile: {metrics['p95']:.2f} ms")
                logger.info(f"  99th percentile: {metrics['p99']:.2f} ms")
    
    def generate_report(self):
        """Generate an HTML report with the results."""
        logger.info(f"Generating performance report: {self.report_path}")
        
        # Create report dataframe
        data = []
        for endpoint, metrics in self.results.items():
            data.append({
                "Endpoint": endpoint,
                "Requests": metrics.get("requests", 0),
                "Success Rate": f"{metrics.get('success_rate', 0):.2%}",
                "Min (ms)": f"{metrics.get('min', 0):.2f}",
                "Avg (ms)": f"{metrics.get('avg', 0):.2f}",
                "Median (ms)": f"{metrics.get('median', 0):.2f}",
                "P95 (ms)": f"{metrics.get('p95', 0):.2f}",
                "P99 (ms)": f"{metrics.get('p99', 0):.2f}",
                "Max (ms)": f"{metrics.get('max', 0):.2f}"
            })
        
        df = pd.DataFrame(data)
        
        # Generate plots
        plots_html = ""
        for endpoint, metrics in self.results.items():
            response_times = metrics.get("response_times", [])
            if response_times:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(response_times, bins=50, alpha=0.7, color='blue')
                ax.set_title(f"Response Time Distribution for {endpoint}")
                ax.set_xlabel("Response Time (ms)")
                ax.set_ylabel("Frequency")
                
                # Save plot to a temporary file
                plot_path = f"plot_{endpoint.replace('/', '_')}.png"
                plt.savefig(plot_path)
                plt.close()
                
                plots_html += f"""
                <div class="plot">
                    <h3>{endpoint} Response Time Distribution</h3>
                    <img src="{plot_path}" alt="{endpoint} plot">
                </div>
                """
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .plot {{ margin-top: 30px; }}
                .summary {{ background-color: #e7f3fe; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Performance Test Report</h1>
            <div class="summary">
                <p><strong>Test Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Test Duration:</strong> {self.test_duration} seconds</p>
                <p><strong>API Base URL:</strong> {self.base_url}</p>
            </div>
            
            <h2>Results Summary</h2>
            {df.to_html(index=False)}
            
            <h2>Response Time Distributions</h2>
            {plots_html}
            
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(self.report_path, "w") as f:
            f.write(html)
            
        logger.info(f"Report generated at {self.report_path}")


class APIPerformanceTest(PerformanceTest):
    """Test the performance of various API endpoints."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 test_duration: int = 300,
                 report_path: str = "./api_performance_report.html",
                 auth_token: Optional[str] = None,
                 endpoints: Optional[List[str]] = None):
        """
        Initialize the API performance test.
        
        Args:
            base_url: Base URL of the API
            test_duration: Test duration in seconds
            report_path: Path to save the HTML report
            auth_token: Authentication token for API requests
            endpoints: List of API endpoints to test
        """
        super().__init__(base_url, test_duration, report_path)
        self.auth_token = auth_token
        
        # Default endpoints to test
        self.endpoints = endpoints or [
            "/health",
            "/api/v1/market/summary",
            "/api/v1/system/status",
            "/api/v1/strategies"
        ]
        
        # Initialize results dictionary
        for endpoint in self.endpoints:
            self.results[endpoint] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "response_times": []
            }
    
    def setup(self):
        """Set up the test."""
        super().setup()
        
        # Get auth token if needed
        if not self.auth_token:
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json={"username": "performance_test", "password": "test123"}
                )
                if response.status_code == 200:
                    self.auth_token = response.json().get("access_token")
                    logger.info("Successfully obtained auth token")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Could not obtain auth token: {e}")
    
    def execute_test_iteration(self):
        """Execute a test iteration by making API requests."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        for endpoint in self.endpoints:
            url = f"{self.base_url}{endpoint}"
            
            try:
                start_time = time.time()
                response = requests.get(url, headers=headers, timeout=10)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                self.results[endpoint]["requests"] += 1
                self.results[endpoint]["response_times"].append(response_time)
                
                if response.status_code < 400:
                    self.results[endpoint]["successes"] += 1
                else:
                    self.results[endpoint]["failures"] += 1
                    logger.warning(f"Request to {endpoint} failed with status {response.status_code}")
                
            except requests.exceptions.RequestException as e:
                self.results[endpoint]["requests"] += 1
                self.results[endpoint]["failures"] += 1
                logger.warning(f"Request to {endpoint} failed: {e}")
            
            # Calculate success rate
            requests = self.results[endpoint]["requests"]
            successes = self.results[endpoint]["successes"]
            self.results[endpoint]["success_rate"] = successes / requests if requests > 0 else 0
            
            # Sleep briefly between requests
            time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance tests for AI Trading Agent API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--report", default="./api_performance_report.html", help="Path to save the HTML report")
    parser.add_argument("--token", help="Authentication token for API requests")
    
    args = parser.parse_args()
    
    test = APIPerformanceTest(
        base_url=args.url,
        test_duration=args.duration,
        report_path=args.report,
        auth_token=args.token
    )
    
    test.run()
