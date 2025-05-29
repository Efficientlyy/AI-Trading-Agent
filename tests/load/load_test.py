#!/usr/bin/env python
"""
Load testing framework for AI Trading Agent.

This module tests the system's ability to handle high concurrent user loads
and evaluates performance under stress conditions.
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
import threading
import multiprocessing
import queue
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoadTest:
    """Base class for load tests."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 test_duration: int = 300,
                 report_path: str = "./load_test_report.html",
                 users: int = 50,
                 ramp_up_time: int = 60):
        """
        Initialize the load test.
        
        Args:
            base_url: Base URL of the API
            test_duration: Test duration in seconds
            report_path: Path to save the HTML report
            users: Number of concurrent users to simulate
            ramp_up_time: Time in seconds to ramp up to full user load
        """
        self.base_url = base_url
        self.test_duration = test_duration
        self.report_path = report_path
        self.users = users
        self.ramp_up_time = ramp_up_time
        
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.user_threads = []
        self.stop_event = threading.Event()
        self.results_queue = multiprocessing.Queue()
        
    def setup(self):
        """Set up the test."""
        logger.info(f"Setting up load test: {self.__class__.__name__}")
        
        # Check if the API is accessible
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            logger.info("API is accessible")
        except requests.exceptions.RequestException as e:
            logger.error(f"API is not accessible: {e}")
            raise
            
        self.start_time = time.time()
        self.end_time = self.start_time + self.test_duration
        
    def run(self):
        """Run the load test."""
        self.setup()
        
        logger.info(f"Running load test with {self.users} users for {self.test_duration} seconds")
        logger.info(f"Ramping up to full load over {self.ramp_up_time} seconds")
        
        try:
            # Start user threads with ramp-up
            sleep_time = self.ramp_up_time / self.users if self.users > 0 else 0
            
            for i in range(self.users):
                thread = threading.Thread(
                    target=self.user_thread_func,
                    args=(i,),
                    daemon=True
                )
                self.user_threads.append(thread)
                thread.start()
                
                if i < self.users - 1:  # Don't sleep after the last user starts
                    time.sleep(sleep_time)
            
            # Wait until test duration completes
            remaining = self.end_time - time.time()
            while remaining > 0:
                logger.info(f"Test running. {remaining:.1f} seconds remaining...")
                time.sleep(min(10, remaining))
                remaining = self.end_time - time.time()
                
            # Signal threads to stop
            logger.info("Test duration complete, stopping user threads...")
            self.stop_event.set()
            
            # Wait for threads to finish (with a timeout)
            for thread in self.user_threads:
                thread.join(timeout=5)
                
            # Collect all results from the queue
            self.collect_results()
                
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            self.stop_event.set()
        except Exception as e:
            logger.exception(f"Error during test: {e}")
            self.stop_event.set()
        finally:
            self.analyze_results()
            self.generate_report()
    
    def user_thread_func(self, user_id: int):
        """Function executed by each user thread."""
        logger.info(f"User {user_id} started")
        
        user_results = {}
        request_count = 0
        
        try:
            while not self.stop_event.is_set() and time.time() < self.end_time:
                # Execute user actions with random think time
                endpoint, result = self.execute_user_action(user_id)
                
                if endpoint not in user_results:
                    user_results[endpoint] = {
                        "requests": 0,
                        "successes": 0,
                        "failures": 0,
                        "response_times": []
                    }
                
                user_results[endpoint]["requests"] += 1
                
                if result.get("success", False):
                    user_results[endpoint]["successes"] += 1
                    user_results[endpoint]["response_times"].append(result.get("response_time", 0))
                else:
                    user_results[endpoint]["failures"] += 1
                
                # Simulate user "think time"
                think_time = random.uniform(1, 5)
                request_count += 1
                
                # Check if we should stop more frequently during think time
                for _ in range(int(think_time * 10)):
                    if self.stop_event.is_set() or time.time() >= self.end_time:
                        break
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.exception(f"Error in user thread {user_id}: {e}")
        finally:
            # Put the results in the queue
            logger.info(f"User {user_id} completed with {request_count} requests")
            self.results_queue.put(user_results)
    
    def execute_user_action(self, user_id: int) -> tuple:
        """
        Execute a single user action.
        
        Args:
            user_id: ID of the user thread
            
        Returns:
            tuple: (endpoint, result_dict)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def collect_results(self):
        """Collect results from all user threads."""
        logger.info("Collecting results from all users...")
        
        # Get all results from the queue
        while not self.results_queue.empty():
            user_result = self.results_queue.get(block=False)
            
            # Merge into main results
            for endpoint, metrics in user_result.items():
                if endpoint not in self.results:
                    self.results[endpoint] = {
                        "requests": 0,
                        "successes": 0,
                        "failures": 0,
                        "response_times": []
                    }
                
                self.results[endpoint]["requests"] += metrics["requests"]
                self.results[endpoint]["successes"] += metrics["successes"]
                self.results[endpoint]["failures"] += metrics["failures"]
                self.results[endpoint]["response_times"].extend(metrics["response_times"])
    
    def analyze_results(self):
        """Analyze the test results."""
        logger.info("Analyzing test results")
        
        # Initialize summary metrics
        total_requests = 0
        total_failures = 0
        
        for endpoint, metrics in self.results.items():
            response_times = metrics.get("response_times", [])
            requests = metrics.get("requests", 0)
            failures = metrics.get("failures", 0)
            successes = metrics.get("successes", 0)
            
            total_requests += requests
            total_failures += failures
            
            if response_times:
                metrics["min"] = min(response_times)
                metrics["max"] = max(response_times)
                metrics["avg"] = statistics.mean(response_times)
                metrics["median"] = statistics.median(response_times)
                metrics["p95"] = np.percentile(response_times, 95)
                metrics["p99"] = np.percentile(response_times, 99)
                metrics["success_rate"] = successes / requests if requests > 0 else 0
                metrics["throughput"] = requests / self.test_duration
                
                logger.info(f"Endpoint: {endpoint}")
                logger.info(f"  Requests: {requests}")
                logger.info(f"  Success rate: {metrics['success_rate']:.2%}")
                logger.info(f"  Throughput: {metrics['throughput']:.2f} req/sec")
                logger.info(f"  Avg response time: {metrics['avg']:.2f} ms")
                logger.info(f"  95th percentile: {metrics['p95']:.2f} ms")
                logger.info(f"  99th percentile: {metrics['p99']:.2f} ms")
        
        # Calculate overall stats
        test_duration = time.time() - self.start_time
        requests_per_second = total_requests / test_duration
        error_rate = total_failures / total_requests if total_requests > 0 else 0
        
        logger.info("Overall Statistics:")
        logger.info(f"  Total duration: {test_duration:.2f} seconds")
        logger.info(f"  Total requests: {total_requests}")
        logger.info(f"  Requests per second: {requests_per_second:.2f}")
        logger.info(f"  Error rate: {error_rate:.2%}")
        
        # Store overall stats
        self.overall_stats = {
            "duration": test_duration,
            "total_requests": total_requests,
            "requests_per_second": requests_per_second,
            "error_rate": error_rate
        }
    
    def generate_report(self):
        """Generate an HTML report with the results."""
        logger.info(f"Generating load test report: {self.report_path}")
        
        # Create report dataframe
        data = []
        for endpoint, metrics in self.results.items():
            data.append({
                "Endpoint": endpoint,
                "Requests": metrics.get("requests", 0),
                "Success Rate": f"{metrics.get('success_rate', 0):.2%}",
                "Throughput (req/s)": f"{metrics.get('throughput', 0):.2f}",
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
                plot_path = f"load_plot_{endpoint.replace('/', '_')}.png"
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
            <title>Load Test Report</title>
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
            <h1>Load Test Report</h1>
            <div class="summary">
                <p><strong>Test Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Test Duration:</strong> {self.overall_stats['duration']:.2f} seconds</p>
                <p><strong>API Base URL:</strong> {self.base_url}</p>
                <p><strong>Simulated Users:</strong> {self.users}</p>
                <p><strong>Total Requests:</strong> {self.overall_stats['total_requests']}</p>
                <p><strong>Requests Per Second:</strong> {self.overall_stats['requests_per_second']:.2f}</p>
                <p><strong>Error Rate:</strong> {self.overall_stats['error_rate']:.2%}</p>
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


class APILoadTest(LoadTest):
    """Test the API under load with multiple users."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 test_duration: int = 300,
                 report_path: str = "./api_load_test_report.html",
                 users: int = 50,
                 ramp_up_time: int = 60,
                 auth_token: Optional[str] = None,
                 endpoints: Optional[List[str]] = None):
        """
        Initialize the API load test.
        
        Args:
            base_url: Base URL of the API
            test_duration: Test duration in seconds
            report_path: Path to save the HTML report
            users: Number of concurrent users to simulate
            ramp_up_time: Time in seconds to ramp up to full user load
            auth_token: Authentication token for API requests
            endpoints: List of API endpoints to test
        """
        super().__init__(base_url, test_duration, report_path, users, ramp_up_time)
        self.auth_tokens = {}  # Dict to store tokens by user_id
        self.default_token = auth_token
        
        # Default endpoints to test
        self.endpoints = endpoints or [
            "/health",
            "/api/v1/market/summary",
            "/api/v1/system/status",
            "/api/v1/strategies"
        ]
    
    def setup(self):
        """Set up the test."""
        super().setup()
        
        # Pre-authenticate with default token if provided
        if self.default_token:
            for i in range(self.users):
                self.auth_tokens[i] = self.default_token
    
    def execute_user_action(self, user_id: int) -> tuple:
        """
        Execute a user action by making an API request.
        
        Args:
            user_id: ID of the user thread
            
        Returns:
            tuple: (endpoint, result_dict)
        """
        # Get auth token for this user
        token = self.auth_tokens.get(user_id)
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        # Select random endpoint to test
        endpoint = random.choice(self.endpoints)
        url = f"{self.base_url}{endpoint}"
        
        result = {
            "success": False,
            "response_time": 0,
            "status_code": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=10)
            end_time = time.time()
            
            result["response_time"] = (end_time - start_time) * 1000  # Convert to ms
            result["status_code"] = response.status_code
            
            if response.status_code < 400:
                result["success"] = True
                
                # If this was a login endpoint, store the token
                if "auth/login" in endpoint and response.status_code == 200:
                    try:
                        token = response.json().get("access_token")
                        if token:
                            self.auth_tokens[user_id] = token
                    except:
                        pass
            else:
                result["error"] = f"HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            result["error"] = str(e)
        
        return endpoint, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run load tests for AI Trading Agent API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--report", default="./api_load_test_report.html", help="Path to save the HTML report")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent users to simulate")
    parser.add_argument("--ramp-up", type=int, default=60, help="Time in seconds to ramp up to full user load")
    parser.add_argument("--token", help="Authentication token for API requests")
    
    args = parser.parse_args()
    
    test = APILoadTest(
        base_url=args.url,
        test_duration=args.duration,
        report_path=args.report,
        users=args.users,
        ramp_up_time=args.ramp_up,
        auth_token=args.token
    )
    
    test.run()
