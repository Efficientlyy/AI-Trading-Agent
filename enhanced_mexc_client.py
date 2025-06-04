#!/usr/bin/env python
"""
Enhanced Error Handling and Logging for MEXC API Client

This module implements detailed logging and comprehensive error handling
for the optimized MEXC API client to ensure robust operation.
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mexc_api_detailed.log")
    ]
)

# Create a separate error logger that writes to a dedicated file
error_logger = logging.getLogger("mexc_api_errors")
error_handler = logging.FileHandler("mexc_api_errors.log")
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)
error_logger.propagate = False  # Don't propagate to root logger

# Create a performance logger for tracking API performance
perf_logger = logging.getLogger("mexc_api_performance")
perf_handler = logging.FileHandler("mexc_api_performance.log")
perf_formatter = logging.Formatter('%(asctime)s - %(message)s')
perf_handler.setFormatter(perf_formatter)
perf_logger.addHandler(perf_handler)
perf_logger.propagate = False  # Don't propagate to root logger

# Main logger
logger = logging.getLogger("mexc_api_enhanced")

# Import optimized MEXC client
try:
    from optimized_mexc_client import OptimizedMEXCClient, RateLimiter
except ImportError as e:
    logger.error(f"Error importing OptimizedMEXCClient: {str(e)}")
    sys.exit(1)

class MEXCAPIError(Exception):
    """Custom exception for MEXC API errors"""
    
    def __init__(self, status_code, message, response_body=None, endpoint=None, params=None):
        """Initialize MEXC API error
        
        Args:
            status_code: HTTP status code
            message: Error message
            response_body: Full response body (optional)
            endpoint: API endpoint (optional)
            params: Request parameters (optional)
        """
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        self.endpoint = endpoint
        self.params = params
        
        # Create detailed error message
        detailed_message = f"MEXC API Error {status_code}: {message}"
        if endpoint:
            detailed_message += f" (Endpoint: {endpoint})"
        
        super().__init__(detailed_message)

class EnhancedMEXCClient(OptimizedMEXCClient):
    """Enhanced MEXC client with detailed logging and error handling"""
    
    def __init__(self, api_key=None, api_secret=None, env_path=None):
        """Initialize enhanced MEXC client
        
        Args:
            api_key: MEXC API key (optional, will load from env if not provided)
            api_secret: MEXC API secret (optional, will load from env if not provided)
            env_path: Path to .env file (optional)
        """
        # Initialize base client
        super().__init__(api_key, api_secret, env_path)
        
        # Enhanced tracking
        self.request_history = []
        self.error_history = []
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_count": 0,
            "total_response_time": 0,
            "avg_response_time": 0,
            "max_response_time": 0,
            "min_response_time": float('inf'),
            "rate_limit_hits": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Maximum history size
        self.max_history_size = 1000
        
        # Error categorization
        self.error_categories = {
            "rate_limit": 0,
            "authentication": 0,
            "validation": 0,
            "server": 0,
            "network": 0,
            "timeout": 0,
            "unknown": 0
        }
        
        logger.info("Enhanced MEXC client initialized with detailed logging and error handling")
    
    def _categorize_error(self, status_code, error_message):
        """Categorize error based on status code and message
        
        Args:
            status_code: HTTP status code
            error_message: Error message
            
        Returns:
            str: Error category
        """
        if status_code == 429:
            return "rate_limit"
        elif status_code == 401 or status_code == 403:
            return "authentication"
        elif status_code == 400 or status_code == 404:
            return "validation"
        elif status_code >= 500:
            return "server"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "network" in error_message.lower() or "connection" in error_message.lower():
            return "network"
        else:
            return "unknown"
    
    def _log_request(self, category, method, endpoint, params, start_time, response=None, error=None):
        """Log request details and update metrics
        
        Args:
            category: Request category
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            start_time: Request start time
            response: Response object (optional)
            error: Error details (optional)
        """
        # Calculate response time
        response_time = time.time() - start_time
        
        # Update performance metrics
        with self.lock:
            self.performance_metrics["total_requests"] += 1
            self.performance_metrics["total_response_time"] += response_time
            self.performance_metrics["avg_response_time"] = (
                self.performance_metrics["total_response_time"] / 
                self.performance_metrics["total_requests"]
            )
            self.performance_metrics["max_response_time"] = max(
                self.performance_metrics["max_response_time"], 
                response_time
            )
            self.performance_metrics["min_response_time"] = min(
                self.performance_metrics["min_response_time"], 
                response_time
            )
            self.performance_metrics["last_updated"] = datetime.now().isoformat()
            
            if error:
                self.performance_metrics["failed_requests"] += 1
                
                # Categorize and count error
                if isinstance(error, dict) and "status_code" in error:
                    error_category = self._categorize_error(error["status_code"], error.get("message", ""))
                    self.error_categories[error_category] += 1
                    
                    # Track rate limit hits
                    if error_category == "rate_limit":
                        self.performance_metrics["rate_limit_hits"] += 1
            else:
                self.performance_metrics["successful_requests"] += 1
        
        # Log request details
        request_details = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "method": method,
            "endpoint": endpoint,
            "params": self._sanitize_params(params),
            "response_time": response_time,
            "success": error is None,
            "status_code": response.status_code if response else None,
            "error": error
        }
        
        # Add to request history
        with self.lock:
            self.request_history.append(request_details)
            
            # Trim history if needed
            if len(self.request_history) > self.max_history_size:
                self.request_history = self.request_history[-self.max_history_size:]
        
        # Log to performance logger
        perf_logger.info(
            f"{category},{method},{endpoint},{response_time:.4f}," +
            f"{error is None},{response.status_code if response else 0}"
        )
        
        # Log detailed error if present
        if error:
            with self.lock:
                self.error_history.append(request_details)
                
                # Trim error history if needed
                if len(self.error_history) > self.max_history_size:
                    self.error_history = self.error_history[-self.max_history_size:]
            
            # Log to error logger
            error_message = f"API Error: {method} {endpoint} failed"
            if isinstance(error, dict):
                if "status_code" in error:
                    error_message += f" with status {error['status_code']}"
                if "message" in error:
                    error_message += f": {error['message']}"
            else:
                error_message += f": {error}"
            
            error_logger.error(error_message)
    
    def _sanitize_params(self, params):
        """Sanitize parameters to remove sensitive information
        
        Args:
            params: Request parameters
            
        Returns:
            dict: Sanitized parameters
        """
        if not params:
            return params
        
        # Create a copy to avoid modifying the original
        sanitized = params.copy()
        
        # Remove sensitive fields
        sensitive_fields = ["signature", "api_key", "api_secret", "apiKey", "apiSecret"]
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***REDACTED***"
        
        return sanitized
    
    def _queue_request(self, category, method, endpoint, params=None, headers=None, authenticated=False):
        """Queue a request with enhanced logging and error handling
        
        Args:
            category: Request category for rate limiting
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            headers: Request headers
            authenticated: Whether request requires authentication
            
        Returns:
            requests.Response: Response object
            
        Raises:
            MEXCAPIError: On API error
            Exception: On other errors
        """
        start_time = time.time()
        
        try:
            # Call base implementation
            response = super()._queue_request(
                category=category,
                method=method,
                endpoint=endpoint,
                params=params,
                headers=headers,
                authenticated=authenticated
            )
            
            # Log successful request
            self._log_request(
                category=category,
                method=method,
                endpoint=endpoint,
                params=params,
                start_time=start_time,
                response=response
            )
            
            return response
            
        except Exception as e:
            # Extract error details
            error_details = {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            
            # Add status code if available
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                error_details["status_code"] = e.response.status_code
                
                # Try to parse response body
                try:
                    error_details["response_body"] = e.response.json()
                except:
                    error_details["response_body"] = e.response.text
            
            # Log failed request
            self._log_request(
                category=category,
                method=method,
                endpoint=endpoint,
                params=params,
                start_time=start_time,
                error=error_details
            )
            
            # Re-raise as MEXCAPIError if it's an API error
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                status_code = e.response.status_code
                message = str(e)
                
                # Try to extract error message from response
                try:
                    response_body = e.response.json()
                    if "msg" in response_body:
                        message = response_body["msg"]
                    elif "message" in response_body:
                        message = response_body["message"]
                except:
                    response_body = e.response.text
                
                raise MEXCAPIError(
                    status_code=status_code,
                    message=message,
                    response_body=response_body,
                    endpoint=endpoint,
                    params=self._sanitize_params(params)
                ) from e
            
            # Re-raise original exception
            raise
    
    def get_performance_metrics(self):
        """Get performance metrics
        
        Returns:
            dict: Performance metrics
        """
        with self.lock:
            # Create a copy to avoid modification during access
            metrics = self.performance_metrics.copy()
            metrics["error_categories"] = self.error_categories.copy()
            
            # Add success rate
            total = metrics["total_requests"]
            if total > 0:
                metrics["success_rate"] = (metrics["successful_requests"] / total) * 100
            else:
                metrics["success_rate"] = 0
            
            return metrics
    
    def get_error_history(self, limit=10):
        """Get recent error history
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            list: Recent errors
        """
        with self.lock:
            return self.error_history[-limit:]
    
    def get_request_history(self, limit=10):
        """Get recent request history
        
        Args:
            limit: Maximum number of requests to return
            
        Returns:
            list: Recent requests
        """
        with self.lock:
            return self.request_history[-limit:]
    
    def get_detailed_status(self):
        """Get detailed client status
        
        Returns:
            dict: Detailed client status
        """
        # Get base status
        status = self.get_status()
        
        # Add enhanced metrics
        status["performance_metrics"] = self.get_performance_metrics()
        status["recent_errors"] = self.get_error_history(5)
        
        return status
    
    def export_metrics(self, filepath):
        """Export performance metrics to file
        
        Args:
            filepath: Path to export file
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            metrics = self.get_performance_metrics()
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Performance metrics exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting performance metrics: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = EnhancedMEXCClient()
    
    # Test with BTCUSDC
    symbol = "BTCUSDC"
    
    # Get ticker
    try:
        ticker = client.get_ticker(symbol)
        print(f"Ticker: {ticker}")
    except MEXCAPIError as e:
        print(f"API Error: {e.status_code} - {e.message}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get klines
    try:
        klines = client.get_klines(symbol, interval="5m", limit=10)
        print(f"Klines: {len(klines)} candles")
        if klines:
            print(f"First candle: {klines[0]}")
            print(f"Last candle: {klines[-1]}")
    except MEXCAPIError as e:
        print(f"API Error: {e.status_code} - {e.message}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get detailed status
    status = client.get_detailed_status()
    print(f"Performance metrics: {status['performance_metrics']}")
    
    # Export metrics
    client.export_metrics("mexc_performance_metrics.json")
