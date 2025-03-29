"""Monitoring service for sentiment analysis system.

This module provides active monitoring of the sentiment analysis system,
including component health checks, performance tracking, and usage statistics.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple

import aiohttp

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.analysis_agents.sentiment.monitoring_alerts import (
    AlertSeverity, AlertType, alert_manager
)


class HealthCheckStatus:
    """Status values for health checks."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, interval_seconds: int = 60):
        """Initialize health check.
        
        Args:
            name: Name of the health check
            interval_seconds: How frequently to run the check in seconds
        """
        self.name = name
        self.interval_seconds = interval_seconds
        self.logger = get_logger("monitoring", f"health_check_{name}")
        self.last_check_time = datetime.min
        self.last_status = None
        self.last_details = {}
    
    async def check(self) -> Tuple[str, Dict[str, Any]]:
        """Run the health check.
        
        Returns:
            Tuple of (status, details)
        """
        raise NotImplementedError("Subclasses must implement check")
    
    async def run(self) -> Dict[str, Any]:
        """Run the health check if due.
        
        Returns:
            Health check result or None if not due
        """
        # Skip if not due
        now = datetime.utcnow()
        if (now - self.last_check_time).total_seconds() < self.interval_seconds:
            return None
        
        # Update last check time
        self.last_check_time = now
        
        try:
            # Run the check
            status, details = self.check()
            
            # Update last status and details
            self.last_status = status
            self.last_details = details
            
            # Create result
            result = {
                "name": self.name,
                "status": status,
                "timestamp": now.isoformat(),
                "details": details
            }
            
            # Log significant status changes
            if self.last_status != status:
                if status == HealthCheckStatus.HEALTHY:
                    self.logger.info(f"Health check {self.name} is now {status}")
                elif status == HealthCheckStatus.DEGRADED:
                    self.logger.warning(f"Health check {self.name} is now {status}: {details}")
                elif status == HealthCheckStatus.UNHEALTHY:
                    self.logger.error(f"Health check {self.name} is now {status}: {details}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running health check {self.name}: {str(e)}")
            
            # Consider the check unhealthy if it raised an exception
            self.last_status = HealthCheckStatus.UNHEALTHY
            self.last_details = {"error": str(e)}
            
            return {
                "name": self.name,
                "status": HealthCheckStatus.UNHEALTHY,
                "timestamp": now.isoformat(),
                "details": {"error": str(e)}
            }


class LLMApiHealthCheck(HealthCheck):
    """Health check for LLM APIs."""
    
    def __init__(
        self,
        provider: str,
        api_key_variable: str,
        test_endpoint: str,
        interval_seconds: int = 300
    ):
        """Initialize LLM API health check.
        
        Args:
            provider: LLM provider name (e.g., openai, anthropic)
            api_key_variable: Environment variable name for API key
            test_endpoint: Endpoint to test
            interval_seconds: Check interval in seconds
        """
        super().__init__(f"llm_api_{provider}", interval_seconds)
        self.provider = provider
        self.api_key_variable = api_key_variable
        self.test_endpoint = test_endpoint
        self.timeout_seconds = 10
    
    async def check(self) -> Tuple[str, Dict[str, Any]]:
        """Check LLM API health.
        
        Returns:
            Tuple of (status, details)
        """
        import os
        
        # Check if API key is configured
        api_key = os.environ.get(self.api_key_variable)
        if not api_key:
            return HealthCheckStatus.UNHEALTHY, {
                "error": f"API key environment variable {self.api_key_variable} not set"
            }
        
        # Setup headers based on provider
        if self.provider == "openai":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        elif self.provider == "anthropic":
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
        else:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.test_endpoint,
                    headers=headers,
                    timeout=self.timeout_seconds
                ) as response:
                    elapsed_time = time.time() - start_time
                    
                    # Check response status
                    if response.status < 300:
                        return HealthCheckStatus.HEALTHY, {
                            "latency_ms": int(elapsed_time * 1000),
                            "status_code": response.status
                        }
                    elif response.status < 500:
                        # Client error (4xx)
                        response_text = response.text()
                        return HealthCheckStatus.DEGRADED, {
                            "latency_ms": int(elapsed_time * 1000),
                            "status_code": response.status,
                            "response": response_text[:200]  # Limit response size
                        }
                    else:
                        # Server error (5xx)
                        response_text = response.text()
                        return HealthCheckStatus.UNHEALTHY, {
                            "latency_ms": int(elapsed_time * 1000),
                            "status_code": response.status,
                            "response": response_text[:200]  # Limit response size
                        }
        except asyncio.TimeoutError:
            return HealthCheckStatus.UNHEALTHY, {
                "error": f"Timeout after {self.timeout_seconds} seconds"
            }
        except Exception as e:
            return HealthCheckStatus.UNHEALTHY, {
                "error": str(e)
            }


class SentimentServiceHealthCheck(HealthCheck):
    """Health check for sentiment service components."""
    
    def __init__(self, service_name: str, check_fn: Callable[[], Tuple[bool, Dict[str, Any]]]):
        """Initialize sentiment service health check.
        
        Args:
            service_name: Name of the service to check
            check_fn: Function that checks the service health
        """
        super().__init__(f"sentiment_service_{service_name}")
        self.service_name = service_name
        self.check_fn = check_fn
    
    async def check(self) -> Tuple[str, Dict[str, Any]]:
        """Check sentiment service health.
        
        Returns:
            Tuple of (status, details)
        """
        try:
            is_healthy, details = self.check_fn()
            
            if is_healthy:
                return HealthCheckStatus.HEALTHY, details
            else:
                return HealthCheckStatus.UNHEALTHY, details
                
        except Exception as e:
            return HealthCheckStatus.UNHEALTHY, {
                "error": str(e)
            }


class FileSystemHealthCheck(HealthCheck):
    """Health check for file system."""
    
    def __init__(self, directory: str, min_free_space_mb: int = 100):
        """Initialize file system health check.
        
        Args:
            directory: Directory to check
            min_free_space_mb: Minimum free space in MB
        """
        super().__init__("file_system")
        self.directory = directory
        self.min_free_space_mb = min_free_space_mb
    
    async def check(self) -> Tuple[str, Dict[str, Any]]:
        """Check file system health.
        
        Returns:
            Tuple of (status, details)
        """
        import shutil
        
        try:
            # Get disk usage
            total, used, free = shutil.disk_usage(self.directory)
            
            # Convert to MB
            total_mb = total / (1024 * 1024)
            used_mb = used / (1024 * 1024)
            free_mb = free / (1024 * 1024)
            
            # Calculate usage percentage
            usage_pct = (used / total) * 100
            
            # Check if free space is below threshold
            if free_mb < self.min_free_space_mb:
                return HealthCheckStatus.UNHEALTHY, {
                    "directory": self.directory,
                    "total_mb": int(total_mb),
                    "used_mb": int(used_mb),
                    "free_mb": int(free_mb),
                    "usage_percent": round(usage_pct, 1),
                    "min_free_space_mb": self.min_free_space_mb
                }
            
            # Check if usage is high
            if usage_pct > 90:
                return HealthCheckStatus.DEGRADED, {
                    "directory": self.directory,
                    "total_mb": int(total_mb),
                    "used_mb": int(used_mb),
                    "free_mb": int(free_mb),
                    "usage_percent": round(usage_pct, 1)
                }
            
            return HealthCheckStatus.HEALTHY, {
                "directory": self.directory,
                "total_mb": int(total_mb),
                "used_mb": int(used_mb),
                "free_mb": int(free_mb),
                "usage_percent": round(usage_pct, 1)
            }
            
        except Exception as e:
            return HealthCheckStatus.UNHEALTHY, {
                "directory": self.directory,
                "error": str(e)
            }


class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory)."""
    
    def __init__(
        self,
        max_cpu_percent: float = 90.0,
        max_memory_percent: float = 90.0
    ):
        """Initialize system resource health check.
        
        Args:
            max_cpu_percent: Maximum healthy CPU percentage
            max_memory_percent: Maximum healthy memory percentage
        """
        super().__init__("system_resources")
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
    
    async def check(self) -> Tuple[str, Dict[str, Any]]:
        """Check system resource health.
        
        Returns:
            Tuple of (status, details)
        """
        try:
            import psutil
        except ImportError:
            return HealthCheckStatus.DEGRADED, {
                "error": "psutil not installed, cannot check system resources"
            }
        
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Check thresholds
            status = HealthCheckStatus.HEALTHY
            if cpu_percent > self.max_cpu_percent or memory_percent > self.max_memory_percent:
                status = HealthCheckStatus.DEGRADED
            
            if cpu_percent > self.max_cpu_percent * 1.1 or memory_percent > self.max_memory_percent * 1.1:
                status = HealthCheckStatus.UNHEALTHY
            
            return status, {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory_percent, 1),
                "memory_total_mb": int(memory.total / (1024 * 1024)),
                "memory_available_mb": int(memory.available / (1024 * 1024))
            }
            
        except Exception as e:
            return HealthCheckStatus.UNHEALTHY, {
                "error": str(e)
            }


class SentimentDataHealthCheck(HealthCheck):
    """Health check for sentiment data freshness."""
    
    def __init__(self, data_dir: str, max_age_hours: int = 24):
        """Initialize sentiment data health check.
        
        Args:
            data_dir: Directory containing sentiment data
            max_age_hours: Maximum acceptable age of data in hours
        """
        super().__init__("sentiment_data")
        self.data_dir = data_dir
        self.max_age_hours = max_age_hours
    
    async def check(self) -> Tuple[str, Dict[str, Any]]:
        """Check sentiment data health.
        
        Returns:
            Tuple of (status, details)
        """
        import os
        from datetime import datetime, timedelta
        
        try:
            # Check if directory exists
            if not os.path.exists(self.data_dir):
                return HealthCheckStatus.UNHEALTHY, {
                    "error": f"Directory {self.data_dir} does not exist"
                }
            
            # Get all JSON files in the directory
            data_files = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.endswith(".json")
            ]
            
            if not data_files:
                return HealthCheckStatus.DEGRADED, {
                    "error": f"No data files found in {self.data_dir}"
                }
            
            # Check modification times
            now = datetime.utcnow()
            max_age_delta = timedelta(hours=self.max_age_hours)
            
            file_ages = []
            stale_files = []
            
            for file_path in data_files:
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age = now - mtime
                    file_ages.append((os.path.basename(file_path), age.total_seconds() / 3600))
                    
                    if age > max_age_delta:
                        stale_files.append(os.path.basename(file_path))
                except Exception:
                    # Skip files with errors
                    pass
            
            # Determine status based on stale files
            if not file_ages:
                return HealthCheckStatus.UNHEALTHY, {
                    "error": f"Could not determine age of any files in {self.data_dir}"
                }
            
            if len(stale_files) > len(file_ages) / 2:
                # More than half of files are stale
                return HealthCheckStatus.UNHEALTHY, {
                    "stale_file_count": len(stale_files),
                    "total_file_count": len(file_ages),
                    "stale_files": stale_files[:5],  # Limit to 5 files
                    "max_age_hours": self.max_age_hours
                }
            
            if stale_files:
                # Some files are stale
                return HealthCheckStatus.DEGRADED, {
                    "stale_file_count": len(stale_files),
                    "total_file_count": len(file_ages),
                    "stale_files": stale_files[:5],  # Limit to 5 files
                    "max_age_hours": self.max_age_hours
                }
            
            # All files are fresh
            return HealthCheckStatus.HEALTHY, {
                "file_count": len(file_ages),
                "oldest_file_hours": round(max(age for _, age in file_ages), 1),
                "newest_file_hours": round(min(age for _, age in file_ages), 1)
            }
            
        except Exception as e:
            return HealthCheckStatus.UNHEALTHY, {
                "error": str(e)
            }


class ModelPerformanceCheck:
    """Monitor model performance metrics."""
    
    def __init__(self):
        """Initialize model performance check."""
        self.logger = get_logger("monitoring", "model_performance")
        self.baseline_performance: Dict[str, Dict[str, float]] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.history_file = os.path.join("data", "monitoring", "model_performance_history.json")
        
        # Create directory for performance history
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the model performance check."""
        # Load performance history
        self._load_history()
        
        # Subscribe to performance events
        event_bus.subscribe("sentiment_model_performance", self.handle_performance_event)
    
    async def _load_history(self) -> None:
        """Load performance history from file."""
        if not os.path.exists(self.history_file):
            self.logger.info("No performance history file found")
            return
        
        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)
            
            self.performance_history = data.get("history", {})
            self.baseline_performance = data.get("baselines", {})
            
            self.logger.info(f"Loaded performance history for {len(self.performance_history)} models")
            
        except Exception as e:
            self.logger.error(f"Failed to load performance history: {str(e)}")
    
    async def _save_history(self) -> None:
        """Save performance history to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # Save data
            with open(self.history_file, "w") as f:
                json.dump({
                    "baselines": self.baseline_performance,
                    "history": self.performance_history
                }, f, indent=2)
                
            self.logger.debug("Saved performance history")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance history: {str(e)}")
    
    async def handle_performance_event(self, event: Event) -> None:
        """Handle model performance event.
        
        Args:
            event: The performance event
        """
        data = event.data
        model = data.get("model", "unknown")
        metrics = data.get("metrics", {})
        
        if not metrics:
            return
        
        # Add timestamp to metrics
        metrics["timestamp"] = datetime.utcnow().isoformat()
        
        # Initialize history for this model if needed
        if model not in self.performance_history:
            self.performance_history[model] = []
        
        # Add metrics to history
        self.performance_history[model].append(metrics)
        
        # Limit history to 100 entries per model
        if len(self.performance_history[model]) > 100:
            self.performance_history[model] = self.performance_history[model][-100:]
        
        # Initialize baseline for this model if needed
        if model not in self.baseline_performance:
            self.baseline_performance[model] = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "timestamp"
            }
        
        # Save history
        self._save_history()
        
        # Check for performance degradation
        await self._check_performance(model, metrics)
    
    async def _check_performance(self, model: str, metrics: Dict[str, Any]) -> None:
        """Check for performance degradation.
        
        Args:
            model: Model name
            metrics: Current metrics
        """
        # Skip if no baseline
        if model not in self.baseline_performance:
            return
        
        baseline = self.baseline_performance[model]
        
        # Check each metric
        for metric, value in metrics.items():
            # Skip non-numeric metrics and timestamp
            if not isinstance(value, (int, float)) or metric == "timestamp":
                continue
            
            # Skip metrics not in baseline
            if metric not in baseline:
                continue
            
            baseline_value = baseline[metric]
            
            # Skip if baseline is zero
            if baseline_value == 0:
                continue
            
            # Calculate percentage change
            pct_change = ((value - baseline_value) / baseline_value) * 100
            
            # Only act on significant changes
            if abs(pct_change) < 5:
                continue
            
            # Check for degradation
            if pct_change < -10:
                # Publish performance degradation event
                event_bus.publish(
                    "model_performance",
                    {
                        "source": model,
                        "metric": metric,
                        "value": value,
                        "baseline": baseline_value,
                        "percent_change": pct_change
                    }
                )
            
            # Update baseline for significant improvements
            if pct_change > 20:
                self.logger.info(f"Updating baseline for {model}/{metric}: {baseline_value:.4f} -> {value:.4f}")
                self.baseline_performance[model][metric] = value
                self._save_history()


class APIUsageTracker:
    """Track API usage and costs."""
    
    def __init__(self):
        """Initialize API usage tracker."""
        self.logger = get_logger("monitoring", "api_usage")
        self.usage_data: Dict[str, Dict[str, Any]] = {}
        self.usage_file = os.path.join("data", "monitoring", "api_usage.json")
        
        # Per-model costs ($ per 1K tokens)
        self.model_costs = {
            "gpt-4o": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
        }
        
        # Rate limits (requests per minute)
        self.rate_limits = {
            "openai": config.get("monitoring.rate_limits.openai", 60),
            "anthropic": config.get("monitoring.rate_limits.anthropic", 30),
            "azure": config.get("monitoring.rate_limits.azure", 120)
        }
        
        # Request tracking for rate limiting
        self.requests: Dict[str, List[datetime]] = {}
        
        # Create directory for usage data
        os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the API usage tracker."""
        # Load usage data
        self._load_usage()
        
        # Subscribe to API usage events
        event_bus.subscribe("llm_api_usage", self.handle_api_usage)
    
    async def _load_usage(self) -> None:
        """Load usage data from file."""
        if not os.path.exists(self.usage_file):
            self.logger.info("No API usage file found")
            return
        
        try:
            with open(self.usage_file, "r") as f:
                self.usage_data = json.load(f)
            
            self.logger.info(f"Loaded API usage data for {len(self.usage_data)} providers")
            
        except Exception as e:
            self.logger.error(f"Failed to load API usage data: {str(e)}")
    
    async def _save_usage(self) -> None:
        """Save usage data to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
            
            # Save data
            with open(self.usage_file, "w") as f:
                json.dump(self.usage_data, f, indent=2)
                
            self.logger.debug("Saved API usage data")
            
        except Exception as e:
            self.logger.error(f"Failed to save API usage data: {str(e)}")
    
    async def handle_api_usage(self, event: Event) -> None:
        """Handle API usage event.
        
        Args:
            event: The API usage event
        """
        data = event.data
        provider = data.get("provider", "unknown")
        model = data.get("model", "unknown")
        input_tokens = data.get("input_tokens", 0)
        output_tokens = data.get("output_tokens", 0)
        operation = data.get("operation", "unknown")
        
        # Track request for rate limiting
        if provider not in self.requests:
            self.requests[provider] = []
        
        self.requests[provider].append(datetime.utcnow())
        
        # Check rate limit
        await self._check_rate_limit(provider)
        
        # Initialize usage data for this provider if needed
        if provider not in self.usage_data:
            self.usage_data[provider] = {
                "models": {},
                "daily": {},
                "total_cost": 0.0,
                "total_tokens": 0
            }
        
        # Initialize model data if needed
        if model not in self.usage_data[provider]["models"]:
            self.usage_data[provider]["models"][model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0
            }
        
        # Initialize daily data if needed
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today not in self.usage_data[provider]["daily"]:
            self.usage_data[provider]["daily"][today] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0
            }
        
        # Update model usage
        model_data = self.usage_data[provider]["models"][model]
        model_data["input_tokens"] += input_tokens
        model_data["output_tokens"] += output_tokens
        
        # Update daily usage
        daily_data = self.usage_data[provider]["daily"][today]
        daily_data["input_tokens"] += input_tokens
        daily_data["output_tokens"] += output_tokens
        
        # Calculate cost
        cost = 0.0
        if model in self.model_costs:
            input_cost = (input_tokens / 1000.0) * self.model_costs[model]["input"]
            output_cost = (output_tokens / 1000.0) * self.model_costs[model]["output"]
            cost = input_cost + output_cost
        
        # Update costs
        model_data["cost"] += cost
        daily_data["cost"] += cost
        self.usage_data[provider]["total_cost"] += cost
        self.usage_data[provider]["total_tokens"] += input_tokens + output_tokens
        
        # Save usage data
        self._save_usage()
        
        # Check cost thresholds
        await self._check_cost_thresholds(provider)
    
    async def _check_rate_limit(self, provider: str) -> None:
        """Check if rate limit is approaching.
        
        Args:
            provider: The API provider
        """
        if provider not in self.rate_limits:
            return
        
        limit = self.rate_limits[provider]
        
        # Count requests in the last minute
        minute_ago = datetime.utcnow() - timedelta(minutes=1)
        recent_requests = [
            req for req in self.requests[provider]
            if req >= minute_ago
        ]
        
        # Clean up old requests
        self.requests[provider] = recent_requests
        
        # Check if approaching limit
        rate_percent = (len(recent_requests) / limit) * 100
        
        if rate_percent > 80:
            # Publish rate limit event
            event_bus.publish(
                "rate_limit",
                {
                    "provider": provider,
                    "current_rate": len(recent_requests),
                    "limit": limit,
                    "percent": rate_percent
                }
            )
    
    async def _check_cost_thresholds(self, provider: str) -> None:
        """Check if cost thresholds are exceeded.
        
        Args:
            provider: The API provider
        """
        # Calculate daily and weekly costs
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily_cost = 0.0
        weekly_cost = 0.0
        
        if provider in self.usage_data and today in self.usage_data[provider]["daily"]:
            daily_cost = self.usage_data[provider]["daily"][today]["cost"]
        
        # Calculate weekly cost (last 7 days)
        if provider in self.usage_data:
            for date, data in self.usage_data[provider]["daily"].items():
                try:
                    date_obj = datetime.strptime(date, "%Y-%m-%d")
                    if date_obj >= datetime.utcnow() - timedelta(days=7):
                        weekly_cost += data["cost"]
                except ValueError:
                    # Skip invalid dates
                    pass
        
        # Publish cost event
        event_bus.publish(
            "sentiment_api_cost",
            {
                "provider": provider,
                "daily_cost": daily_cost,
                "weekly_cost": weekly_cost,
                "details": {
                    "total_cost": self.usage_data[provider]["total_cost"],
                    "total_tokens": self.usage_data[provider]["total_tokens"]
                }
            }
        )


class SentimentMonitoringService:
    """Monitoring service for sentiment analysis system."""
    
    def __init__(self):
        """Initialize monitoring service."""
        self.logger = get_logger("monitoring", "sentiment_service")
        self.health_checks: List[HealthCheck] = []
        self.is_running = False
        self.check_interval = config.get("monitoring.check_interval", 60)  # seconds
        self.status = HealthCheckStatus.HEALTHY
        
        # Initialize model performance monitor
        self.performance_monitor = ModelPerformanceCheck()
        
        # Initialize API usage tracker
        self.api_usage_tracker = APIUsageTracker()
    
    def _initialize_health_checks(self) -> None:
        """Initialize health checks."""
        # System checks
        self.health_checks.append(
            FileSystemHealthCheck("./data")
        )
        
        self.health_checks.append(
            SystemResourceHealthCheck()
        )
        
        # Sentiment data checks
        sentiment_data_dir = os.path.join("data", "sentiment")
        if os.path.exists(sentiment_data_dir):
            self.health_checks.append(
                SentimentDataHealthCheck(sentiment_data_dir)
            )
        
        # API health checks
        if os.environ.get("OPENAI_API_KEY"):
            self.health_checks.append(
                LLMApiHealthCheck(
                    provider="openai",
                    api_key_variable="OPENAI_API_KEY",
                    test_endpoint="https://api.openai.com/v1/models",
                    interval_seconds=300
                )
            )
        
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.health_checks.append(
                LLMApiHealthCheck(
                    provider="anthropic",
                    api_key_variable="ANTHROPIC_API_KEY",
                    test_endpoint="https://api.anthropic.com/v1/messages",
                    interval_seconds=300
                )
            )
        
        self.logger.info(f"Initialized {len(self.health_checks)} health checks")
    
    async def initialize(self) -> None:
        """Initialize the monitoring service."""
        self.logger.info("Initializing sentiment monitoring service")
        
        # Initialize alert manager
        alert_manager.initialize()
        
        # Initialize health checks
        self._initialize_health_checks()
        
        # Initialize performance monitor
        await self.performance_monitor.initialize()
        
        # Initialize API usage tracker
        await self.api_usage_tracker.initialize()
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Health check results
        """
        # Run all checks in parallel
        check_tasks = [check.run() for check in self.health_checks]
        results = await asyncio.gather(*check_tasks)
        
        # Filter out None results (checks not due yet)
        results = [result for result in results if result is not None]
        
        # Calculate overall status
        if any(result["status"] = = HealthCheckStatus.UNHEALTHY for result in results):
            overall_status = HealthCheckStatus.UNHEALTHY
        elif any(result["status"] = = HealthCheckStatus.DEGRADED for result in results):
            overall_status = HealthCheckStatus.DEGRADED
        else:
            overall_status = HealthCheckStatus.HEALTHY
        
        # Update service status
        self.status = overall_status
        
        # Publish health status event
        event_bus.publish(
            "sentiment_health_status",
            {
                "status": overall_status,
                "checks": results,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Create alerts for unhealthy checks if status changed
        for result in results:
            if result["status"] = = HealthCheckStatus.UNHEALTHY:
                await alert_manager.create_alert(
                    alert_type=AlertType.COMPONENT_FAILURE,
                    severity=AlertSeverity.ERROR,
                    source=f"health_check_{result['name']}",
                    message=f"Health check failed: {result['name']}",
                    details=result["details"]
                )
            elif result["status"] = = HealthCheckStatus.DEGRADED:
                await alert_manager.create_alert(
                    alert_type=AlertType.COMPONENT_FAILURE,
                    severity=AlertSeverity.WARNING,
                    source=f"health_check_{result['name']}",
                    message=f"Health check degraded: {result['name']}",
                    details=result["details"]
                )
        
        return {
            "status": overall_status,
            "checks": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Run health checks
                self.run_health_checks()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.check_interval)
    
    async def start(self) -> None:
        """Start the monitoring service."""
        if self.is_running:
            return
        
        self.logger.info("Starting sentiment monitoring service")
        self.is_running = True
        
        # Start monitoring loop
        asyncio.create_task(self.monitoring_loop())
    
    async def stop(self) -> None:
        """Stop the monitoring service."""
        self.logger.info("Stopping sentiment monitoring service")
        self.is_running = False


# Singleton instance
monitoring_service = SentimentMonitoringService()