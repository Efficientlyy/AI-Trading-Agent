"""
Health monitoring for External API Gateway.

This module implements health monitoring capabilities for the External API Gateway,
tracking system metrics, endpoint performance, and alerting on issues.
"""
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import json
import statistics
from enum import Enum
import socket
import platform
import os
import psutil

# Setup logging
logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Status values for health checks."""
    OK = "ok"
    DEGRADED = "degraded"
    FAILING = "failing"
    UNKNOWN = "unknown"


class HealthCheckType(str, Enum):
    """Types of health checks."""
    PING = "ping"
    DATABASE = "database"
    DEPENDENCY = "dependency"
    RATE_LIMITER = "rate_limiter"
    AUTHENTICATION = "authentication"
    QUOTA = "quota"
    ENDPOINT = "endpoint"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"


class HealthMonitor:
    """
    Health monitoring for External API Gateway.
    
    Tracks system metrics, endpoint performance, and provides
    alerts for potential issues.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize health monitor.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Store health check results
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        
        # Endpoint performance tracking
        # {endpoint: {"count": int, "total_time": float, "success": int, "error": int, "response_times": List[float]}}
        self.endpoint_stats: Dict[str, Dict[str, Any]] = {}
        
        # System metrics history (keep recent history for trending)
        self.system_metrics_history: List[Dict[str, Any]] = []
        
        # Maximum history length
        self.max_history_length = self.config.get("max_history_length", 100)
        
        # Alert thresholds
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "response_time_ms": 1000,  # Alert if response time > 1s
            "error_rate_percent": 5,   # Alert if error rate > 5%
            "cpu_percent": 80,         # Alert if CPU > 80%
            "memory_percent": 85,      # Alert if memory > 85%
            "disk_percent": 90         # Alert if disk > 90%
        })
        
        # Alert handlers
        self.alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Initialize monitoring task
        self.monitoring_task = None
        self.monitoring_interval = self.config.get("monitoring_interval", 60)  # seconds
        
        logger.info("Health monitor initialized")
    
    async def start_monitoring(self):
        """Start the background monitoring task."""
        if self.monitoring_task is not None:
            logger.warning("Monitoring task already running")
            return
        
        logger.info("Starting health monitoring task")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop the background monitoring task."""
        if self.monitoring_task is None:
            logger.warning("No monitoring task running")
            return
        
        logger.info("Stopping health monitoring task")
        self.monitoring_task.cancel()
        try:
            await self.monitoring_task
        except asyncio.CancelledError:
            pass
        self.monitoring_task = None
    
    async def _monitoring_loop(self):
        """Background task for periodic health checks."""
        try:
            while True:
                try:
                    await self.collect_system_metrics()
                    await self.run_health_checks()
                    self._process_alerts()
                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {str(e)}")
                
                await asyncio.sleep(self.monitoring_interval)
        except asyncio.CancelledError:
            logger.info("Health monitoring task cancelled")
            raise
    
    async def collect_system_metrics(self):
        """Collect current system metrics."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "cores": psutil.cpu_count(),
                "logical_cores": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            },
            "network": {
                "connections": len(psutil.net_connections())
            },
            "system": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "uptime": time.time() - psutil.boot_time()
            },
            "process": {
                "pid": os.getpid(),
                "memory_percent": psutil.Process(os.getpid()).memory_percent(),
                "cpu_percent": psutil.Process(os.getpid()).cpu_percent(interval=0.5),
                "threads": psutil.Process(os.getpid()).num_threads(),
                "open_files": len(psutil.Process(os.getpid()).open_files()),
                "connections": len(psutil.Process(os.getpid()).connections())
            }
        }
        
        # Add to history and maintain max length
        self.system_metrics_history.append(metrics)
        if len(self.system_metrics_history) > self.max_history_length:
            self.system_metrics_history = self.system_metrics_history[-self.max_history_length:]
        
        return metrics
    
    async def run_health_checks(self):
        """Run all configured health checks."""
        # Example health checks - these would be more sophisticated in production
        checks = [
            self._check_memory(),
            self._check_cpu(),
            self._check_disk(),
            self._check_endpoint_performance()
        ]
        
        # Add additional checks from configuration
        for check_name, check_func in self.config.get("additional_checks", {}).items():
            try:
                result = await check_func()
                checks.append(result)
            except Exception as e:
                logger.error(f"Error running health check {check_name}: {str(e)}")
                checks.append({
                    "name": check_name,
                    "type": HealthCheckType.UNKNOWN,
                    "status": HealthStatus.UNKNOWN,
                    "message": f"Error running check: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Update stored health checks
        for check in checks:
            self.health_checks[check["name"]] = check
        
        return checks
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        if not self.system_metrics_history:
            return {
                "name": "memory",
                "type": HealthCheckType.MEMORY,
                "status": HealthStatus.UNKNOWN,
                "message": "No system metrics available",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        current = self.system_metrics_history[-1]["memory"]
        threshold = self.alert_thresholds.get("memory_percent", 85)
        
        if current["percent"] > threshold:
            status = HealthStatus.FAILING
            message = f"Memory usage is critical: {current['percent']}% > {threshold}%"
        elif current["percent"] > threshold * 0.8:  # Warning at 80% of threshold
            status = HealthStatus.DEGRADED
            message = f"Memory usage is high: {current['percent']}%"
        else:
            status = HealthStatus.OK
            message = f"Memory usage is normal: {current['percent']}%"
        
        return {
            "name": "memory",
            "type": HealthCheckType.MEMORY,
            "status": status,
            "message": message,
            "metrics": {
                "total_gb": round(current["total"] / (1024**3), 2),
                "available_gb": round(current["available"] / (1024**3), 2),
                "used_percent": current["percent"]
            },
            "threshold": threshold,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check system CPU usage."""
        if not self.system_metrics_history:
            return {
                "name": "cpu",
                "type": HealthCheckType.CPU,
                "status": HealthStatus.UNKNOWN,
                "message": "No system metrics available",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        current = self.system_metrics_history[-1]["cpu"]
        threshold = self.alert_thresholds.get("cpu_percent", 80)
        
        if current["percent"] > threshold:
            status = HealthStatus.FAILING
            message = f"CPU usage is critical: {current['percent']}% > {threshold}%"
        elif current["percent"] > threshold * 0.8:  # Warning at 80% of threshold
            status = HealthStatus.DEGRADED
            message = f"CPU usage is high: {current['percent']}%"
        else:
            status = HealthStatus.OK
            message = f"CPU usage is normal: {current['percent']}%"
        
        return {
            "name": "cpu",
            "type": HealthCheckType.CPU,
            "status": status,
            "message": message,
            "metrics": {
                "percent": current["percent"],
                "cores": current["cores"],
                "logical_cores": current["logical_cores"]
            },
            "threshold": threshold,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check system disk usage."""
        if not self.system_metrics_history:
            return {
                "name": "disk",
                "type": HealthCheckType.DISK,
                "status": HealthStatus.UNKNOWN,
                "message": "No system metrics available",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        current = self.system_metrics_history[-1]["disk"]
        threshold = self.alert_thresholds.get("disk_percent", 90)
        
        if current["percent"] > threshold:
            status = HealthStatus.FAILING
            message = f"Disk usage is critical: {current['percent']}% > {threshold}%"
        elif current["percent"] > threshold * 0.8:  # Warning at 80% of threshold
            status = HealthStatus.DEGRADED
            message = f"Disk usage is high: {current['percent']}%"
        else:
            status = HealthStatus.OK
            message = f"Disk usage is normal: {current['percent']}%"
        
        return {
            "name": "disk",
            "type": HealthCheckType.DISK,
            "status": status,
            "message": message,
            "metrics": {
                "total_gb": round(current["total"] / (1024**3), 2),
                "free_gb": round(current["free"] / (1024**3), 2),
                "used_percent": current["percent"]
            },
            "threshold": threshold,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _check_endpoint_performance(self) -> Dict[str, Any]:
        """Check API endpoint performance."""
        if not self.endpoint_stats:
            return {
                "name": "endpoint_performance",
                "type": HealthCheckType.ENDPOINT,
                "status": HealthStatus.UNKNOWN,
                "message": "No endpoint statistics available",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Calculate overall statistics
        total_requests = sum(stats.get("count", 0) for stats in self.endpoint_stats.values())
        total_errors = sum(stats.get("error", 0) for stats in self.endpoint_stats.values())
        
        if total_requests == 0:
            error_rate = 0
        else:
            error_rate = (total_errors / total_requests) * 100
        
        # Get response times across all endpoints
        all_response_times = []
        for stats in self.endpoint_stats.values():
            all_response_times.extend(stats.get("response_times", []))
        
        if not all_response_times:
            avg_response_time = 0
            p95_response_time = 0
            max_response_time = 0
        else:
            avg_response_time = statistics.mean(all_response_times)
            all_response_times.sort()
            p95_index = int(len(all_response_times) * 0.95)
            p95_response_time = all_response_times[p95_index]
            max_response_time = max(all_response_times)
        
        # Check against thresholds
        response_time_threshold = self.alert_thresholds.get("response_time_ms", 1000)
        error_rate_threshold = self.alert_thresholds.get("error_rate_percent", 5)
        
        if error_rate > error_rate_threshold or p95_response_time > response_time_threshold:
            status = HealthStatus.FAILING
            issues = []
            if error_rate > error_rate_threshold:
                issues.append(f"Error rate {error_rate:.2f}% > {error_rate_threshold}%")
            if p95_response_time > response_time_threshold:
                issues.append(f"P95 response time {p95_response_time:.2f}ms > {response_time_threshold}ms")
            message = "API performance issues: " + ", ".join(issues)
        elif error_rate > error_rate_threshold * 0.5 or p95_response_time > response_time_threshold * 0.8:
            status = HealthStatus.DEGRADED
            message = f"API performance degraded: Error rate {error_rate:.2f}%, P95 response time {p95_response_time:.2f}ms"
        else:
            status = HealthStatus.OK
            message = f"API performance normal: Error rate {error_rate:.2f}%, Avg response time {avg_response_time:.2f}ms"
        
        return {
            "name": "endpoint_performance",
            "type": HealthCheckType.ENDPOINT,
            "status": status,
            "message": message,
            "metrics": {
                "total_requests": total_requests,
                "error_rate_percent": round(error_rate, 2),
                "avg_response_time_ms": round(avg_response_time, 2),
                "p95_response_time_ms": round(p95_response_time, 2),
                "max_response_time_ms": round(max_response_time, 2)
            },
            "thresholds": {
                "response_time_ms": response_time_threshold,
                "error_rate_percent": error_rate_threshold
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _process_alerts(self):
        """Process alerts for failing health checks."""
        alerts = []
        
        for check_name, check in self.health_checks.items():
            if check.get("status") == HealthStatus.FAILING:
                alerts.append({
                    "check": check_name,
                    "type": check.get("type"),
                    "message": check.get("message"),
                    "metrics": check.get("metrics"),
                    "timestamp": check.get("timestamp")
                })
        
        # Process alerts through handlers
        for alert in alerts:
            for handler in self.alert_handlers:
                try:
                    handler(alert["check"], alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {str(e)}")
    
    def register_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """
        Register an alert handler function.
        
        Args:
            handler: Function that will be called with (check_name, alert_data)
        """
        self.alert_handlers.append(handler)
    
    def track_request(
        self,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        method: str
    ):
        """
        Track an API request for performance monitoring.
        
        Args:
            endpoint: API endpoint path
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            method: HTTP method (GET, POST, etc.)
        """
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                "count": 0,
                "total_time": 0,
                "success": 0,
                "error": 0,
                "response_times": [],
                "status_codes": {},
                "methods": {}
            }
        
        stats = self.endpoint_stats[endpoint]
        stats["count"] += 1
        stats["total_time"] += response_time_ms
        
        # Track success/error
        is_error = status_code >= 400
        if is_error:
            stats["error"] += 1
        else:
            stats["success"] += 1
        
        # Keep recent response times for percentile calculations
        stats["response_times"].append(response_time_ms)
        if len(stats["response_times"]) > self.max_history_length:
            stats["response_times"] = stats["response_times"][-self.max_history_length:]
        
        # Track status codes
        status_str = str(status_code)
        stats["status_codes"][status_str] = stats["status_codes"].get(status_str, 0) + 1
        
        # Track methods
        stats["methods"][method] = stats["methods"].get(method, 0) + 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status.
        
        Returns:
            Dictionary with overall health status and individual checks
        """
        # Determine overall status based on worst individual status
        overall = HealthStatus.OK
        for check in self.health_checks.values():
            status = check.get("status", HealthStatus.UNKNOWN)
            if status == HealthStatus.FAILING:
                overall = HealthStatus.FAILING
                break
            elif status == HealthStatus.DEGRADED and overall != HealthStatus.FAILING:
                overall = HealthStatus.DEGRADED
            elif status == HealthStatus.UNKNOWN and overall == HealthStatus.OK:
                overall = HealthStatus.UNKNOWN
        
        # Create the response
        return {
            "status": overall,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": self.health_checks,
            "system_metrics": self.system_metrics_history[-1] if self.system_metrics_history else None,
            "uptime": time.time() - psutil.boot_time()
        }
    
    def get_endpoint_stats(self) -> Dict[str, Any]:
        """
        Get endpoint performance statistics.
        
        Returns:
            Dictionary with endpoint performance statistics
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall": {
                "total_requests": sum(stats.get("count", 0) for stats in self.endpoint_stats.values()),
                "success_requests": sum(stats.get("success", 0) for stats in self.endpoint_stats.values()),
                "error_requests": sum(stats.get("error", 0) for stats in self.endpoint_stats.values())
            },
            "endpoints": {}
        }
        
        # Calculate error rate
        if result["overall"]["total_requests"] > 0:
            result["overall"]["error_rate"] = (result["overall"]["error_requests"] / result["overall"]["total_requests"]) * 100
        else:
            result["overall"]["error_rate"] = 0
        
        # Add endpoint details
        for endpoint, stats in self.endpoint_stats.items():
            if stats["count"] > 0:
                avg_time = stats["total_time"] / stats["count"]
                error_rate = (stats["error"] / stats["count"]) * 100 if stats["count"] > 0 else 0
                
                # Calculate percentiles if we have response times
                percentiles = {}
                if stats["response_times"]:
                    sorted_times = sorted(stats["response_times"])
                    percentiles = {
                        "p50": sorted_times[int(len(sorted_times) * 0.5)],
                        "p90": sorted_times[int(len(sorted_times) * 0.9)],
                        "p95": sorted_times[int(len(sorted_times) * 0.95)],
                        "p99": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) >= 100 else None
                    }
                
                result["endpoints"][endpoint] = {
                    "count": stats["count"],
                    "success": stats["success"],
                    "error": stats["error"],
                    "error_rate_percent": round(error_rate, 2),
                    "avg_response_time_ms": round(avg_time, 2),
                    "percentiles": {k: round(v, 2) if v is not None else None for k, v in percentiles.items()},
                    "status_codes": stats["status_codes"],
                    "methods": stats["methods"]
                }
        
        return result


class HealthMiddleware:
    """
    FastAPI middleware for tracking request performance.
    
    This middleware tracks request performance and reports to the
    health monitor.
    """
    
    def __init__(self, health_monitor: HealthMonitor):
        """
        Initialize middleware.
        
        Args:
            health_monitor: HealthMonitor instance
        """
        self.health_monitor = health_monitor
    
    async def __call__(self, request, call_next):
        """
        Process a request and track performance.
        
        Args:
            request: FastAPI request
            call_next: Next middleware in chain
            
        Returns:
            FastAPI response
        """
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            status_code = 500
            raise
        finally:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Track the request
            self.health_monitor.track_request(
                endpoint=request.url.path,
                response_time_ms=response_time_ms,
                status_code=status_code,
                method=request.method
            )
        
        return response
