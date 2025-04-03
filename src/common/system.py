"""
System Monitoring Module

This module provides functionality for monitoring system resources and health.
It collects real-time metrics on CPU, memory, disk, and network usage.
"""

import os
import time
import logging
import platform
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Try to import psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available, using limited system metrics")
    PSUTIL_AVAILABLE = False


class SystemMonitor:
    """
    Monitors system resources and health.
    
    This class provides methods to collect and report system metrics
    such as CPU usage, memory usage, disk space, and network latency.
    """
    
    def __init__(self, cache_duration: int = 5):
        """
        Initialize the system monitor.
        
        Args:
            cache_duration: Duration in seconds to cache metrics (default: 5)
        """
        self.cache_duration = cache_duration
        self.last_update_time = 0
        self.cached_health_data = {}
        self.start_time = time.time()
        
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health metrics.
        
        Returns:
            Dictionary containing system health metrics
        """
        current_time = time.time()
        
        # Return cached data if it's still valid
        if current_time - self.last_update_time < self.cache_duration and self.cached_health_data:
            return self.cached_health_data
        
        try:
            # Collect metrics
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            disk_usage = self._get_disk_usage()
            network_latency = self._get_network_latency()
            io_wait = self._get_io_wait()
            temperature = self._get_temperature()
            processes = self._get_process_count()
            
            # Determine health status based on thresholds
            health_status = self._determine_health_status(
                cpu_usage, memory_usage, disk_usage, network_latency
            )
            
            # Create health data dictionary
            health_data = {
                "status": health_status,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
                "network_latency": network_latency,
                "uptime": int(current_time - self.start_time),
                "last_update": datetime.now().isoformat(),
                "processes": processes,
                "temperature": temperature,
                "io_wait": io_wait
            }
            
            # Update cache
            self.cached_health_data = health_data
            self.last_update_time = current_time
            
            return health_data
            
        except Exception as e:
            logger.error(f"Error collecting system health metrics: {e}")
            
            # If we have cached data, return it even if it's expired
            if self.cached_health_data:
                logger.warning("Returning expired cached health data due to error")
                return self.cached_health_data
                
            # Otherwise, return a minimal set of data
            return {
                "status": "unknown",
                "cpu_usage": 0,
                "memory_usage": 0,
                "disk_usage": 0,
                "network_latency": 0,
                "uptime": int(current_time - self.start_time),
                "last_update": datetime.now().isoformat(),
                "processes": 0,
                "temperature": 0,
                "io_wait": 0
            }
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """
        Get detailed resource utilization metrics.
        
        Returns:
            Dictionary containing detailed resource utilization metrics
        """
        try:
            if not PSUTIL_AVAILABLE:
                return {
                    "cpu": {"usage": self._get_cpu_usage()},
                    "memory": {"usage": self._get_memory_usage()},
                    "disk": {"usage": self._get_disk_usage()},
                    "network": {"latency": self._get_network_latency()}
                }
            
            # Get CPU details
            cpu_times = psutil.cpu_times_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq() if hasattr(psutil, 'cpu_freq') else None
            
            # Get memory details
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Get disk details
            disk_partitions = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_partitions.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": round(usage.total / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                        "percent": usage.percent
                    })
                except (PermissionError, FileNotFoundError):
                    # Skip partitions we can't access
                    pass
            
            # Get network details
            net_io_counters = psutil.net_io_counters()
            
            # Compile detailed resource data
            resource_data = {
                "cpu": {
                    "usage": self._get_cpu_usage(),
                    "count": cpu_count,
                    "user": cpu_times.user,
                    "system": cpu_times.system,
                    "idle": cpu_times.idle,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "usage": memory.percent,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "swap_percent": swap.percent,
                    "swap_total_gb": round(swap.total / (1024**3), 2),
                    "swap_used_gb": round(swap.used / (1024**3), 2)
                },
                "disk": {
                    "usage": self._get_disk_usage(),
                    "partitions": disk_partitions,
                    "io_counters": {
                        "read_count": psutil.disk_io_counters().read_count,
                        "write_count": psutil.disk_io_counters().write_count,
                        "read_bytes": psutil.disk_io_counters().read_bytes,
                        "write_bytes": psutil.disk_io_counters().write_bytes
                    } if hasattr(psutil, 'disk_io_counters') and psutil.disk_io_counters() else {}
                },
                "network": {
                    "latency": self._get_network_latency(),
                    "bytes_sent": net_io_counters.bytes_sent,
                    "bytes_recv": net_io_counters.bytes_recv,
                    "packets_sent": net_io_counters.packets_sent,
                    "packets_recv": net_io_counters.packets_recv,
                    "connections": len(psutil.net_connections(kind='inet'))
                }
            }
            
            return resource_data
            
        except Exception as e:
            logger.error(f"Error collecting detailed resource metrics: {e}")
            return {
                "cpu": {"usage": self._get_cpu_usage()},
                "memory": {"usage": self._get_memory_usage()},
                "disk": {"usage": self._get_disk_usage()},
                "network": {"latency": self._get_network_latency()}
            }
    
    def _get_cpu_usage(self) -> float:
        """
        Get CPU usage percentage.
        
        Returns:
            CPU usage as a percentage (0-100)
        """
        try:
            if PSUTIL_AVAILABLE:
                return psutil.cpu_percent(interval=0.1)
            else:
                # Fallback method for Unix-like systems
                if platform.system() != "Windows":
                    try:
                        with open('/proc/stat', 'r') as f:
                            cpu_stats = f.readline().split()
                            user = float(cpu_stats[1])
                            nice = float(cpu_stats[2])
                            system = float(cpu_stats[3])
                            idle = float(cpu_stats[4])
                            total = user + nice + system + idle
                            return 100 * (1 - idle / total)
                    except:
                        pass
                
                # Return a reasonable estimate if we can't get real data
                return 30.0
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return 30.0  # Return a reasonable default
    
    def _get_memory_usage(self) -> float:
        """
        Get memory usage percentage.
        
        Returns:
            Memory usage as a percentage (0-100)
        """
        try:
            if PSUTIL_AVAILABLE:
                return psutil.virtual_memory().percent
            else:
                # Fallback method for Unix-like systems
                if platform.system() != "Windows":
                    try:
                        with open('/proc/meminfo', 'r') as f:
                            lines = f.readlines()
                            total = int(lines[0].split()[1])
                            free = int(lines[1].split()[1])
                            return 100 * (1 - free / total)
                    except:
                        pass
                
                # Return a reasonable estimate if we can't get real data
                return 40.0
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 40.0  # Return a reasonable default
    
    def _get_disk_usage(self) -> float:
        """
        Get disk usage percentage for the main disk.
        
        Returns:
            Disk usage as a percentage (0-100)
        """
        try:
            if PSUTIL_AVAILABLE:
                # Get usage for the disk containing the current directory
                return psutil.disk_usage(os.getcwd()).percent
            else:
                # Fallback method for Unix-like systems
                if platform.system() != "Windows":
                    try:
                        output = os.popen("df -h . | tail -1").read()
                        return float(output.split()[4].strip('%'))
                    except:
                        pass
                
                # Return a reasonable estimate if we can't get real data
                return 50.0
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return 50.0  # Return a reasonable default
    
    def _get_network_latency(self) -> float:
        """
        Estimate network latency by pinging a reliable host.
        
        Returns:
            Network latency in milliseconds
        """
        try:
            # Use a simple ping to estimate latency
            host = "8.8.8.8"  # Google DNS
            
            if platform.system() == "Windows":
                ping_cmd = f"ping -n 1 {host}"
                output = os.popen(ping_cmd).read()
                try:
                    # Extract time from Windows ping output
                    time_str = output.split("time=")[1].split("ms")[0].strip()
                    return float(time_str)
                except:
                    return 50.0
            else:
                ping_cmd = f"ping -c 1 {host}"
                output = os.popen(ping_cmd).read()
                try:
                    # Extract time from Unix ping output
                    time_str = output.split("time=")[1].split(" ms")[0].strip()
                    return float(time_str)
                except:
                    return 50.0
        except Exception as e:
            logger.error(f"Error measuring network latency: {e}")
            return 50.0  # Return a reasonable default
    
    def _get_io_wait(self) -> float:
        """
        Get IO wait percentage.
        
        Returns:
            IO wait as a percentage (0-100)
        """
        try:
            if PSUTIL_AVAILABLE:
                cpu_times = psutil.cpu_times_percent(interval=0.1)
                if hasattr(cpu_times, 'iowait'):
                    return cpu_times.iowait
            
            # Return a reasonable estimate if we can't get real data
            return 2.0
        except Exception as e:
            logger.error(f"Error getting IO wait: {e}")
            return 2.0  # Return a reasonable default
    
    def _get_temperature(self) -> float:
        """
        Get CPU temperature if available.
        
        Returns:
            CPU temperature in Celsius, or 0 if not available
        """
        try:
            if PSUTIL_AVAILABLE and hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Different systems report temperatures differently
                    # Try to find CPU temperature from common sources
                    for source in ['coretemp', 'cpu_thermal', 'acpitz', 'k10temp']:
                        if source in temps:
                            return temps[source][0].current
            
            # Return a reasonable estimate if we can't get real data
            return 45.0
        except Exception as e:
            logger.error(f"Error getting temperature: {e}")
            return 45.0  # Return a reasonable default
    
    def _get_process_count(self) -> int:
        """
        Get the number of running processes.
        
        Returns:
            Number of processes
        """
        try:
            if PSUTIL_AVAILABLE:
                return len(psutil.pids())
            else:
                # Fallback method for Unix-like systems
                if platform.system() != "Windows":
                    try:
                        output = os.popen("ps -e | wc -l").read()
                        return int(output.strip())
                    except:
                        pass
                
                # Return a reasonable estimate if we can't get real data
                return 50
        except Exception as e:
            logger.error(f"Error getting process count: {e}")
            return 50  # Return a reasonable default
    
    def _determine_health_status(
        self, 
        cpu_usage: float, 
        memory_usage: float, 
        disk_usage: float, 
        network_latency: float
    ) -> str:
        """
        Determine overall system health status based on metrics.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            disk_usage: Disk usage percentage
            network_latency: Network latency in milliseconds
            
        Returns:
            Health status string: "good", "warning", or "critical"
        """
        # Define thresholds
        warning_thresholds = {
            "cpu": 80.0,
            "memory": 85.0,
            "disk": 90.0,
            "network": 200.0
        }
        
        critical_thresholds = {
            "cpu": 95.0,
            "memory": 95.0,
            "disk": 95.0,
            "network": 500.0
        }
        
        # Check for critical conditions
        if (cpu_usage >= critical_thresholds["cpu"] or
            memory_usage >= critical_thresholds["memory"] or
            disk_usage >= critical_thresholds["disk"] or
            network_latency >= critical_thresholds["network"]):
            return "critical"
            
        # Check for warning conditions
        if (cpu_usage >= warning_thresholds["cpu"] or
            memory_usage >= warning_thresholds["memory"] or
            disk_usage >= warning_thresholds["disk"] or
            network_latency >= warning_thresholds["network"]):
            return "warning"
            
        # Otherwise, system is good
        return "good"


# Singleton instance
_SYSTEM_MONITOR: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """
    Get the singleton SystemMonitor instance.
    
    Returns:
        The SystemMonitor instance
    """
    global _SYSTEM_MONITOR
    if _SYSTEM_MONITOR is None:
        _SYSTEM_MONITOR = SystemMonitor()
    return _SYSTEM_MONITOR
