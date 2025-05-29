"""
Component Health for Health Monitoring System.

This module defines the data structures for representing component health status
and provides functionality for health status tracking and updates.
"""

import time
from typing import Any, Dict, List, Optional, Union

# Import from core_definitions to avoid circular dependencies
from .core_definitions import HealthStatus


class ComponentHealth:
    """
    Data structure for component health status.
    
    Represents the health status and metrics of an individual component.
    """
    
    def __init__(
        self,
        component_id: str,
        description: Optional[str] = None,
        status: HealthStatus = HealthStatus.UNKNOWN
    ):
        """
        Initialize component health data.
        
        Args:
            component_id: Unique identifier for the component
            description: Description of the component
            status: Initial health status
        """
        self.component_id = component_id
        self.description = description or f"Component {component_id}"
        self.status = status
        self.last_heartbeat = None
        self.first_heartbeat = None
        self.heartbeat_count = 0
        self.metrics = {}
        self.last_status_change = time.time()
        self.recovery_attempts = 0
        self.last_recovery_time = None
        self.diagnostics = {}
        self.last_updated = time.time()
    
    @property
    def uptime(self) -> Optional[float]:
        """Calculate component uptime in seconds."""
        if self.first_heartbeat is None:
            return None
        return time.time() - self.first_heartbeat
    
    def update_status(self, status: HealthStatus) -> None:
        """
        Update the component status.
        
        Args:
            status: New health status
        """
        if self.status != status:
            self.status = status
            self.last_status_change = time.time()
            
        self.last_updated = time.time()
    
    def record_heartbeat(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a heartbeat from the component.
        
        Args:
            data: Optional data provided with the heartbeat
        """
        current_time = time.time()
        self.last_heartbeat = current_time
        
        if self.first_heartbeat is None:
            self.first_heartbeat = current_time
            
        self.heartbeat_count += 1
        
        if data:
            # Update diagnostics with heartbeat data
            self.diagnostics.update(data)
            
        self.last_updated = current_time
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record metrics from the component.
        
        Args:
            metrics: Dictionary of metrics from the component
        """
        self.metrics.update(metrics)
        self.last_updated = time.time()
    
    def record_recovery_attempt(self) -> None:
        """Record a recovery attempt for this component."""
        self.recovery_attempts += 1
        self.last_recovery_time = time.time()
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the component health data to a dictionary representation."""
        return {
            "component_id": self.component_id,
            "description": self.description,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "first_heartbeat": self.first_heartbeat,
            "heartbeat_count": self.heartbeat_count,
            "uptime": self.uptime,
            "metrics": self.metrics,
            "last_status_change": self.last_status_change,
            "recovery_attempts": self.recovery_attempts,
            "last_recovery_time": self.last_recovery_time,
            "diagnostics": self.diagnostics,
            "last_updated": self.last_updated
        }


class SystemHealth:
    """
    Data structure for system-wide health status.
    
    Represents the aggregated health status of the entire system.
    """
    
    def __init__(self):
        """Initialize system health data."""
        self.overall_status = HealthStatus.UNKNOWN
        self.component_statuses = {}  # Dict[str, ComponentHealth]
        self.system_metrics = {}
        self.started_at = time.time()
        self.last_updated = time.time()
    
    def update_overall_status(self) -> None:
        """Update the overall system status based on component statuses."""
        if not self.component_statuses:
            self.overall_status = HealthStatus.UNKNOWN
            return
            
        # Determine worst status among components
        status_priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.UNKNOWN: 1,
            HealthStatus.RECOVERING: 2,
            HealthStatus.DEGRADED: 3,
            HealthStatus.UNHEALTHY: 4,
            HealthStatus.CRITICAL: 5
        }
        
        worst_status = HealthStatus.HEALTHY
        
        for component in self.component_statuses.values():
            if status_priority[component.status] > status_priority[worst_status]:
                worst_status = component.status
                
        self.overall_status = worst_status
        self.last_updated = time.time()
    
    def add_component_status(self, component_data: ComponentHealth) -> None:
        """
        Add or update a component status.
        
        Args:
            component_data: Component health data to add/update
        """
        self.component_statuses[component_data.component_id] = component_data
        self.update_overall_status()
    
    def update_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update system-wide metrics.
        
        Args:
            metrics: Dictionary of system metrics
        """
        self.system_metrics.update(metrics)
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the system health data to a dictionary representation."""
        return {
            "overall_status": self.overall_status.value,
            "component_statuses": {
                component_id: component.to_dict() 
                for component_id, component in self.component_statuses.items()
            },
            "system_metrics": self.system_metrics,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "uptime": time.time() - self.started_at
        }
