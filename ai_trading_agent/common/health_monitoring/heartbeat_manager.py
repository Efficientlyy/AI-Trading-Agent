"""
Heartbeat Manager for Health Monitoring System.

This module implements the heartbeat mechanism for detecting component liveness,
tracking missed heartbeats, and issuing alerts when components fail to report.
"""

import threading
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

# Import from core definitions to avoid circular dependencies
from .core_definitions import HealthStatus, AlertSeverity
from .health_status import AlertData

# Set up logger
logger = logging.getLogger(__name__)


class HeartbeatConfig:
    """Configuration for component heartbeat monitoring."""
    
    def __init__(
        self,
        interval: float = 5.0,
        tolerance: float = 1.5,
        missing_threshold: int = 1,
        degraded_threshold: int = 2,
        unhealthy_threshold: int = 3
    ):
        """
        Initialize heartbeat configuration.
        
        Args:
            interval: Expected time between heartbeats in seconds
            tolerance: Multiplier applied to interval for tolerance
            missing_threshold: Missed heartbeats before first alert
            degraded_threshold: Missed heartbeats before degraded status
            unhealthy_threshold: Missed heartbeats before unhealthy status
        """
        self.interval = interval
        self.tolerance = tolerance
        self.missing_threshold = missing_threshold
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold


class HeartbeatData:
    """Internal data structure for tracking component heartbeats."""
    
    def __init__(
        self,
        component_id: str,
        config: HeartbeatConfig
    ):
        """
        Initialize heartbeat tracking data.
        
        Args:
            component_id: ID of the component to track
            config: Heartbeat configuration for this component
        """
        self.component_id = component_id
        self.config = config
        self.last_heartbeat = None
        self.first_heartbeat = None
        self.missed_count = 0
        self.total_count = 0
        self.alerted = False
        self.status = HealthStatus.UNKNOWN
    
    def record_heartbeat(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a heartbeat from the component.
        
        Args:
            data: Optional data provided with the heartbeat
        """
        current_time = time.time()
        
        if self.first_heartbeat is None:
            self.first_heartbeat = current_time
            self.status = HealthStatus.HEALTHY
            
        self.last_heartbeat = current_time
        self.total_count += 1
        
        # Reset missed count and alert status when heartbeat received
        if self.missed_count > 0:
            logger.info(f"Component {self.component_id} recovered after "
                       f"{self.missed_count} missed heartbeats")
            
        self.missed_count = 0
        self.alerted = False
        
        # If status was unhealthy, set to recovering
        if self.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            self.status = HealthStatus.RECOVERING
        # If recovering and we've had enough good heartbeats, set to healthy
        elif self.status == HealthStatus.RECOVERING and self.total_count % 3 == 0:
            self.status = HealthStatus.HEALTHY
    
    def check_missed_heartbeat(self) -> Tuple[bool, Optional[HealthStatus]]:
        """
        Check if a heartbeat has been missed.
        
        Returns:
            Tuple of (should_alert, new_status)
            - should_alert: True if an alert should be generated
            - new_status: New health status if changed, None otherwise
        """
        if self.last_heartbeat is None:
            # Never received a heartbeat, can't determine if missed
            return False, None
            
        current_time = time.time()
        elapsed = current_time - self.last_heartbeat
        max_interval = self.config.interval * self.config.tolerance
        
        if elapsed <= max_interval:
            # Heartbeat is still within acceptable range
            return False, None
        
        # Heartbeat has been missed
        self.missed_count += 1
        
        # Determine if we should change status
        new_status = None
        
        if self.missed_count >= self.config.unhealthy_threshold:
            new_status = HealthStatus.UNHEALTHY
        elif self.missed_count >= self.config.degraded_threshold:
            new_status = HealthStatus.DEGRADED
            
        # Update status if changed
        if new_status is not None and self.status != new_status:
            self.status = new_status
            
        # Determine if we should alert
        should_alert = False
        
        if self.missed_count >= self.config.missing_threshold and not self.alerted:
            should_alert = True
            self.alerted = True
            
        return should_alert, new_status


class HeartbeatManager:
    """
    Manager for tracking component heartbeats.
    
    Monitors heartbeats from system components, detects missed heartbeats,
    and triggers alerts when components fail to report in the expected interval.
    """
    
    def __init__(
        self,
        alert_callback: Optional[Callable[[AlertData], None]] = None,
        status_callback: Optional[Callable[[str, HealthStatus], None]] = None,
        check_interval: float = 1.0
    ):
        """
        Initialize the heartbeat manager.
        
        Args:
            alert_callback: Callback function for alerts
            status_callback: Callback function for status changes
            check_interval: Interval for checking missed heartbeats
        """
        self.components = {}  # Dict[str, HeartbeatData]
        self.alert_callback = alert_callback
        self.status_callback = status_callback
        self.check_interval = check_interval
        
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
    
    def register_component(
        self,
        component_id: str,
        config: Optional[HeartbeatConfig] = None
    ) -> None:
        """
        Register a component for heartbeat tracking.
        
        Args:
            component_id: ID of the component to register
            config: Optional custom heartbeat configuration
        """
        with self._lock:
            if component_id in self.components:
                logger.warning(f"Component {component_id} already registered for heartbeat tracking")
                return
                
            # Use provided config or default
            heartbeat_config = config or HeartbeatConfig()
            
            # Create heartbeat tracking data
            component_data = HeartbeatData(component_id, heartbeat_config)
            self.components[component_id] = component_data
            
            logger.info(f"Registered component {component_id} for heartbeat tracking "
                       f"with interval {heartbeat_config.interval}s")
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component from heartbeat tracking.
        
        Args:
            component_id: ID of the component to unregister
            
        Returns:
            True if component was unregistered, False if not found
        """
        with self._lock:
            if component_id not in self.components:
                logger.warning(f"Component {component_id} not found for unregistration")
                return False
                
            del self.components[component_id]
            logger.info(f"Unregistered component {component_id} from heartbeat tracking")
            return True
    
    def record_heartbeat(
        self,
        component_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a heartbeat from a component.
        
        Args:
            component_id: ID of the component sending heartbeat
            data: Optional data provided with the heartbeat
            
        Returns:
            True if heartbeat was recorded, False if component not registered
        """
        with self._lock:
            if component_id not in self.components:
                logger.warning(f"Received heartbeat from unregistered component {component_id}")
                return False
                
            component_data = self.components[component_id]
            
            # Get previous status for change detection
            prev_status = component_data.status
            
            # Record heartbeat
            component_data.record_heartbeat(data)
            
            # Check for status change
            if prev_status != component_data.status and self.status_callback:
                self.status_callback(component_id, component_data.status)
                
            return True
    
    def get_component_status(self, component_id: str) -> Optional[HealthStatus]:
        """
        Get the current heartbeat-derived status of a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            Current health status or None if component not found
        """
        with self._lock:
            if component_id not in self.components:
                return None
                
            return self.components[component_id].status
    
    def start(self) -> None:
        """Start the heartbeat monitoring thread."""
        with self._lock:
            if self._running:
                logger.warning("Heartbeat manager already running")
                return
                
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_heartbeats,
                name="HeartbeatMonitor",
                daemon=True
            )
            self._thread.start()
            
            logger.info("Heartbeat manager started")
    
    def stop(self) -> None:
        """Stop the heartbeat monitoring thread."""
        with self._lock:
            if not self._running:
                logger.warning("Heartbeat manager already stopped")
                return
                
            self._running = False
            if self._thread:
                self._thread.join(timeout=5.0)
                self._thread = None
                
            logger.info("Heartbeat manager stopped")
    
    def _monitor_heartbeats(self) -> None:
        """Background thread for monitoring missed heartbeats."""
        logger.info("Heartbeat monitoring thread started")
        
        while self._running:
            try:
                self._check_missed_heartbeats()
            except Exception as e:
                logger.error(f"Error in heartbeat monitoring: {str(e)}")
                
            # Sleep for check interval
            time.sleep(self.check_interval)
            
        logger.info("Heartbeat monitoring thread stopped")
    
    def _check_missed_heartbeats(self) -> None:
        """Check all components for missed heartbeats."""
        with self._lock:
            for component_id, component_data in list(self.components.items()):
                should_alert, new_status = component_data.check_missed_heartbeat()
                
                # Generate alert if needed
                if should_alert and self.alert_callback:
                    alert = self._create_missed_heartbeat_alert(component_id, component_data)
                    self.alert_callback(alert)
                
                # Notify of status change if needed
                if new_status is not None and self.status_callback:
                    self.status_callback(component_id, new_status)
    
    def _create_missed_heartbeat_alert(
        self,
        component_id: str,
        component_data: HeartbeatData
    ) -> AlertData:
        """
        Create an alert for a missed heartbeat.
        
        Args:
            component_id: ID of the component with missed heartbeat
            component_data: Heartbeat tracking data for the component
            
        Returns:
            AlertData instance for the missed heartbeat
        """
        # Determine severity based on missed count
        severity = AlertSeverity.WARNING
        if component_data.missed_count >= component_data.config.unhealthy_threshold:
            severity = AlertSeverity.ERROR
        
        # Calculate time since last heartbeat
        time_since_last = "Never" if component_data.last_heartbeat is None else \
                         f"{time.time() - component_data.last_heartbeat:.2f}s"
        
        # Create alert
        alert_id = f"heartbeat_{component_id}_{int(time.time())}"
        
        message = (f"Component {component_id} missed {component_data.missed_count} "
                  f"heartbeats. Last heartbeat: {time_since_last} ago.")
        
        details = {
            "component_id": component_id,
            "missed_count": component_data.missed_count,
            "last_heartbeat": component_data.last_heartbeat,
            "expected_interval": component_data.config.interval,
            "status": component_data.status.value
        }
        
        return AlertData(
            alert_id=alert_id,
            component_id=component_id,
            severity=severity,
            message=message,
            details=details
        )
