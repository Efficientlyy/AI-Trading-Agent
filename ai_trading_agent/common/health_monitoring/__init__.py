"""
Health Monitoring System for AI Trading Agent.

This package provides comprehensive monitoring and diagnostic capabilities
for detecting component failures, performance degradation, and system-level
issues while facilitating autonomous recovery actions.

Main components:
- HealthMonitor: Central monitoring coordination
- ComponentHealth: Individual component health tracking
- HeartbeatManager: Component liveness detection
- AlertManager: Health alert processing and routing
- RecoveryCoordinator: Autonomous recovery actions
"""

# First import common definitions to avoid circular imports
from .core_definitions import HealthStatus, AlertSeverity, ThresholdType

# Then import components in dependency order
from .component_health import ComponentHealth
from .health_status import AlertData
from .health_metrics import MetricThreshold, HealthMetrics
from .alert_manager import AlertManager
from .heartbeat_manager import HeartbeatManager, HeartbeatConfig
from .recovery_coordinator import RecoveryCoordinator, RecoveryAction

# Import health monitor last since it depends on all of the above
from .health_monitor import HealthMonitor

__all__ = [
    'HealthMonitor',
    'ComponentHealth',
    'HealthStatus',
    'AlertManager',
    'AlertData',
    'AlertSeverity',
    'HeartbeatManager',
    'HeartbeatConfig',
    'MetricThreshold',
    'ThresholdType',
    'HealthMetrics',
    'RecoveryCoordinator',
    'RecoveryAction'
]
