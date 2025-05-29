"""
Core definitions for the Health Monitoring System.

This module contains the basic enums and data structures used across
the health monitoring system. It should have minimal dependencies
to avoid circular import issues.
"""

import enum
import time
from typing import Any, Dict, List, Optional


class HealthStatus(enum.Enum):
    """Enum representing the health status of a component or system."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    RECOVERING = "recovering"


class AlertSeverity(enum.Enum):
    """Enum representing alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ThresholdType(enum.Enum):
    """Types of thresholds for metrics."""
    UPPER = "upper"  # Alert if value exceeds threshold
    LOWER = "lower"  # Alert if value falls below threshold
    EQUALITY = "equality"  # Alert if value equals threshold
    CHANGE_RATE = "change_rate"  # Alert if rate of change exceeds threshold
