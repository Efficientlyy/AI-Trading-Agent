"""
Monitoring Module.

This module provides monitoring and dashboard capabilities for the trading system,
including real-time status visualization, performance tracking, and alerts.
"""

from src.monitoring.service import MonitoringService, get_monitoring_service

__all__ = [
    'MonitoringService',
    'get_monitoring_service',
] 