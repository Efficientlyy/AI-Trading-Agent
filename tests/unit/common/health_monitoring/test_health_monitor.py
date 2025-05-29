"""
Unit tests for the Health Monitoring System.

This module contains tests for the Health Monitoring System components,
verifying their functionality both individually and when integrated.
"""

import unittest
import time
import threading
import logging
from unittest.mock import MagicMock, patch
import tempfile
import os
from pathlib import Path

from ai_trading_agent.common.health_monitoring import (
    HealthMonitor,
    ComponentHealth,
    HealthStatus,
    AlertData,
    AlertSeverity,
    HeartbeatManager,
    HeartbeatConfig,
    MetricThreshold,
    ThresholdType,
    RecoveryAction
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestHeartbeatManager(unittest.TestCase):
    """Test cases for the heartbeat manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alert_callback = MagicMock()
        self.status_callback = MagicMock()
        self.manager = HeartbeatManager(
            alert_callback=self.alert_callback,
            status_callback=self.status_callback,
            check_interval=0.1
        )
        self.manager.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.manager.stop()
    
    def test_register_component(self):
        """Test registering a component for heartbeat tracking."""
        self.manager.register_component("test_component")
        self.assertIn("test_component", self.manager.components)
    
    def test_record_heartbeat(self):
        """Test recording a heartbeat from a component."""
        self.manager.register_component("test_component")
        result = self.manager.record_heartbeat("test_component")
        self.assertTrue(result)
        self.assertEqual(
            self.manager.components["test_component"].status,
            HealthStatus.HEALTHY
        )
    
    def test_missed_heartbeat(self):
        """Test detection of missed heartbeats."""
        # Register with short interval for testing
        config = HeartbeatConfig(
            interval=0.1,
            missing_threshold=1,
            degraded_threshold=1,
            unhealthy_threshold=2
        )
        self.manager.register_component("test_component", config)
        
        # Record initial heartbeat
        self.manager.record_heartbeat("test_component")
        
        # Wait for heartbeat to be missed
        time.sleep(0.3)
        
        # Check that alert was generated
        self.alert_callback.assert_called()
        self.assertEqual(
            self.alert_callback.call_args[0][0].severity,
            AlertSeverity.WARNING
        )
        
        # Check that status was updated
        self.status_callback.assert_called()
        self.assertEqual(
            self.status_callback.call_args[0][1],
            HealthStatus.DEGRADED
        )
        
        # Wait longer for unhealthy threshold
        time.sleep(0.3)
        
        # Check that status was updated to unhealthy
        self.assertEqual(
            self.status_callback.call_args[0][1],
            HealthStatus.UNHEALTHY
        )


class TestAlertManager(unittest.TestCase):
    """Test cases for the alert manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tempdir = tempfile.mkdtemp()
        self.alert_log_path = os.path.join(self.tempdir, "alerts.log")
        self.dashboard_callback = MagicMock()
        
        from ai_trading_agent.common.health_monitoring.alert_manager import AlertManager, AlertChannel
        self.manager = AlertManager(
            channels=[AlertChannel.CONSOLE, AlertChannel.FILE],
            alert_log_path=self.alert_log_path,
            dashboard_callback=self.dashboard_callback
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        import shutil
        shutil.rmtree(self.tempdir)
    
    def test_add_alert(self):
        """Test adding an alert."""
        alert = AlertData(
            alert_id="test_alert",
            component_id="test_component",
            severity=AlertSeverity.WARNING,
            message="Test alert message"
        )
        
        self.manager.add_alert(alert)
        
        # Check that alert was added to history
        self.assertIn(alert, self.manager.alert_history)
        
        # Check that alert was added to active alerts
        self.assertIn(alert.alert_id, self.manager.active_alerts)
        
        # Check that alert log file was created
        self.assertTrue(os.path.exists(self.alert_log_path))
    
    def test_resolve_alert(self):
        """Test resolving an alert."""
        alert = AlertData(
            alert_id="test_alert",
            component_id="test_component",
            severity=AlertSeverity.WARNING,
            message="Test alert message"
        )
        
        self.manager.add_alert(alert)
        result = self.manager.resolve_alert("test_alert")
        
        self.assertTrue(result)
        self.assertTrue(alert.resolved)
        self.assertNotIn(alert.alert_id, self.manager.active_alerts)
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Add multiple alerts
        for i in range(3):
            alert = AlertData(
                alert_id=f"test_alert_{i}",
                component_id="test_component",
                severity=AlertSeverity.WARNING,
                message=f"Test alert message {i}"
            )
            self.manager.add_alert(alert)
        
        # Resolve one alert
        self.manager.resolve_alert("test_alert_1")
        
        # Get active alerts
        active_alerts = self.manager.get_active_alerts()
        
        self.assertEqual(len(active_alerts), 2)
        self.assertIn("test_alert_0", [a.alert_id for a in active_alerts])
        self.assertIn("test_alert_2", [a.alert_id for a in active_alerts])
        self.assertNotIn("test_alert_1", [a.alert_id for a in active_alerts])


class TestHealthMetrics(unittest.TestCase):
    """Test cases for the health metrics component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alert_callback = MagicMock()
        
        from ai_trading_agent.common.health_monitoring.health_metrics import HealthMetrics
        self.metrics = HealthMetrics(
            alert_callback=self.alert_callback,
            check_interval=0.1
        )
    
    def test_add_metric(self):
        """Test adding a metric."""
        self.metrics.add_metric("test_component", "test_metric", 42.0)
        
        # Get metrics
        metrics = self.metrics.get_metrics()
        
        self.assertIn("test_component", metrics)
        self.assertIn("test_metric", metrics["test_component"])
        self.assertEqual(metrics["test_component"]["test_metric"]["last_value"], 42.0)
    
    def test_threshold_alert(self):
        """Test threshold alerting."""
        # Start metrics monitoring
        self.metrics.start()
        
        try:
            # Add threshold
            from ai_trading_agent.common.health_monitoring.health_metrics import MetricThreshold, ThresholdType
            threshold = MetricThreshold(
                metric_name="test_metric",
                threshold_type=ThresholdType.UPPER,
                warning_threshold=10.0,
                critical_threshold=20.0,
                component_id="test_component"
            )
            self.metrics.add_threshold(threshold)
            
            # Add metric below threshold
            self.metrics.add_metric("test_component", "test_metric", 5.0)
            
            # No alert should be generated
            self.alert_callback.assert_not_called()
            
            # Add metric above warning threshold
            self.metrics.add_metric("test_component", "test_metric", 15.0)
            
            # Wait for check
            time.sleep(0.2)
            
            # Alert should be generated
            self.alert_callback.assert_called_once()
            self.assertEqual(
                self.alert_callback.call_args[0][0].severity,
                AlertSeverity.WARNING
            )
            
            # Reset mock
            self.alert_callback.reset_mock()
            
            # Add metric above critical threshold
            self.metrics.add_metric("test_component", "test_metric", 25.0)
            
            # Wait for check
            time.sleep(0.2)
            
            # Critical alert should be generated
            self.alert_callback.assert_called_once()
            self.assertEqual(
                self.alert_callback.call_args[0][0].severity,
                AlertSeverity.CRITICAL
            )
        finally:
            self.metrics.stop()


class TestRecoveryCoordinator(unittest.TestCase):
    """Test cases for the recovery coordinator."""
    
    def setUp(self):
        """Set up test fixtures."""
        from ai_trading_agent.common.health_monitoring.recovery_coordinator import RecoveryCoordinator, RecoveryAction
        self.coordinator = RecoveryCoordinator()
        
        # Mock recovery action function
        self.action_func = MagicMock(return_value=True)
        
        # Create recovery action
        self.action = RecoveryAction(
            action_id="test_action",
            description="Test recovery action",
            action_func=self.action_func,
            component_id="test_component",
            severity_threshold=AlertSeverity.ERROR
        )
        
        # Register action
        self.coordinator.register_action(self.action)
    
    def test_register_action(self):
        """Test registering a recovery action."""
        self.assertIn("test_action", self.coordinator.actions)
        self.assertIn("test_component", self.coordinator.component_actions)
        self.assertIn("test_action", self.coordinator.component_actions["test_component"])
    
    def test_handle_alert(self):
        """Test handling an alert with recovery action."""
        # Create alert
        alert = AlertData(
            alert_id="test_alert",
            component_id="test_component",
            severity=AlertSeverity.ERROR,
            message="Test alert message"
        )
        
        # Handle alert
        result = self.coordinator.handle_alert(alert)
        
        # Action should be executed
        self.assertTrue(result)
        self.action_func.assert_called_once()
        
        # Check recovery history
        history = self.coordinator.get_recovery_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["alert_id"], "test_alert")
        self.assertEqual(history[0]["action_id"], "test_action")
        self.assertTrue(history[0]["success"])
    
    def test_severity_threshold(self):
        """Test that actions respect severity threshold."""
        # Create warning alert (below threshold)
        alert = AlertData(
            alert_id="test_alert",
            component_id="test_component",
            severity=AlertSeverity.WARNING,
            message="Test alert message"
        )
        
        # Handle alert
        result = self.coordinator.handle_alert(alert)
        
        # Action should not be executed
        self.assertFalse(result)
        self.action_func.assert_not_called()


class TestHealthMonitor(unittest.TestCase):
    """Test cases for the integrated health monitor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tempdir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.tempdir, "health_logs")
        
        # Create monitor
        self.monitor = HealthMonitor(
            log_dir=self.log_dir
        )
        
        # Start monitoring
        self.monitor.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.monitor.stop()
        import shutil
        shutil.rmtree(self.tempdir)
    
    def test_register_component(self):
        """Test registering a component."""
        self.monitor.register_component(
            component_id="test_component",
            description="Test Component"
        )
        
        # Check component health
        component_health = self.monitor.get_component_health("test_component")
        self.assertIn("test_component", component_health)
        self.assertEqual(
            component_health["test_component"]["status"],
            HealthStatus.UNKNOWN.value
        )
    
    def test_heartbeat_integration(self):
        """Test heartbeat integration."""
        # Register component
        self.monitor.register_component(
            component_id="test_component",
            description="Test Component",
            heartbeat_config=HeartbeatConfig(
                interval=0.1,
                missing_threshold=1,
                degraded_threshold=1,
                unhealthy_threshold=2
            )
        )
        
        # Send heartbeat
        self.monitor.record_heartbeat("test_component")
        
        # Check status
        component_health = self.monitor.get_component_health("test_component")
        self.assertEqual(
            component_health["test_component"]["status"],
            HealthStatus.HEALTHY.value
        )
        
        # Wait for heartbeat to be missed
        time.sleep(0.3)
        
        # Check alerts
        alerts = self.monitor.get_active_alerts()
        self.assertTrue(len(alerts) > 0)
        
        # Check component status
        component_health = self.monitor.get_component_health("test_component")
        self.assertEqual(
            component_health["test_component"]["status"],
            HealthStatus.DEGRADED.value
        )
    
    def test_metrics_integration(self):
        """Test metrics integration."""
        # Register component
        self.monitor.register_component(
            component_id="test_component",
            description="Test Component"
        )
        
        # Add metric threshold
        self.monitor.add_metric_threshold(
            metric_name="test_metric",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=10.0,
            critical_threshold=20.0,
            component_id="test_component",
            description="Test metric threshold"
        )
        
        # Add metric below threshold
        self.monitor.add_metric(
            component_id="test_component",
            metric_name="test_metric",
            value=5.0
        )
        
        # Check metrics
        metrics = self.monitor.get_metrics("test_component", "test_metric")
        self.assertEqual(
            metrics["test_component"]["test_metric"]["last_value"],
            5.0
        )
        
        # Add metric above threshold
        self.monitor.add_metric(
            component_id="test_component",
            metric_name="test_metric",
            value=25.0
        )
        
        # Wait for check
        time.sleep(0.2)
        
        # Check alerts
        alerts = self.monitor.get_active_alerts()
        metric_alerts = [a for a in alerts if "metric" in a.alert_id]
        self.assertTrue(len(metric_alerts) > 0)
        self.assertEqual(metric_alerts[0].severity, AlertSeverity.CRITICAL)
    
    def test_recovery_integration(self):
        """Test recovery action integration."""
        # Mock recovery function
        action_func = MagicMock(return_value=True)
        
        # Register component
        self.monitor.register_component(
            component_id="test_component",
            description="Test Component"
        )
        
        # Register recovery action
        self.monitor.register_recovery_action(
            action_id="test_action",
            description="Test recovery action",
            action_func=action_func,
            component_id="test_component",
            severity_threshold=AlertSeverity.ERROR
        )
        
        # Create alert that should trigger recovery
        alert = AlertData(
            alert_id="test_alert",
            component_id="test_component",
            severity=AlertSeverity.ERROR,
            message="Test alert message"
        )
        
        # Add alert
        self.monitor.alert_manager.add_alert(alert)
        
        # Check recovery history
        history = self.monitor.get_recovery_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["action_id"], "test_action")
        self.assertTrue(history[0]["success"])
        
        # Check that action was called
        action_func.assert_called_once()
    
    def test_system_health(self):
        """Test system health aggregation."""
        # Register components with different status
        self.monitor.register_component(
            component_id="healthy_component",
            description="Healthy Component"
        )
        
        self.monitor.register_component(
            component_id="degraded_component",
            description="Degraded Component"
        )
        
        self.monitor.register_component(
            component_id="unhealthy_component",
            description="Unhealthy Component"
        )
        
        # Set component status
        # For healthy component, just send heartbeat
        self.monitor.record_heartbeat("healthy_component")
        
        # Manually set status for others
        self.monitor._update_component_status(
            component_id="degraded_component",
            status=HealthStatus.DEGRADED
        )
        
        self.monitor._update_component_status(
            component_id="unhealthy_component",
            status=HealthStatus.UNHEALTHY
        )
        
        # Get system health
        system_health = self.monitor.get_system_health()
        
        # Overall status should reflect worst component
        self.assertEqual(
            system_health["overall_status"],
            HealthStatus.UNHEALTHY.value
        )
        
        # Should have 3 components
        self.assertEqual(system_health["component_count"], 3)
        
        # Check status counts
        self.assertEqual(system_health["status_counts"][HealthStatus.HEALTHY.value], 1)
        self.assertEqual(system_health["status_counts"][HealthStatus.DEGRADED.value], 1)
        self.assertEqual(system_health["status_counts"][HealthStatus.UNHEALTHY.value], 1)


if __name__ == "__main__":
    unittest.main()
