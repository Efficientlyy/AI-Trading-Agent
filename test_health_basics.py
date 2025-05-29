"""
Simple Test for Health Monitoring Basics

This script tests only the most basic functionality of the health monitoring components
to identify and diagnose import issues.
"""

import os
import sys
import logging
from importlib import import_module

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of component modules to test individually
components_to_test = [
    "ai_trading_agent.common.health_monitoring.component_health",
    "ai_trading_agent.common.health_monitoring.health_status",
    "ai_trading_agent.common.health_monitoring.alert_manager",
    "ai_trading_agent.common.health_monitoring.heartbeat_manager",
    "ai_trading_agent.common.health_monitoring.health_metrics",
    "ai_trading_agent.common.health_monitoring.recovery_coordinator",
    "ai_trading_agent.common.health_monitoring.health_monitor",
    "ai_trading_agent.agent.health_integrated_orchestrator"
]

# Test each component individually to pinpoint issues
def test_individual_components():
    results = {}
    
    for component in components_to_test:
        logger.info(f"Testing import of {component}...")
        try:
            module = import_module(component)
            classes = [name for name in dir(module) if not name.startswith("_") and name[0].isupper()]
            logger.info(f"  Success! Found classes: {', '.join(classes)}")
            results[component] = {"success": True, "classes": classes}
        except Exception as e:
            logger.error(f"  Failed to import {component}: {e}")
            results[component] = {"success": False, "error": str(e)}
    
    return results

# Create a mock agent for basic testing
class MockAgent:
    def __init__(self, agent_id="test_agent", name="Test Agent"):
        self.agent_id = agent_id
        self.name = name
        self.status = "INIT"
        
    def start(self):
        self.status = "RUNNING"
        logger.info(f"Agent {self.agent_id} started")
        
    def stop(self):
        self.status = "STOPPED"
        logger.info(f"Agent {self.agent_id} stopped")
        
    def process(self, inputs=None):
        logger.info(f"Agent {self.agent_id} processing inputs")
        return [{"data": "test_output"}]

# Test creating health metrics
def test_health_metrics():
    try:
        logger.info("Testing health_metrics.py...")
        from ai_trading_agent.common.health_monitoring.health_metrics import HealthMetrics, MetricThreshold, ThresholdType
        
        metrics = HealthMetrics()
        metrics.add_metric("test_component", "cpu_usage", 45.0)
        metrics.add_metric("test_component", "memory_usage", 128.5)
        
        # Add threshold
        metrics.add_metric_threshold(
            component_id="test_component",
            metric_name="cpu_usage",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=70.0,
            critical_threshold=90.0
        )
        
        # Get metrics
        all_metrics = metrics.get_metrics("test_component")
        logger.info(f"Metrics for test_component: {all_metrics}")
        
        # Check if threshold is exceeded
        exceeded = metrics.check_thresholds("test_component", "cpu_usage")
        logger.info(f"CPU usage threshold exceeded: {exceeded}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to test health metrics: {e}")
        return False

# Test component health tracking
def test_component_health():
    try:
        logger.info("Testing component_health.py...")
        from ai_trading_agent.common.health_monitoring.component_health import ComponentHealth
        from ai_trading_agent.common.health_monitoring.health_status import HealthStatus
        
        # Create component health
        component = ComponentHealth(
            component_id="test_component",
            description="Test Component"
        )
        
        # Update status
        component.update_status(HealthStatus.HEALTHY)
        logger.info(f"Component status: {component.status}")
        
        # Update heartbeat
        component.update_heartbeat()
        logger.info(f"Last heartbeat: {component.last_heartbeat}")
        
        # Check health
        is_healthy = component.is_healthy()
        logger.info(f"Component is healthy: {is_healthy}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to test component health: {e}")
        return False

# Test alert manager
def test_alert_manager():
    try:
        logger.info("Testing alert_manager.py...")
        from ai_trading_agent.common.health_monitoring.alert_manager import AlertManager, AlertData, AlertSeverity
        
        # Create alert manager
        alert_manager = AlertManager()
        
        # Add alert
        alert_manager.add_alert(
            component_id="test_component",
            severity=AlertSeverity.WARNING,
            message="Test warning message"
        )
        
        # Get alerts
        alerts = alert_manager.get_alerts()
        logger.info(f"Active alerts: {len(alerts)}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to test alert manager: {e}")
        return False

# Run all tests
def run_tests():
    logger.info("Starting basic health monitoring component tests")
    
    # Test individual component imports
    logger.info("\n=== Testing Individual Component Imports ===")
    import_results = test_individual_components()
    successful_imports = sum(1 for r in import_results.values() if r["success"])
    logger.info(f"Successfully imported {successful_imports}/{len(components_to_test)} components")
    
    # Test specific components if their imports succeeded
    if import_results.get("ai_trading_agent.common.health_monitoring.health_metrics", {}).get("success", False):
        logger.info("\n=== Testing Health Metrics ===")
        metrics_success = test_health_metrics()
        logger.info(f"Health Metrics Test: {'PASSED' if metrics_success else 'FAILED'}")
    
    if import_results.get("ai_trading_agent.common.health_monitoring.component_health", {}).get("success", False):
        logger.info("\n=== Testing Component Health ===")
        component_success = test_component_health()
        logger.info(f"Component Health Test: {'PASSED' if component_success else 'FAILED'}")
    
    if import_results.get("ai_trading_agent.common.health_monitoring.alert_manager", {}).get("success", False):
        logger.info("\n=== Testing Alert Manager ===")
        alert_success = test_alert_manager()
        logger.info(f"Alert Manager Test: {'PASSED' if alert_success else 'FAILED'}")
    
    logger.info("\n=== Test Summary ===")
    for component, result in import_results.items():
        status = "PASSED" if result["success"] else "FAILED"
        logger.info(f"{component}: {status}")
    
    # Overall success
    overall_success = all(r["success"] for r in import_results.values())
    logger.info(f"\nOverall Test Result: {'PASSED' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
