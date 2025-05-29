"""
Functional tests for health monitoring integration with trading agents.

This test focuses on practical functionality rather than unit testing.
"""

import os
import sys
import time
import unittest
import logging
from pathlib import Path

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import components with absolute imports to avoid circular dependencies
from ai_trading_agent.agent.agent_definitions import BaseAgent, AgentStatus
from ai_trading_agent.common.health_monitoring.health_monitor import HealthMonitor
from ai_trading_agent.agent.trading_orchestrator import TradingOrchestrator


# Create a test agent class
class TestAgent(BaseAgent):
    """Test agent for health monitoring integration testing."""

    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id=agent_id, name=name)
        self.execution_count = 0
        self.should_fail = False
        self.process_called = False
        
    def process(self, inputs=None):
        """Process inputs and produce outputs."""
        self.process_called = True
        self.execution_count += 1
        
        if self.should_fail:
            raise ValueError("Test agent failure")
            
        return [{"data": "test_output", "source": self.agent_id}]


class TestHealthIntegration(unittest.TestCase):
    """Integration tests for health monitoring with trading agents."""
    
    def setUp(self):
        """Set up test case."""
        # Create a temporary log directory for tests
        self.test_dir = Path("test_health_logs")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create health monitor
        self.health_monitor = HealthMonitor(log_dir=str(self.test_dir))
        
        # Create orchestrator
        self.orchestrator = TradingOrchestrator()
        
        # Create test agents
        self.agent1 = TestAgent("test_data_agent", "Test Data Agent")
        self.agent2 = TestAgent("test_strategy_agent", "Test Strategy Agent")
        
        # Set up dependencies
        self.agent2.dependencies = ["test_data_agent"]
        
        # Register agents
        self.orchestrator.register_agent(self.agent1)
        self.orchestrator.register_agent(self.agent2)
        
        # Register components with health monitor
        self.health_monitor.register_component(
            component_id="test_data_agent", 
            description="Test Data Agent"
        )
        
        self.health_monitor.register_component(
            component_id="test_strategy_agent",
            description="Test Strategy Agent"
        )
        
        # Start health monitor
        self.health_monitor.start()
    
    def tearDown(self):
        """Clean up after test case."""
        try:
            # Stop health monitoring
            self.health_monitor.stop()
            
            # Clean up temporary files
            for file in self.test_dir.glob("*"):
                try:
                    file.unlink()
                except Exception:
                    pass
                    
            self.test_dir.rmdir()
        except Exception as e:
            logging.warning(f"Error during tearDown: {e}")
    
    def test_health_status_updates(self):
        """Test that health status is updated properly."""
        # Start agent
        self.agent1.start()
        
        # Record heartbeat
        self.health_monitor.record_heartbeat(
            component_id="test_data_agent",
            data={"status": self.agent1.status.name}
        )
        
        # Get component health
        health = self.health_monitor.get_component_health("test_data_agent")
        
        # Check status
        self.assertIsNotNone(health)
        self.assertEqual(health.status, "healthy")
        
        # Stop agent
        self.agent1.stop()
        
        # Record heartbeat with stopped status
        self.health_monitor.record_heartbeat(
            component_id="test_data_agent",
            data={"status": self.agent1.status.name}
        )
    
    def test_metrics_collection(self):
        """Test that metrics are collected properly."""
        # Add a test metric
        self.health_monitor.add_metric(
            component_id="test_data_agent",
            metric_name="execution_time",
            value=0.1
        )
        
        # Get metrics
        metrics = self.health_monitor.get_metrics("test_data_agent")
        
        # Check metrics
        self.assertIn("execution_time", metrics)
        self.assertEqual(metrics["execution_time"].current_value, 0.1)
    
    def test_alert_generation(self):
        """Test that alerts are generated properly."""
        # Add an alert
        self.health_monitor.add_alert(
            component_id="test_data_agent",
            severity="warning",
            message="Test alert message"
        )
        
        # Get active alerts
        alerts = self.health_monitor.get_active_alerts("test_data_agent")
        
        # Check alerts
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].component_id, "test_data_agent")
        self.assertEqual(alerts[0].severity, "warning")
    
    def test_execution_metrics(self):
        """Test tracking execution metrics during agent processing."""
        # Start the agent
        self.agent1.start()
        
        # Process some data
        start_time = time.time()
        outputs = self.agent1.process([{"data": "test_input"}])
        duration = time.time() - start_time
        
        # Record metrics
        self.health_monitor.add_metric(
            component_id="test_data_agent",
            metric_name="execution_time",
            value=duration
        )
        
        self.health_monitor.add_metric(
            component_id="test_data_agent",
            metric_name="output_count",
            value=len(outputs)
        )
        
        # Get metrics
        metrics = self.health_monitor.get_metrics("test_data_agent")
        
        # Check metrics
        self.assertIn("execution_time", metrics)
        self.assertIn("output_count", metrics)
        self.assertEqual(metrics["output_count"].current_value, 1)


if __name__ == "__main__":
    unittest.main()
