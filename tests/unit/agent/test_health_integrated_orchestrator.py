"""
Unit tests for the HealthIntegratedOrchestrator.

These tests verify that the health monitoring capabilities
are properly integrated with the trading orchestrator.
"""

import time
import unittest
from unittest.mock import patch, MagicMock, call

import sys
import os

# Add project root to the Python path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator
from ai_trading_agent.agent.agent_definitions import BaseAgent, AgentStatus
from ai_trading_agent.common.health_monitoring.health_monitor import HealthMonitor
from ai_trading_agent.common.health_monitoring.alert_manager import AlertSeverity
from ai_trading_agent.common.health_monitoring.health_status import HealthStatus


class TestAgent(BaseAgent):
    """A simple test agent for testing the orchestrator."""

    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id=agent_id, name=name)
        self.execution_count = 0
        self.process_called = False
        self.should_fail = False
        
    def process(self, inputs=None):
        self.process_called = True
        self.execution_count += 1
        
        if self.should_fail:
            raise ValueError("Test agent failure")
            
        return [{"data": "test_output"}]


class TestHealthIntegratedOrchestrator(unittest.TestCase):
    """Test cases for the HealthIntegratedOrchestrator class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a mock health monitor
        self.health_monitor = MagicMock(spec=HealthMonitor)
        
        # Create orchestrator
        self.orchestrator = HealthIntegratedOrchestrator(
            health_monitor=self.health_monitor,
            heartbeat_interval=1.0,
            monitor_components=True
        )
        
        # Create test agents
        self.agent1 = TestAgent("agent1", "Test Agent 1")
        self.agent2 = TestAgent("agent2", "Test Agent 2")
        
    def tearDown(self):
        """Clean up after tests."""
        self.orchestrator.stop_all_agents()
        
    def test_orchestrator_initialization(self):
        """Test that the orchestrator initializes properly with health monitoring."""
        # Verify health monitor was registered with the component
        self.health_monitor.register_component.assert_called_with(
            component_id="trading_orchestrator",
            description="Trading Orchestrator",
            heartbeat_config=unittest.mock.ANY
        )
        
        # Verify thresholds were added
        self.health_monitor.add_metric_threshold.assert_called()
        
        # Verify recovery actions were registered
        self.health_monitor.register_recovery_action.assert_called()
        
    def test_agent_registration(self):
        """Test that agents are properly registered with health monitoring."""
        # Register an agent
        self.orchestrator.register_agent(self.agent1)
        
        # Verify agent was registered with health monitor
        self.health_monitor.register_component.assert_any_call(
            component_id="agent1",
            description="Agent: Test Agent 1",
            heartbeat_config=unittest.mock.ANY
        )
        
        # Verify agent thresholds were added
        self.health_monitor.add_metric_threshold.assert_called()
        
        # Verify agent recovery actions were registered
        self.health_monitor.register_recovery_action.assert_called()
        
    def test_start_agent(self):
        """Test starting an agent with health monitoring."""
        # Register an agent
        self.orchestrator.register_agent(self.agent1)
        
        # Start the agent
        result = self.orchestrator.start_agent("agent1")
        
        # Verify agent was started
        self.assertTrue(result)
        self.assertEqual(self.agent1.status, AgentStatus.RUNNING)
        
        # Verify heartbeat was recorded
        self.health_monitor.record_heartbeat.assert_any_call(
            component_id="agent1",
            data=unittest.mock.ANY
        )
        
    def test_stop_agent(self):
        """Test stopping an agent with health monitoring."""
        # Register and start an agent
        self.orchestrator.register_agent(self.agent1)
        self.orchestrator.start_agent("agent1")
        
        # Reset mock to clear previous calls
        self.health_monitor.record_heartbeat.reset_mock()
        
        # Stop the agent
        result = self.orchestrator.stop_agent("agent1")
        
        # Verify agent was stopped
        self.assertTrue(result)
        self.assertNotEqual(self.agent1.status, AgentStatus.RUNNING)
        
        # Verify heartbeat was recorded
        self.health_monitor.record_heartbeat.assert_called_with(
            component_id="agent1",
            data=unittest.mock.ANY
        )
        
    def test_run_agent_success(self):
        """Test running an agent with successful execution."""
        # Register and start an agent
        self.orchestrator.register_agent(self.agent1)
        self.orchestrator.start_agent("agent1")
        
        # Reset mock to clear previous calls
        self.health_monitor.record_heartbeat.reset_mock()
        self.health_monitor.add_metric.reset_mock()
        
        # Run the agent
        outputs = self.orchestrator.run_agent("agent1", [{"data": "test_input"}])
        
        # Verify agent was executed
        self.assertTrue(self.agent1.process_called)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["data"], "test_output")
        
        # Verify heartbeat was recorded
        self.health_monitor.record_heartbeat.assert_called_with(
            component_id="agent1",
            data=unittest.mock.ANY
        )
        
        # Verify metrics were recorded
        self.health_monitor.add_metric.assert_any_call(
            component_id="agent1",
            metric_name="execution_time",
            value=unittest.mock.ANY
        )
        
        self.health_monitor.add_metric.assert_any_call(
            component_id="agent1",
            metric_name="output_count",
            value=1
        )
        
    def test_run_agent_failure(self):
        """Test running an agent with execution failure."""
        # Register and start an agent that will fail
        self.orchestrator.register_agent(self.agent1)
        self.agent1.should_fail = True
        self.orchestrator.start_agent("agent1")
        
        # Reset mock to clear previous calls
        self.health_monitor.record_heartbeat.reset_mock()
        self.health_monitor.add_metric.reset_mock()
        self.health_monitor.add_alert.reset_mock()
        
        # Run the agent (should fail)
        outputs = self.orchestrator.run_agent("agent1", [{"data": "test_input"}])
        
        # Verify execution failed but didn't crash
        self.assertEqual(len(outputs), 0)
        
        # Verify error metric was recorded
        self.health_monitor.add_metric.assert_any_call(
            component_id="agent1",
            metric_name="errors",
            value=1,
            increment=True
        )
        
        # Verify alert was generated
        self.health_monitor.add_alert.assert_called_with(
            component_id="agent1",
            severity=AlertSeverity.ERROR,
            message=unittest.mock.ANY,
            details=unittest.mock.ANY
        )
        
    def test_run_cycle_with_health_tracking(self):
        """Test running a complete orchestrator cycle with health tracking."""
        # Register two agents with a dependency
        self.agent1.dependencies = []
        self.agent2.dependencies = ["agent1"]
        
        self.orchestrator.register_agent(self.agent1)
        self.orchestrator.register_agent(self.agent2)
        
        # Start both agents
        self.orchestrator.start_all_agents()
        
        # Reset mock to clear previous calls
        self.health_monitor.record_heartbeat.reset_mock()
        self.health_monitor.add_metric.reset_mock()
        
        # Run a cycle
        self.orchestrator.run_cycle()
        
        # Verify both agents were executed in the right order
        self.assertTrue(self.agent1.process_called)
        self.assertTrue(self.agent2.process_called)
        
        # Verify cycle metrics were recorded
        self.health_monitor.add_metric.assert_any_call(
            component_id="trading_orchestrator",
            metric_name="cycle_duration",
            value=unittest.mock.ANY
        )
        
    def test_health_metrics_retrieval(self):
        """Test retrieving health metrics."""
        # Mock the health monitor's get methods
        self.health_monitor.get_system_health.return_value = {"overall_status": "healthy"}
        self.health_monitor.get_active_alerts.return_value = []
        
        # Get metrics
        metrics = self.orchestrator.get_health_metrics()
        
        # Verify metrics contain expected data
        self.assertIn("system_health", metrics)
        self.assertIn("active_alerts", metrics)
        self.assertIn("cycle_metrics", metrics)
        self.assertIn("agent_stats", metrics)
        
    def test_recovery_actions(self):
        """Test that recovery actions are properly registered and can be executed."""
        # Mock the recovery coordinator's register_recovery_action method
        self.health_monitor.recovery_coordinator.register_recovery_action = MagicMock()
        
        # Register an agent
        self.orchestrator.register_agent(self.agent1)
        
        # Verify orchestrator recovery actions were registered
        calls = self.health_monitor.register_recovery_action.call_args_list
        self.assertGreater(len(calls), 0)
        
        # Find the restart agent action for agent1
        restart_action = None
        for call in calls:
            args, kwargs = call
            if len(args) >= 3 and isinstance(args[0], str) and "restart_agent" in args[0] and "agent1" in args[0]:
                restart_action = args[2]
                break
                
        # Verify we found the restart action
        self.assertIsNotNone(restart_action)
        
        # Now execute the restart action function manually
        with patch.object(self.agent1, 'stop') as mock_stop, \
             patch.object(self.agent1, 'start') as mock_start:
            
            result = restart_action()
            
            # Verify agent was restarted
            self.assertTrue(result)
            mock_stop.assert_called_once()
            mock_start.assert_called_once()
            

if __name__ == "__main__":
    unittest.main()
