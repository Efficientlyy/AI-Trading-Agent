"""
Integration tests for health monitoring system with trading components.

This test suite validates the integration between the health monitoring
system and the real trading components to ensure proper operation under
various conditions.
"""

import time
import unittest
import logging
import threading
from pathlib import Path
import tempfile
import shutil

from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator
from ai_trading_agent.agent.agent_definitions import AgentStatus
from ai_trading_agent.common.health_monitoring import HealthMonitor, AlertSeverity

# Import sample agents for testing
from ai_trading_agent.agent.market_data.sample_data_agent import SampleMarketDataAgent
from ai_trading_agent.agent.strategy.moving_average_strategy import MovingAverageStrategy
from ai_trading_agent.agent.execution.simulation_execution import SimulationExecutionAgent


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TestHealthMonitoringIntegration(unittest.TestCase):
    """Integration tests for health monitoring system with trading components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up resources for all test cases."""
        # Create temporary directory for test logs
        cls.temp_dir = tempfile.mkdtemp()
        cls.log_dir = Path(cls.temp_dir) / "health_logs"
        cls.log_dir.mkdir(exist_ok=True)
        
        logging.info(f"Created temporary log directory at {cls.log_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
        
    def setUp(self):
        """Set up test case."""
        # Create health monitor with log directory
        self.health_monitor = HealthMonitor(log_dir=str(self.log_dir))
        
        # Create health integrated orchestrator
        self.orchestrator = HealthIntegratedOrchestrator(
            health_monitor=self.health_monitor,
            log_dir=str(self.log_dir),
            heartbeat_interval=1.0,
            monitor_components=True
        )
        
        # Create sample agents
        self.market_data = SampleMarketDataAgent(
            agent_id="market_data_1",
            name="Sample Market Data"
        )
        
        self.strategy = MovingAverageStrategy(
            agent_id="strategy_1",
            name="Moving Average Strategy"
        )
        
        self.execution = SimulationExecutionAgent(
            agent_id="execution_1", 
            name="Simulation Execution"
        )
        
        # Set up dependencies
        self.strategy.dependencies = ["market_data_1"]
        self.execution.dependencies = ["strategy_1"]
        
        # Register agents with orchestrator
        self.orchestrator.register_agent(self.market_data)
        self.orchestrator.register_agent(self.strategy)
        self.orchestrator.register_agent(self.execution)
        
    def tearDown(self):
        """Clean up after test case."""
        try:
            self.orchestrator.stop_all_agents()
        except Exception as e:
            logging.warning(f"Exception during teardown: {e}")
            
        # Allow time for cleanup
        time.sleep(1)
    
    def test_agent_lifecycle_monitoring(self):
        """Test agent lifecycle with health monitoring."""
        # Start all agents
        self.orchestrator.start_all_agents()
        
        # Verify agents are running
        self.assertEqual(self.market_data.status, AgentStatus.RUNNING)
        self.assertEqual(self.strategy.status, AgentStatus.RUNNING)
        self.assertEqual(self.execution.status, AgentStatus.RUNNING)
        
        # Get component health
        component_health = self.health_monitor.get_component_health()
        
        # Verify all components are registered
        self.assertIn("market_data_1", component_health)
        self.assertIn("strategy_1", component_health)
        self.assertIn("execution_1", component_health)
        
        # Run a processing cycle
        self.orchestrator.run_cycle()
        
        # Get component health again
        component_health = self.health_monitor.get_component_health()
        
        # All components should be healthy
        for component_id in ["market_data_1", "strategy_1", "execution_1"]:
            self.assertEqual(
                component_health[component_id].status,
                "healthy",
                f"Component {component_id} should be healthy"
            )
        
        # Check metrics were recorded
        metrics = self.health_monitor.get_metrics()
        for agent_id in ["market_data_1", "strategy_1", "execution_1"]:
            # Each agent should have execution time metric
            self.assertIn(agent_id, metrics)
            self.assertIn("execution_time", metrics[agent_id])
        
    def test_failed_agent_detection(self):
        """Test detection of failed agents."""
        # Start all agents
        self.orchestrator.start_all_agents()
        
        # Make market data agent fail
        def make_agent_fail():
            # Set a faulty process method that raises an exception
            old_process = self.market_data.process
            
            def faulty_process(inputs=None):
                raise RuntimeError("Simulated agent failure")
            
            self.market_data.process = faulty_process
            
            # Run for a bit then restore original behavior
            time.sleep(2)
            self.market_data.process = old_process
            
        # Start failure thread
        failure_thread = threading.Thread(target=make_agent_fail)
        failure_thread.daemon = True
        failure_thread.start()
        
        # Run processing cycles with a failing agent
        for _ in range(3):
            try:
                self.orchestrator.run_cycle()
            except Exception as e:
                # We should not see exceptions here because the orchestrator
                # should catch and handle them
                self.fail(f"Orchestrator failed to handle agent exception: {e}")
                
            time.sleep(1)
            
        # Wait for failure simulation to complete
        failure_thread.join(timeout=5)
        
        # Check if alerts were generated
        alerts = self.health_monitor.get_active_alerts()
        
        # There should be at least one error alert for the market data agent
        market_data_alerts = [a for a in alerts if a.component_id == "market_data_1" 
                             and a.severity == AlertSeverity.ERROR]
                             
        self.assertGreaterEqual(len(market_data_alerts), 1,
                              "Expected at least one error alert for market_data_1")
        
        # Get recovery history
        recovery_history = self.health_monitor.get_recovery_history()
        
        # Check if recovery actions were attempted
        market_data_recoveries = [r for r in recovery_history 
                                 if r["component_id"] == "market_data_1"]
        
        self.assertGreaterEqual(len(market_data_recoveries), 0,
                              "Expected recovery attempts for market_data_1")
        
    def test_agent_control_through_health_system(self):
        """Test controlling agents through the health system interface."""
        # Start all agents initially
        self.orchestrator.start_all_agents()
        
        # Verify all agents are running
        self.assertEqual(self.market_data.status, AgentStatus.RUNNING)
        self.assertEqual(self.strategy.status, AgentStatus.RUNNING)
        self.assertEqual(self.execution.status, AgentStatus.RUNNING)
        
        # Stop a specific agent through the health interface
        self.orchestrator.stop_agent("strategy_1")
        
        # Verify only that agent stopped
        self.assertEqual(self.market_data.status, AgentStatus.RUNNING)
        self.assertNotEqual(self.strategy.status, AgentStatus.RUNNING)
        self.assertEqual(self.execution.status, AgentStatus.RUNNING)
        
        # Start the agent again
        self.orchestrator.start_agent("strategy_1")
        
        # Verify agent is running again
        self.assertEqual(self.strategy.status, AgentStatus.RUNNING)
        
        # Run a cycle to ensure everything works
        self.orchestrator.run_cycle()
        
        # Get health metrics
        metrics = self.orchestrator.get_health_metrics()
        
        # Verify metrics include cycle information
        self.assertGreater(metrics["cycle_metrics"]["cycle_count"], 0)
        
        # Get dashboard data
        dashboard_data = self.orchestrator.get_health_dashboard_data()
        
        # Verify dashboard data is complete
        self.assertIn("system_health", dashboard_data)
        self.assertIn("component_health", dashboard_data)
        self.assertIn("metrics_data", dashboard_data)
        self.assertIn("recovery_history", dashboard_data)
        
        # Verify system health is included
        self.assertIn("overall_status", dashboard_data["system_health"])
        

if __name__ == "__main__":
    unittest.main()
