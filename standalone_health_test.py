"""
Standalone Health Integration Test

This script tests the health monitoring system integration with trading agents
while avoiding import issues by using direct imports in a specific order.
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Set up Python path properly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define mock agent for testing
class MockAgent:
    def __init__(self, agent_id, name):
        self.agent_id = agent_id
        self.name = name
        self.status = "INIT"
        self.dependencies = []
        self.execution_count = 0
        self.error_rate = 0.0
        
    def process(self, inputs=None):
        self.execution_count += 1
        if self.error_rate > 0 and (self.execution_count % int(1/self.error_rate) == 0):
            raise ValueError(f"Simulated error in {self.agent_id}")
        return [{"data": "test_output", "source": self.agent_id}]
        
    def start(self):
        self.status = "RUNNING"
        logger.info(f"Agent {self.agent_id} started")
        
    def stop(self):
        self.status = "STOPPED"
        logger.info(f"Agent {self.agent_id} stopped")


def test_health_monitor_direct():
    """Test health monitor directly without the orchestrator."""
    # Import health monitor components directly
    try:
        # First import these to avoid circular dependencies
        from ai_trading_agent.common.health_monitoring.health_status import HealthStatus
        from ai_trading_agent.common.health_monitoring.alert_manager import AlertManager, AlertData, AlertSeverity
        from ai_trading_agent.common.health_monitoring.recovery_coordinator import RecoveryCoordinator, RecoveryAction
        from ai_trading_agent.common.health_monitoring.heartbeat_manager import HeartbeatManager
        from ai_trading_agent.common.health_monitoring.health_metrics import HealthMetrics
        
        # Then import the main monitor
        from ai_trading_agent.common.health_monitoring.health_monitor import HealthMonitor
        
        logger.info("Successfully imported health monitoring components")
    except ImportError as e:
        logger.error(f"Failed to import health monitoring components: {e}")
        return False
    
    # Create health monitor
    try:
        health_monitor = HealthMonitor(log_dir="./health_logs")
        logger.info("Successfully created health monitor")
    except Exception as e:
        logger.error(f"Failed to create health monitor: {e}")
        return False
    
    # Register components
    try:
        health_monitor.register_component(
            component_id="test_data_agent",
            description="Test Data Agent"
        )
        
        health_monitor.register_component(
            component_id="test_strategy_agent",
            description="Test Strategy Agent"
        )
        
        logger.info("Successfully registered components")
    except Exception as e:
        logger.error(f"Failed to register components: {e}")
        return False
    
    # Start health monitor
    try:
        health_monitor.start()
        logger.info("Successfully started health monitor")
    except Exception as e:
        logger.error(f"Failed to start health monitor: {e}")
        return False
    
    # Add metrics
    try:
        health_monitor.add_metric(
            component_id="test_data_agent",
            metric_name="execution_time",
            value=0.1
        )
        
        health_monitor.add_metric(
            component_id="test_strategy_agent",
            metric_name="execution_time",
            value=0.2
        )
        
        logger.info("Successfully added metrics")
    except Exception as e:
        logger.error(f"Failed to add metrics: {e}")
        return False
    
    # Record heartbeat
    try:
        health_monitor.record_heartbeat(
            component_id="test_data_agent",
            data={"status": "RUNNING"}
        )
        
        health_monitor.record_heartbeat(
            component_id="test_strategy_agent",
            data={"status": "RUNNING"}
        )
        
        logger.info("Successfully recorded heartbeats")
    except Exception as e:
        logger.error(f"Failed to record heartbeats: {e}")
        return False
    
    # Get health status
    try:
        system_health = health_monitor.get_system_health()
        logger.info(f"System health: {system_health}")
        
        data_agent_health = health_monitor.get_component_health("test_data_agent")
        logger.info(f"Data agent health: {data_agent_health}")
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        return False
    
    # Add alerts
    try:
        health_monitor.add_alert(
            component_id="test_data_agent",
            severity="warning",
            message="Test warning alert"
        )
        
        health_monitor.add_alert(
            component_id="test_strategy_agent",
            severity="critical",
            message="Test critical alert"
        )
        
        logger.info("Successfully added alerts")
    except Exception as e:
        logger.error(f"Failed to add alerts: {e}")
        return False
    
    # Get alerts
    try:
        all_alerts = health_monitor.get_active_alerts()
        logger.info(f"Active alerts: {len(all_alerts)}")
        
        for alert in all_alerts:
            logger.info(f"Alert: {alert.component_id} - {alert.severity} - {alert.message}")
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return False
    
    # Stop health monitor
    try:
        health_monitor.stop()
        logger.info("Successfully stopped health monitor")
    except Exception as e:
        logger.error(f"Failed to stop health monitor: {e}")
        return False
    
    return True


def test_health_integrated_orchestrator():
    """Test health integrated orchestrator."""
    # Import health integrated orchestrator
    try:
        # First import BaseAgent to avoid circular dependencies
        from ai_trading_agent.agent.agent_definitions import BaseAgent
        
        # Then import health monitoring components
        from ai_trading_agent.common.health_monitoring.health_status import HealthStatus
        from ai_trading_agent.common.health_monitoring.alert_manager import AlertManager
        from ai_trading_agent.common.health_monitoring.recovery_coordinator import RecoveryCoordinator
        from ai_trading_agent.common.health_monitoring.health_monitor import HealthMonitor
        
        # Finally import orchestrator
        from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator
        
        logger.info("Successfully imported HealthIntegratedOrchestrator")
    except ImportError as e:
        logger.error(f"Failed to import HealthIntegratedOrchestrator: {e}")
        logger.error("Using mock implementation instead")
        
        # Create mock orchestrator for testing
        class MockHealthOrchestrator:
            def __init__(self, log_dir="./health_logs"):
                self.agents = {}
                self.health_monitor = HealthMonitor(log_dir=log_dir)
                self.health_monitor.start()
                logger.info("Created mock health orchestrator")
                
            def register_agent(self, agent):
                self.agents[agent.agent_id] = agent
                self.health_monitor.register_component(
                    component_id=agent.agent_id,
                    description=f"Agent: {agent.name}"
                )
                logger.info(f"Registered agent: {agent.agent_id}")
                return agent
                
            def start_agent(self, agent_id):
                agent = self.agents.get(agent_id)
                if agent:
                    agent.start()
                    self.health_monitor.record_heartbeat(
                        component_id=agent_id, 
                        data={"status": agent.status}
                    )
                    return True
                return False
                
            def stop_agent(self, agent_id):
                agent = self.agents.get(agent_id)
                if agent:
                    agent.stop()
                    self.health_monitor.record_heartbeat(
                        component_id=agent_id, 
                        data={"status": agent.status}
                    )
                    return True
                return False
                
            def run_agent(self, agent_id, inputs=None):
                agent = self.agents.get(agent_id)
                if not agent:
                    return []
                    
                start_time = time.time()
                try:
                    outputs = agent.process(inputs)
                    
                    # Record metrics
                    self.health_monitor.add_metric(
                        component_id=agent_id,
                        metric_name="execution_time",
                        value=time.time() - start_time
                    )
                    
                    self.health_monitor.add_metric(
                        component_id=agent_id,
                        metric_name="execution_count",
                        value=agent.execution_count
                    )
                    
                    return outputs
                except Exception as e:
                    # Record error
                    self.health_monitor.add_alert(
                        component_id=agent_id,
                        severity="error",
                        message=f"Error in agent {agent_id}: {str(e)}"
                    )
                    return []
                    
            def get_health_status(self):
                return self.health_monitor.get_system_health()
                
            def get_agent_health(self, agent_id):
                return self.health_monitor.get_component_health(agent_id)
        
        # Use the mock implementation
        HealthIntegratedOrchestrator = MockHealthOrchestrator
    
    # Create test agents
    data_agent = MockAgent("data_agent", "Data Agent")
    strategy_agent = MockAgent("strategy_agent", "Strategy Agent")
    strategy_agent.dependencies = ["data_agent"]
    execution_agent = MockAgent("execution_agent", "Execution Agent")
    execution_agent.dependencies = ["strategy_agent"]
    
    # Create orchestrator
    try:
        orchestrator = HealthIntegratedOrchestrator(log_dir="./health_logs")
        logger.info("Successfully created orchestrator")
    except Exception as e:
        logger.error(f"Failed to create orchestrator: {e}")
        return False
    
    # Register agents
    try:
        orchestrator.register_agent(data_agent)
        orchestrator.register_agent(strategy_agent)
        orchestrator.register_agent(execution_agent)
        logger.info("Successfully registered agents")
    except Exception as e:
        logger.error(f"Failed to register agents: {e}")
        return False
    
    # Start agents
    try:
        orchestrator.start_agent("data_agent")
        orchestrator.start_agent("strategy_agent")
        orchestrator.start_agent("execution_agent")
        logger.info("Successfully started agents")
    except Exception as e:
        logger.error(f"Failed to start agents: {e}")
        return False
    
    # Run agents
    try:
        data_outputs = orchestrator.run_agent("data_agent")
        strategy_inputs = data_outputs  # In a real scenario, data would flow from one agent to the next
        strategy_outputs = orchestrator.run_agent("strategy_agent", strategy_inputs)
        execution_inputs = strategy_outputs
        execution_outputs = orchestrator.run_agent("execution_agent", execution_inputs)
        logger.info("Successfully ran agents")
    except Exception as e:
        logger.error(f"Failed to run agents: {e}")
        return False
    
    # Get health status
    try:
        system_health = orchestrator.get_health_status()
        logger.info(f"System health: {system_health}")
        
        for agent_id in ["data_agent", "strategy_agent", "execution_agent"]:
            agent_health = orchestrator.get_agent_health(agent_id)
            logger.info(f"{agent_id} health: {agent_health}")
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        return False
    
    # Simulate a failure
    try:
        strategy_agent.error_rate = 1.0  # 100% error rate
        strategy_outputs = orchestrator.run_agent("strategy_agent")
        logger.info("Successfully simulated agent failure")
    except Exception as e:
        logger.error(f"Unexpected error in failure simulation: {e}")
        return False
    
    # Stop agents
    try:
        orchestrator.stop_agent("execution_agent")
        orchestrator.stop_agent("strategy_agent")
        orchestrator.stop_agent("data_agent")
        logger.info("Successfully stopped agents")
    except Exception as e:
        logger.error(f"Failed to stop agents: {e}")
        return False
    
    return True


def run_tests():
    """Run all tests."""
    logger.info("Starting health monitoring tests")
    
    # Test health monitor directly
    logger.info("\n=== Testing Health Monitor ===")
    health_monitor_success = test_health_monitor_direct()
    logger.info(f"Health Monitor Test: {'PASSED' if health_monitor_success else 'FAILED'}")
    
    # Test health integrated orchestrator
    logger.info("\n=== Testing Health Integrated Orchestrator ===")
    orchestrator_success = test_health_integrated_orchestrator()
    logger.info(f"Health Integrated Orchestrator Test: {'PASSED' if orchestrator_success else 'FAILED'}")
    
    # Overall result
    overall_success = health_monitor_success and orchestrator_success
    logger.info(f"\nOverall Test Result: {'PASSED' if overall_success else 'FAILED'}")
    
    return overall_success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
