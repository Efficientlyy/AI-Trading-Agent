"""
Health Monitoring Integration Test Script

This script tests the integration between the Health Monitoring System and
the Trading Orchestrator, verifying that the fixed import structure works correctly.
"""
import os
import sys
import time
import logging
import threading
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_trading_agent.agent.agent_definitions import BaseAgent, AgentStatus, AgentRole
from ai_trading_agent.common.health_monitoring.core_definitions import (
    HealthStatus,
    AlertSeverity,
    ThresholdType
)
from ai_trading_agent.common.health_monitoring.health_monitor import HealthMonitor
from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator

class TestAgent(BaseAgent):
    """Sample agent for testing health monitoring."""
    
    def __init__(self, agent_id: str, name: str, fail_rate: float = 0.0):
        # Call the parent constructor with all required parameters
        super().__init__(
            agent_id=agent_id,
            name=name,
            agent_role=AgentRole.SPECIALIZED_TECHNICAL,  # Using proper AgentRole enum
            agent_type="TestAgent",
            status=AgentStatus.IDLE,
            inputs_from=[],
            outputs_to=[]
        )
        self.fail_rate = fail_rate
        self.execution_count = 0
        
    def process(self, inputs=None):
        """Process inputs and produce outputs."""
        self.execution_count += 1
        
        # Potentially simulate a failure
        if self.fail_rate > 0 and self.execution_count % int(1/self.fail_rate) == 0:
            logger.warning(f"Simulating failure in {self.agent_id}")
            raise ValueError(f"Simulated failure in {self.agent_id}")
            
        logger.info(f"Agent {self.agent_id} processing, execution #{self.execution_count}")
        
        # Update metrics for health monitoring
        self.update_metrics({
            "execution_count": self.execution_count,
            "last_execution_timestamp": time.time(),
            "processing_success": True
        })
        
        # Produce some sample outputs
        return [{"result": f"Output from {self.agent_id}", "timestamp": time.time()}]

def test_health_integrated_orchestrator():
    """Test the health integrated orchestrator with multiple agents."""
    log_dir = os.path.join(project_root, "health_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create orchestrator with health monitoring
    orchestrator = HealthIntegratedOrchestrator(log_dir=log_dir)
    
    # Create test agents
    agents = [
        TestAgent("data_agent", "Data Agent"),
        TestAgent("strategy_agent", "Strategy Agent", fail_rate=0.5),  # Fails every 2nd execution
        TestAgent("execution_agent", "Execution Agent")
    ]
    
    # Register dependencies
    agents[1].dependencies = ["data_agent"]
    agents[2].dependencies = ["strategy_agent"]
    
    # Register agents
    for agent in agents:
        orchestrator.register_agent(agent)
    
    # Start agents
    orchestrator.start_all_agents()
    
    try:
        # Run several cycles
        logger.info("Running normal processing cycles...")
        for i in range(5):
            logger.info(f"Starting cycle {i+1}")
            try:
                orchestrator.run_cycle()
                
                # Sleep between cycles
                time.sleep(0.5)
                
                # Get health status
                health = orchestrator.get_health_metrics()
                logger.info(f"System health: {health['system_health']['overall_status']}")
                
                # Get alerts if any
                alerts = orchestrator.health_monitor.get_active_alerts()
                if alerts:
                    logger.info(f"Active alerts: {len(alerts)}")
                    for alert in alerts:
                        logger.info(f"Alert: {alert.severity.value} - {alert.message}")
                
            except Exception as e:
                logger.error(f"Error in cycle {i+1}: {e}")
    
    finally:
        # Stop agents
        orchestrator.stop_all_agents()
        logger.info("Test completed")

def test_recovery_actions():
    """Test the recovery actions in the health integrated orchestrator."""
    log_dir = os.path.join(project_root, "health_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create orchestrator with health monitoring
    orchestrator = HealthIntegratedOrchestrator(log_dir=log_dir)
    
    # Create test agents with higher failure rates
    agents = [
        TestAgent("data_agent", "Data Agent"),
        TestAgent("strategy_agent", "Strategy Agent", fail_rate=1.0),  # Always fails
        TestAgent("execution_agent", "Execution Agent")
    ]
    
    # Register agents
    for agent in agents:
        orchestrator.register_agent(agent)
    
    # Start agents
    orchestrator.start_all_agents()
    
    try:
        # Run a cycle with guaranteed failure
        logger.info("Running cycle with expected failure...")
        orchestrator.run_cycle()
        
        # Check for alerts
        alerts = orchestrator.health_monitor.get_active_alerts()
        logger.info(f"Active alerts after failure: {len(alerts)}")
        for alert in alerts:
            logger.info(f"Alert: {alert.severity.value} - {alert.message}")
        
        # Get health status
        health = orchestrator.get_health_metrics()
        logger.info(f"System health after failure: {health['system_health']['overall_status']}")
        
        # Check component health
        strategy_health = orchestrator.health_monitor.get_component_health("strategy_agent")
        if strategy_health:
            # Access the health status from the dictionary
            if isinstance(strategy_health, dict) and 'status' in strategy_health:
                logger.info(f"Strategy agent health status: {strategy_health['status']}")
            # If it's an object with a status attribute
            elif hasattr(strategy_health, 'status'):
                logger.info(f"Strategy agent health status: {strategy_health.status.value}")
            else:
                logger.info(f"Strategy agent health info: {strategy_health}")
    
    finally:
        # Stop agents
        orchestrator.stop_all_agents()
        logger.info("Test completed")

if __name__ == "__main__":
    logger.info("Starting health monitoring integration tests")
    
    logger.info("\n=== Testing Health Integrated Orchestrator ===")
    test_health_integrated_orchestrator()
    
    logger.info("\n=== Testing Recovery Actions ===")
    test_recovery_actions()
    
    logger.info("\nAll tests completed")
