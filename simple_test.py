"""
Simple test script to verify basic functionality of the Health Monitoring system.
"""

import logging
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the required modules directly
from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator
from ai_trading_agent.agent.agent_definitions import BaseAgent

# Define a simple test agent
class SimpleTestAgent(BaseAgent):
    def __init__(self, agent_id, name):
        super().__init__(agent_id=agent_id, name=name)
        
    def process(self, inputs=None):
        print(f"Agent {self.agent_id} processing...")
        return [{"result": f"output from {self.agent_id}"}]

def main():
    print("Starting Health Monitoring System test...")
    
    try:
        # Create orchestrator
        print("Creating HealthIntegratedOrchestrator...")
        orchestrator = HealthIntegratedOrchestrator(
            log_dir=None,
            heartbeat_interval=1.0
        )
        
        print("HealthIntegratedOrchestrator created successfully!")
        
        # Create and register test agents
        print("Creating and registering test agents...")
        agent1 = SimpleTestAgent("test_agent_1", "Test Agent 1")
        agent2 = SimpleTestAgent("test_agent_2", "Test Agent 2")
        
        # Set up dependencies
        agent2.dependencies = ["test_agent_1"]
        
        # Register with orchestrator
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)
        
        print("Agents registered successfully!")
        
        # Start agents
        print("Starting agents...")
        orchestrator.start_all_agents()
        
        print("Running a processing cycle...")
        orchestrator.run_cycle()
        
        # Get health metrics
        print("Getting health metrics...")
        metrics = orchestrator.get_health_metrics()
        
        print("Health metrics:")
        print(f"  System health: {metrics['system_health'].get('overall_status', 'unknown')}")
        print(f"  Active alerts: {len(metrics['active_alerts'])}")
        print(f"  Cycle count: {metrics['cycle_metrics']['cycle_count']}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
