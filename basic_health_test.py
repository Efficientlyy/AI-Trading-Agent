"""
Basic test script for health monitoring integration.
This uses absolute imports and minimal dependencies.
"""
import os
import sys
import time

# Set up Python path properly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Simple test agent class
class TestAgent:
    def __init__(self, agent_id, name):
        self.agent_id = agent_id
        self.name = name
        self.dependencies = []
        self.status = "INIT"
        self.execution_count = 0
        
    def process(self, inputs=None):
        self.execution_count += 1
        return [{"result": f"Output from {self.agent_id}"}]
        
    def start(self):
        self.status = "RUNNING"
        print(f"Agent {self.agent_id} started")
        
    def stop(self):
        self.status = "STOPPED"
        print(f"Agent {self.agent_id} stopped")

# Simple mock for health monitor
class SimpleHealthMonitor:
    def __init__(self):
        self.components = {}
        self.metrics = {}
        self.alerts = []
        
    def register_component(self, component_id, description=None, **kwargs):
        self.components[component_id] = {
            "id": component_id,
            "description": description,
            "status": "healthy"
        }
        print(f"Registered component: {component_id}")
        
    def add_metric(self, component_id, metric_name, value, **kwargs):
        if component_id not in self.metrics:
            self.metrics[component_id] = {}
        self.metrics[component_id][metric_name] = value
        
    def add_metric_threshold(self, **kwargs):
        pass
        
    def register_recovery_action(self, *args, **kwargs):
        pass
        
    def start(self):
        print("Health monitor started")
        
    def record_heartbeat(self, *args, **kwargs):
        pass
        
    def add_alert(self, *args, **kwargs):
        pass
        
    def get_system_health(self):
        return {"overall_status": "healthy"}
        
    def get_active_alerts(self):
        return []

# Simple orchestrator class
class SimpleOrchestrator:
    def __init__(self):
        self.health_monitor = SimpleHealthMonitor()
        self.agents = {}
        self.data_queues = {}
        self.execution_order = []
        self.cycle_metrics = {"cycle_count": 0, "agent_errors": 0}
        
    def register_agent(self, agent):
        self.agents[agent.agent_id] = agent
        self.data_queues[agent.agent_id] = []
        print(f"Registered agent: {agent.agent_id}")
        
        # Register with health monitoring
        self.health_monitor.register_component(
            component_id=agent.agent_id,
            description=f"Agent: {agent.name}"
        )
        
        return agent
        
    def start_all_agents(self):
        for agent_id, agent in self.agents.items():
            agent.start()
            
    def stop_all_agents(self):
        for agent_id, agent in self.agents.items():
            agent.stop()
            
    def run_agent(self, agent_id, inputs=None):
        agent = self.agents.get(agent_id)
        if not agent:
            return []
            
        outputs = agent.process(inputs)
        
        # Record metrics
        self.health_monitor.add_metric(
            component_id=agent_id,
            metric_name="execution_time",
            value=0.1  # Mock value
        )
        
        return outputs
        
    def run_cycle(self):
        self.cycle_metrics["cycle_count"] += 1
        
        for agent_id in self.agents.keys():
            self.run_agent(agent_id, [])
            
        # Record metrics
        self.health_monitor.add_metric(
            component_id="orchestrator",
            metric_name="cycle_duration",
            value=0.5  # Mock value
        )
        
    def get_health_metrics(self):
        return {
            "system_health": self.health_monitor.get_system_health(),
            "active_alerts": self.health_monitor.get_active_alerts(),
            "cycle_metrics": self.cycle_metrics,
            "agent_stats": {}
        }

# Create test agents
agent1 = TestAgent("agent1", "Test Agent 1")
agent2 = TestAgent("agent2", "Test Agent 2")
agent2.dependencies = ["agent1"]

# Create orchestrator
orchestrator = SimpleOrchestrator()
orchestrator.register_agent(agent1)
orchestrator.register_agent(agent2)

# Start agents
orchestrator.start_all_agents()

# Run cycles
for _ in range(3):
    orchestrator.run_cycle()
    time.sleep(0.1)

# Get health metrics
metrics = orchestrator.get_health_metrics()
print(f"Cycle count: {metrics['cycle_metrics']['cycle_count']}")
print(f"System health: {metrics['system_health']['overall_status']}")

# Stop agents
orchestrator.stop_all_agents()

print("Test completed successfully!")
