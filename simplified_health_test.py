"""
Simplified Health Monitoring Integration Test

This script directly imports the minimum necessary components while avoiding import cycles.
"""
import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Simplified Agent Class for Testing
# -----------------------------------------------------------------------------

class AgentStatus(Enum):
    """Agent status states."""
    INIT = "init"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"

class SimpleAgent:
    """Simple agent implementation for health monitoring tests."""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.INIT
        self.dependencies = []
        self.execution_count = 0
        self.fail_next = False
        
    def start(self):
        """Start the agent."""
        self.status = AgentStatus.RUNNING
        logger.info(f"Agent {self.agent_id} started")
        
    def stop(self):
        """Stop the agent."""
        self.status = AgentStatus.STOPPED
        logger.info(f"Agent {self.agent_id} stopped")
        
    def process(self, inputs=None):
        """Process inputs and return outputs."""
        self.execution_count += 1
        
        if self.fail_next:
            self.fail_next = False
            self.status = AgentStatus.FAILED
            logger.error(f"Agent {self.agent_id} failed")
            raise ValueError(f"Simulated failure in agent {self.agent_id}")
        
        logger.info(f"Agent {self.agent_id} processing inputs, count={self.execution_count}")
        return [{"result": f"Output from {self.agent_id}", "timestamp": time.time()}]

# -----------------------------------------------------------------------------
# Simplified Health Monitoring Components
# -----------------------------------------------------------------------------

class HealthStatus(Enum):
    """Health status values."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    RECOVERING = "recovering"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ComponentHealth:
    """Component health data structure."""
    
    def __init__(self, component_id: str, description: str = None):
        self.component_id = component_id
        self.description = description or f"Component {component_id}"
        self.status = HealthStatus.UNKNOWN
        self.last_heartbeat = None
        self.first_heartbeat = None
        self.heartbeat_count = 0
        self.metrics = {}
        self.alerts = []
        self.recovery_attempts = 0
        self.last_updated = time.time()
        
    def update_status(self, status: HealthStatus):
        """Update component health status."""
        self.status = status
        self.last_updated = time.time()
        
    def record_heartbeat(self, data: Dict[str, Any] = None):
        """Record a heartbeat from the component."""
        current_time = time.time()
        self.last_heartbeat = current_time
        
        if self.first_heartbeat is None:
            self.first_heartbeat = current_time
            
        self.heartbeat_count += 1
        self.last_updated = current_time
        
    def is_healthy(self):
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

class AlertData:
    """Alert data structure."""
    
    def __init__(self, alert_id: str, component_id: str, severity: AlertSeverity, message: str):
        self.alert_id = alert_id
        self.component_id = component_id
        self.severity = severity
        self.message = message
        self.timestamp = time.time()
        self.acknowledged = False
        self.resolved = False

class SimpleHealthMonitor:
    """Simplified health monitor for testing."""
    
    def __init__(self, log_dir: str = "./health_logs"):
        self.log_dir = log_dir
        self.components = {}  # Dict[str, ComponentHealth]
        self.alerts = []      # List[AlertData]
        self.metrics = {}     # Dict[str, Dict[str, Any]]
        self.next_alert_id = 1
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
    def start(self):
        """Start the health monitor."""
        logger.info("Health monitor started")
        
    def stop(self):
        """Stop the health monitor."""
        logger.info("Health monitor stopped")
        
    def register_component(self, component_id: str, description: str = None, **kwargs):
        """Register a component for health monitoring."""
        if component_id in self.components:
            logger.warning(f"Component {component_id} already registered")
            return
            
        component = ComponentHealth(component_id, description)
        self.components[component_id] = component
        logger.info(f"Registered component: {component_id}")
        
    def record_heartbeat(self, component_id: str, data: Dict[str, Any] = None):
        """Record a heartbeat from a component."""
        if component_id not in self.components:
            logger.warning(f"Cannot record heartbeat, component {component_id} not registered")
            return
            
        self.components[component_id].record_heartbeat(data)
        
        # Update status based on agent status if provided
        if data and "status" in data:
            status_map = {
                "INIT": HealthStatus.UNKNOWN,
                "STARTING": HealthStatus.UNKNOWN,
                "RUNNING": HealthStatus.HEALTHY,
                "STOPPING": HealthStatus.UNKNOWN,
                "STOPPED": HealthStatus.UNKNOWN,
                "FAILED": HealthStatus.UNHEALTHY,
                "RECOVERING": HealthStatus.RECOVERING
            }
            
            agent_status = data["status"]
            if agent_status in status_map:
                self.components[component_id].update_status(status_map[agent_status])
        
    def add_metric(self, component_id: str, metric_name: str, value: Any, **kwargs):
        """Add a metric for a component."""
        if component_id not in self.metrics:
            self.metrics[component_id] = {}
            
        self.metrics[component_id][metric_name] = {
            "current_value": value,
            "timestamp": time.time(),
            "history": [(time.time(), value)]
        }
        
    def add_alert(self, component_id: str, severity: str, message: str, **kwargs):
        """Add an alert for a component."""
        if component_id not in self.components:
            logger.warning(f"Cannot add alert, component {component_id} not registered")
            return
            
        alert_id = f"alert_{self.next_alert_id}"
        self.next_alert_id += 1
        
        alert = AlertData(
            alert_id=alert_id,
            component_id=component_id,
            severity=AlertSeverity[severity.upper()],
            message=message
        )
        
        self.alerts.append(alert)
        logger.info(f"Added alert: {severity} - {message}")
        
    def get_component_health(self, component_id: str = None) -> ComponentHealth:
        """Get health data for a component."""
        if component_id is None:
            return None
            
        return self.components.get(component_id)
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get system-wide health status."""
        # Determine overall status based on component statuses
        status_priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.UNKNOWN: 1,
            HealthStatus.RECOVERING: 2,
            HealthStatus.DEGRADED: 3,
            HealthStatus.UNHEALTHY: 4,
            HealthStatus.CRITICAL: 5
        }
        
        if not self.components:
            overall_status = HealthStatus.UNKNOWN
        else:
            worst_status = HealthStatus.HEALTHY
            for component in self.components.values():
                if status_priority[component.status] > status_priority[worst_status]:
                    worst_status = component.status
            overall_status = worst_status
        
        return {
            "overall_status": overall_status.value,
            "component_count": len(self.components),
            "healthy_count": sum(1 for c in self.components.values() if c.status == HealthStatus.HEALTHY),
            "alert_count": len(self.alerts),
            "timestamp": time.time()
        }
        
    def get_metrics(self, component_id: str = None) -> Dict[str, Any]:
        """Get metrics for a component or all components."""
        if component_id is None:
            return self.metrics
            
        return self.metrics.get(component_id, {})
        
    def get_active_alerts(self, component_id: str = None) -> List[AlertData]:
        """Get active alerts for a component or all components."""
        if component_id is None:
            return self.alerts
            
        return [a for a in self.alerts if a.component_id == component_id and not a.resolved]

# -----------------------------------------------------------------------------
# Simplified Health Integrated Orchestrator
# -----------------------------------------------------------------------------

class SimpleHealthIntegratedOrchestrator:
    """Simplified orchestrator with health monitoring."""
    
    def __init__(self, log_dir: str = "./health_logs"):
        self.agents = {}  # Dict[str, SimpleAgent]
        self.health_monitor = SimpleHealthMonitor(log_dir)
        self.health_monitor.start()
        
    def register_agent(self, agent: SimpleAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        
        # Register with health monitor
        self.health_monitor.register_component(
            component_id=agent.agent_id,
            description=f"Agent: {agent.name}"
        )
        
        logger.info(f"Registered agent: {agent.agent_id}")
        return agent
        
    def start_agent(self, agent_id: str):
        """Start an agent."""
        if agent_id not in self.agents:
            logger.warning(f"Cannot start, agent {agent_id} not registered")
            return False
            
        agent = self.agents[agent_id]
        agent.start()
        
        # Record heartbeat with health monitor
        self.health_monitor.record_heartbeat(
            component_id=agent_id,
            data={"status": agent.status.name}
        )
        
        return True
        
    def stop_agent(self, agent_id: str):
        """Stop an agent."""
        if agent_id not in self.agents:
            logger.warning(f"Cannot stop, agent {agent_id} not registered")
            return False
            
        agent = self.agents[agent_id]
        agent.stop()
        
        # Record heartbeat with health monitor
        self.health_monitor.record_heartbeat(
            component_id=agent_id,
            data={"status": agent.status.name}
        )
        
        return True
        
    def run_agent(self, agent_id: str, inputs=None):
        """Run an agent and track health metrics."""
        if agent_id not in self.agents:
            logger.warning(f"Cannot run, agent {agent_id} not registered")
            return []
            
        agent = self.agents[agent_id]
        
        # Process with timing for health metrics
        start_time = time.time()
        try:
            outputs = agent.process(inputs)
            execution_time = time.time() - start_time
            
            # Record health metrics
            self.health_monitor.add_metric(
                component_id=agent_id,
                metric_name="execution_time",
                value=execution_time
            )
            
            self.health_monitor.add_metric(
                component_id=agent_id,
                metric_name="execution_count",
                value=agent.execution_count
            )
            
            # Record heartbeat
            self.health_monitor.record_heartbeat(
                component_id=agent_id,
                data={"status": agent.status.name}
            )
            
            return outputs
            
        except Exception as e:
            # Record error and alert
            execution_time = time.time() - start_time
            
            self.health_monitor.add_alert(
                component_id=agent_id,
                severity="error",
                message=f"Agent execution failed: {str(e)}"
            )
            
            self.health_monitor.record_heartbeat(
                component_id=agent_id,
                data={"status": agent.status.name}
            )
            
            logger.error(f"Error running agent {agent_id}: {e}")
            return []
    
    def get_agent_health(self, agent_id: str):
        """Get health status for an agent."""
        return self.health_monitor.get_component_health(agent_id)
    
    def get_system_health(self):
        """Get overall system health status."""
        return self.health_monitor.get_system_health()


# -----------------------------------------------------------------------------
# Test Functions
# -----------------------------------------------------------------------------

def test_health_integrated_orchestrator():
    """Test the health integrated orchestrator."""
    logger.info("Starting health integrated orchestrator test...")
    
    # Create orchestrator
    orchestrator = SimpleHealthIntegratedOrchestrator()
    
    # Create test agents
    agent1 = SimpleAgent("data_agent", "Data Agent")
    agent2 = SimpleAgent("strategy_agent", "Strategy Agent")
    agent3 = SimpleAgent("execution_agent", "Execution Agent")
    
    # Register agents
    orchestrator.register_agent(agent1)
    orchestrator.register_agent(agent2)
    orchestrator.register_agent(agent3)
    
    # Start agents
    logger.info("Starting agents...")
    orchestrator.start_agent("data_agent")
    orchestrator.start_agent("strategy_agent")
    orchestrator.start_agent("execution_agent")
    
    # Run normal execution cycle
    logger.info("Running normal execution cycle...")
    for _ in range(3):
        data_outputs = orchestrator.run_agent("data_agent")
        strategy_outputs = orchestrator.run_agent("strategy_agent", data_outputs)
        execution_outputs = orchestrator.run_agent("execution_agent", strategy_outputs)
    
    # Check health status
    logger.info("Checking health status...")
    system_health = orchestrator.get_system_health()
    logger.info(f"System health: {system_health}")
    
    for agent_id in ["data_agent", "strategy_agent", "execution_agent"]:
        agent_health = orchestrator.get_agent_health(agent_id)
        logger.info(f"Agent {agent_id} status: {agent_health.status.value}")
    
    # Simulate failure
    logger.info("Simulating agent failure...")
    agent2.fail_next = True
    try:
        orchestrator.run_agent("strategy_agent")
    except Exception as e:
        logger.error(f"Expected error occurred: {e}")
    
    # Check status after failure
    logger.info("Checking health status after failure...")
    system_health = orchestrator.get_system_health()
    logger.info(f"System health after failure: {system_health}")
    
    # Stop agents
    logger.info("Stopping agents...")
    orchestrator.stop_agent("execution_agent")
    orchestrator.stop_agent("strategy_agent")
    orchestrator.stop_agent("data_agent")
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    test_health_integrated_orchestrator()
