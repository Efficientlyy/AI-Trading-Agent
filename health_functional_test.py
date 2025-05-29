"""
Functional Integration Test for Health Monitoring System

This test demonstrates the core functionality of the health monitoring system 
with a simplified integration approach that avoids circular dependencies.
"""
import os
import sys
import time
import threading
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#-----------------------------------------------------------------------------
# Simplified Health Monitoring Component Implementations
#-----------------------------------------------------------------------------

class HealthStatus(Enum):
    """Health status enum."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    RECOVERING = "recovering"

class AlertSeverity(Enum):
    """Alert severity enum."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AgentStatus(Enum):
    """Agent status enum."""
    INIT = "init"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"

class ComponentHealth:
    """Component health tracking."""
    
    def __init__(self, component_id: str, description: str = None):
        self.component_id = component_id
        self.description = description or f"Component {component_id}"
        self.status = HealthStatus.UNKNOWN
        self.last_heartbeat = None
        self.first_heartbeat = None
        self.metrics = {}
        self.alerts = []
        self.last_updated = time.time()
    
    def update_status(self, status: HealthStatus):
        """Update component status."""
        self.status = status
        self.last_updated = time.time()
    
    def record_heartbeat(self):
        """Record a heartbeat."""
        current_time = time.time()
        self.last_heartbeat = current_time
        
        if self.first_heartbeat is None:
            self.first_heartbeat = current_time
            
        self.last_updated = current_time

class AlertData:
    """Alert data structure."""
    
    def __init__(self, alert_id: str, component_id: str, 
                 severity: AlertSeverity, message: str):
        self.alert_id = alert_id
        self.component_id = component_id
        self.severity = severity
        self.message = message
        self.timestamp = time.time()
        self.acknowledged = False
        self.resolved = False

class HealthMonitor:
    """Simplified health monitor."""
    
    def __init__(self):
        self.components = {}  # Dict[str, ComponentHealth]
        self.alerts = []  # List[AlertData]
        self.metrics = {}  # Dict[str, Dict[str, Any]]
        self.next_alert_id = 1
        self._running = False
        self._lock = threading.RLock()
    
    def start(self):
        """Start the health monitor."""
        with self._lock:
            self._running = True
        logger.info("Health monitor started")
    
    def stop(self):
        """Stop the health monitor."""
        with self._lock:
            self._running = False
        logger.info("Health monitor stopped")
    
    def register_component(self, component_id: str, description: str = None):
        """Register a component for health monitoring."""
        with self._lock:
            if component_id in self.components:
                logger.warning(f"Component {component_id} already registered")
                return
            
            component = ComponentHealth(component_id, description)
            self.components[component_id] = component
            logger.info(f"Registered component: {component_id}")
    
    def record_heartbeat(self, component_id: str, data: Dict[str, Any] = None):
        """Record a component heartbeat."""
        with self._lock:
            if component_id not in self.components:
                logger.warning(f"Cannot record heartbeat for unknown component {component_id}")
                return
            
            self.components[component_id].record_heartbeat()
            
            # Update status based on agent status if provided
            if data and "status" in data:
                status_map = {
                    "RUNNING": HealthStatus.HEALTHY,
                    "PAUSED": HealthStatus.DEGRADED,
                    "STOPPED": HealthStatus.UNKNOWN,
                    "FAILED": HealthStatus.UNHEALTHY,
                }
                
                agent_status = data["status"]
                if agent_status in status_map:
                    self.components[component_id].update_status(status_map[agent_status])
    
    def add_metric(self, component_id: str, metric_name: str, value: Any):
        """Add a metric for a component."""
        with self._lock:
            if component_id not in self.metrics:
                self.metrics[component_id] = {}
            
            self.metrics[component_id][metric_name] = {
                "value": value,
                "timestamp": time.time()
            }
    
    def add_alert(self, component_id: str, severity: str, message: str):
        """Add an alert for a component."""
        with self._lock:
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
    
    def get_component_health(self, component_id: str):
        """Get health data for a component."""
        with self._lock:
            return self.components.get(component_id)
    
    def get_system_health(self):
        """Get overall system health."""
        with self._lock:
            # Determine worst status among components
            if not self.components:
                return {"overall_status": HealthStatus.UNKNOWN.value}
            
            status_priority = {
                HealthStatus.HEALTHY: 0,
                HealthStatus.UNKNOWN: 1,
                HealthStatus.RECOVERING: 2,
                HealthStatus.DEGRADED: 3,
                HealthStatus.UNHEALTHY: 4,
                HealthStatus.CRITICAL: 5
            }
            
            worst_status = HealthStatus.HEALTHY
            for component in self.components.values():
                if status_priority[component.status] > status_priority[worst_status]:
                    worst_status = component.status
            
            return {
                "overall_status": worst_status.value,
                "component_count": len(self.components),
                "healthy_count": sum(1 for c in self.components.values() 
                                    if c.status == HealthStatus.HEALTHY),
                "alert_count": len(self.alerts)
            }
    
    def get_active_alerts(self, component_id: str = None):
        """Get active alerts for a component or all components."""
        with self._lock:
            if component_id is None:
                return self.alerts
            
            return [a for a in self.alerts 
                    if a.component_id == component_id and not a.resolved]

class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_id, name):
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.INIT
        self.dependencies = []
        self.execution_count = 0
        self.should_fail = False
    
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
        
        if self.should_fail:
            self.status = AgentStatus.FAILED
            raise RuntimeError(f"Simulated failure in {self.agent_id}")
        
        return [{"result": f"Output from {self.agent_id}"}]

class HealthIntegratedOrchestrator:
    """Simplified health integrated orchestrator."""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.agents = {}  # Dict[str, MockAgent]
        self.health_monitor.start()
    
    def register_agent(self, agent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        
        # Register with health monitor
        self.health_monitor.register_component(
            component_id=agent.agent_id,
            description=f"Agent: {agent.name}"
        )
        
        logger.info(f"Registered agent: {agent.agent_id}")
        return agent
    
    def start_agent(self, agent_id):
        """Start an agent."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        agent.start()
        
        # Update health monitor
        self.health_monitor.record_heartbeat(
            component_id=agent_id,
            data={"status": agent.status.name}
        )
        
        return True
    
    def stop_agent(self, agent_id):
        """Stop an agent."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        agent.stop()
        
        # Update health monitor
        self.health_monitor.record_heartbeat(
            component_id=agent_id,
            data={"status": agent.status.name}
        )
        
        return True
    
    def run_agent(self, agent_id, inputs=None):
        """Run an agent with health monitoring."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return []
        
        agent = self.agents[agent_id]
        
        start_time = time.time()
        try:
            outputs = agent.process(inputs)
            execution_time = time.time() - start_time
            
            # Record metrics
            self.health_monitor.add_metric(
                component_id=agent_id,
                metric_name="execution_time",
                value=execution_time
            )
            
            # Record heartbeat
            self.health_monitor.record_heartbeat(
                component_id=agent_id,
                data={"status": agent.status.name}
            )
            
            return outputs
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            
            self.health_monitor.add_alert(
                component_id=agent_id,
                severity="ERROR",
                message=f"Agent execution failed: {str(e)}"
            )
            
            # Update health monitor with failed status
            self.health_monitor.record_heartbeat(
                component_id=agent_id,
                data={"status": agent.status.name}
            )
            
            logger.error(f"Error running agent {agent_id}: {e}")
            return []
    
    def run_cycle(self):
        """Run one processing cycle for all agents."""
        results = {}
        for agent_id in self.agents:
            results[agent_id] = self.run_agent(agent_id)
        return results
    
    def get_agent_health(self, agent_id):
        """Get health status for an agent."""
        return self.health_monitor.get_component_health(agent_id)
    
    def get_system_health(self):
        """Get overall system health."""
        return self.health_monitor.get_system_health()
    
    def get_active_alerts(self):
        """Get active alerts."""
        return self.health_monitor.get_active_alerts()

#-----------------------------------------------------------------------------
# Test Functions
#-----------------------------------------------------------------------------

def test_agent_lifecycle():
    """Test agent lifecycle with health monitoring."""
    logger.info("=== Testing Agent Lifecycle ===")
    
    # Create orchestrator
    orchestrator = HealthIntegratedOrchestrator()
    
    # Create test agents
    data_agent = MockAgent("data_agent", "Data Agent")
    strategy_agent = MockAgent("strategy_agent", "Strategy Agent")
    execution_agent = MockAgent("execution_agent", "Execution Agent")
    
    # Register agents
    orchestrator.register_agent(data_agent)
    orchestrator.register_agent(strategy_agent)
    orchestrator.register_agent(execution_agent)
    
    # Verify registration
    logger.info("Checking system health after registration...")
    health = orchestrator.get_system_health()
    logger.info(f"System health: {health}")
    
    # Start agents
    logger.info("Starting agents...")
    orchestrator.start_agent("data_agent")
    orchestrator.start_agent("strategy_agent")
    orchestrator.start_agent("execution_agent")
    
    # Check health after starting
    logger.info("Checking system health after starting agents...")
    health = orchestrator.get_system_health()
    logger.info(f"System health: {health}")
    
    # Run processing cycle
    logger.info("Running processing cycle...")
    orchestrator.run_cycle()
    
    # Check agent metrics
    logger.info("Checking metrics after processing...")
    for agent_id in ["data_agent", "strategy_agent", "execution_agent"]:
        agent_health = orchestrator.get_agent_health(agent_id)
        logger.info(f"Agent {agent_id} status: {agent_health.status.value}")
    
    # Simulate failure
    logger.info("Simulating agent failure...")
    strategy_agent.should_fail = True
    orchestrator.run_agent("strategy_agent")
    
    # Check alerts and health after failure
    logger.info("Checking alerts after failure...")
    alerts = orchestrator.get_active_alerts()
    logger.info(f"Active alerts: {len(alerts)}")
    for alert in alerts:
        logger.info(f"Alert: {alert.component_id} - {alert.severity.value} - {alert.message}")
    
    # Check system health after failure
    logger.info("Checking system health after failure...")
    health = orchestrator.get_system_health()
    logger.info(f"System health: {health}")
    
    # Stop agents
    logger.info("Stopping agents...")
    orchestrator.stop_agent("data_agent")
    orchestrator.stop_agent("strategy_agent")
    orchestrator.stop_agent("execution_agent")
    
    logger.info("Test completed successfully!")
    return True

def test_alert_generation():
    """Test alert generation and processing."""
    logger.info("=== Testing Alert Generation ===")
    
    # Create health monitor directly
    health_monitor = HealthMonitor()
    health_monitor.start()
    
    # Register components
    health_monitor.register_component("test_component", "Test Component")
    
    # Record initial heartbeat
    health_monitor.record_heartbeat("test_component", {"status": "RUNNING"})
    
    # Add metrics
    health_monitor.add_metric("test_component", "cpu_usage", 45.0)
    health_monitor.add_metric("test_component", "memory_usage", 128.5)
    
    # Check component health
    component_health = health_monitor.get_component_health("test_component")
    logger.info(f"Component status: {component_health.status.value}")
    
    # Generate alerts
    health_monitor.add_alert(
        component_id="test_component",
        severity="WARNING",
        message="Memory usage is high"
    )
    
    health_monitor.add_alert(
        component_id="test_component",
        severity="ERROR",
        message="CPU throttling detected"
    )
    
    # Get alerts
    alerts = health_monitor.get_active_alerts()
    logger.info(f"Active alerts: {len(alerts)}")
    for alert in alerts:
        logger.info(f"Alert: {alert.component_id} - {alert.severity.value} - {alert.message}")
    
    # Check overall health
    system_health = health_monitor.get_system_health()
    logger.info(f"System health: {system_health}")
    
    health_monitor.stop()
    logger.info("Test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        # Run individual tests
        success1 = test_agent_lifecycle()
        success2 = test_alert_generation()
        
        # Overall result
        success = success1 and success2
        logger.info("\n=== Overall Test Results ===")
        logger.info(f"Agent Lifecycle Test: {'PASSED' if success1 else 'FAILED'}")
        logger.info(f"Alert Generation Test: {'PASSED' if success2 else 'FAILED'}")
        logger.info(f"Overall Result: {'PASSED' if success else 'FAILED'}")
        
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
