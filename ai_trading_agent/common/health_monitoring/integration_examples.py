"""
Health Monitoring Integration Examples.

This module provides example integrations of the Health Monitoring System
with other components of the AI Trading Agent, particularly the Trading Orchestrator.
"""

import time
import logging
import threading
from typing import Any, Dict, Optional, List, Union, Callable

from ai_trading_agent.common.health_monitoring import (
    HealthMonitor,
    HeartbeatConfig,
    HealthStatus,
    AlertSeverity,
    ThresholdType,
    MetricThreshold,
    RecoveryAction
)

# Set up logger
logger = logging.getLogger(__name__)


class OrchestratorHealthAdapter:
    """
    Adapter for integrating health monitoring with the Trading Orchestrator.
    
    Provides hooks for heartbeat generation, metric collection, and recovery actions
    to be used with the trading orchestrator component.
    """
    
    def __init__(
        self,
        health_monitor: HealthMonitor,
        orchestrator_id: str = "trading_orchestrator",
        heartbeat_interval: float = 5.0
    ):
        """
        Initialize the orchestrator health adapter.
        
        Args:
            health_monitor: Health monitoring system instance
            orchestrator_id: ID for the orchestrator component
            heartbeat_interval: Interval for heartbeat generation in seconds
        """
        self.health_monitor = health_monitor
        self.orchestrator_id = orchestrator_id
        self.heartbeat_interval = heartbeat_interval
        
        # Configure heartbeat
        heartbeat_config = HeartbeatConfig(
            interval=heartbeat_interval,
            missing_threshold=1,
            degraded_threshold=2,
            unhealthy_threshold=3
        )
        
        # Register orchestrator with health monitor
        self.health_monitor.register_component(
            component_id=orchestrator_id,
            description="Trading Orchestrator",
            heartbeat_config=heartbeat_config,
            monitors=["heartbeat"]
        )
        
        # Add performance metrics thresholds
        self._add_metric_thresholds()
        
        # Register recovery actions
        self._register_recovery_actions()
        
        # Setup heartbeat thread
        self._heartbeat_thread = None
        self._running = False
    
    def _add_metric_thresholds(self) -> None:
        """Add metric thresholds for orchestrator monitoring."""
        # Cycle duration threshold
        self.health_monitor.add_metric_threshold(
            metric_name="cycle_duration",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=1.0,  # seconds
            critical_threshold=5.0,  # seconds
            component_id=self.orchestrator_id,
            description="Maximum time for a trading cycle"
        )
        
        # Agent processing errors
        self.health_monitor.add_metric_threshold(
            metric_name="agent_errors",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=1,
            critical_threshold=3,
            component_id=self.orchestrator_id,
            description="Number of agent processing errors per cycle"
        )
        
        # Trading decision latency
        self.health_monitor.add_metric_threshold(
            metric_name="decision_latency",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=0.5,  # seconds
            critical_threshold=2.0,  # seconds
            component_id=self.orchestrator_id,
            description="Maximum latency for trading decisions"
        )
    
    def _register_recovery_actions(self) -> None:
        """Register recovery actions for orchestrator issues."""
        # Action to restart orchestrator cycle
        self.health_monitor.register_recovery_action(
            action_id="restart_orchestrator_cycle",
            description="Restart the trading orchestrator cycle",
            action_func=self._restart_cycle_action,
            component_id=self.orchestrator_id,
            severity_threshold=AlertSeverity.ERROR
        )
        
        # Action to reset agent states
        self.health_monitor.register_recovery_action(
            action_id="reset_agent_states",
            description="Reset all agent states to recover from inconsistent state",
            action_func=self._reset_agent_states_action,
            component_id=self.orchestrator_id,
            severity_threshold=AlertSeverity.ERROR
        )
    
    def _restart_cycle_action(self) -> bool:
        """
        Recovery action to restart orchestrator cycle.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing recovery action: Restart orchestrator cycle")
        # In a real implementation, this would hook into the actual orchestrator
        # This is a placeholder for demonstration
        time.sleep(0.5)  # Simulate some work
        return True
    
    def _reset_agent_states_action(self) -> bool:
        """
        Recovery action to reset agent states.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing recovery action: Reset agent states")
        # In a real implementation, this would hook into the actual agents
        # This is a placeholder for demonstration
        time.sleep(1.0)  # Simulate some work
        return True
    
    def start_heartbeat(self) -> None:
        """Start the heartbeat generation thread."""
        if self._running:
            logger.warning("Heartbeat already running")
            return
            
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._generate_heartbeats,
            name="OrchestratorHeartbeat",
            daemon=True
        )
        self._heartbeat_thread.start()
        
        logger.info(f"Started heartbeat generation for {self.orchestrator_id}")
    
    def stop_heartbeat(self) -> None:
        """Stop the heartbeat generation thread."""
        if not self._running:
            logger.warning("Heartbeat already stopped")
            return
            
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None
            
        logger.info(f"Stopped heartbeat generation for {self.orchestrator_id}")
    
    def _generate_heartbeats(self) -> None:
        """Background thread for generating heartbeats."""
        logger.info("Heartbeat generation thread started")
        
        while self._running:
            try:
                # Generate heartbeat with diagnostics data
                diagnostics = {
                    "active_agents": 5,  # Example value
                    "pending_signals": 2,  # Example value
                    "memory_usage_mb": 256.5  # Example value
                }
                
                self.health_monitor.record_heartbeat(
                    component_id=self.orchestrator_id,
                    data=diagnostics
                )
                
                # Sleep until next heartbeat
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat generation: {str(e)}")
                time.sleep(1.0)  # Sleep briefly before retry
                
        logger.info("Heartbeat generation thread stopped")
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record metrics from orchestrator operation.
        
        Args:
            metrics: Dictionary of metrics to record
        """
        # Record each metric individually
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.health_monitor.add_metric(
                    component_id=self.orchestrator_id,
                    metric_name=name,
                    value=value
                )


class AgentHealthAdapter:
    """
    Adapter for integrating health monitoring with Trading Agents.
    
    Provides hooks for heartbeat generation, metric collection, and recovery actions
    to be used with individual trading agent components.
    """
    
    def __init__(
        self,
        health_monitor: HealthMonitor,
        agent_id: str,
        agent_type: str,
        heartbeat_interval: float = 10.0
    ):
        """
        Initialize the agent health adapter.
        
        Args:
            health_monitor: Health monitoring system instance
            agent_id: ID for the agent component
            agent_type: Type of the agent (e.g., "data", "strategy", "execution")
            heartbeat_interval: Interval for heartbeat generation in seconds
        """
        self.health_monitor = health_monitor
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.heartbeat_interval = heartbeat_interval
        
        # Configure heartbeat
        heartbeat_config = HeartbeatConfig(
            interval=heartbeat_interval,
            missing_threshold=1,
            degraded_threshold=2,
            unhealthy_threshold=3
        )
        
        # Register agent with health monitor
        self.health_monitor.register_component(
            component_id=agent_id,
            description=f"{agent_type.capitalize()} Agent: {agent_id}",
            heartbeat_config=heartbeat_config,
            monitors=["heartbeat"]
        )
        
        # Add performance metrics thresholds
        self._add_metric_thresholds()
        
        # Register recovery actions
        self._register_recovery_actions()
        
        # Setup heartbeat thread
        self._heartbeat_thread = None
        self._running = False
    
    def _add_metric_thresholds(self) -> None:
        """Add metric thresholds for agent monitoring."""
        # Processing time threshold
        self.health_monitor.add_metric_threshold(
            metric_name="processing_time",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=0.5,  # seconds
            critical_threshold=2.0,  # seconds
            component_id=self.agent_id,
            description="Maximum processing time for agent"
        )
        
        # Error rate threshold
        self.health_monitor.add_metric_threshold(
            metric_name="error_rate",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=0.05,  # 5%
            critical_threshold=0.20,  # 20%
            component_id=self.agent_id,
            description="Maximum error rate for agent"
        )
    
    def _register_recovery_actions(self) -> None:
        """Register recovery actions for agent issues."""
        # Action to restart agent
        self.health_monitor.register_recovery_action(
            action_id=f"restart_{self.agent_id}",
            description=f"Restart the {self.agent_id} agent",
            action_func=self._restart_agent_action,
            component_id=self.agent_id,
            severity_threshold=AlertSeverity.ERROR
        )
        
        # Action to reset agent state
        self.health_monitor.register_recovery_action(
            action_id=f"reset_{self.agent_id}_state",
            description=f"Reset the {self.agent_id} agent state",
            action_func=self._reset_agent_state_action,
            component_id=self.agent_id,
            severity_threshold=AlertSeverity.WARNING
        )
    
    def _restart_agent_action(self) -> bool:
        """
        Recovery action to restart agent.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing recovery action: Restart agent {self.agent_id}")
        # In a real implementation, this would hook into the actual agent
        # This is a placeholder for demonstration
        time.sleep(0.5)  # Simulate some work
        return True
    
    def _reset_agent_state_action(self) -> bool:
        """
        Recovery action to reset agent state.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing recovery action: Reset state for agent {self.agent_id}")
        # In a real implementation, this would hook into the actual agent
        # This is a placeholder for demonstration
        time.sleep(0.3)  # Simulate some work
        return True
    
    def start_heartbeat(self) -> None:
        """Start the heartbeat generation thread."""
        if self._running:
            logger.warning("Heartbeat already running")
            return
            
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._generate_heartbeats,
            name=f"AgentHeartbeat_{self.agent_id}",
            daemon=True
        )
        self._heartbeat_thread.start()
        
        logger.info(f"Started heartbeat generation for agent {self.agent_id}")
    
    def stop_heartbeat(self) -> None:
        """Stop the heartbeat generation thread."""
        if not self._running:
            logger.warning("Heartbeat already stopped")
            return
            
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None
            
        logger.info(f"Stopped heartbeat generation for agent {self.agent_id}")
    
    def _generate_heartbeats(self) -> None:
        """Background thread for generating heartbeats."""
        logger.info(f"Heartbeat generation thread started for agent {self.agent_id}")
        
        while self._running:
            try:
                # Generate heartbeat with diagnostics data
                diagnostics = {
                    "state": "processing",
                    "queue_size": 3,  # Example value
                    "last_cycle_time": 0.23  # Example value
                }
                
                self.health_monitor.record_heartbeat(
                    component_id=self.agent_id,
                    data=diagnostics
                )
                
                # Sleep until next heartbeat
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat generation for agent {self.agent_id}: {str(e)}")
                time.sleep(1.0)  # Sleep briefly before retry
                
        logger.info(f"Heartbeat generation thread stopped for agent {self.agent_id}")
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record metrics from agent operation.
        
        Args:
            metrics: Dictionary of metrics to record
        """
        # Record each metric individually
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.health_monitor.add_metric(
                    component_id=self.agent_id,
                    metric_name=name,
                    value=value
                )
    
    def record_processing_time(self, processing_time: float) -> None:
        """
        Record agent processing time metric.
        
        Args:
            processing_time: Time taken to process data in seconds
        """
        self.health_monitor.add_metric(
            component_id=self.agent_id,
            metric_name="processing_time",
            value=processing_time
        )
    
    def record_error_rate(self, error_rate: float) -> None:
        """
        Record agent error rate metric.
        
        Args:
            error_rate: Error rate as a fraction (0.0 to 1.0)
        """
        self.health_monitor.add_metric(
            component_id=self.agent_id,
            metric_name="error_rate",
            value=error_rate
        )


# Example usage in a trading orchestrator
def example_trading_orchestrator_integration():
    """Example of integrating health monitoring with a trading orchestrator."""
    # Create health monitor
    health_monitor = HealthMonitor(
        log_dir="logs/health"
    )
    
    # Start health monitoring
    health_monitor.start()
    
    # Create orchestrator adapter
    orchestrator_adapter = OrchestratorHealthAdapter(
        health_monitor=health_monitor,
        orchestrator_id="main_orchestrator",
        heartbeat_interval=5.0
    )
    
    # Create agent adapters
    agent_adapters = {}
    agent_types = {
        "market_data_agent": "data",
        "sentiment_agent": "data",
        "technical_strategy_agent": "strategy",
        "sentiment_strategy_agent": "strategy",
        "risk_management_agent": "risk",
        "execution_agent": "execution"
    }
    
    for agent_id, agent_type in agent_types.items():
        agent_adapters[agent_id] = AgentHealthAdapter(
            health_monitor=health_monitor,
            agent_id=agent_id,
            agent_type=agent_type,
            heartbeat_interval=10.0
        )
    
    # Start heartbeats
    orchestrator_adapter.start_heartbeat()
    for adapter in agent_adapters.values():
        adapter.start_heartbeat()
    
    # Record some example metrics in a simulated trading cycle
    for _ in range(5):
        # Simulate a trading cycle
        cycle_start = time.time()
        
        # Record orchestrator metrics
        orchestrator_metrics = {
            "cycle_duration": 0.8,
            "agent_errors": 0,
            "decision_latency": 0.3,
            "active_strategies": 2,
            "pending_orders": 1
        }
        orchestrator_adapter.record_metrics(orchestrator_metrics)
        
        # Record agent metrics
        for agent_id, adapter in agent_adapters.items():
            # Simulate different metrics for different agent types
            if "data" in agent_types[agent_id]:
                adapter.record_metrics({
                    "processing_time": 0.15,
                    "error_rate": 0.01,
                    "data_points_processed": 1000,
                    "data_freshness_seconds": 0.5
                })
            elif "strategy" in agent_types[agent_id]:
                adapter.record_metrics({
                    "processing_time": 0.25,
                    "error_rate": 0.0,
                    "signals_generated": 3,
                    "confidence_score": 0.85
                })
            elif "risk" in agent_types[agent_id]:
                adapter.record_metrics({
                    "processing_time": 0.12,
                    "error_rate": 0.0,
                    "portfolio_var": 0.05,
                    "max_position_size": 0.15
                })
            elif "execution" in agent_types[agent_id]:
                adapter.record_metrics({
                    "processing_time": 0.18,
                    "error_rate": 0.02,
                    "orders_executed": 2,
                    "slippage_bps": 1.5
                })
        
        # Simulate a cycle completion
        cycle_duration = time.time() - cycle_start
        orchestrator_adapter.record_metrics({"cycle_duration": cycle_duration})
        
        # Sleep to simulate time between cycles
        time.sleep(1.0)
    
    # Get system health status
    system_health = health_monitor.get_system_health()
    logger.info(f"System health: {system_health['overall_status']}")
    
    # Get active alerts
    active_alerts = health_monitor.get_active_alerts()
    logger.info(f"Active alerts: {len(active_alerts)}")
    
    # Clean up
    orchestrator_adapter.stop_heartbeat()
    for adapter in agent_adapters.values():
        adapter.stop_heartbeat()
    
    health_monitor.stop()
    
    return health_monitor, orchestrator_adapter, agent_adapters


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the example
    example_trading_orchestrator_integration()
