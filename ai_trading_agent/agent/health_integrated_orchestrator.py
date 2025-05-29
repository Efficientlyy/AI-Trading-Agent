"""
Health-Integrated Trading Orchestrator.

This module extends the TradingOrchestrator with health monitoring capabilities,
providing real-time health tracking, performance metrics, and autonomous recovery.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from collections import deque

from ai_trading_agent.agent.trading_orchestrator import TradingOrchestrator
from ai_trading_agent.agent.agent_definitions import BaseAgent, AgentStatus

# Import health monitoring components with proper structure
from ai_trading_agent.common.health_monitoring.core_definitions import (
    HealthStatus,
    AlertSeverity,
    ThresholdType
)
from ai_trading_agent.common.health_monitoring.heartbeat_manager import HeartbeatConfig
from ai_trading_agent.common.health_monitoring.health_metrics import MetricThreshold
from ai_trading_agent.common.health_monitoring.health_monitor import HealthMonitor

# Set up logger
logger = logging.getLogger(__name__)


class HealthIntegratedOrchestrator(TradingOrchestrator):
    """
    Trading Orchestrator with integrated health monitoring capabilities.
    
    Extends the base TradingOrchestrator to track health metrics, detect issues,
    and perform autonomous recovery actions.
    """
    
    def __init__(
        self,
        health_monitor: Optional[HealthMonitor] = None,
        log_dir: Optional[str] = None,
        heartbeat_interval: float = 5.0,
        monitor_components: bool = True
    ):
        """
        Initialize the health-integrated orchestrator.
        
        Args:
            health_monitor: Optional existing health monitor instance
            log_dir: Directory for health monitoring logs
            heartbeat_interval: Interval for orchestrator heartbeats in seconds
            monitor_components: Whether to monitor individual agents as components
        """
        super().__init__()
        
        # Set up health monitoring
        self.health_monitor = health_monitor or HealthMonitor(log_dir=log_dir)
        self.heartbeat_interval = heartbeat_interval
        self.monitor_components = monitor_components
        
        # Performance metrics
        self.cycle_metrics = {
            "cycle_count": 0,
            "cycle_duration": 0.0,
            "total_cycle_time": 0.0,
            "avg_cycle_time": 0.0,
            "max_cycle_time": 0.0,
            "min_cycle_time": float('inf'),
            "agent_errors": 0,
            "agent_warnings": 0,
            "total_signals_processed": 0,
            "last_cycle_time": 0.0
        }
        
        # Health monitoring status
        self._heartbeat_thread = None
        self._running = False
        
        # Register orchestrator with health monitor
        self._register_with_health_monitor()
    
    def _register_with_health_monitor(self) -> None:
        """Register the orchestrator with the health monitoring system."""
        # Register orchestrator component
        orchestrator_id = "trading_orchestrator"
        
        # Configure heartbeat
        heartbeat_config = HeartbeatConfig(
            interval=self.heartbeat_interval,
            missing_threshold=1,
            degraded_threshold=2,
            unhealthy_threshold=3
        )
        
        # Register orchestrator with health monitor
        self.health_monitor.register_component(
            component_id=orchestrator_id,
            description="Trading Orchestrator",
            heartbeat_config=heartbeat_config
        )
        
        # Add metric thresholds
        self._add_orchestrator_thresholds()
        
        # Register recovery actions
        self._register_recovery_actions()
        
        # Start health monitoring
        self.health_monitor.start()
        
        logger.info("Trading orchestrator registered with health monitoring system")
    
    def _add_orchestrator_thresholds(self) -> None:
        """Add metric thresholds for orchestrator monitoring."""
        orchestrator_id = "trading_orchestrator"
        
        # Cycle duration threshold (detect slow processing)
        self.health_monitor.add_metric_threshold(
            metric_name="cycle_duration",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=1.0,  # seconds
            critical_threshold=5.0,  # seconds
            component_id=orchestrator_id,
            description="Maximum time for a trading cycle"
        )
        
        # Agent error threshold (detect agent failures)
        self.health_monitor.add_metric_threshold(
            metric_name="agent_errors",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=1,
            critical_threshold=3,
            component_id=orchestrator_id,
            description="Number of agent errors per cycle"
        )
        
        # Signal queue size threshold (detect bottlenecks)
        self.health_monitor.add_metric_threshold(
            metric_name="max_queue_size",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=50,
            critical_threshold=200,
            component_id=orchestrator_id,
            description="Maximum agent queue size"
        )
        
        # Agent warning threshold (detect agent issues)
        self.health_monitor.add_metric_threshold(
            metric_name="agent_warnings",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=3,
            critical_threshold=10,
            component_id=orchestrator_id,
            description="Number of agent warnings per cycle"
        )
    
    def _register_recovery_actions(self) -> None:
        """Register recovery actions for orchestrator issues."""
        orchestrator_id = "trading_orchestrator"
        
        # Action to reset agent queues
        self.health_monitor.register_recovery_action(
            action_id="reset_agent_queues",
            description="Reset all agent queues to recover from bottlenecks",
            action_func=self._reset_agent_queues_action,
            component_id=orchestrator_id,
            severity_threshold=AlertSeverity.WARNING
        )
        
        # Action to restart failing agents
        self.health_monitor.register_recovery_action(
            action_id="restart_failing_agents",
            description="Restart agents in ERROR state",
            action_func=self._restart_failing_agents_action,
            component_id=orchestrator_id,
            severity_threshold=AlertSeverity.ERROR
        )
        
        # Action to redetermine execution order
        self.health_monitor.register_recovery_action(
            action_id="redetermine_execution_order",
            description="Recalculate agent execution order",
            action_func=self._redetermine_execution_order_action,
            component_id=orchestrator_id,
            severity_threshold=AlertSeverity.WARNING
        )
        
        # Action to throttle processing
        self.health_monitor.register_recovery_action(
            action_id="throttle_processing",
            description="Throttle processing to reduce system load",
            action_func=self._throttle_processing_action,
            component_id=orchestrator_id,
            severity_threshold=AlertSeverity.CRITICAL
        )
    
    def _reset_agent_queues_action(self) -> bool:
        """
        Recovery action to reset agent queues.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Count items being cleared for logging
            total_items = sum(len(queue) for queue in self.data_queues.values())
            
            # Clear all queues
            for agent_id in self.data_queues:
                self.data_queues[agent_id].clear()
                
            logger.info(f"Reset all agent queues, cleared {total_items} items")
            return True
        except Exception as e:
            logger.error(f"Error resetting agent queues: {str(e)}")
            return False
    
    def _restart_failing_agents_action(self) -> bool:
        """
        Recovery action to restart failing agents.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find agents in ERROR state
            error_agents = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.status == AgentStatus.ERROR
            ]
            
            # No failing agents
            if not error_agents:
                logger.info("No failing agents to restart")
                return True
                
            # Restart each failing agent
            success_count = 0
            for agent_id in error_agents:
                agent = self.agents[agent_id]
                try:
                    logger.info(f"Restarting agent {agent_id}")
                    agent.stop()
                    time.sleep(0.5)  # Brief pause
                    agent.start()
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to restart agent {agent_id}: {str(e)}")
                    
            logger.info(f"Restarted {success_count}/{len(error_agents)} failing agents")
            return success_count == len(error_agents)
        except Exception as e:
            logger.error(f"Error restarting failing agents: {str(e)}")
            return False
    
    def _redetermine_execution_order_action(self) -> bool:
        """
        Recovery action to recalculate agent execution order.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            old_order = self.agent_execution_order.copy()
            self._determine_execution_order()
            
            if self.agent_execution_order:
                logger.info(f"Recalculated agent execution order: {self.agent_execution_order}")
                return True
            else:
                # Restore previous order if new calculation failed
                self.agent_execution_order = old_order
                logger.error("Failed to determine new execution order")
                return False
        except Exception as e:
            logger.error(f"Error redetermining execution order: {str(e)}")
            return False
    
    def _throttle_processing_action(self) -> bool:
        """
        Recovery action to throttle processing.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # This is a placeholder for actual throttling logic
            # In a real implementation, this might adjust cycle timing,
            # reduce processing frequency, or limit queue sizes
            logger.info("Throttling processing to reduce system load")
            time.sleep(1.0)  # Simulate throttling
            return True
        except Exception as e:
            logger.error(f"Error throttling processing: {str(e)}")
            return False
    
    def start_all_agents(self) -> None:
        """Start all registered agents and health monitoring."""
        super().start_all_agents()
        
        # Start heartbeat monitoring
        self._start_heartbeat()
        
        # Register each agent as a component if enabled
        if self.monitor_components:
            self._register_agents_as_components()
    
    def stop_all_agents(self) -> None:
        """Stop all registered agents and health monitoring."""
        super().stop_all_agents()
        
        # Stop heartbeat monitoring
        self._stop_heartbeat()
    
    def _start_heartbeat(self) -> None:
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
        
        logger.info("Started heartbeat generation for trading orchestrator")
    
    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat generation thread."""
        if not self._running:
            return
            
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None
            
        logger.info("Stopped heartbeat generation for trading orchestrator")
    
    def _generate_heartbeats(self) -> None:
        """Background thread for generating heartbeats."""
        logger.info("Heartbeat generation thread started")
        
        while self._running:
            try:
                # Generate heartbeat with diagnostics data
                agent_stats = self._get_agent_stats()
                
                diagnostics = {
                    "active_agents": agent_stats["active_count"],
                    "error_agents": agent_stats["error_count"],
                    "pending_signals": agent_stats["queued_signals"],
                    "cycle_count": self.cycle_metrics["cycle_count"],
                    "avg_cycle_time": self.cycle_metrics["avg_cycle_time"]
                }
                
                self.health_monitor.record_heartbeat(
                    component_id="trading_orchestrator",
                    data=diagnostics
                )
                
                # Sleep until next heartbeat
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat generation: {str(e)}")
                time.sleep(1.0)  # Sleep briefly before retry
                
        logger.info("Heartbeat generation thread stopped")
    
    def _get_agent_stats(self) -> Dict[str, Any]:
        """
        Get statistics about agents.
        
        Returns:
            Dictionary of agent statistics
        """
        active_count = 0
        error_count = 0
        queued_signals = 0
        
        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.RUNNING:
                active_count += 1
            elif agent.status == AgentStatus.ERROR:
                error_count += 1
                
            queued_signals += len(self.data_queues.get(agent_id, []))
            
        return {
            "active_count": active_count,
            "error_count": error_count,
            "queued_signals": queued_signals,
            "total_count": len(self.agents)
        }
    
    def _register_agents_as_components(self) -> None:
        """Register each agent as a monitored component."""
        for agent_id, agent in self.agents.items():
            # Skip already registered agents
            if agent_id in self.health_monitor.component_health:
                continue
                
            # Configure heartbeat for agent
            heartbeat_config = HeartbeatConfig(
                interval=10.0,  # Longer interval for agents
                missing_threshold=2,
                degraded_threshold=3,
                unhealthy_threshold=4
            )
            
            # Determine agent type for description
            agent_type = getattr(agent, "agent_type", "Unknown")
            
            # Register agent with health monitor
            self.health_monitor.register_component(
                component_id=agent_id,
                description=f"{agent.name} ({agent_type})",
                heartbeat_config=heartbeat_config
            )
            
            # Add agent-specific thresholds
            self._add_agent_thresholds(agent_id, agent)
            
            # Register agent-specific recovery actions
            self._register_agent_recovery_actions(agent_id, agent)
            
            logger.info(f"Registered agent {agent_id} as a monitored component")
    
    def _add_agent_thresholds(self, agent_id: str, agent: BaseAgent) -> None:
        """
        Add thresholds for agent monitoring.
        
        Args:
            agent_id: ID of the agent
            agent: Agent instance
        """
        # Processing time threshold
        self.health_monitor.add_metric_threshold(
            metric_name="processing_time",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=0.5,  # seconds
            critical_threshold=2.0,  # seconds
            component_id=agent_id,
            description="Maximum processing time for agent"
        )
        
        # Error rate threshold
        self.health_monitor.add_metric_threshold(
            metric_name="error_rate",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=0.05,  # 5%
            critical_threshold=0.20,  # 20%
            component_id=agent_id,
            description="Maximum error rate for agent"
        )
        
        # Signal success rate threshold
        self.health_monitor.add_metric_threshold(
            metric_name="signal_success_rate",
            threshold_type=ThresholdType.LOWER,
            warning_threshold=0.9,  # 90%
            critical_threshold=0.7,  # 70%
            component_id=agent_id,
            description="Minimum success rate for signal generation"
        )
    
    def _register_agent_recovery_actions(self, agent_id: str, agent: BaseAgent) -> None:
        """
        Register recovery actions for an agent.
        
        Args:
            agent_id: ID of the agent
            agent: Agent instance
        """
        # Action to restart agent
        self.health_monitor.register_recovery_action(
            action_id=f"restart_{agent_id}",
            description=f"Restart the {agent_id} agent",
            action_func=self._get_restart_agent_action(agent_id),
            component_id=agent_id,
            severity_threshold=AlertSeverity.ERROR
        )
        
        # Action to reset agent state
        self.health_monitor.register_recovery_action(
            action_id=f"reset_{agent_id}_state",
            description=f"Reset the {agent_id} agent state",
            action_func=self._get_reset_agent_state_action(agent_id),
            component_id=agent_id,
            severity_threshold=AlertSeverity.WARNING
        )
    
    def _get_restart_agent_action(self, agent_id: str) -> callable:
        """
        Create a restart action function for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Restart action function
        """
        def restart_action() -> bool:
            try:
                agent = self.agents.get(agent_id)
                if not agent:
                    logger.error(f"Agent {agent_id} not found for restart")
                    return False
                    
                logger.info(f"Restarting agent {agent_id}")
                agent.stop()
                time.sleep(0.5)  # Brief pause
                agent.start()
                
                return True
            except Exception as e:
                logger.error(f"Error restarting agent {agent_id}: {str(e)}")
                return False
                
        return restart_action
    
    def _get_reset_agent_state_action(self, agent_id: str) -> callable:
        """
        Create a state reset action function for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            State reset action function
        """
        def reset_state_action() -> bool:
            try:
                agent = self.agents.get(agent_id)
                if not agent:
                    logger.error(f"Agent {agent_id} not found for state reset")
                    return False
                    
                logger.info(f"Resetting state for agent {agent_id}")
                
                # Clear agent's queue
                if agent_id in self.data_queues:
                    self.data_queues[agent_id].clear()
                
                # Reset internal state if agent supports it
                if hasattr(agent, "reset_state") and callable(getattr(agent, "reset_state")):
                    agent.reset_state()
                    
                return True
            except Exception as e:
                logger.error(f"Error resetting state for agent {agent_id}: {str(e)}")
                return False
                
        return reset_state_action
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the orchestrator and health monitoring.
        
        Args:
            agent: The agent to register
        """
        super().register_agent(agent)
        
        # Register with health monitoring if components are monitored
        # and health monitoring is already running
        if self.monitor_components and self._running:
            self._register_agent_as_component(agent.agent_id, agent)
    
    def _register_agent_as_component(self, agent_id: str, agent: BaseAgent) -> None:
        """
        Register a single agent as a monitored component.
        
        Args:
            agent_id: ID of the agent
            agent: Agent instance
        """
        # Skip if already registered
        if agent_id in self.health_monitor.component_health:
            return
            
        # Configure heartbeat for agent
        heartbeat_config = HeartbeatConfig(
            interval=10.0,  # Longer interval for agents
            missing_threshold=2,
            degraded_threshold=3,
            unhealthy_threshold=4
        )
        
        # Determine agent type for description
        agent_type = getattr(agent, "agent_type", "Unknown")
        
        # Register agent with health monitor
        self.health_monitor.register_component(
            component_id=agent_id,
            description=f"{agent.name} ({agent_type})",
            heartbeat_config=heartbeat_config
        )
        
        # Add agent-specific thresholds
        self._add_agent_thresholds(agent_id, agent)
        
        # Register agent-specific recovery actions
        self._register_agent_recovery_actions(agent_id, agent)
        
        logger.info(f"Registered agent {agent_id} as a monitored component")
    
    def run_cycle(self, external_inputs: Optional[Dict[str, List[Dict]]] = None) -> None:
        """
        Run one cycle of processing with health monitoring.
        
        Args:
            external_inputs: Optional external inputs for agents
        """
        # Record metrics for cycle
        cycle_start = time.time()
        
        # Reset cycle error count
        self.cycle_metrics["agent_errors"] = 0
        self.cycle_metrics["agent_warnings"] = 0
        
        # Run the cycle
        super().run_cycle(external_inputs)
        
        # Update cycle metrics
        cycle_duration = time.time() - cycle_start
        self.cycle_metrics["cycle_count"] += 1
        self.cycle_metrics["cycle_duration"] = cycle_duration
        self.cycle_metrics["total_cycle_time"] += cycle_duration
        self.cycle_metrics["avg_cycle_time"] = (
            self.cycle_metrics["total_cycle_time"] / self.cycle_metrics["cycle_count"]
        )
        self.cycle_metrics["last_cycle_time"] = cycle_duration
        
        if cycle_duration > self.cycle_metrics["max_cycle_time"]:
            self.cycle_metrics["max_cycle_time"] = cycle_duration
            
        if cycle_duration < self.cycle_metrics["min_cycle_time"]:
            self.cycle_metrics["min_cycle_time"] = cycle_duration
            
        # Get queue sizes
        max_queue_size = max([len(q) for q in self.data_queues.values()], default=0)
        total_queued = sum([len(q) for q in self.data_queues.values()])
        
        # Record metrics in health monitor
        self.health_monitor.add_metric(
            component_id="trading_orchestrator",
            metric_name="cycle_duration",
            value=cycle_duration
        )
        
        self.health_monitor.add_metric(
            component_id="trading_orchestrator",
            metric_name="agent_errors",
            value=self.cycle_metrics["agent_errors"]
        )
        
        self.health_monitor.add_metric(
            component_id="trading_orchestrator",
            metric_name="agent_warnings",
            value=self.cycle_metrics["agent_warnings"]
        )
        
        self.health_monitor.add_metric(
            component_id="trading_orchestrator",
            metric_name="max_queue_size",
            value=max_queue_size
        )
        
        self.health_monitor.add_metric(
            component_id="trading_orchestrator",
            metric_name="total_queued_signals",
            value=total_queued
        )
        
        logger.debug(
            f"Cycle {self.cycle_metrics['cycle_count']} completed in "
            f"{cycle_duration:.4f}s with {self.cycle_metrics['agent_errors']} errors"
        )
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get health metrics for the orchestrator.
        
        Returns:
            Dictionary of health metrics
        """
        system_health = self.health_monitor.get_system_health()
        active_alerts = self.health_monitor.get_active_alerts()
        
        return {
            "system_health": system_health,
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "cycle_metrics": self.cycle_metrics,
            "agent_stats": self._get_agent_stats()
        }
    
    def get_component_health(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health status of components.
        
        Args:
            component_id: Optional component ID to filter by
            
        Returns:
            Dictionary of component health status
        """
        return self.health_monitor.get_component_health(component_id)
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """
        Get recovery action history.
        
        Returns:
            List of recovery attempt records
        """
        return self.health_monitor.get_recovery_history()
    
    def get_health_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete data for health dashboard.
        
        Returns:
            Dictionary with all health monitoring data
        """
        # Get health metrics
        health_metrics = self.get_health_metrics()
        
        # Get component health
        component_health = self.health_monitor.get_component_health()
        
        # Get detailed metrics data for graphs
        metrics_data = self.health_monitor.get_metrics()
        
        # Get recovery history
        recovery_history = self.health_monitor.get_recovery_history()
        
        # Get alert history
        alert_history = self.health_monitor.get_alert_history(limit=50)
        
        return {
            "system_health": health_metrics["system_health"],
            "active_alerts": health_metrics["active_alerts"],
            "component_health": component_health,
            "cycle_metrics": health_metrics["cycle_metrics"],
            "agent_stats": health_metrics["agent_stats"],
            "metrics_data": metrics_data,
            "recovery_history": recovery_history,
            "alert_history": [alert.to_dict() for alert in alert_history]
        }
        
    def register_agent(self, agent: BaseAgent):
        """
        Registers an agent with the orchestrator and health monitoring system.
        
        Extends the base TradingOrchestrator register_agent method to also register
        the agent with the health monitoring system.
        
        Args:
            agent: Agent to register
        """
        # Register with base orchestrator first
        super().register_agent(agent)
        
        # Now register with health monitoring system if enabled
        if self.monitor_components and self.health_monitor is not None:
            agent_id = agent.agent_id
            
            # Only register the agent if it hasn't been registered yet
            existing_health = self.health_monitor.get_component_health(agent_id)
            if not existing_health:
                logger.info(f"Registering agent {agent_id} with health monitoring system")
                
                # Register the agent as a component
                heartbeat_config = HeartbeatConfig(
                    interval=self.heartbeat_interval * 1.5,  # Allow slightly longer interval
                    missing_threshold=1,
                    degraded_threshold=2,
                    unhealthy_threshold=3
                )
                
                self.health_monitor.register_component(
                    component_id=agent_id,
                    description=f"Agent: {agent.name}",
                    heartbeat_config=heartbeat_config
                )
                
                # Add agent-specific thresholds
                self._add_agent_thresholds(agent_id, agent)
                
                # Register recovery actions
                self._register_agent_recovery_actions(agent_id, agent)
                
                # Initial heartbeat
                self.health_monitor.record_heartbeat(
                    component_id=agent_id,
                    data={
                        "status": agent.status.name,
                        "last_execution": None,
                        "execution_count": 0
                    }
                )
        
        return agent
        
    def start_agent(self, agent_id: str) -> bool:
        """
        Start a specific agent and update its health status.
        
        Args:
            agent_id: ID of the agent to start
            
        Returns:
            True if agent was started successfully, False otherwise
        """
        if agent_id not in self.agents:
            logger.warning(f"Cannot start agent {agent_id}: not registered")
            return False
        
        agent = self.agents[agent_id]
        try:
            # Check if already running
            if agent.status == AgentStatus.RUNNING:
                logger.info(f"Agent {agent_id} is already running")
                return True
                
            # Start the agent
            agent.start()
            
            # Update health monitoring
            if self.monitor_components and self.health_monitor is not None:
                self.health_monitor.record_heartbeat(
                    component_id=agent_id,
                    data={
                        "status": agent.status.name,
                        "last_execution": None,
                        "execution_count": 0,
                        "event": "agent_started"
                    }
                )
                
            logger.info(f"Agent {agent_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent {agent_id}: {str(e)}")
            
            # Record error in health monitoring
            if self.monitor_components and self.health_monitor is not None:
                self.health_monitor.add_alert(
                    component_id=agent_id,
                    severity=AlertSeverity.ERROR,
                    message=f"Failed to start agent: {str(e)}"
                )
                
            return False
    
    def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a specific agent and update its health status.
        
        Args:
            agent_id: ID of the agent to stop
            
        Returns:
            True if agent was stopped successfully, False otherwise
        """
        if agent_id not in self.agents:
            logger.warning(f"Cannot stop agent {agent_id}: not registered")
            return False
        
        agent = self.agents[agent_id]
        try:
            # Check if already stopped
            if agent.status != AgentStatus.RUNNING:
                logger.info(f"Agent {agent_id} is already stopped")
                return True
                
            # Stop the agent
            agent.stop()
            
            # Update health monitoring
            if self.monitor_components and self.health_monitor is not None:
                self.health_monitor.record_heartbeat(
                    component_id=agent_id,
                    data={
                        "status": agent.status.name,
                        "event": "agent_stopped"
                    }
                )
                
            logger.info(f"Agent {agent_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}: {str(e)}")
            
            # Record error in health monitoring
            if self.monitor_components and self.health_monitor is not None:
                self.health_monitor.add_alert(
                    component_id=agent_id,
                    severity=AlertSeverity.ERROR,
                    message=f"Failed to stop agent: {str(e)}"
                )
                
            return False
    
    def run_agent(self, agent_id: str, inputs: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Run a specific agent with health monitoring.
        
        Extends the base run_agent method to include health monitoring, heartbeat
        generation, and metrics collection.
        
        Args:
            agent_id: ID of the agent to run
            inputs: Optional list of input data
            
        Returns:
            List of output data from the agent
        """
        if agent_id not in self.agents:
            logger.error(f"Cannot run agent {agent_id}: not registered")
            return []
        
        agent = self.agents[agent_id]
        start_time = time.time()
        
        try:
            # Run the agent using the base method
            outputs = super().run_agent(agent_id, inputs)
            
            # Record metrics
            duration = time.time() - start_time
            
            # Update health monitoring
            if self.monitor_components and self.health_monitor is not None:
                # Record heartbeat
                self.health_monitor.record_heartbeat(
                    component_id=agent_id,
                    data={
                        "status": agent.status.name,
                        "last_execution": time.time(),
                        "execution_count": agent.execution_count,
                        "inputs": len(inputs) if inputs else 0,
                        "outputs": len(outputs)
                    }
                )
                
                # Record execution time metric
                self.health_monitor.add_metric(
                    component_id=agent_id,
                    metric_name="execution_time",
                    value=duration
                )
                
                # Record output count metric
                self.health_monitor.add_metric(
                    component_id=agent_id,
                    metric_name="output_count",
                    value=len(outputs)
                )
            
            return outputs
            
        except Exception as e:
            # Record the error
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error(f"Error running agent {agent_id}: {str(e)}")
            
            # Update health monitoring
            if self.monitor_components and self.health_monitor is not None:
                # Record error metric
                self.health_monitor.add_metric(
                    component_id=agent_id,
                    metric_name="errors",
                    value=1,
                    increment=True
                )
                
                # Add an alert
                self.health_monitor.add_alert(
                    component_id=agent_id,
                    severity=AlertSeverity.ERROR,
                    message=f"Error during agent execution: {str(e)}",
                    details={
                        "exception": str(e),
                        "execution_time": duration,
                        "timestamp": end_time
                    }
                )
                
                # Record execution time metric (even for failed executions)
                self.health_monitor.add_metric(
                    component_id=agent_id,
                    metric_name="execution_time",
                    value=duration
                )
                
            # Increment error count
            self.cycle_metrics["agent_errors"] += 1
            
            # Default return value
            return []
