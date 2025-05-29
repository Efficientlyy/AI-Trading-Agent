# Health Monitoring System Guide

**Date:** May 15, 2025  
**Author:** Cascade AI Assistant  

## Overview

The Health Monitoring System provides comprehensive monitoring, alerting, and autonomous recovery capabilities for the AI Trading Agent. It enables real-time tracking of component health, performance metrics, and automatic recovery actions when issues are detected.

## Core Components

The Health Monitoring System consists of the following core components:

1. **HealthMonitor**: Central system that integrates all monitoring components
2. **HeartbeatManager**: Tracks component liveness through regular heartbeats
3. **AlertManager**: Processes and routes alerts through multiple channels
4. **HealthMetrics**: Monitors performance metrics with configurable thresholds
5. **RecoveryCoordinator**: Manages automatic recovery actions
6. **HealthDashboard**: Web-based visualization of system health
7. **HealthIntegratedOrchestrator**: Trading orchestrator with health monitoring capabilities

## Getting Started

### Basic Setup

To use the Health Monitoring System with your trading agents, replace the standard `TradingOrchestrator` with the `HealthIntegratedOrchestrator`:

```python
from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator

# Create health-integrated orchestrator
orchestrator = HealthIntegratedOrchestrator(
    log_dir="logs/health_monitoring",
    heartbeat_interval=5.0,  # 5 seconds
    monitor_components=True   # Monitor individual agents
)

# Register agents as normal
orchestrator.register_agent(market_data_agent)
orchestrator.register_agent(strategy_agent)
orchestrator.register_agent(execution_agent)

# Start all agents
orchestrator.start_all_agents()

# Run processing cycles
while True:
    orchestrator.run_cycle()
    time.sleep(1)  # Cycle interval
```

### Starting the Health Dashboard

To visualize the health monitoring data, start the health dashboard:

```python
from ai_trading_agent.common.health_monitoring.health_dashboard import run_standalone_dashboard

# Run the dashboard using the orchestrator's health monitor
dashboard = run_standalone_dashboard(
    health_monitor=orchestrator.health_monitor,
    host="127.0.0.1",
    port=5000
)

# The dashboard will be available at http://127.0.0.1:5000
```

## Agent Health Monitoring

### Agent Lifecycle Monitoring

The health monitoring system tracks the entire lifecycle of each agent:

- **Registration**: When agents are registered with the orchestrator, they are automatically registered with the health monitoring system.
- **Startup**: Agent startup events are recorded and monitored.
- **Processing**: Each agent execution is tracked for performance metrics and errors.
- **Heartbeats**: Regular heartbeats ensure the agent is still alive.
- **Shutdown**: Graceful shutdown events are recorded.

### Agent Metrics

The following metrics are tracked for each agent:

- **Execution Time**: How long it takes for the agent to process inputs
- **Output Count**: Number of outputs produced by the agent
- **Error Rate**: Frequency of errors during processing
- **Warning Rate**: Frequency of warnings during processing
- **Missed Heartbeats**: Number of consecutive missed heartbeats

### Agent Controls

The health monitoring system provides individual controls for each agent:

```python
# Start a specific agent
orchestrator.start_agent("strategy_1")

# Stop a specific agent
orchestrator.stop_agent("execution_1")

# Check agent status
status = orchestrator.get_health_metrics()
```

## Autonomous Recovery

### Recovery Actions

The health monitoring system supports the following recovery actions:

- **Agent Restart**: Restarts a failed agent
- **Queue Reset**: Clears queued data when backlogs are detected
- **Execution Order Recalculation**: Recalculates agent execution order
- **Processing Throttling**: Slows down processing during high load

### Configuring Recovery

Custom recovery actions can be registered for specific issues:

```python
def custom_recovery_action():
    # Custom recovery logic
    return True  # Return success/failure

# Register the recovery action
orchestrator.health_monitor.register_recovery_action(
    "custom_recovery", 
    "Component ID", 
    custom_recovery_action,
    description="Custom recovery action"
)
```

### Auto-Recovery

Auto-recovery can be enabled or disabled:

```python
# Enable auto-recovery
orchestrator.health_monitor.recovery_coordinator.set_auto_recovery(True)

# Disable auto-recovery
orchestrator.health_monitor.recovery_coordinator.set_auto_recovery(False)
```

## Health Dashboard

The health dashboard provides a web-based interface for monitoring system health. It includes:

- **System Status Overview**: Overall health status and component counts
- **Component Status**: Status of individual components with detailed metrics
- **Alerts View**: Active alerts and alert history
- **Metrics View**: Performance metrics with interactive charts
- **Controls Panel**: Controls for starting/stopping agents and enabling/disabling auto-recovery

The dashboard is accessible at `http://<host>:<port>/` after starting it.

## Integration with Trading System

### Paper Trading Mode

When running in paper trading mode, the health monitoring system provides:

- **Real-time Agent Status**: See which agents are running or stopped
- **Global Controls**: Start or stop all agents at once
- **Individual Controls**: Control specific agents independently
- **Performance Metrics**: Monitor real-time performance of each agent
- **Alert Notifications**: Get notified when issues occur

### Usage Example: Paper Trading Setup

```python
from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator
from ai_trading_agent.common.health_monitoring.health_dashboard import run_standalone_dashboard
from ai_trading_agent.agent.market_data import AlpacaMarketDataAgent
from ai_trading_agent.agent.strategy import MovingAverageStrategy
from ai_trading_agent.agent.execution import PaperTradingExecutionAgent

# Create paper trading agents
market_data = AlpacaMarketDataAgent(
    agent_id="alpaca_market_data",
    name="Alpaca Market Data",
    symbols=["AAPL", "MSFT", "GOOG"]
)

strategy = MovingAverageStrategy(
    agent_id="ma_strategy",
    name="Moving Average Strategy",
    short_window=20,
    long_window=50
)

execution = PaperTradingExecutionAgent(
    agent_id="paper_execution",
    name="Paper Trading Execution"
)

# Set up dependencies
strategy.dependencies = ["alpaca_market_data"]
execution.dependencies = ["ma_strategy"]

# Create health-integrated orchestrator
orchestrator = HealthIntegratedOrchestrator(
    log_dir="logs/paper_trading",
    heartbeat_interval=5.0
)

# Register agents
orchestrator.register_agent(market_data)
orchestrator.register_agent(strategy)
orchestrator.register_agent(execution)

# Start health dashboard in a separate thread
import threading
def run_dashboard():
    dashboard = run_standalone_dashboard(
        health_monitor=orchestrator.health_monitor,
        host="127.0.0.1",
        port=5000
    )
    
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

print("Health dashboard available at http://127.0.0.1:5000")

# Start all agents
orchestrator.start_all_agents()

# Run trading loop
try:
    while True:
        orchestrator.run_cycle()
        time.sleep(1)  # Wait between cycles
except KeyboardInterrupt:
    print("Shutting down...")
    orchestrator.stop_all_agents()
```

## Advanced Configuration

### Custom Metrics

You can define custom metrics for your agents:

```python
# Add custom metric
orchestrator.health_monitor.add_metric(
    component_id="strategy_1",
    metric_name="signal_strength",
    value=0.85
)

# Add metric threshold
orchestrator.health_monitor.add_metric_threshold(
    metric_name="signal_strength",
    warning_threshold=0.2,   # Alert if below 0.2
    critical_threshold=0.1,  # Critical if below 0.1
    threshold_type="lower",  # Alert on low values
    component_id="strategy_1"
)
```

### Custom Alerts

Custom alerts can be created for specific conditions:

```python
from ai_trading_agent.common.health_monitoring import AlertSeverity

# Create custom alert
orchestrator.health_monitor.add_alert(
    component_id="market_data_1",
    severity=AlertSeverity.WARNING,
    message="Market data latency increased",
    details={"latency": 250, "threshold": 200}
)
```

## Conclusion

The Health Monitoring System provides a comprehensive solution for monitoring, alerting, and autonomous recovery of the AI Trading Agent system. By using the `HealthIntegratedOrchestrator` instead of the standard `TradingOrchestrator`, you gain full visibility into the health and performance of your trading system and enable autonomous recovery from failures.

---

## API Reference

For detailed API documentation, please refer to:

- `ai_trading_agent.common.health_monitoring.health_monitor.HealthMonitor`
- `ai_trading_agent.common.health_monitoring.heartbeat_manager.HeartbeatManager`
- `ai_trading_agent.common.health_monitoring.alert_manager.AlertManager`
- `ai_trading_agent.common.health_monitoring.health_metrics.HealthMetrics`
- `ai_trading_agent.common.health_monitoring.recovery_coordinator.RecoveryCoordinator`
- `ai_trading_agent.common.health_monitoring.health_dashboard.HealthDashboard`
- `ai_trading_agent.agent.health_integrated_orchestrator.HealthIntegratedOrchestrator`
