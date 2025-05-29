"""
Trading-Specific Health Metrics Configuration.

This module defines trading-specific metrics, thresholds, and health checks
tailored for monitoring trading system components.
"""

from typing import Dict, Any, List, Optional
import logging

from ai_trading_agent.common.health_monitoring import (
    HealthMonitor,
    MetricThreshold,
    ThresholdType,
    AlertSeverity
)

# Set up logger
logger = logging.getLogger(__name__)


def configure_market_data_agent_metrics(
    health_monitor: HealthMonitor,
    agent_id: str,
    symbol: Optional[str] = None
) -> None:
    """
    Configure health metrics for a market data agent.
    
    Args:
        health_monitor: Health monitoring system
        agent_id: ID of the market data agent
        symbol: Optional specific symbol the agent monitors
    """
    symbol_str = f" for {symbol}" if symbol else ""
    description_suffix = f"{symbol_str}" if symbol else " (all symbols)"
    
    # Data freshness threshold
    health_monitor.add_metric_threshold(
        metric_name="data_freshness_seconds",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=60.0,  # 1 minute
        critical_threshold=300.0,  # 5 minutes
        component_id=agent_id,
        description=f"Maximum age of market data{description_suffix}"
    )
    
    # Data completeness threshold
    health_monitor.add_metric_threshold(
        metric_name="data_completeness",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.95,  # 95%
        critical_threshold=0.80,  # 80% 
        component_id=agent_id,
        description=f"Minimum completeness of market data{description_suffix}"
    )
    
    # API error rate threshold
    health_monitor.add_metric_threshold(
        metric_name="api_error_rate",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.05,  # 5%
        critical_threshold=0.20,  # 20%
        component_id=agent_id,
        description=f"Maximum API error rate{description_suffix}"
    )
    
    # Data points per second threshold
    health_monitor.add_metric_threshold(
        metric_name="data_points_per_second",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.5,  # At least 0.5 data points per second
        critical_threshold=0.1,  # At least 0.1 data points per second
        component_id=agent_id,
        description=f"Minimum data throughput{description_suffix}"
    )
    
    logger.info(f"Configured market data agent metrics for {agent_id}{description_suffix}")


def configure_strategy_agent_metrics(
    health_monitor: HealthMonitor,
    agent_id: str,
    strategy_type: Optional[str] = None
) -> None:
    """
    Configure health metrics for a trading strategy agent.
    
    Args:
        health_monitor: Health monitoring system
        agent_id: ID of the strategy agent
        strategy_type: Optional strategy type for specialized thresholds
    """
    type_str = f" for {strategy_type}" if strategy_type else ""
    
    # Signal generation rate threshold
    health_monitor.add_metric_threshold(
        metric_name="signals_per_hour",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=1.0,  # At least 1 signal per hour
        critical_threshold=0.2,  # At least 1 signal per 5 hours
        component_id=agent_id,
        description=f"Minimum signal generation rate{type_str}"
    )
    
    # Signal quality threshold
    health_monitor.add_metric_threshold(
        metric_name="signal_quality",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.6,  # 60% quality
        critical_threshold=0.4,  # 40% quality
        component_id=agent_id,
        description=f"Minimum signal quality{type_str}"
    )
    
    # Calculation error rate threshold
    health_monitor.add_metric_threshold(
        metric_name="calculation_error_rate",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.01,  # 1%
        critical_threshold=0.05,  # 5%
        component_id=agent_id,
        description=f"Maximum calculation error rate{type_str}"
    )
    
    # Input data utilization threshold
    health_monitor.add_metric_threshold(
        metric_name="input_data_utilization",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.8,  # 80%
        critical_threshold=0.5,  # 50%
        component_id=agent_id,
        description=f"Minimum input data utilization{type_str}"
    )
    
    # Special thresholds for specific strategy types
    if strategy_type == "technical":
        # Indicator consistency threshold
        health_monitor.add_metric_threshold(
            metric_name="indicator_consistency",
            threshold_type=ThresholdType.LOWER,
            warning_threshold=0.85,  # 85%
            critical_threshold=0.7,  # 70%
            component_id=agent_id,
            description="Minimum consistency between technical indicators"
        )
    elif strategy_type == "sentiment":
        # Sentiment source diversity threshold
        health_monitor.add_metric_threshold(
            metric_name="source_diversity",
            threshold_type=ThresholdType.LOWER,
            warning_threshold=3.0,  # At least 3 sources
            critical_threshold=1.0,  # At least 1 source
            component_id=agent_id,
            description="Minimum diversity of sentiment sources"
        )
    
    logger.info(f"Configured strategy agent metrics for {agent_id}{type_str}")


def configure_decision_agent_metrics(
    health_monitor: HealthMonitor,
    agent_id: str
) -> None:
    """
    Configure health metrics for a decision agent.
    
    Args:
        health_monitor: Health monitoring system
        agent_id: ID of the decision agent
    """
    # Decision latency threshold
    health_monitor.add_metric_threshold(
        metric_name="decision_latency",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.5,  # 500ms
        critical_threshold=2.0,  # 2 seconds
        component_id=agent_id,
        description="Maximum decision calculation time"
    )
    
    # Decision confidence threshold
    health_monitor.add_metric_threshold(
        metric_name="decision_confidence",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.7,  # 70%
        critical_threshold=0.5,  # 50%
        component_id=agent_id,
        description="Minimum decision confidence level"
    )
    
    # Signal input diversity threshold
    health_monitor.add_metric_threshold(
        metric_name="signal_input_diversity",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=2.0,  # At least 2 signal types
        critical_threshold=1.0,  # At least 1 signal type
        component_id=agent_id,
        description="Minimum diversity of signal inputs"
    )
    
    # Decision consistency threshold
    health_monitor.add_metric_threshold(
        metric_name="decision_consistency",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.8,  # 80%
        critical_threshold=0.6,  # 60%
        component_id=agent_id,
        description="Minimum consistency in decision making"
    )
    
    # Conflicting signal ratio threshold
    health_monitor.add_metric_threshold(
        metric_name="conflicting_signal_ratio",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.3,  # 30%
        critical_threshold=0.5,  # 50%
        component_id=agent_id,
        description="Maximum ratio of conflicting signals"
    )
    
    logger.info(f"Configured decision agent metrics for {agent_id}")


def configure_execution_agent_metrics(
    health_monitor: HealthMonitor,
    agent_id: str,
    execution_type: Optional[str] = None
) -> None:
    """
    Configure health metrics for an execution agent.
    
    Args:
        health_monitor: Health monitoring system
        agent_id: ID of the execution agent
        execution_type: Optional execution type (e.g., "paper", "live")
    """
    type_str = f" for {execution_type}" if execution_type else ""
    
    # Execution latency threshold
    health_monitor.add_metric_threshold(
        metric_name="execution_latency",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=1.0,  # 1 second
        critical_threshold=5.0,  # 5 seconds
        component_id=agent_id,
        description=f"Maximum trade execution latency{type_str}"
    )
    
    # Order success rate threshold
    health_monitor.add_metric_threshold(
        metric_name="order_success_rate",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.95,  # 95%
        critical_threshold=0.8,  # 80%
        component_id=agent_id,
        description=f"Minimum order success rate{type_str}"
    )
    
    # Slippage threshold
    health_monitor.add_metric_threshold(
        metric_name="slippage_bps",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=10.0,  # 10 basis points
        critical_threshold=50.0,  # 50 basis points
        component_id=agent_id,
        description=f"Maximum acceptable slippage{type_str}"
    )
    
    # API error rate threshold
    health_monitor.add_metric_threshold(
        metric_name="api_error_rate",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.01,  # 1%
        critical_threshold=0.05,  # 5%
        component_id=agent_id,
        description=f"Maximum API error rate{type_str}"
    )
    
    # Additional thresholds for live trading
    if execution_type == "live":
        # Connection stability threshold
        health_monitor.add_metric_threshold(
            metric_name="connection_stability",
            threshold_type=ThresholdType.LOWER,
            warning_threshold=0.99,  # 99%
            critical_threshold=0.95,  # 95%
            component_id=agent_id,
            description="Minimum connection stability for live trading"
        )
        
        # Balance verification frequency threshold
        health_monitor.add_metric_threshold(
            metric_name="balance_verification_interval",
            threshold_type=ThresholdType.UPPER,
            warning_threshold=300.0,  # 5 minutes
            critical_threshold=1800.0,  # 30 minutes
            component_id=agent_id,
            description="Maximum time between balance verifications"
        )
    
    logger.info(f"Configured execution agent metrics for {agent_id}{type_str}")


def configure_risk_management_metrics(
    health_monitor: HealthMonitor,
    agent_id: str
) -> None:
    """
    Configure health metrics for a risk management agent.
    
    Args:
        health_monitor: Health monitoring system
        agent_id: ID of the risk management agent
    """
    # Position limit check frequency threshold
    health_monitor.add_metric_threshold(
        metric_name="limit_check_interval",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=60.0,  # 1 minute
        critical_threshold=300.0,  # 5 minutes
        component_id=agent_id,
        description="Maximum time between position limit checks"
    )
    
    # Risk calculation error rate threshold
    health_monitor.add_metric_threshold(
        metric_name="risk_calculation_error_rate",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.001,  # 0.1%
        critical_threshold=0.01,  # 1%
        component_id=agent_id,
        description="Maximum risk calculation error rate"
    )
    
    # Portfolio VaR threshold
    health_monitor.add_metric_threshold(
        metric_name="portfolio_var",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.05,  # 5%
        critical_threshold=0.1,  # 10%
        component_id=agent_id,
        description="Maximum portfolio Value-at-Risk"
    )
    
    # Max drawdown threshold
    health_monitor.add_metric_threshold(
        metric_name="max_drawdown",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.1,  # 10%
        critical_threshold=0.2,  # 20%
        component_id=agent_id,
        description="Maximum acceptable drawdown"
    )
    
    # Position concentration threshold
    health_monitor.add_metric_threshold(
        metric_name="position_concentration",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.3,  # 30%
        critical_threshold=0.5,  # 50%
        component_id=agent_id,
        description="Maximum position concentration in single asset"
    )
    
    logger.info(f"Configured risk management metrics for {agent_id}")


def configure_portfolio_metrics(
    health_monitor: HealthMonitor,
    component_id: str = "portfolio_manager"
) -> None:
    """
    Configure health metrics for portfolio management.
    
    Args:
        health_monitor: Health monitoring system
        component_id: ID of the portfolio management component
    """
    # Sharpe ratio threshold
    health_monitor.add_metric_threshold(
        metric_name="sharpe_ratio",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.5,
        critical_threshold=0.0,
        component_id=component_id,
        description="Minimum acceptable Sharpe ratio"
    )
    
    # Profit factor threshold
    health_monitor.add_metric_threshold(
        metric_name="profit_factor",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=1.2,
        critical_threshold=1.0,
        component_id=component_id,
        description="Minimum acceptable profit factor"
    )
    
    # Win rate threshold
    health_monitor.add_metric_threshold(
        metric_name="win_rate",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=0.4,  # 40%
        critical_threshold=0.3,  # 30%
        component_id=component_id,
        description="Minimum acceptable win rate"
    )
    
    # Drawdown duration threshold
    health_monitor.add_metric_threshold(
        metric_name="drawdown_duration",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=20.0,  # 20 cycles
        critical_threshold=50.0,  # 50 cycles
        component_id=component_id,
        description="Maximum acceptable drawdown duration"
    )
    
    # Trade frequency threshold
    health_monitor.add_metric_threshold(
        metric_name="trades_per_day",
        threshold_type=ThresholdType.LOWER,
        warning_threshold=1.0,
        critical_threshold=0.2,
        component_id=component_id,
        description="Minimum acceptable trading frequency"
    )
    
    logger.info(f"Configured portfolio metrics for {component_id}")


def configure_system_metrics(
    health_monitor: HealthMonitor,
    component_id: str = "system_monitor"
) -> None:
    """
    Configure health metrics for system monitoring.
    
    Args:
        health_monitor: Health monitoring system
        component_id: ID of the system monitoring component
    """
    # CPU usage threshold
    health_monitor.add_metric_threshold(
        metric_name="cpu_usage",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=70.0,  # 70%
        critical_threshold=90.0,  # 90%
        component_id=component_id,
        description="Maximum CPU usage"
    )
    
    # Memory usage threshold
    health_monitor.add_metric_threshold(
        metric_name="memory_usage",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=70.0,  # 70%
        critical_threshold=90.0,  # 90%
        component_id=component_id,
        description="Maximum memory usage"
    )
    
    # Disk space threshold
    health_monitor.add_metric_threshold(
        metric_name="disk_usage",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=80.0,  # 80%
        critical_threshold=95.0,  # 95%
        component_id=component_id,
        description="Maximum disk usage"
    )
    
    # Network latency threshold
    health_monitor.add_metric_threshold(
        metric_name="network_latency",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=200.0,  # 200ms
        critical_threshold=1000.0,  # 1 second
        component_id=component_id,
        description="Maximum network latency"
    )
    
    # Database query time threshold
    health_monitor.add_metric_threshold(
        metric_name="db_query_time",
        threshold_type=ThresholdType.UPPER,
        warning_threshold=0.1,  # 100ms
        critical_threshold=1.0,  # 1 second
        component_id=component_id,
        description="Maximum database query time"
    )
    
    logger.info(f"Configured system metrics for {component_id}")


def configure_all_trading_metrics(
    health_monitor: HealthMonitor,
    agents: Dict[str, Dict[str, Any]]
) -> None:
    """
    Configure all trading metrics for a set of agents.
    
    Args:
        health_monitor: Health monitoring system
        agents: Dictionary of agent configurations
    """
    # Configure system-wide metrics
    configure_system_metrics(health_monitor)
    configure_portfolio_metrics(health_monitor)
    
    # Configure agent-specific metrics
    for agent_id, config in agents.items():
        agent_type = config.get("type", "unknown")
        
        if agent_type == "market_data":
            configure_market_data_agent_metrics(
                health_monitor, 
                agent_id,
                config.get("symbol")
            )
        elif agent_type == "strategy":
            configure_strategy_agent_metrics(
                health_monitor,
                agent_id,
                config.get("strategy_type")
            )
        elif agent_type == "decision":
            configure_decision_agent_metrics(
                health_monitor,
                agent_id
            )
        elif agent_type == "execution":
            configure_execution_agent_metrics(
                health_monitor,
                agent_id,
                config.get("execution_type")
            )
        elif agent_type == "risk":
            configure_risk_management_metrics(
                health_monitor,
                agent_id
            )
    
    logger.info(f"Configured all trading metrics for {len(agents)} agents")
