"""
Factory module for creating agent components based on configuration.

This module provides factory functions to create instances of:
- DataManager
- Strategy
- RiskManager
- PortfolioManager
- ExecutionHandler
- Orchestrator

Each factory function takes a configuration dictionary and returns an instance
of the appropriate component.
"""
from typing import Dict, Any, Type, Optional, List, Union
import importlib

# Import all component types
from .data_manager import (
    DataManagerABC,
    BaseDataManager,
    SimpleDataManager,
)
from .strategy import (
    BaseStrategy,
    StrategyManagerABC,
    BaseStrategyManager,
    SentimentStrategy,
    SentimentStrategyManager,
)
from .risk_manager import (
    RiskManagerABC,
    BaseRiskManager,
    SimpleRiskManager,
)
from .execution_handler import (
    ExecutionHandlerABC,
    BaseExecutionHandler,
    SimulatedExecutionHandler,
)
from .portfolio import (
    PortfolioManagerABC,
    BasePortfolioManager,
)
from .orchestrator import (
    OrchestratorABC,
    BaseOrchestrator,
    BacktestOrchestrator,
)

# Import trading engine components
from ..trading_engine.portfolio_manager import PortfolioManager

from ..common.logging_config import logger

# Component registries
DATA_MANAGER_REGISTRY = {
    "SimpleDataManager": SimpleDataManager,
}

STRATEGY_REGISTRY = {
    "SentimentStrategy": SentimentStrategy,
}

STRATEGY_MANAGER_REGISTRY = {
    "SentimentStrategyManager": SentimentStrategyManager,
    "BaseStrategyManager": BaseStrategyManager,
}

RISK_MANAGER_REGISTRY = {
    "SimpleRiskManager": SimpleRiskManager,
}

PORTFOLIO_MANAGER_REGISTRY = {
    "PortfolioManager": PortfolioManager,
}

EXECUTION_HANDLER_REGISTRY = {
    "SimulatedExecutionHandler": SimulatedExecutionHandler,
}

ORCHESTRATOR_REGISTRY = {
    "BacktestOrchestrator": BacktestOrchestrator,
}

# Check if Rust components are available
try:
    from ..backtesting.rust_backtester import RustBacktester
    ORCHESTRATOR_REGISTRY["RustBacktester"] = RustBacktester
    RUST_AVAILABLE = True
    logger.info("Rust components are available")
except ImportError:
    RUST_AVAILABLE = False
    logger.info("Rust components are not available")

def is_rust_available() -> bool:
    """
    Check if Rust components are available.
    
    Returns:
        bool: True if Rust components are available, False otherwise.
    """
    return RUST_AVAILABLE

def register_component(registry: Dict[str, Type], name: str, component_class: Type) -> None:
    """
    Register a component in a registry.
    
    Args:
        registry: The registry to register the component in.
        name: The name of the component.
        component_class: The component class.
    """
    registry[name] = component_class
    logger.info(f"Registered component {name} in registry")

def create_data_manager(config: Dict[str, Any]) -> DataManagerABC:
    """
    Create a data manager instance based on configuration.
    
    Args:
        config: Configuration dictionary with 'type' and 'config' keys.
        
    Returns:
        An instance of DataManagerABC.
        
    Raises:
        ValueError: If the data manager type is not recognized.
    """
    data_manager_type = config.get("type")
    data_manager_config = config.get("config", {})
    
    if data_manager_type not in DATA_MANAGER_REGISTRY:
        raise ValueError(f"Unknown data manager type: {data_manager_type}")
    
    data_manager_class = DATA_MANAGER_REGISTRY[data_manager_type]
    logger.info(f"Creating {data_manager_type} with config: {data_manager_config}")
    
    return data_manager_class(config=data_manager_config)

def create_strategy(config: Dict[str, Any]) -> BaseStrategy:
    """
    Create a strategy instance based on configuration.
    
    Args:
        config: Configuration dictionary with 'type' and 'config' keys.
        
    Returns:
        An instance of BaseStrategy.
        
    Raises:
        ValueError: If the strategy type is not recognized.
    """
    strategy_type = config.get("type")
    strategy_config = config.get("config", {})
    
    if strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_type]
    logger.info(f"Creating {strategy_type} with config: {strategy_config}")
    
    # Extract name from config or use default
    name = strategy_config.pop("name", strategy_type)
    
    return strategy_class(name=name, config=strategy_config)

def create_strategy_manager(strategy: BaseStrategy, manager_type: str = "BaseStrategyManager", data_manager=None, config=None) -> StrategyManagerABC:
    """
    Create a strategy manager instance based on configuration.
    
    Args:
        strategy: The strategy instance to manage.
        manager_type: The type of strategy manager to create.
        data_manager: The data manager instance, if needed by the manager.
    Returns:
        An instance of StrategyManagerABC.
    Raises:
        ValueError: If the strategy manager type is not recognized.
    """
    if manager_type not in STRATEGY_MANAGER_REGISTRY:
        raise ValueError(f"Unknown strategy manager type: {manager_type}")
    
    manager_class = STRATEGY_MANAGER_REGISTRY[manager_type]
    logger.info(f"Creating {manager_type} for strategy: {strategy.name}")
    
    if manager_type == "SentimentStrategyManager":
        # Use explicit config if provided, otherwise fallback to strategy.config
        final_config = config if config is not None else strategy.config
        return manager_class(final_config, data_manager)
    else:
        return manager_class(strategy)


def create_risk_manager(config: Dict[str, Any]) -> RiskManagerABC:
    """
    Create a risk manager instance based on configuration.
    
    Args:
        config: Configuration dictionary with 'type' and 'config' keys.
        
    Returns:
        An instance of RiskManagerABC.
        
    Raises:
        ValueError: If the risk manager type is not recognized.
    """
    risk_manager_type = config.get("type")
    risk_manager_config = config.get("config", {})
    
    if risk_manager_type not in RISK_MANAGER_REGISTRY:
        raise ValueError(f"Unknown risk manager type: {risk_manager_type}")
    
    risk_manager_class = RISK_MANAGER_REGISTRY[risk_manager_type]
    logger.info(f"Creating {risk_manager_type} with config: {risk_manager_config}")
    
    return risk_manager_class(config=risk_manager_config)

def create_portfolio_manager(config: Dict[str, Any]) -> PortfolioManager:
    """
    Create a portfolio manager instance based on configuration.
    
    Args:
        config: Configuration dictionary with 'type' and 'config' keys.
        
    Returns:
        An instance of PortfolioManager.
        
    Raises:
        ValueError: If the portfolio manager type is not recognized.
    """
    portfolio_manager_type = config.get("type")
    portfolio_manager_config = config.get("config", {})
    
    if portfolio_manager_type not in PORTFOLIO_MANAGER_REGISTRY:
        raise ValueError(f"Unknown portfolio manager type: {portfolio_manager_type}")
    
    portfolio_manager_class = PORTFOLIO_MANAGER_REGISTRY[portfolio_manager_type]
    logger.info(f"Creating {portfolio_manager_type} with config: {portfolio_manager_config}")
    
    # Extract parameters for the constructor
    initial_capital = portfolio_manager_config.get("initial_capital", 10000.0)
    risk_per_trade = portfolio_manager_config.get("risk_per_trade", 0.02)
    max_position_size = portfolio_manager_config.get("max_position_size", 0.2)
    max_correlation = portfolio_manager_config.get("max_correlation", 0.7)
    rebalance_frequency = portfolio_manager_config.get("rebalance_frequency", "weekly")
    
    return portfolio_manager_class(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        max_position_size=max_position_size,
        max_correlation=max_correlation,
        rebalance_frequency=rebalance_frequency,
    )

def create_execution_handler(config: Dict[str, Any], portfolio_manager: PortfolioManager) -> ExecutionHandlerABC:
    """
    Create an execution handler instance based on configuration.
    
    Args:
        config: Configuration dictionary with 'type' and 'config' keys.
        portfolio_manager: The portfolio manager instance to use.
        
    Returns:
        An instance of ExecutionHandlerABC.
        
    Raises:
        ValueError: If the execution handler type is not recognized.
    """
    execution_handler_type = config.get("type")
    execution_handler_config = config.get("config", {})
    
    if execution_handler_type not in EXECUTION_HANDLER_REGISTRY:
        raise ValueError(f"Unknown execution handler type: {execution_handler_type}")
    
    execution_handler_class = EXECUTION_HANDLER_REGISTRY[execution_handler_type]
    logger.info(f"Creating {execution_handler_type} with config: {execution_handler_config}")
    
    return execution_handler_class(
        portfolio_manager=portfolio_manager,
        config=execution_handler_config,
    )

def create_orchestrator(
    data_manager: DataManagerABC,
    strategy_manager: StrategyManagerABC,
    portfolio_manager: PortfolioManager,
    risk_manager: RiskManagerABC,
    execution_handler: ExecutionHandlerABC,
    config: Dict[str, Any],
    orchestrator_type: str = "BacktestOrchestrator"
) -> OrchestratorABC:
    """
    Create an orchestrator instance based on configuration.
    
    Args:
        data_manager: The data manager instance.
        strategy_manager: The strategy manager instance.
        portfolio_manager: The portfolio manager instance.
        risk_manager: The risk manager instance.
        execution_handler: The execution handler instance.
        config: Configuration dictionary for the orchestrator.
        orchestrator_type: The type of orchestrator to create.
        
    Returns:
        An instance of OrchestratorABC.
        
    Raises:
        ValueError: If the orchestrator type is not recognized.
    """
    # Check if Rust is requested but not available
    if orchestrator_type == "RustBacktester" and not RUST_AVAILABLE:
        logger.warning("Rust backtester requested but not available, falling back to Python backtester")
        orchestrator_type = "BacktestOrchestrator"
    
    if orchestrator_type not in ORCHESTRATOR_REGISTRY:
        raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")
    
    orchestrator_class = ORCHESTRATOR_REGISTRY[orchestrator_type]
    logger.info(f"Creating {orchestrator_type} with config: {config}")
    
    return orchestrator_class(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=config,
    )

def create_agent_from_config(config: Dict[str, Any], use_rust: bool = False) -> OrchestratorABC:
    """
    Create a complete agent (orchestrator with all components) from configuration.
    
    Args:
        config: The complete agent configuration.
        use_rust: Whether to use Rust-accelerated components if available.
        
    Returns:
        An instance of OrchestratorABC with all components initialized.
    """
    # Create data manager
    data_manager = create_data_manager(config["data_manager"])
    
    # Create strategy and strategy manager
    strategy = create_strategy(config["strategy"])
    strategy_manager = create_strategy_manager(strategy)
    
    # Create portfolio manager
    portfolio_manager = create_portfolio_manager(config["portfolio_manager"])
    
    # Create risk manager
    risk_manager = create_risk_manager(config["risk_manager"])
    
    # Create execution handler
    execution_handler = create_execution_handler(
        config["execution_handler"],
        portfolio_manager
    )
    
    # Determine orchestrator type
    orchestrator_type = "RustBacktester" if use_rust and RUST_AVAILABLE else "BacktestOrchestrator"
    
    # Create orchestrator
    orchestrator = create_orchestrator(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=config.get("backtest", {}),
        orchestrator_type=orchestrator_type
    )
    
    return orchestrator

def load_custom_component(module_path: str, class_name: str) -> Type:
    """
    Dynamically load a custom component class.
    
    Args:
        module_path: The path to the module containing the component.
        class_name: The name of the component class.
        
    Returns:
        The component class.
        
    Raises:
        ImportError: If the module or class cannot be imported.
    """
    try:
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        return component_class
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load custom component {class_name} from {module_path}: {e}")
        raise ImportError(f"Failed to load custom component {class_name} from {module_path}: {e}")

def register_custom_component(component_type: str, name: str, module_path: str, class_name: str) -> None:
    """
    Register a custom component in the appropriate registry.
    
    Args:
        component_type: The type of component ('data_manager', 'strategy', etc.).
        name: The name to register the component under.
        module_path: The path to the module containing the component.
        class_name: The name of the component class.
        
    Raises:
        ValueError: If the component type is not recognized.
        ImportError: If the module or class cannot be imported.
    """
    component_class = load_custom_component(module_path, class_name)
    
    if component_type == "data_manager":
        register_component(DATA_MANAGER_REGISTRY, name, component_class)
    elif component_type == "strategy":
        register_component(STRATEGY_REGISTRY, name, component_class)
    elif component_type == "strategy_manager":
        register_component(STRATEGY_MANAGER_REGISTRY, name, component_class)
    elif component_type == "risk_manager":
        register_component(RISK_MANAGER_REGISTRY, name, component_class)
    elif component_type == "portfolio_manager":
        register_component(PORTFOLIO_MANAGER_REGISTRY, name, component_class)
    elif component_type == "execution_handler":
        register_component(EXECUTION_HANDLER_REGISTRY, name, component_class)
    elif component_type == "orchestrator":
        register_component(ORCHESTRATOR_REGISTRY, name, component_class)
    else:
        raise ValueError(f"Unknown component type: {component_type}")
    
    logger.info(f"Registered custom {component_type} component: {name}")
