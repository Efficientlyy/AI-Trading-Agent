# AI Trading Agent - Agent Module
"""
The agent module provides the core components for building and running trading agents.

This module includes:
- Data management (DataManagerABC, BaseDataManager, SimpleDataManager)
- Strategy management (StrategyManagerABC, BaseStrategyManager, SentimentStrategy)
- Risk management (RiskManagerABC, BaseRiskManager, SimpleRiskManager)
- Execution handling (ExecutionHandlerABC, BaseExecutionHandler, SimulatedExecutionHandler)
- Portfolio management (PortfolioManagerABC, BasePortfolioManager)
- Orchestration (OrchestratorABC, BaseOrchestrator, BacktestOrchestrator)
"""

# Data Management
from .data_manager import (
    DataManagerABC,
    BaseDataManager,
    SimpleDataManager,
)

# Strategy Management
from .strategy import (
    BaseStrategy,
    StrategyManagerABC,
    BaseStrategyManager,
    SentimentStrategy,
    SentimentStrategyManager,
)

# Risk Management
from .risk_manager import (
    RiskManagerABC,
    BaseRiskManager,
    SimpleRiskManager,
)

# Execution Handling
from .execution_handler import (
    ExecutionHandlerABC,
    BaseExecutionHandler,
    SimulatedExecutionHandler,
)

# Portfolio Management
from .portfolio import (
    PortfolioManagerABC,
    BasePortfolioManager,
)

# Orchestration
from .orchestrator import (
    OrchestratorABC,
    BaseOrchestrator,
    BacktestOrchestrator,
)

__all__ = [
    # Data Management
    'DataManagerABC', 'BaseDataManager', 'SimpleDataManager',
    
    # Strategy Management
    'BaseStrategy', 'StrategyManagerABC', 'BaseStrategyManager', 
    'SentimentStrategy', 'SentimentStrategyManager',
    
    # Risk Management
    'RiskManagerABC', 'BaseRiskManager', 'SimpleRiskManager',
    
    # Execution Handling
    'ExecutionHandlerABC', 'BaseExecutionHandler', 'SimulatedExecutionHandler',
    
    # Portfolio Management
    'PortfolioManagerABC', 'BasePortfolioManager',
    
    # Orchestration
    'OrchestratorABC', 'BaseOrchestrator', 'BacktestOrchestrator',
]
