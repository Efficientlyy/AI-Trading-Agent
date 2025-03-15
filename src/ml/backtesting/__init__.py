"""Market Regime Backtesting module.

This module provides tools for backtesting trading strategies based on market regime detection.
"""

# Import from core backtesting modules
from .regime_strategy import RegimeStrategy
from .backtester import (
    Backtester, 
    TransactionCosts, 
    Position, 
    TradeResult, 
    BacktestResult, 
    BacktestMetrics
)

# Import from performance metrics module
from .performance_metrics import (
    calculate_returns,
    calculate_log_returns,
    calculate_cumulative_returns,
    calculate_drawdowns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_expectancy,
    calculate_regime_metrics,
    calculate_comprehensive_metrics
)

# Import from position sizing module
from .position_sizing import (
    PositionSizer,
    FixedPositionSizer,
    PercentPositionSizer,
    KellyPositionSizer,
    VolatilityPositionSizer,
    PositionSizerFactory
)

# Import from risk management module
from .risk_management import (
    RiskManager,
    BasicRiskManager,
    TrailingStopManager,
    VolatilityBasedRiskManager,
    RiskManagerFactory,
    ExitType
)

# Import from visualization module
from .visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_regime_performance,
    plot_trade_analysis,
    plot_returns_heatmap,
    create_comprehensive_report
)

__all__ = [
    # Core backtesting
    'RegimeStrategy',
    'Backtester',
    'TransactionCosts',
    'Position',
    'TradeResult',
    'BacktestResult',
    'BacktestMetrics',
    
    # Performance metrics
    'calculate_returns',
    'calculate_log_returns',
    'calculate_cumulative_returns',
    'calculate_drawdowns',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_expectancy',
    'calculate_regime_metrics',
    'calculate_comprehensive_metrics',
    
    # Position sizing
    'PositionSizer',
    'FixedPositionSizer',
    'PercentPositionSizer',
    'KellyPositionSizer',
    'VolatilityPositionSizer',
    'PositionSizerFactory',
    
    # Risk management
    'RiskManager',
    'BasicRiskManager',
    'TrailingStopManager',
    'VolatilityBasedRiskManager',
    'RiskManagerFactory',
    'ExitType',
    
    # Visualization
    'plot_equity_curve',
    'plot_drawdown',
    'plot_regime_performance',
    'plot_trade_analysis',
    'plot_returns_heatmap',
    'create_comprehensive_report'
] 