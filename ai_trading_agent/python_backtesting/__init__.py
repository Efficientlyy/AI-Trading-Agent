# Python implementation of the backtesting module
# This serves as a temporary replacement for the Rust extension

from .models import (
    OrderSide, OrderType, OrderStatus, Order, Fill, Trade, Position,
    Portfolio, PortfolioSnapshot, OHLCVBar, BacktestConfig, PerformanceMetrics, BacktestResult
)
from .backtester import (
    calculate_execution_price, apply_transaction_costs, update_portfolio_from_trade,
    update_portfolio_value, update_position_market_price, calculate_performance_metrics,
    run_backtest
)
from .wrapper import run_backtest_py, convert_numpy_arrays_to_lists

# Provide the same interface as the Rust extension would
# This allows for seamless switching between implementations
run_backtest_rs = run_backtest_py
