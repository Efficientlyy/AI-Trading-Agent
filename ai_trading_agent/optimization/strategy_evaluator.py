import pandas as pd
from typing import List, Dict, Any, Callable
from ai_trading_agent.backtesting.performance_metrics import calculate_metrics

class StrategyEvaluator:
    """
    Evaluate and compare multiple trading strategies or parameter sets.
    """

    def __init__(
        self,
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        metrics_fn: Callable[..., Any] = calculate_metrics,
    ):
        """
        backtest_fn: function that takes a parameter set and returns backtest results (portfolio_history, trade_history, initial_capital)
        metrics_fn: function that takes backtest results and returns performance metrics
        """
        self.backtest_fn = backtest_fn
        self.metrics_fn = metrics_fn

    def evaluate_strategies(self, param_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluate a list of strategies/parameter sets and return a DataFrame of metrics.
        """
        results = []
        for params in param_list:
            backtest_result = self.backtest_fn(params)
            metrics = self.metrics_fn(
                backtest_result["portfolio_history"],
                backtest_result["trade_history"],
                backtest_result["initial_capital"]
            )
            results.append({
                "params": params,
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "calmar_ratio": metrics.calmar_ratio,
                "omega_ratio": metrics.omega_ratio,
                "annualized_return": metrics.annualized_return,
                "volatility": metrics.volatility,
            })
        return pd.DataFrame(results)