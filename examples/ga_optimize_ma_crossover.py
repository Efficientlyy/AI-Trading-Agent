"""
Example: Optimize Moving Average Crossover strategy parameters using GA.
"""

from src.strategies.ga_optimizer import GAOptimizer, sample_fitness_function

# Define parameter space
param_space = {
    "fast_period": list(range(5, 30)),
    "slow_period": list(range(20, 100)),
    "threshold": [0.0, 0.01, 0.02, 0.05]
}

def run_backtest_with_params(params):
    """
    Run a backtest with the given strategy parameters and return a fitness score.

    Args:
        params: Dictionary of strategy parameters.

    Returns:
        Fitness score (higher is better).
    """
    # Penalize invalid parameter combinations
    if params["fast_period"] >= params["slow_period"]:
        return -1.0

    import pandas as pd
    import numpy as np
    from src.backtesting.backtester import Backtester

    # Generate sample price data for multiple assets
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    market_data = {}
    for symbol, base_price in {"BTC/USD": 50000, "ETH/USD": 3000, "SOL/USD": 150}.items():
        prices = base_price + np.cumsum(np.random.randn(200) * base_price * 0.02)
        df = pd.DataFrame({"close": prices})
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 1000
        df.index = dates
        market_data[symbol] = df

    backtester = Backtester(market_data, initial_capital=100000)

    def strategy_fn(data_dict, portfolio, idx):
        orders = []
        for symbol, df in data_dict.items():
            if idx < params["slow_period"]:
                continue

            fast_ma = df["close"].iloc[idx - params["fast_period"]:idx].mean()
            slow_ma = df["close"].iloc[idx - params["slow_period"]:idx].mean()

            if fast_ma > slow_ma * (1 + params["threshold"]):
                orders.append({
                    "symbol": symbol,
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 0.1
                })
            elif fast_ma < slow_ma * (1 - params["threshold"]):
                orders.append({
                    "symbol": symbol,
                    "side": "sell",
                    "order_type": "market",
                    "quantity": 0.1
                })
        return orders

    metrics = backtester.run(strategy_fn)
    return metrics.sharpe_ratio if hasattr(metrics, "sharpe_ratio") else 0.0

optimizer = GAOptimizer(
    param_space=param_space,
    fitness_func=run_backtest_with_params,
    population_size=10,
    generations=20,
    crossover_rate=0.7,
    mutation_rate=0.2
)

best_params = optimizer.evolve()

print("Best parameters found:")
print(best_params)