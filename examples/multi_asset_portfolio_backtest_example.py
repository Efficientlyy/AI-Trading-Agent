"""
Example: Multi-Asset Portfolio-Level Backtesting

This script demonstrates how to use MultiAssetBacktester to run a portfolio-level backtest
with asset allocation, correlation analysis, and portfolio-wide risk management.
"""

import pandas as pd
import numpy as np
from ai_trading_agent.backtesting.multi_asset_backtester import MultiAssetBacktester
from ai_trading_agent.backtesting.asset_allocation import equal_weight_allocation, momentum_weight_allocation, risk_parity_allocation

# --- Mock Data Preparation ---
# Generate synthetic OHLCV data for 3 assets
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=200, freq="D")
assets = ["BTC", "ETH", "SOL"]
data = {}

for asset in assets:
    price = 100 + np.cumsum(np.random.randn(len(dates))) + np.random.uniform(-5, 5)
    df = pd.DataFrame({
        "open": price + np.random.uniform(-1, 1, len(dates)),
        "high": price + np.random.uniform(0, 2, len(dates)),
        "low": price - np.random.uniform(0, 2, len(dates)),
        "close": price,
        "volume": np.random.uniform(100, 1000, len(dates))
    }, index=dates)
    data[asset] = df

# --- Buy-and-Hold Strategy that Initializes Positions ---
def buy_and_hold_strategy(data, portfolio, bar_idx, allocation_fn=None):
    from ai_trading_agent.trading_engine.models import Order
    from ai_trading_agent.trading_engine.enums import OrderSide, OrderType
    
    # Only place initial buy orders on the first bar
    if bar_idx == 0 and allocation_fn is not None:
        # Determine allocation weights
        allocations = allocation_fn(data, bar_idx)
        total_value = portfolio.total_value
        current_prices = {symbol: df.iloc[bar_idx]["close"] for symbol, df in data.items()}
        orders = []
        for symbol, weight in allocations.items():
            if weight > 0 and symbol in current_prices:
                alloc_value = total_value * weight
                price = current_prices[symbol]
                quantity = alloc_value / price
                orders.append(Order(
                    symbol=symbol,
                    quantity=quantity,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    timestamp=df.index[bar_idx] if hasattr(df.index, "__getitem__") else None
                ))
        return orders
    else:
        return []

# --- DEBUG: Print MultiAssetBacktester class and method presence ---
print("DEBUG: MultiAssetBacktester class:", MultiAssetBacktester)
print("DEBUG: Has _initialize_portfolio_state?", hasattr(MultiAssetBacktester, '_initialize_portfolio_state'))
print("DEBUG: File loaded from:", MultiAssetBacktester.__module__)

# --- Portfolio-Level Backtesting (Equal Weight) ---
print("\n=== Equal Weight Allocation ===")
backtester_eq = MultiAssetBacktester(
    data=data,
    initial_capital=10000.0,
    max_position_pct=0.5,
    max_correlation=0.8
)
metrics_eq, additional_results_eq = backtester_eq.run(
    lambda d, p, i: buy_and_hold_strategy(d, p, i, allocation_fn=equal_weight_allocation),
    allocation_fn=equal_weight_allocation,
    rebalance_period=10
)
print("\nPortfolio Performance Metrics (Equal Weight):")
print(metrics_eq)
print("\nAsset Allocation (final, Equal Weight):")
print(additional_results_eq["allocation_history"])  # Show full allocation history
print("\nCorrelation Matrix (final, Equal Weight):")
print(additional_results_eq.get("correlation_matrix", "Not computed"))
print("\nRisk Metrics (Equal Weight):")
# If metrics_eq has risk metrics, print them; else print from additional_results_eq if present
if hasattr(metrics_eq, "max_drawdown"):
    print(f"Max Drawdown: {metrics_eq.max_drawdown}")
if hasattr(metrics_eq, "sharpe_ratio"):
    print(f"Sharpe Ratio: {metrics_eq.sharpe_ratio}")

# --- Portfolio-Level Backtesting (Momentum) ---
def momentum_alloc(data, bar_idx):
    return momentum_weight_allocation(data, bar_idx, lookback_period=30, top_n=None)

print("\n=== Momentum Allocation ===")
backtester_mom = MultiAssetBacktester(
    data=data,
    initial_capital=10000.0,
    max_position_pct=0.5,
    max_correlation=0.8
)
metrics_mom, additional_results_mom = backtester_mom.run(
    lambda d, p, i: buy_and_hold_strategy(d, p, i, allocation_fn=momentum_alloc),
    allocation_fn=momentum_alloc,
    rebalance_period=10
)
print("\nPortfolio Performance Metrics (Momentum):")
print(metrics_mom)
print("\nAsset Allocation (final, Momentum):")
print(additional_results_mom["allocation_history"])  # Show full allocation history
print("\nCorrelation Matrix (final, Momentum):")
print(additional_results_mom.get("correlation_matrix", "Not computed"))
print("\nRisk Metrics (Momentum):")
if hasattr(metrics_mom, "max_drawdown"):
    print(f"Max Drawdown: {metrics_mom.max_drawdown}")
if hasattr(metrics_mom, "sharpe_ratio"):
    print(f"Sharpe Ratio: {metrics_mom.sharpe_ratio}")

# --- Portfolio-Level Backtesting (Risk Parity) ---
print("\n=== Risk Parity Allocation ===")
backtester_rp = MultiAssetBacktester(
    data=data,
    initial_capital=10000.0,
    max_position_pct=0.5,
    max_correlation=0.8
)
metrics_rp, additional_results_rp = backtester_rp.run(
    lambda d, p, i: buy_and_hold_strategy(d, p, i, allocation_fn=risk_parity_allocation),
    allocation_fn=risk_parity_allocation,
    rebalance_period=10
)
print("\nPortfolio Performance Metrics (Risk Parity):")
print(metrics_rp)
print("\nAsset Allocation (final, Risk Parity):")
print(additional_results_rp["allocation_history"])
print("\nCorrelation Matrix (final, Risk Parity):")
print(additional_results_rp.get("correlation_matrix", "Not computed"))
print("\nRisk Metrics (Risk Parity):")
if hasattr(metrics_rp, "max_drawdown"):
    print(f"Max Drawdown: {metrics_rp.max_drawdown}")
if hasattr(metrics_rp, "sharpe_ratio"):
    print(f"Sharpe Ratio: {metrics_rp.sharpe_ratio}")

print("\nExample complete. Portfolio-level backtest executed for all allocation strategies (equal weight, momentum, risk parity).")
