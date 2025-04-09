rech"""
Plot portfolio performance metrics and visualizations.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(equity_curve: pd.Series):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid()
    plt.show()

def plot_asset_allocations(portfolio_history: list):
    df = pd.DataFrame(portfolio_history)
    df.set_index("timestamp", inplace=True)
    allocations = pd.DataFrame([
        snapshot.get("positions", {}) for snapshot in portfolio_history
    ], index=df.index).fillna(0)
    allocations.plot.area(figsize=(12, 6), title="Asset Allocations Over Time")
    plt.xlabel("Date")
    plt.ylabel("Position Value")
    plt.grid()
    plt.show()

def plot_drawdowns(drawdown_curve: pd.Series):
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown_curve.index, drawdown_curve.values)
    plt.title("Portfolio Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid()
    plt.show()

def main():
    # Example: Load saved portfolio history and metrics
    with open("portfolio_history.json") as f:
        portfolio_history = json.load(f)

    equity_curve = pd.Series(
        {snap["timestamp"]: snap["total_value"] for snap in portfolio_history}
    )
    equity_curve.index = pd.to_datetime(equity_curve.index)

    # Plot equity curve
    plot_equity_curve(equity_curve)

    # Plot asset allocations
    plot_asset_allocations(portfolio_history)

    # Example: Load saved drawdown curve
    # with open("drawdown_curve.json") as f:
    #     drawdown_data = json.load(f)
    # drawdown_curve = pd.Series(drawdown_data)
    # plot_drawdowns(drawdown_curve)

if __name__ == "__main__":
    main()