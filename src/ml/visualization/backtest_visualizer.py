"""Visualization tools for backtest analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional
from ..backtesting.backtester import BacktestResult, TradeResult

class BacktestVisualizer:
    """Visualization tools for backtest analysis."""
    
    def __init__(self, results: BacktestResult):
        """Initialize visualizer with backtest results."""
        self.results = results
        self.trades_df = pd.DataFrame(results["trades"])
        self.equity_curve = results["equity_curve"]
        self.returns = results["returns"]
    
    def plot_equity_curve(self) -> go.Figure:
        """Plot equity curve with drawdown overlay.
        
        Returns:
            Plotly figure with equity curve and drawdown
        """
        # Calculate drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                y=self.equity_curve,
                name="Equity",
                line=dict(color="blue", width=2)
            ),
            secondary_y=False
        )
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                y=drawdown * 100,  # Convert to percentage
                name="Drawdown",
                line=dict(color="red", width=1),
                fill="tonexty"
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Equity Curve and Drawdown",
            xaxis_title="Trade Number",
            yaxis_title="Equity ($)",
            yaxis2_title="Drawdown (%)",
            hovermode="x unified"
        )
        
        return fig
    
    def plot_returns_distribution(self) -> go.Figure:
        """Plot distribution of returns with normal distribution overlay.
        
        Returns:
            Plotly figure with returns distribution
        """
        fig = go.Figure()
        
        # Add histogram of returns
        fig.add_trace(
            go.Histogram(
                x=self.returns * 100,  # Convert to percentage
                name="Returns",
                nbinsx=50,
                histnorm="probability"
            )
        )
        
        # Add normal distribution overlay
        x = np.linspace(
            min(self.returns) * 100,
            max(self.returns) * 100,
            100
        )
        mu = np.mean(self.returns) * 100
        sigma = np.std(self.returns) * 100
        y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name="Normal Distribution",
                line=dict(color="red", dash="dash")
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Probability",
            showlegend=True
        )
        
        return fig
    
    def plot_trade_analysis(self) -> go.Figure:
        """Plot trade analysis dashboard.
        
        Returns:
            Plotly figure with trade analysis
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Win Rate by Month",
                "Average Trade PnL by Month",
                "Trade Duration Distribution",
                "PnL vs. Position Size"
            )
        )
        
        # Convert timestamps to datetime
        self.trades_df["entry_time"] = pd.to_datetime(
            self.trades_df["entry_time"]
        )
        
        # Group by month
        monthly = self.trades_df.set_index("entry_time").resample("M")
        
        # Win rate by month
        win_rate = monthly.apply(
            lambda x: (x["pnl"] > 0).mean()
        ).fillna(0)
        
        fig.add_trace(
            go.Bar(
                x=win_rate.index,
                y=win_rate.values * 100,
                name="Win Rate"
            ),
            row=1,
            col=1
        )
        
        # Average PnL by month
        avg_pnl = monthly["pnl"].mean()
        
        fig.add_trace(
            go.Bar(
                x=avg_pnl.index,
                y=avg_pnl.values,
                name="Avg PnL"
            ),
            row=1,
            col=2
        )
        
        # Trade duration distribution
        fig.add_trace(
            go.Histogram(
                x=self.trades_df["holding_period"],
                name="Duration"
            ),
            row=2,
            col=1
        )
        
        # PnL vs. Position Size scatter
        fig.add_trace(
            go.Scatter(
                x=abs(self.trades_df["size"]),
                y=self.trades_df["pnl"],
                mode="markers",
                name="PnL vs Size",
                marker=dict(
                    color=self.trades_df["return_pct"],
                    colorscale="RdYlGn",
                    showscale=True
                )
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Trade Analysis Dashboard"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Duration (days)", row=2, col=1)
        fig.update_xaxes(title_text="Position Size", row=2, col=2)
        
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Average PnL ($)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="PnL ($)", row=2, col=2)
        
        return fig
    
    def plot_risk_metrics(self) -> go.Figure:
        """Plot risk metrics dashboard.
        
        Returns:
            Plotly figure with risk metrics
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Rolling Sharpe Ratio",
                "Rolling Volatility",
                "Rolling Max Drawdown",
                "Rolling Win Rate"
            )
        )
        
        # Calculate rolling metrics
        window = 20  # 20-trade window
        
        # Rolling Sharpe
        returns_series = pd.Series(self.returns)
        rolling_sharpe = (
            returns_series.rolling(window).mean() /
            returns_series.rolling(window).std()
        ) * np.sqrt(252)  # Annualize
        
        fig.add_trace(
            go.Scatter(
                y=rolling_sharpe,
                name="Sharpe Ratio",
                line=dict(color="blue")
            ),
            row=1,
            col=1
        )
        
        # Rolling Volatility
        rolling_vol = returns_series.rolling(window).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(
                y=rolling_vol,
                name="Volatility",
                line=dict(color="red")
            ),
            row=1,
            col=2
        )
        
        # Rolling Max Drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_dd = equity_series.rolling(window).apply(
            lambda x: (x.max() - x.min()) / x.max()
        )
        
        fig.add_trace(
            go.Scatter(
                y=rolling_dd * 100,
                name="Max Drawdown",
                line=dict(color="orange")
            ),
            row=2,
            col=1
        )
        
        # Rolling Win Rate
        pnl_series = pd.Series([t["pnl"] for t in self.results["trades"]])
        rolling_winrate = (
            (pnl_series > 0).rolling(window).mean() * 100
        )
        
        fig.add_trace(
            go.Scatter(
                y=rolling_winrate,
                name="Win Rate",
                line=dict(color="green")
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Risk Metrics Dashboard"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Trade Number", row=1, col=1)
        fig.update_xaxes(title_text="Trade Number", row=1, col=2)
        fig.update_xaxes(title_text="Trade Number", row=2, col=1)
        fig.update_xaxes(title_text="Trade Number", row=2, col=2)
        
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Annualized Volatility", row=1, col=2)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)
        
        return fig
    
    def generate_report(self) -> str:
        """Generate a text report summarizing backtest results.
        
        Returns:
            Formatted string with backtest summary
        """
        report = []
        report.append("Backtest Results Summary")
        report.append("=" * 50)
        
        # Overall metrics
        report.append("\nOverall Performance:")
        report.append(f"Total Return: {(self.equity_curve[-1] / self.equity_curve[0] - 1) * 100:.2f}%")
        report.append(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {self.results['max_drawdown'] * 100:.2f}%")
        report.append(f"Profit Factor: {self.results['profit_factor']:.2f}")
        
        # Trade statistics
        report.append("\nTrade Statistics:")
        report.append(f"Number of Trades: {len(self.results['trades'])}")
        report.append(f"Win Rate: {self.results['win_rate'] * 100:.2f}%")
        report.append(f"Average Trade: ${self.results['avg_trade']:.2f}")
        report.append(f"Average Winner: ${self.results['avg_winning_trade']:.2f}")
        report.append(f"Average Loser: ${self.results['avg_losing_trade']:.2f}")
        report.append(f"Max Consecutive Losses: {self.results['max_consecutive_losses']}")
        
        # Risk metrics
        report.append("\nRisk Metrics:")
        report.append(f"Annualized Volatility: {np.std(self.returns) * np.sqrt(252) * 100:.2f}%")
        report.append(f"Skewness: {pd.Series(self.returns).skew():.2f}")
        report.append(f"Kurtosis: {pd.Series(self.returns).kurtosis():.2f}")
        
        # Cost analysis
        report.append("\nTransaction Cost Analysis:")
        report.append(f"Total Costs: ${self.results['total_costs']:.2f}")
        report.append(f"Total Slippage: ${self.results['total_slippage']:.2f}")
        report.append(f"Total Market Impact: ${self.results['total_market_impact']:.2f}")
        report.append(f"Cost per Trade: ${self.results['total_costs'] / len(self.results['trades']):.2f}")
        
        # Monthly analysis
        monthly_returns = (
            pd.Series(self.returns)
            .groupby(pd.date_range(start=self.trades_df["entry_time"].min(),
                                 end=self.trades_df["entry_time"].max(),
                                 freq="M"))
            .sum()
        )
        
        report.append("\nMonthly Analysis:")
        report.append(f"Best Month: {monthly_returns.max() * 100:.2f}%")
        report.append(f"Worst Month: {monthly_returns.min() * 100:.2f}%")
        report.append(f"Average Month: {monthly_returns.mean() * 100:.2f}%")
        report.append(f"Monthly Std Dev: {monthly_returns.std() * 100:.2f}%")
        
        return "\n".join(report) 