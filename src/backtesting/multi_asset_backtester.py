"""
Multi-Asset Backtesting Framework for AI Trading Agent.

This module extends the backtesting functionality to support portfolio-level
analysis across multiple assets, including correlation analysis, asset allocation,
and portfolio-level risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from datetime import datetime
import time
from dataclasses import dataclass, field
import os

from src.trading_engine.models import Order, Trade, Position, Portfolio, OrderSide, OrderType
from src.trading_engine.order_manager import OrderManager
from src.common import logger
from .performance_metrics import calculate_metrics, PerformanceMetrics, calculate_asset_metrics, calculate_portfolio_diversification
from .portfolio_risk import PortfolioRiskManager
from .backtester import Backtester
from .correlation_analysis import calculate_correlation_matrix, analyze_correlations, CorrelationStats
from .portfolio_visualization import (
    plot_portfolio_performance, 
    plot_asset_allocation, 
    plot_asset_performance_comparison,
    plot_drawdown_comparison,
    create_performance_dashboard
)
from .diversification_analysis import (
    analyze_diversification_benefits,
    plot_efficient_frontier,
    plot_risk_contributions,
    plot_correlation_impact
)


class MultiAssetBacktester(Backtester):
    """
    Multi-Asset Backtester class for simulating portfolio-level strategies across multiple assets.
    
    This class extends the base Backtester to provide:
    - Portfolio-level backtesting with proper asset allocation
    - Correlation analysis between different assets
    - Risk management across the entire portfolio
    - Performance metrics that consider the entire portfolio
    - Visualization of portfolio performance including diversification benefits
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.0,
        enable_fractional: bool = True,
        max_position_pct: float = 0.2,
        max_correlation: float = 0.7,
        rebalance_threshold: float = 0.05,
    ):
        """
        Initialize the multi-asset backtester.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            initial_capital: Starting capital for the portfolio
            commission_rate: Commission rate as a decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as a decimal (e.g., 0.001 = 0.1%)
            enable_fractional: Whether to allow fractional position sizes
            max_position_pct: Maximum percentage of portfolio in a single position
            max_correlation: Maximum allowed correlation between assets
            rebalance_threshold: Threshold for rebalancing (as percentage difference)
        """
        super().__init__(
            data=data,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage,
            enable_fractional=enable_fractional,
        )
        
        # Initialize portfolio risk manager
        self.risk_manager = PortfolioRiskManager(
            max_position_pct=max_position_pct,
            max_correlation=max_correlation
        )
        
        # Additional parameters
        self.rebalance_threshold = rebalance_threshold
        
        # Initialize correlation tracking
        self.correlation_matrix = {}
        self.returns_data = {symbol: [] for symbol in self.symbols}
        self.correlation_stats_history = []
        
        # Initialize asset allocation tracking
        self.target_allocations = {}
        self.current_allocations = {symbol: 0.0 for symbol in self.symbols}
        self.allocation_history = {symbol: [] for symbol in self.symbols}
        self.allocation_timestamps = []
        
        # Initialize asset-level performance tracking
        self.asset_performance = {symbol: {'returns': [], 'drawdowns': []} for symbol in self.symbols}
        self.asset_equity_curves = {symbol: [] for symbol in self.symbols}
        self.asset_drawdown_curves = {symbol: [] for symbol in self.symbols}
        
        logger.info(f"Initialized multi-asset backtester with {len(self.symbols)} symbols")
    
    def run(
        self, 
        strategy_fn: Callable[[Dict[str, pd.DataFrame], Portfolio, int], List[Order]],
        allocation_fn: Optional[Callable[[Dict[str, pd.DataFrame], int], Dict[str, float]]] = None,
        rebalance_period: int = 20,  # e.g., rebalance every 20 bars
    ) -> Tuple[PerformanceMetrics, Dict[str, Any]]:
        """
        Run the multi-asset backtest.
        
        Args:
            strategy_fn: A function that takes (data, portfolio, current_idx) and returns a list of orders
            allocation_fn: Optional function that takes (data, current_idx) and returns target allocations
            rebalance_period: Number of bars between rebalancing
            
        Returns:
            Tuple of (performance metrics, additional results)
        """
        logger.info("Starting multi-asset backtest")
        start_time = time.time()
        
        # Initialize
        self.portfolio_history = []
        self.trade_history = []
        
        # Get common date range
        self.align_data_dates()
        
        # Main backtest loop
        for bar_idx in range(len(self.common_dates)):
            current_date = self.common_dates[bar_idx]
            
            # Update portfolio with current prices
            self._update_portfolio(bar_idx)
            
            # Store portfolio snapshot
            self._store_portfolio_snapshot(current_date)
            
            # Update returns data for correlation calculation
            self._update_returns_data(bar_idx)
            
            # Calculate correlations periodically
            if bar_idx % 5 == 0 or bar_idx == len(self.common_dates) - 1:
                self._calculate_correlations()
                
                # Store correlation stats
                if self.correlation_matrix:
                    corr_df = pd.DataFrame(self.correlation_matrix)
                    corr_stats = analyze_correlations(corr_df)
                    self.correlation_stats_history.append((current_date, corr_stats))
            
            # Update current allocations
            self._update_current_allocations(bar_idx)
            
            # Store allocation history
            self.allocation_timestamps.append(current_date)
            for symbol in self.symbols:
                self.allocation_history[symbol].append(self.current_allocations.get(symbol, 0.0))
            
            # Update asset-level performance
            self._update_asset_performance(bar_idx)
            
            # Determine if rebalancing is needed
            should_rebalance = (bar_idx % rebalance_period == 0) and allocation_fn is not None
            
            # Get target allocations if rebalancing
            if should_rebalance:
                self.target_allocations = allocation_fn(self.data, bar_idx)
                logger.debug(f"Target allocations at bar {bar_idx}: {self.target_allocations}")
                
                # Generate rebalance orders
                rebalance_orders = self._generate_rebalance_orders(current_date, bar_idx)
                
                # Process rebalance orders
                for order in rebalance_orders:
                    # Check risk limits
                    if self._check_risk_limits(order, bar_idx):
                        # Process order
                        trade = self._process_order(order, bar_idx)
                        if trade:
                            self.trade_history.append(trade)
            
            # Get strategy orders
            strategy_orders = strategy_fn(self.data, self.portfolio, bar_idx)
            
            # Process strategy orders
            for order in strategy_orders:
                # Check risk limits
                if self._check_risk_limits(order, bar_idx):
                    # Process order
                    trade = self._process_order(order, bar_idx)
                    if trade:
                        self.trade_history.append(trade)
        
        # Calculate performance metrics
        metrics = calculate_metrics(
            portfolio_history=self.portfolio_history,
            trade_history=self.trade_history,
            initial_capital=self.initial_capital
        )
        
        # Calculate asset-level metrics
        asset_metrics = calculate_asset_metrics(self.portfolio_history)
        
        # Calculate portfolio diversification score
        diversification_score = calculate_portfolio_diversification(self.correlation_matrix)
        
        # Prepare asset drawdown curves for visualization
        asset_drawdowns = {}
        for symbol, values in self.asset_equity_curves.items():
            if values:
                equity_series = pd.Series(values, index=self.common_dates[:len(values)])
                drawdown_series = (equity_series / equity_series.cummax() - 1)
                asset_drawdowns[symbol] = drawdown_series
        
        # Additional results
        additional_results = {
            "asset_metrics": asset_metrics,
            "correlation_matrix": pd.DataFrame(self.correlation_matrix),
            "diversification_score": diversification_score,
            "allocation_history": {symbol: values for symbol, values in self.allocation_history.items()},
            "allocation_timestamps": self.allocation_timestamps,
            "asset_drawdowns": asset_drawdowns,
            "runtime_seconds": time.time() - start_time
        }
        
        logger.info(f"Backtest completed in {additional_results['runtime_seconds']:.2f} seconds")
        logger.info(f"Total return: {metrics.total_return:.2%}")
        logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {metrics.max_drawdown:.2%}")
        logger.info(f"Diversification score: {diversification_score:.2f}")
        
        return metrics, additional_results
    
    def _update_returns_data(self, bar_idx: int) -> None:
        """Update returns data for correlation calculation."""
        if bar_idx == 0:
            return  # Skip first bar
            
        current_prices = self._get_current_prices(bar_idx)
        prev_prices = self._get_current_prices(bar_idx - 1)
        
        for symbol in self.symbols:
            if symbol in current_prices and symbol in prev_prices:
                current = current_prices[symbol]
                prev = prev_prices[symbol]
                
                if prev > 0:
                    ret = (current / prev) - 1
                    self.returns_data[symbol].append(ret)
                    
                    # Keep a fixed window of returns data
                    if len(self.returns_data[symbol]) > 60:  # 60-day window
                        self.returns_data[symbol].pop(0)
    
    def _calculate_correlations(self) -> None:
        """Calculate correlation matrix between assets."""
        # Only calculate if we have enough data
        min_length = min([len(returns) for returns in self.returns_data.values()])
        
        if min_length >= 10:  # Need at least 10 data points
            # Convert to DataFrame for correlation calculation
            returns_df = pd.DataFrame(self.returns_data)
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr().to_dict()
            
            # Update correlation matrix
            self.correlation_matrix = corr_matrix
    
    def _update_current_allocations(self, bar_idx: int) -> None:
        """Update current portfolio allocations."""
        current_prices = self._get_current_prices(bar_idx)
        total_value = self.portfolio.total_value
        
        if total_value <= 0:
            return
            
        # Calculate current allocations
        self.current_allocations = {symbol: 0.0 for symbol in self.symbols}
        
        for symbol, position in self.portfolio.positions.items():
            if position.quantity != 0 and symbol in current_prices:
                price = current_prices[symbol]
                position_value = position.quantity * price
                self.current_allocations[symbol] = position_value / total_value
    
    def _generate_rebalance_orders(self, timestamp: pd.Timestamp, bar_idx: int) -> List[Order]:
        """Generate orders to rebalance the portfolio to target allocations."""
        from src.trading_engine.models import OrderSide, OrderType
        
        rebalance_signals = self.risk_manager.generate_rebalance_signals(
            current_allocations=self.current_allocations,
            target_allocations=self.target_allocations,
            threshold=self.rebalance_threshold
        )
        
        orders = []
        current_prices = self._get_current_prices(bar_idx)
        
        for symbol, weight_diff in rebalance_signals.items():
            if symbol not in current_prices:
                continue
                
            price = current_prices[symbol]
            if price <= 0:
                continue
                
            # Calculate target position value
            target_value = self.portfolio.total_value * weight_diff
            
            # Calculate quantity to trade
            quantity = target_value / price
            
            if abs(quantity) < 1e-6 and not self.enable_fractional:
                continue  # Skip tiny orders if fractional not enabled
            
            # Determine order side
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            
            # Create rebalance order
            order = Order(
                symbol=symbol,
                quantity=abs(quantity),
                side=side,
                type=OrderType.MARKET,
                timestamp=timestamp
            )
            
            orders.append(order)
        
        return orders
    
    def _check_risk_limits(self, order: Order, bar_idx: int) -> bool:
        """Check if an order passes portfolio risk limits."""
        from src.trading_engine.models import OrderSide
        
        current_prices = self._get_current_prices(bar_idx)
        if order.symbol not in current_prices:
            return False
            
        price = current_prices[order.symbol]
        
        # Get current position quantities
        current_positions = {
            symbol: position.quantity for symbol, position in self.portfolio.positions.items()
        }
        
        # Calculate proposed position after order
        proposed_qty = current_positions.get(order.symbol, 0)
        if order.side == OrderSide.BUY:
            proposed_qty += order.quantity
        else:  # SELL
            proposed_qty -= order.quantity
        
        # Check position size limits
        size_ok = self.risk_manager.check_position_size(
            portfolio_value=self.portfolio.total_value,
            symbol=order.symbol,
            current_positions=current_positions,
            proposed_qty=proposed_qty,
            price=price
        )
        
        if not size_ok:
            return False
        
        # Check correlation limits
        existing_symbols = [s for s, q in current_positions.items() if q != 0 and s != order.symbol]
        corr_ok = self.risk_manager.check_correlation(
            symbol=order.symbol,
            existing_symbols=existing_symbols,
            correlation_matrix=self.correlation_matrix
        )
        
        return corr_ok
    
    def _update_asset_performance(self, bar_idx: int) -> None:
        """Update asset-level performance metrics."""
        if bar_idx == 0:
            return  # Skip first bar
            
        current_prices = self._get_current_prices(bar_idx)
        
        for symbol, position in self.portfolio.positions.items():
            if symbol not in current_prices:
                continue
                
            # Calculate asset value
            price = current_prices[symbol]
            position_value = position.quantity * price
            
            # Store asset equity value
            self.asset_equity_curves[symbol].append(position_value)
            
            # Calculate drawdown if we have enough data
            equity_values = self.asset_equity_curves[symbol]
            if len(equity_values) > 1:
                peak = max(equity_values)
                drawdown = (position_value / peak) - 1 if peak > 0 else 0
                self.asset_drawdown_curves[symbol].append(drawdown)
    
    def generate_report(self, output_dir: str = "./reports") -> str:
        """
        Generate a comprehensive backtest report with visualizations.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Path to the report directory
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = calculate_metrics(
            portfolio_history=self.portfolio_history,
            trade_history=self.trade_history,
            initial_capital=self.initial_capital
        )
        
        # Calculate asset-level metrics
        asset_metrics = calculate_asset_metrics(self.portfolio_history)
        
        # Prepare correlation matrix
        correlation_df = pd.DataFrame(self.correlation_matrix)
        
        # Prepare asset allocation history
        asset_allocations = {symbol: values for symbol, values in self.allocation_history.items()}
        
        # Prepare asset drawdown curves
        asset_drawdowns = {}
        for symbol, values in self.asset_equity_curves.items():
            if values:
                equity_series = pd.Series(values, index=self.common_dates[:len(values)])
                drawdown_series = (equity_series / equity_series.cummax() - 1)
                asset_drawdowns[symbol] = drawdown_series
        
        # Create performance dashboard
        create_performance_dashboard(
            metrics=metrics,
            asset_metrics=asset_metrics,
            correlation_matrix=correlation_df,
            asset_allocations=asset_allocations,
            allocation_timestamps=self.allocation_timestamps,
            asset_drawdowns=asset_drawdowns,
            output_dir=output_dir
        )
        
        # Create summary report
        self._create_summary_report(
            metrics=metrics,
            asset_metrics=asset_metrics,
            output_path=os.path.join(output_dir, "summary_report.html")
        )
        
        # Generate diversification analysis if we have enough data
        if len(asset_metrics) > 1 and not correlation_df.empty:
            # Create diversification directory
            diversification_dir = os.path.join(output_dir, "diversification")
            os.makedirs(diversification_dir, exist_ok=True)
            
            # Prepare returns data
            returns_data = {}
            for symbol, df in self.data.items():
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns
            
            # Get final portfolio weights
            portfolio_weights = {}
            for symbol, values in asset_allocations.items():
                if values:
                    portfolio_weights[symbol] = values[-1]
            
            # Prepare portfolio metrics
            portfolio_metrics = {
                'total_return': metrics.total_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown
            }
            
            # Analyze diversification benefits
            try:
                analyze_diversification_benefits(
                    returns_data=returns_data,
                    portfolio_weights=portfolio_weights,
                    portfolio_metrics=portfolio_metrics,
                    single_asset_metrics=asset_metrics,
                    correlation_matrix=correlation_df,
                    output_dir=diversification_dir
                )
                print(f"Diversification analysis saved to {diversification_dir}")
            except Exception as e:
                print(f"Error generating diversification analysis: {e}")
        
        return output_dir
    
    def _create_summary_report(self, metrics: PerformanceMetrics, asset_metrics: Dict[str, Dict[str, float]], output_path: str) -> None:
        """
        Create an HTML summary report.
        
        Args:
            metrics: Performance metrics
            asset_metrics: Asset-level metrics
            output_path: Path to save the report
        """
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Asset Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .section {{ margin-bottom: 30px; }}
                .image-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .image-container img {{ max-width: 100%; height: auto; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>Multi-Asset Backtest Report</h1>
            
            <div class="section">
                <h2>Portfolio Performance Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Return</td>
                        <td class="{self._get_color_class(metrics.total_return)}">{metrics.total_return:.2%}</td>
                    </tr>
                    <tr>
                        <td>Annualized Return</td>
                        <td class="{self._get_color_class(metrics.annualized_return)}">{metrics.annualized_return:.2%}</td>
                    </tr>
                    <tr>
                        <td>Volatility</td>
                        <td>{metrics.volatility:.2%}</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td class="{self._get_color_class(metrics.sharpe_ratio)}">{metrics.sharpe_ratio:.2f}</td>
                    </tr>
                    <tr>
                        <td>Sortino Ratio</td>
                        <td class="{self._get_color_class(metrics.sortino_ratio)}">{metrics.sortino_ratio:.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td class="negative">{metrics.max_drawdown:.2%}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown Duration</td>
                        <td>{metrics.max_drawdown_duration} days</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td>{metrics.win_rate:.2%}</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>{metrics.profit_factor:.2f}</td>
                    </tr>
                    <tr>
                        <td>Calmar Ratio</td>
                        <td class="{self._get_color_class(metrics.calmar_ratio)}">{metrics.calmar_ratio:.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Asset Performance</h2>
                <table>
                    <tr>
                        <th>Asset</th>
                        <th>Total Return</th>
                        <th>Volatility</th>
                        <th>Max Drawdown</th>
                    </tr>
                    {self._generate_asset_metrics_rows(asset_metrics)}
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="image-container">
                    <div>
                        <h3>Portfolio Performance</h3>
                        <img src="portfolio_performance.png" alt="Portfolio Performance">
                    </div>
                    <div>
                        <h3>Asset Allocation</h3>
                        <img src="asset_allocation.png" alt="Asset Allocation">
                    </div>
                </div>
                <div class="image-container">
                    <div>
                        <h3>Asset Performance Comparison</h3>
                        <img src="asset_performance.png" alt="Asset Performance">
                    </div>
                    <div>
                        <h3>Correlation Matrix</h3>
                        <img src="correlation_matrix.png" alt="Correlation Matrix">
                    </div>
                </div>
                <div class="image-container">
                    <div>
                        <h3>Correlation Network</h3>
                        <img src="correlation_network.png" alt="Correlation Network">
                    </div>
                    <div>
                        <h3>Drawdown Comparison</h3>
                        <img src="drawdown_comparison.png" alt="Drawdown Comparison">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, "w") as f:
            f.write(html)
    
    def _get_color_class(self, value: float) -> str:
        """Get CSS class based on value sign."""
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        else:
            return ""
    
    def _generate_asset_metrics_rows(self, asset_metrics: Dict[str, Dict[str, float]]) -> str:
        """Generate HTML table rows for asset metrics."""
        rows = ""
        for symbol, metrics in asset_metrics.items():
            total_return = metrics.get("total_return", 0)
            volatility = metrics.get("volatility", 0)
            max_drawdown = metrics.get("max_drawdown", 0)
            
            rows += f"""
            <tr>
                <td>{symbol}</td>
                <td class="{self._get_color_class(total_return)}">{total_return:.2%}</td>
                <td>{volatility:.2%}</td>
                <td class="negative">{max_drawdown:.2%}</td>
            </tr>
            """
        
        return rows
    
    def align_data_dates(self) -> None:
        """Align data to a common date range."""
        # Find common date range
        all_dates = []
        for symbol, df in self.data.items():
            if 'timestamp' in df.columns:
                dates = df['timestamp'].tolist()
            else:
                dates = df.index.tolist()
            all_dates.append(set(dates))
        
        if not all_dates:
            raise ValueError("No data provided")
            
        # Find intersection of all date sets
        common_dates = set.intersection(*all_dates)
        
        if not common_dates:
            raise ValueError("No common dates found across all symbols")
            
        # Sort dates
        self.common_dates = sorted(list(common_dates))
        
        logger.info(f"Aligned data to {len(self.common_dates)} common dates")
