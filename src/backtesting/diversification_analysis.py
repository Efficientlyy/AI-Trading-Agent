"""
Diversification Analysis for Multi-Asset Backtesting.

This module provides tools for analyzing and visualizing the
diversification benefits of multi-asset portfolios, including
efficient frontier analysis, correlation impact, and risk contribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import scipy.optimize as sco

from .performance_metrics import PerformanceMetrics


def calculate_efficient_frontier(
    returns_data: Dict[str, pd.Series],
    risk_free_rate: float = 0.0,
    num_portfolios: int = 10000,
    target_returns: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate the efficient frontier for a set of assets.
    
    Args:
        returns_data: Dictionary mapping symbols to Series of returns
        risk_free_rate: Risk-free rate (annual)
        num_portfolios: Number of random portfolios to generate
        target_returns: Optional list of target returns for the efficient frontier
        
    Returns:
        Tuple of (DataFrame with portfolio allocations and metrics, Dict with optimal portfolios)
    """
    # Convert returns data to DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns_df.mean() * 252  # Annualize returns
    cov_matrix = returns_df.cov() * 252  # Annualize covariance
    
    # Number of assets
    num_assets = len(returns_data)
    
    # Generate random portfolios
    results = np.zeros((4, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Store results
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        results[2, i] = sharpe_ratio
        
        # Store position in results array
        results[3, i] = i
    
    # Convert results to DataFrame
    columns = ['volatility', 'return', 'sharpe', 'index']
    results_df = pd.DataFrame(results.T, columns=columns)
    
    # Add weights to results
    for i, symbol in enumerate(returns_data.keys()):
        results_df[symbol] = [w[i] for w in weights_record]
    
    # Find portfolio with maximum Sharpe ratio
    max_sharpe_idx = results_df['sharpe'].idxmax()
    max_sharpe_portfolio = results_df.loc[max_sharpe_idx]
    
    # Find portfolio with minimum volatility
    min_vol_idx = results_df['volatility'].idxmin()
    min_vol_portfolio = results_df.loc[min_vol_idx]
    
    # Calculate efficient frontier if target returns are provided
    efficient_frontier = None
    if target_returns is not None:
        efficient_frontier = []
        
        for target_return in target_returns:
            # Define optimization problem
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Constraints
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return}  # Target return
            )
            
            # Bounds (no short selling)
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Initial guess
            initial_weights = np.array([1.0 / num_assets] * num_assets)
            
            # Optimize
            result = sco.minimize(
                portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result['success']:
                volatility = portfolio_volatility(result['x'])
                weights = result['x']
                
                # Store result
                portfolio = {
                    'return': target_return,
                    'volatility': volatility,
                    'sharpe': (target_return - risk_free_rate) / volatility,
                    'weights': {symbol: weight for symbol, weight in zip(returns_data.keys(), weights)}
                }
                
                efficient_frontier.append(portfolio)
    
    # Prepare result
    optimal_portfolios = {
        'max_sharpe': {
            'return': max_sharpe_portfolio['return'],
            'volatility': max_sharpe_portfolio['volatility'],
            'sharpe': max_sharpe_portfolio['sharpe'],
            'weights': {symbol: max_sharpe_portfolio[symbol] for symbol in returns_data.keys()}
        },
        'min_volatility': {
            'return': min_vol_portfolio['return'],
            'volatility': min_vol_portfolio['volatility'],
            'sharpe': min_vol_portfolio['sharpe'],
            'weights': {symbol: min_vol_portfolio[symbol] for symbol in returns_data.keys()}
        },
        'efficient_frontier': efficient_frontier
    }
    
    return results_df, optimal_portfolios


def plot_efficient_frontier(
    results_df: pd.DataFrame,
    optimal_portfolios: Dict[str, Any],
    asset_returns: Dict[str, float],
    asset_volatilities: Dict[str, float],
    title: str = "Portfolio Efficient Frontier",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the efficient frontier with optimal portfolios and individual assets.
    
    Args:
        results_df: DataFrame with portfolio allocations and metrics
        optimal_portfolios: Dict with optimal portfolios
        asset_returns: Dictionary mapping symbols to annualized returns
        asset_volatilities: Dictionary mapping symbols to annualized volatilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot random portfolios
    ax.scatter(
        results_df['volatility'],
        results_df['return'],
        c=results_df['sharpe'],
        cmap='viridis',
        marker='o',
        s=10,
        alpha=0.3,
        label='Portfolios'
    )
    
    # Plot optimal portfolios
    max_sharpe = optimal_portfolios['max_sharpe']
    min_vol = optimal_portfolios['min_volatility']
    
    ax.scatter(
        max_sharpe['volatility'],
        max_sharpe['return'],
        marker='*',
        color='red',
        s=300,
        label='Maximum Sharpe Ratio'
    )
    
    ax.scatter(
        min_vol['volatility'],
        min_vol['return'],
        marker='*',
        color='green',
        s=300,
        label='Minimum Volatility'
    )
    
    # Plot efficient frontier if available
    if optimal_portfolios['efficient_frontier'] is not None:
        ef_volatilities = [p['volatility'] for p in optimal_portfolios['efficient_frontier']]
        ef_returns = [p['return'] for p in optimal_portfolios['efficient_frontier']]
        
        ax.plot(
            ef_volatilities,
            ef_returns,
            'b--',
            linewidth=3,
            label='Efficient Frontier'
        )
    
    # Plot individual assets
    for symbol, ret in asset_returns.items():
        vol = asset_volatilities[symbol]
        ax.scatter(
            vol,
            ret,
            marker='o',
            s=100,
            label=symbol
        )
        
        # Add label
        ax.annotate(
            symbol,
            (vol, ret),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10
        )
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Annualized Volatility', fontsize=12)
    ax.set_ylabel('Annualized Return', fontsize=12)
    
    # Format axes as percentage
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def calculate_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate risk contribution of each asset in a portfolio.
    
    Args:
        weights: Array of portfolio weights
        cov_matrix: Covariance matrix
        
    Returns:
        Array of risk contributions
    """
    # Calculate portfolio volatility
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate marginal risk contribution
    marginal_risk = np.dot(cov_matrix, weights)
    
    # Calculate risk contribution
    risk_contribution = np.multiply(marginal_risk, weights) / portfolio_vol
    
    return risk_contribution


def plot_risk_contributions(
    weights: Dict[str, float],
    returns_data: Dict[str, pd.Series],
    title: str = "Portfolio Risk Contributions",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot risk contributions of each asset in a portfolio.
    
    Args:
        weights: Dictionary mapping symbols to weights
        returns_data: Dictionary mapping symbols to Series of returns
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Convert returns data to DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate covariance matrix
    cov_matrix = returns_df.cov() * 252  # Annualize covariance
    
    # Convert weights to array
    symbols = list(weights.keys())
    weights_array = np.array([weights[symbol] for symbol in symbols])
    
    # Calculate risk contributions
    risk_contrib = calculate_risk_contributions(weights_array, cov_matrix.values)
    
    # Calculate total risk contribution
    total_risk_contrib = np.sum(risk_contrib)
    
    # Calculate percentage risk contribution
    risk_pct = risk_contrib / total_risk_contrib if total_risk_contrib != 0 else np.zeros_like(risk_contrib)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot weight allocation
    ax1.pie(
        weights_array,
        labels=symbols,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontsize': 10}
    )
    ax1.set_title('Weight Allocation', fontsize=12)
    
    # Plot risk contribution
    ax2.pie(
        risk_pct,
        labels=symbols,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontsize': 10}
    )
    ax2.set_title('Risk Contribution', fontsize=12)
    
    # Set title for the entire figure
    fig.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_correlation_impact(
    correlation_matrix: pd.DataFrame,
    portfolio_metrics: Dict[str, float],
    single_asset_metrics: Dict[str, Dict[str, float]],
    title: str = "Correlation Impact on Portfolio",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the impact of correlations on portfolio performance.
    
    Args:
        correlation_matrix: Correlation matrix as DataFrame
        portfolio_metrics: Dictionary with portfolio metrics
        single_asset_metrics: Dictionary mapping symbols to dictionaries of metrics
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract metrics
    symbols = list(single_asset_metrics.keys())
    returns = [metrics['total_return'] for symbol, metrics in single_asset_metrics.items()]
    volatilities = [metrics['volatility'] for symbol, metrics in single_asset_metrics.items()]
    
    # Calculate weighted average return and volatility
    weights = [1.0 / len(symbols)] * len(symbols)  # Equal weights
    weighted_return = np.sum(np.array(returns) * np.array(weights))
    
    # Calculate volatility with and without correlation
    # With correlation = actual portfolio volatility
    with_corr_vol = portfolio_metrics['volatility']
    
    # Without correlation = weighted sum of individual volatilities
    without_corr_vol = np.sqrt(np.sum(np.array(volatilities)**2 * np.array(weights)**2))
    
    # Calculate diversification benefit
    diversification_benefit = without_corr_vol - with_corr_vol
    diversification_pct = diversification_benefit / without_corr_vol if without_corr_vol != 0 else 0
    
    # Plot risk-return profile
    ax1.scatter(volatilities, returns, s=100, alpha=0.7)
    
    # Add labels to points
    for i, symbol in enumerate(symbols):
        ax1.annotate(
            symbol, 
            (volatilities[i], returns[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10
        )
    
    # Plot portfolio point
    ax1.scatter(
        with_corr_vol,
        portfolio_metrics['total_return'],
        marker='*',
        color='red',
        s=300,
        label='Portfolio (with correlation)'
    )
    
    # Plot theoretical point without correlation
    ax1.scatter(
        without_corr_vol,
        weighted_return,
        marker='*',
        color='blue',
        s=300,
        label='Portfolio (without correlation)'
    )
    
    # Connect the points to show diversification benefit
    ax1.plot(
        [with_corr_vol, without_corr_vol],
        [portfolio_metrics['total_return'], weighted_return],
        'k--',
        alpha=0.5
    )
    
    # Set title and labels
    ax1.set_title("Risk-Return Profile", fontsize=12)
    ax1.set_xlabel("Volatility", fontsize=10)
    ax1.set_ylabel("Total Return", fontsize=10)
    
    # Format axes as percentage
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend()
    
    # Plot correlation heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        ax=ax2,
        fmt='.2f',
        cbar_kws={'label': 'Correlation'}
    )
    
    # Set title
    ax2.set_title("Correlation Matrix", fontsize=12)
    
    # Set title for the entire figure
    fig.suptitle(f"{title}\nDiversification Benefit: {diversification_pct:.2%}", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def analyze_diversification_benefits(
    returns_data: Dict[str, pd.Series],
    portfolio_weights: Dict[str, float],
    portfolio_metrics: Dict[str, float],
    single_asset_metrics: Dict[str, Dict[str, float]],
    correlation_matrix: pd.DataFrame,
    output_dir: str = "./reports/diversification"
) -> None:
    """
    Analyze and visualize diversification benefits of a multi-asset portfolio.
    
    Args:
        returns_data: Dictionary mapping symbols to Series of returns
        portfolio_weights: Dictionary mapping symbols to weights
        portfolio_metrics: Dictionary with portfolio metrics
        single_asset_metrics: Dictionary mapping symbols to dictionaries of metrics
        correlation_matrix: Correlation matrix as DataFrame
        output_dir: Directory to save visualizations
        
    Returns:
        None
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate annualized returns and volatilities for efficient frontier
    asset_returns = {}
    asset_volatilities = {}
    
    for symbol, metrics in single_asset_metrics.items():
        asset_returns[symbol] = metrics['total_return']
        asset_volatilities[symbol] = metrics['volatility']
    
    # Calculate efficient frontier
    target_returns = np.linspace(
        min(asset_returns.values()) - 0.02,
        max(asset_returns.values()) + 0.02,
        20
    )
    
    results_df, optimal_portfolios = calculate_efficient_frontier(
        returns_data=returns_data,
        target_returns=target_returns
    )
    
    # Plot efficient frontier
    plot_efficient_frontier(
        results_df=results_df,
        optimal_portfolios=optimal_portfolios,
        asset_returns=asset_returns,
        asset_volatilities=asset_volatilities,
        save_path=os.path.join(output_dir, "efficient_frontier.png")
    )
    
    # Plot risk contributions
    plot_risk_contributions(
        weights=portfolio_weights,
        returns_data=returns_data,
        save_path=os.path.join(output_dir, "risk_contributions.png")
    )
    
    # Plot correlation impact
    plot_correlation_impact(
        correlation_matrix=correlation_matrix,
        portfolio_metrics=portfolio_metrics,
        single_asset_metrics=single_asset_metrics,
        save_path=os.path.join(output_dir, "correlation_impact.png")
    )
    
    print(f"Diversification analysis visualizations created in {output_dir}")
