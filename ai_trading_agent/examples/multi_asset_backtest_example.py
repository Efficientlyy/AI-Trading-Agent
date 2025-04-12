"""
Multi-Asset Backtesting Example

This script demonstrates how to use the multi-asset backtesting framework
to test a sentiment-based trading strategy across multiple assets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Correct the imports - separate models and enums
from ..trading_engine.models import Order, Portfolio
from ..trading_engine.enums import OrderSide, OrderType
from ..backtesting.multi_asset_backtester import MultiAssetBacktester
from ..backtesting.asset_allocation import (
    equal_weight_allocation,
    minimum_variance_allocation,
    sentiment_weighted_allocation,
    momentum_allocation
)
from ..backtesting.diversification_analysis import (
    analyze_diversification_benefits,
    plot_efficient_frontier,
    plot_risk_contributions,
    plot_correlation_impact
)


def generate_mock_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    frequency: str = 'D'
) -> Dict[str, pd.DataFrame]:
    """
    Generate mock price data for multiple assets.
    
    Args:
        symbols: List of asset symbols
        start_date: Start date for data
        end_date: End date for data
        frequency: Data frequency ('D' for daily, 'H' for hourly)
        
    Returns:
        Dictionary mapping symbols to DataFrames with OHLCV data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Generate data for each symbol
    data = {}
    
    for symbol in symbols:
        # Generate random price series with trend and volatility
        # Use different seed for each symbol to get different price patterns
        np.random.seed(hash(symbol) % 2**32)
        
        # Base parameters
        initial_price = np.random.uniform(50, 200)
        trend = np.random.uniform(-0.0001, 0.0002)
        volatility = np.random.uniform(0.005, 0.02)
        
        # Generate price series
        prices = [initial_price]
        for i in range(1, len(date_range)):
            # Add random walk with drift
            change = trend + np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        ohlcv = []
        for i, date in enumerate(date_range):
            price = prices[i]
            
            # Generate OHLC based on the price
            open_price = price * (1 + np.random.uniform(-0.005, 0.005))
            high_price = price * (1 + np.random.uniform(0, 0.01))
            low_price = price * (1 - np.random.uniform(0, 0.01))
            close_price = price
            
            # Generate volume
            volume = np.random.randint(100000, 1000000)
            
            ohlcv.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(ohlcv)
        df.set_index('timestamp', inplace=True)
        
        data[symbol] = df
    
    return data


def generate_mock_sentiment_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    frequency: str = 'D'
) -> Dict[str, pd.DataFrame]:
    """
    Generate mock sentiment data for multiple assets.
    
    Args:
        symbols: List of asset symbols
        start_date: Start date for data
        end_date: End date for data
        frequency: Data frequency ('D' for daily, 'H' for hourly)
        
    Returns:
        Dictionary mapping symbols to DataFrames with sentiment data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Generate data for each symbol
    sentiment_data = {}
    
    for symbol in symbols:
        # Use different seed for each symbol
        np.random.seed(hash(symbol + 'sentiment') % 2**32)
        
        # Generate sentiment scores
        sentiment_scores = []
        
        # Start with a random sentiment
        current_sentiment = np.random.uniform(-0.5, 0.5)
        
        for date in date_range:
            # Add some mean reversion and randomness to sentiment
            change = -0.1 * current_sentiment + np.random.normal(0, 0.1)
            current_sentiment += change
            
            # Clip to range [-1, 1]
            current_sentiment = max(-1, min(1, current_sentiment))
            
            sentiment_scores.append({
                'timestamp': date,
                'sentiment': current_sentiment
            })
        
        # Create DataFrame
        df = pd.DataFrame(sentiment_scores)
        df.set_index('timestamp', inplace=True)
        
        sentiment_data[symbol] = df
    
    return sentiment_data


def simple_sentiment_strategy(
    data: Dict[str, pd.DataFrame],
    portfolio: Portfolio,
    bar_idx: int,
    sentiment_data: Dict[str, pd.DataFrame] = None,
    sentiment_threshold: float = 0.3
) -> List[Order]:
    """
    Simple sentiment-based trading strategy.
    
    Buy when sentiment is above threshold, sell when below negative threshold.
    
    Args:
        data: Dictionary mapping symbols to DataFrames with OHLCV data
        portfolio: Current portfolio state
        bar_idx: Current bar index
        sentiment_data: Dictionary mapping symbols to DataFrames with sentiment data
        sentiment_threshold: Threshold for sentiment-based decisions
        
    Returns:
        List of orders to execute
    """
    orders = []
    
    # If no sentiment data, return empty list
    if not sentiment_data:
        return orders
    
    # Get current date
    current_date = list(data.values())[0].index[bar_idx]
    
    for symbol, df in data.items():
        # Skip if we don't have sentiment data for this symbol
        if symbol not in sentiment_data:
            continue
            
        # Get current sentiment
        sentiment_df = sentiment_data[symbol]
        if current_date not in sentiment_df.index:
            continue
            
        sentiment = sentiment_df.loc[current_date, 'sentiment']
        
        # Get current position
        current_position = portfolio.positions.get(symbol, None)
        current_quantity = current_position.quantity if current_position else 0
        
        # Get current price
        current_price = df.loc[current_date, 'close']
        
        # Calculate position size (1% of portfolio per trade)
        position_value = portfolio.total_value * 0.01
        position_size = position_value / current_price
        
        # Generate trading signals based on sentiment
        if sentiment > sentiment_threshold and current_quantity <= 0:
            # Buy signal
            order = Order(
                symbol=symbol,
                quantity=position_size,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                timestamp=current_date
            )
            orders.append(order)
        elif sentiment < -sentiment_threshold and current_quantity > 0:
            # Sell signal
            order = Order(
                symbol=symbol,
                quantity=current_quantity,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                timestamp=current_date
            )
            orders.append(order)
    
    return orders


def prepare_returns_data(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Prepare returns data for diversification analysis.
    
    Args:
        price_data: Dictionary mapping symbols to DataFrames with OHLCV data
        
    Returns:
        Dictionary mapping symbols to Series of returns
    """
    returns_data = {}
    
    for symbol, df in price_data.items():
        # Calculate daily returns
        returns = df['close'].pct_change().dropna()
        returns_data[symbol] = returns
    
    return returns_data


def run_multi_asset_backtest():
    """Run a multi-asset backtest with different allocation strategies."""
    # Define parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    initial_capital = 100000.0
    
    # Generate mock data
    print("Generating mock price data...")
    price_data = generate_mock_data(symbols, start_date, end_date)
    
    print("Generating mock sentiment data...")
    sentiment_data = generate_mock_sentiment_data(symbols, start_date, end_date)
    
    # Create sentiment scores dictionary for allocation
    sentiment_scores = {}
    for symbol, df in sentiment_data.items():
        # Use the most recent sentiment score
        sentiment_scores[symbol] = df['sentiment'].iloc[-1]
    
    # Define allocation strategies to test
    allocation_strategies = {
        'Equal Weight': equal_weight_allocation,
        'Minimum Variance': lambda data, bar_idx: minimum_variance_allocation(data, bar_idx, lookback_period=60),
        'Momentum': lambda data, bar_idx: momentum_allocation(data, bar_idx, lookback_period=60, top_n=5),
        'Sentiment Weighted': lambda data, bar_idx: sentiment_weighted_allocation(
            data, bar_idx, sentiment_scores, equal_weight_allocation, sentiment_weight=0.5
        )
    }
    
    # Run backtest for each allocation strategy
    results = {}
    
    for strategy_name, allocation_fn in allocation_strategies.items():
        print(f"\nRunning backtest with {strategy_name} allocation strategy...")
        
        # Initialize backtester
        backtester = MultiAssetBacktester(
            data=price_data,
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage=0.001,
            enable_fractional=True,
            max_position_pct=0.3,
            max_correlation=0.7,
            rebalance_threshold=0.05
        )
        
        # Run backtest
        metrics, additional_results = backtester.run(
            strategy_fn=lambda data, portfolio, bar_idx: simple_sentiment_strategy(
                data, portfolio, bar_idx, sentiment_data, sentiment_threshold=0.3
            ),
            allocation_fn=allocation_fn,
            rebalance_period=20  # Rebalance every 20 trading days (roughly monthly)
        )
        
        # Store results
        results[strategy_name] = {
            'metrics': metrics,
            'additional_results': additional_results
        }
        
        # Generate report
        report_dir = f"./reports/{strategy_name.replace(' ', '_').lower()}"
        backtester.generate_report(output_dir=report_dir)
        print(f"Report generated in {report_dir}")
        
        # Print key metrics
        print(f"  Total Return: {metrics.total_return:.2%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  Diversification Score: {additional_results['diversification_score']:.2f}")
        
        # Perform diversification analysis
        if strategy_name == 'Equal Weight':  # Only do this for one strategy to save time
            print("\nPerforming diversification analysis...")
            
            # Prepare returns data
            returns_data = prepare_returns_data(price_data)
            
            # Get final portfolio weights
            portfolio_weights = {}
            for symbol, values in additional_results['allocation_history'].items():
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
            analyze_diversification_benefits(
                returns_data=returns_data,
                portfolio_weights=portfolio_weights,
                portfolio_metrics=portfolio_metrics,
                single_asset_metrics=additional_results['asset_metrics'],
                correlation_matrix=additional_results['correlation_matrix'],
                output_dir=f"{report_dir}/diversification"
            )
            
            print(f"Diversification analysis completed and saved to {report_dir}/diversification")
    
    # Compare strategies
    compare_strategies(results)


def compare_strategies(results: Dict[str, Dict[str, Any]]):
    """
    Compare performance of different allocation strategies.
    
    Args:
        results: Dictionary mapping strategy names to results
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract metrics for comparison
    strategy_names = []
    total_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    diversification_scores = []
    
    for strategy_name, result in results.items():
        metrics = result['metrics']
        additional_results = result['additional_results']
        
        strategy_names.append(strategy_name)
        total_returns.append(metrics.total_return)
        sharpe_ratios.append(metrics.sharpe_ratio)
        max_drawdowns.append(metrics.max_drawdown)
        diversification_scores.append(additional_results['diversification_score'])
    
    # Plot returns and Sharpe ratios
    x = np.arange(len(strategy_names))
    width = 0.35
    
    ax1.bar(x - width/2, total_returns, width, label='Total Return', color='green', alpha=0.7)
    ax1.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', color='blue', alpha=0.7)
    
    ax1.set_title('Returns and Sharpe Ratios by Strategy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_names, rotation=45)
    ax1.legend()
    
    # Format y-axis for returns
    ax1.set_ylabel('Value')
    
    # Plot drawdowns and diversification
    ax2.bar(x - width/2, [abs(d) for d in max_drawdowns], width, label='Max Drawdown', color='red', alpha=0.7)
    ax2.bar(x + width/2, diversification_scores, width, label='Diversification Score', color='purple', alpha=0.7)
    
    ax2.set_title('Risk Metrics by Strategy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_names, rotation=45)
    ax2.legend()
    
    # Format y-axis for drawdowns
    ax2.set_ylabel('Value')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs('./reports', exist_ok=True)
    plt.savefig('./reports/strategy_comparison.png', dpi=300, bbox_inches='tight')
    
    print("\nStrategy comparison chart saved to './reports/strategy_comparison.png'")
    
    # Create a more detailed comparison table
    create_comparison_table(results, './reports/strategy_comparison.html')


def create_comparison_table(results: Dict[str, Dict[str, Any]], output_path: str):
    """
    Create an HTML comparison table for different strategies.
    
    Args:
        results: Dictionary mapping strategy names to results
        output_path: Path to save the HTML file
    """
    # Create HTML table
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategy Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .positive { color: green; }
            .negative { color: red; }
        </style>
    </head>
    <body>
        <h1>Multi-Asset Strategy Comparison</h1>
        
        <table>
            <tr>
                <th>Metric</th>
    """
    
    # Add strategy names to header
    for strategy_name in results.keys():
        html += f"<th>{strategy_name}</th>\n"
    
    html += "</tr>\n"
    
    # Add metrics rows
    metrics_to_display = [
        ('Total Return', lambda r: r['metrics'].total_return, '{:.2%}', True),
        ('Annualized Return', lambda r: r['metrics'].annualized_return, '{:.2%}', True),
        ('Volatility', lambda r: r['metrics'].volatility, '{:.2%}', False),
        ('Sharpe Ratio', lambda r: r['metrics'].sharpe_ratio, '{:.2f}', True),
        ('Sortino Ratio', lambda r: r['metrics'].sortino_ratio, '{:.2f}', True),
        ('Max Drawdown', lambda r: r['metrics'].max_drawdown, '{:.2%}', False),
        ('Win Rate', lambda r: r['metrics'].win_rate, '{:.2%}', True),
        ('Profit Factor', lambda r: r['metrics'].profit_factor, '{:.2f}', True),
        ('Diversification Score', lambda r: r['additional_results']['diversification_score'], '{:.2f}', True),
        ('Avg Exposure', lambda r: r['metrics'].avg_exposure, '{:.2%}', False),
        ('Time in Market', lambda r: r['metrics'].time_in_market, '{:.2%}', False),
        ('Total Trades', lambda r: r['metrics'].total_trades, '{:d}', False),
    ]
    
    for metric_name, metric_fn, format_str, is_higher_better in metrics_to_display:
        html += f"<tr>\n<td>{metric_name}</td>\n"
        
        for strategy_name, result in results.items():
            value = metric_fn(result)
            
            # Determine CSS class based on whether higher is better
            css_class = ""
            if is_higher_better and value > 0:
                css_class = "positive"
            elif not is_higher_better and value < 0:
                css_class = "negative"
            
            # Format value
            formatted_value = format_str.format(value)
            
            html += f"<td class='{css_class}'>{formatted_value}</td>\n"
        
        html += "</tr>\n"
    
    html += """
        </table>
        
        <h2>Visualizations</h2>
        <p>See the individual strategy reports for detailed visualizations.</p>
        <img src="strategy_comparison.png" alt="Strategy Comparison" style="max-width: 100%;">
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"Detailed strategy comparison saved to {output_path}")


if __name__ == "__main__":
    run_multi_asset_backtest()
