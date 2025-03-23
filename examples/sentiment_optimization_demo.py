"""
Sentiment Strategy Parameter Optimization Demo.

This example demonstrates how to use the sentiment backtesting framework to optimize
strategy parameters for maximum performance.
"""

import asyncio
import datetime
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.backtesting.sentiment_backtester import SentimentBacktester, AdvancedSentimentStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure reports directory exists
reports_dir = Path('reports')
reports_dir.mkdir(exist_ok=True)

def run_optimization_demo():
    """Run the parameter optimization demo."""
    logger.info("Running sentiment strategy parameter optimization demo")
    
    # Define base configuration for optimization
    config = {
        'symbol': 'BTC-USD',
        'start_date': datetime.datetime.now() - datetime.timedelta(days=90),
        'end_date': datetime.datetime.now(),
        'sources': ['fear_greed'],  # Using Fear & Greed as our primary sentiment source
        'price_data_path': 'data/historical/BTC-USD_1h.csv',
        'strategy': 'AdvancedSentimentStrategy',
        'strategy_config': {
            'symbol': 'BTC-USD',
            'contrarian': False,
            'position_size': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'technical_confirmation': True
        },
        'initial_capital': 10000,
        'commission_rate': 0.001
    }
    
    # Initialize backtester with base configuration
    backtester = SentimentBacktester(config)
    
    # Define parameter grid for optimization
    # We'll optimize sentiment thresholds, trend parameters, and position sizing
    param_grid = {
        # Sentiment threshold parameters
        'sentiment_threshold_buy': [0.6, 0.65, 0.7, 0.75, 0.8],
        'sentiment_threshold_sell': [0.2, 0.25, 0.3, 0.35, 0.4],
        
        # Trend analysis parameters
        'trend_window': [7, 14, 21, 28],
        'trend_strength_threshold': [0.2, 0.3, 0.4, 0.5],
        
        # Risk management parameters
        'stop_loss_pct': [0.03, 0.05, 0.07, 0.1],
        'take_profit_pct': [0.06, 0.08, 0.1, 0.15, 0.2],
        
        # Regime adaptation settings
        'use_adaptive_thresholds': [True, False]
    }
    
    # We'll optimize in stages to make the search space more manageable
    
    # Stage 1: Optimize sentiment thresholds
    logger.info("Stage 1: Optimizing sentiment thresholds")
    stage1_grid = {
        'sentiment_threshold_buy': param_grid['sentiment_threshold_buy'],
        'sentiment_threshold_sell': param_grid['sentiment_threshold_sell']
    }
    
    stage1_results = backtester.run_parameter_optimization(
        param_grid=stage1_grid,
        metric='sharpe_ratio',
        report=True
    )
    
    # Update configuration with best parameters from stage 1
    logger.info(f"Best parameters from stage 1: {stage1_results['best_params']}")
    config['strategy_config'].update(stage1_results['best_params'])
    backtester = SentimentBacktester(config)
    
    # Stage 2: Optimize trend parameters
    logger.info("Stage 2: Optimizing trend parameters")
    stage2_grid = {
        'trend_window': param_grid['trend_window'],
        'trend_strength_threshold': param_grid['trend_strength_threshold']
    }
    
    stage2_results = backtester.run_parameter_optimization(
        param_grid=stage2_grid,
        metric='sharpe_ratio',
        report=True
    )
    
    # Update configuration with best parameters from stage 2
    logger.info(f"Best parameters from stage 2: {stage2_results['best_params']}")
    config['strategy_config'].update(stage2_results['best_params'])
    backtester = SentimentBacktester(config)
    
    # Stage 3: Optimize risk management parameters
    logger.info("Stage 3: Optimizing risk management parameters")
    stage3_grid = {
        'stop_loss_pct': param_grid['stop_loss_pct'],
        'take_profit_pct': param_grid['take_profit_pct']
    }
    
    stage3_results = backtester.run_parameter_optimization(
        param_grid=stage3_grid,
        metric='sharpe_ratio',
        report=True
    )
    
    # Update configuration with best parameters from stage 3
    logger.info(f"Best parameters from stage 3: {stage3_results['best_params']}")
    config['strategy_config'].update(stage3_results['best_params'])
    backtester = SentimentBacktester(config)
    
    # Stage 4: Final optimization with adaptive thresholds
    logger.info("Stage 4: Optimizing adaptive thresholds")
    stage4_grid = {
        'use_adaptive_thresholds': param_grid['use_adaptive_thresholds']
    }
    
    stage4_results = backtester.run_parameter_optimization(
        param_grid=stage4_grid,
        metric='sharpe_ratio',
        report=True
    )
    
    # Collect final optimized parameters
    final_params = config['strategy_config'].copy()
    final_params.update(stage4_results['best_params'])
    
    logger.info("Final optimized parameters:")
    for param, value in final_params.items():
        logger.info(f"  {param}: {value}")
    
    # Run backtest with optimized parameters
    logger.info("Running backtest with optimized parameters")
    config['strategy_config'] = final_params
    backtester = SentimentBacktester(config)
    
    # Run the backtest
    results = backtester.run_backtest()
    
    # Generate report
    report = backtester.generate_report(
        results,
        output_path=reports_dir / "optimized_sentiment_backtest_report.txt"
    )
    print("\nFINAL OPTIMIZED BACKTEST RESULTS:")
    print(report)
    
    # Visualize results
    backtester.visualize_results(
        results,
        output_path=reports_dir / "optimized_sentiment_backtest_plot.png"
    )
    
    # Save optimized parameters to file
    with open(reports_dir / "sentiment_optimized_parameters.txt", "w") as f:
        f.write("# Optimized Sentiment Strategy Parameters\n\n")
        f.write("These parameters were optimized using the sentiment backtesting framework\n")
        f.write("to maximize the Sharpe ratio on historical data.\n\n")
        f.write("```yaml\n")
        for param, value in final_params.items():
            f.write(f"{param}: {value}\n")
        f.write("```\n\n")
        f.write("## Performance Metrics\n\n")
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                f.write(f"- {metric}: {value:.4f}\n")
            else:
                f.write(f"- {metric}: {value}\n")
        
    logger.info(f"Saved optimized parameters to {reports_dir}/sentiment_optimized_parameters.txt")
    
    # Performance comparison with baseline strategies
    run_comparison_with_baseline(final_params)

def run_comparison_with_baseline(optimized_params):
    """Compare optimized strategy with baseline strategies."""
    logger.info("Running comparison with baseline strategies")
    
    # Define baseline strategies
    strategies = {
        "Basic Sentiment (Buy > 0.7, Sell < 0.3)": {
            'sentiment_threshold_buy': 0.7,
            'sentiment_threshold_sell': 0.3,
            'contrarian': False,
            'position_size': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1,
            'trend_window': 14,
            'trend_strength_threshold': 0.3,
            'technical_confirmation': True,
            'use_adaptive_thresholds': False
        },
        "Contrarian Sentiment": {
            'sentiment_threshold_buy': 0.7,
            'sentiment_threshold_sell': 0.3,
            'contrarian': True,  # Inverted signals
            'position_size': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1,
            'trend_window': 14,
            'trend_strength_threshold': 0.3,
            'technical_confirmation': True,
            'use_adaptive_thresholds': False
        },
        "Optimized Strategy": optimized_params
    }
    
    # Base configuration
    base_config = {
        'symbol': 'BTC-USD',
        'start_date': datetime.datetime.now() - datetime.timedelta(days=90),
        'end_date': datetime.datetime.now(),
        'sources': ['fear_greed'],
        'price_data_path': 'data/historical/BTC-USD_1h.csv',
        'strategy': 'AdvancedSentimentStrategy',
        'initial_capital': 10000,
        'commission_rate': 0.001
    }
    
    # Run backtest for each strategy
    results = {}
    for name, params in strategies.items():
        logger.info(f"Running backtest for: {name}")
        config = base_config.copy()
        config['strategy_config'] = params
        backtester = SentimentBacktester(config)
        results[name] = backtester.run_backtest()
        
    # Create comparison report
    comparison_file = reports_dir / "sentiment_strategy_comparison.txt"
    
    with open(comparison_file, "w") as f:
        f.write("# Sentiment Strategy Comparison\n\n")
        
        # Performance metrics table
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Basic Sentiment | Contrarian Sentiment | Optimized Strategy |\n")
        f.write("|--------|----------------|----------------------|--------------------|\n")
        
        metrics_to_show = [
            "total_return", "sharpe_ratio", "max_drawdown", "win_rate", 
            "total_trades", "winning_trades", "losing_trades"
        ]
        
        for metric in metrics_to_show:
            f.write(f"| {metric} |")
            for strategy in ["Basic Sentiment (Buy > 0.7, Sell < 0.3)", "Contrarian Sentiment", "Optimized Strategy"]:
                value = results[strategy]["metrics"].get(metric, "N/A")
                if isinstance(value, float):
                    f.write(f" {value:.4f} |")
                else:
                    f.write(f" {value} |")
            f.write("\n")
        
        f.write("\n## Strategy Parameters\n\n")
        for name, params in strategies.items():
            f.write(f"### {name}\n\n")
            f.write("```yaml\n")
            for param, value in params.items():
                f.write(f"{param}: {value}\n")
            f.write("```\n\n")
    
    logger.info(f"Saved strategy comparison to {comparison_file}")
    
    # Create equity curve comparison chart
    plt.figure(figsize=(14, 8))
    
    for name, result in results.items():
        equity_curve = pd.DataFrame(result['equity_curve'])
        plt.plot(equity_curve['timestamp'], equity_curve['equity'], label=name)
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    comparison_plot = reports_dir / "sentiment_strategy_comparison.png"
    plt.savefig(comparison_plot)
    plt.close()
    
    logger.info(f"Saved equity curve comparison to {comparison_plot}")

if __name__ == "__main__":
    # Run the optimization demo
    run_optimization_demo()