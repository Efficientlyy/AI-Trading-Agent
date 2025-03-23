#!/usr/bin/env python
"""
Sentiment Strategy Backtest Script

This script runs a backtest of the sentiment strategy against historical data
to evaluate its performance.
"""

import asyncio
import os
import sys
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtesting.sentiment_backtester import SentimentBacktester
from src.data.sentiment_collector import SentimentCollector
from src.strategy.sentiment_strategy import SentimentStrategy
from src.strategy.enhanced_sentiment_strategy import EnhancedSentimentStrategy
from src.common.logging import setup_logging, get_logger


# Configure logging
setup_logging(level=logging.INFO)
logger = get_logger("test_script", "sentiment_backtest")


class SentimentBacktestRunner:
    """Runner for sentiment strategy backtests."""
    
    def __init__(self):
        """Initialize the backtest runner."""
        self.logger = get_logger("test_script", "sentiment_backtest")
        self.results = {}
        
        # Create output directory
        os.makedirs("test_output", exist_ok=True)
    
    async def load_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Load historical price and sentiment data.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            
        Returns:
            Dictionary with historical data
        """
        self.logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date}")
        
        # Create sentiment collector
        collector = SentimentCollector()
        await collector.initialize()
        
        # Load historical price data
        historical_prices = await collector.load_price_data(
            symbol=symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # Load historical sentiment data
        historical_sentiment = await collector.load_sentiment_data(
            symbol=symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        
        return {
            "prices": historical_prices,
            "sentiment": historical_sentiment
        }
    
    async def run_backtest(self, symbol: str, start_date: str, end_date: str, strategy_type: str = "enhanced"):
        """Run a backtest for the specified period.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            strategy_type: Type of strategy to test ("basic" or "enhanced")
        
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Running {strategy_type} sentiment strategy backtest for {symbol}")
        
        # Load historical data
        historical_data = await self.load_historical_data(symbol, start_date, end_date)
        
        # Create backtester
        backtester = SentimentBacktester()
        await backtester.initialize()
        
        # Set up backtest parameters
        params = {
            "symbol": symbol,
            "strategy_type": strategy_type,
            "initial_capital": 10000,
            "position_size": 0.1,  # 10% of capital per trade
            "commission_rate": 0.001,  # 0.1% commission
            "use_stop_loss": True,
            "stop_loss_pct": 0.05,  # 5% stop loss
            "use_take_profit": True,
            "take_profit_pct": 0.1,  # 10% take profit
        }
        
        # Run backtest
        if strategy_type == "basic":
            results = await backtester.run_basic_sentiment_backtest(
                historical_prices=historical_data["prices"],
                historical_sentiment=historical_data["sentiment"],
                **params
            )
        else:
            results = await backtester.run_enhanced_sentiment_backtest(
                historical_prices=historical_data["prices"],
                historical_sentiment=historical_data["sentiment"],
                **params
            )
        
        # Generate performance metrics
        metrics = backtester.calculate_performance_metrics(results)
        
        # Save results
        self.results[f"{strategy_type}_{symbol.replace('/', '_')}"] = {
            "parameters": params,
            "metrics": metrics,
            "trades": results["trades"]
        }
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    def visualize_results(self, backtest_id: str, results: Dict[str, Any]):
        """Visualize backtest results.
        
        Args:
            backtest_id: Identifier for the backtest
            results: Backtest results dictionary
        """
        self.logger.info(f"Generating visualizations for {backtest_id}")
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        equity_data = results["results"]["equity_curve"]
        timestamps = [entry["timestamp"] for entry in equity_data]
        equity = [entry["equity"] for entry in equity_data]
        
        axs[0].plot(timestamps, equity, label="Equity", color="blue")
        axs[0].set_title(f"Equity Curve - {backtest_id}")
        axs[0].set_ylabel("Portfolio Value (USD)")
        axs[0].grid(True)
        axs[0].legend()
        
        # Add buy/sell markers
        for trade in results["results"]["trades"]:
            entry_time = trade["entry_time"]
            exit_time = trade["exit_time"] if trade["exit_time"] else timestamps[-1]
            
            if trade["direction"] == "long":
                axs[0].scatter(entry_time, trade["entry_price"], color="green", marker="^", s=100)
                if trade["exit_price"]:
                    axs[0].scatter(exit_time, trade["exit_price"], color="red", marker="v", s=100)
            else:
                axs[0].scatter(entry_time, trade["entry_price"], color="red", marker="v", s=100)
                if trade["exit_price"]:
                    axs[0].scatter(exit_time, trade["exit_price"], color="green", marker="^", s=100)
        
        # Plot drawdown
        drawdowns = [entry["drawdown"] for entry in equity_data]
        axs[1].fill_between(timestamps, 0, drawdowns, color="red", alpha=0.3)
        axs[1].set_title("Drawdown")
        axs[1].set_ylabel("Drawdown (%)")
        axs[1].grid(True)
        
        # Plot sentiment
        sentiment_data = results["results"]["sentiment_values"]
        sent_timestamps = [entry["timestamp"] for entry in sentiment_data]
        sent_values = [entry["value"] for entry in sentiment_data]
        
        axs[2].plot(sent_timestamps, sent_values, label="Sentiment", color="purple")
        axs[2].axhline(y=0.5, color="gray", linestyle="--")
        axs[2].set_title("Sentiment Values")
        axs[2].set_ylabel("Sentiment (0-1)")
        axs[2].set_ylim(0, 1)
        axs[2].grid(True)
        axs[2].legend()
        
        # Format x-axis dates
        for ax in axs:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        output_path = f"test_output/{backtest_id}_results.png"
        plt.savefig(output_path)
        self.logger.info(f"Visualization saved to {output_path}")
    
    def generate_summary_report(self):
        """Generate a summary report of all backtest results."""
        self.logger.info("Generating summary report")
        
        # Create summary table
        summary_data = []
        
        for backtest_id, result in self.results.items():
            metrics = result["metrics"]
            
            summary_data.append({
                "Backtest ID": backtest_id,
                "Total Return (%)": metrics["total_return_pct"],
                "Annualized Return (%)": metrics["annual_return_pct"],
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Max Drawdown (%)": metrics["max_drawdown_pct"],
                "Win Rate (%)": metrics["win_rate"] * 100,
                "Profit Factor": metrics["profit_factor"],
                "Total Trades": metrics["total_trades"],
                "Winning Trades": metrics["winning_trades"],
                "Losing Trades": metrics["losing_trades"]
            })
        
        # Create DataFrame and save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = "test_output/backtest_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Save detailed results to JSON
        results_path = "test_output/backtest_detailed_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Summary report saved to {summary_path}")
        self.logger.info(f"Detailed results saved to {results_path}")
        
        return summary_df
    
    async def run_all_backtests(self):
        """Run all sentiment strategy backtests."""
        self.logger.info("Starting sentiment strategy backtests")
        
        # Define test parameters
        symbols = ["BTC/USDT", "ETH/USDT"]
        strategy_types = ["basic", "enhanced"]
        
        # Define test period (6 months)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        
        # Run backtests for each symbol and strategy type
        for symbol in symbols:
            for strategy_type in strategy_types:
                backtest_id = f"{strategy_type}_{symbol.replace('/', '_')}"
                
                try:
                    # Run backtest
                    results = await self.run_backtest(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        strategy_type=strategy_type
                    )
                    
                    # Visualize results
                    self.visualize_results(backtest_id, results)
                    
                except Exception as e:
                    self.logger.error(f"Error in backtest {backtest_id}: {str(e)}")
        
        # Generate summary report
        summary = self.generate_summary_report()
        
        return summary


async def main():
    """Run the sentiment backtest script."""
    # Display banner
    print("=" * 80)
    print("SENTIMENT STRATEGY BACKTEST".center(80))
    print("=" * 80)
    
    # Create and run backtest
    runner = SentimentBacktestRunner()
    summary = await runner.run_all_backtests()
    
    # Display summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY".center(80))
    print("=" * 80)
    print(f"\n{summary.to_string(index=False)}")
    
    print("\nDetailed results and visualizations saved to test_output/ directory")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        sys.exit(1)