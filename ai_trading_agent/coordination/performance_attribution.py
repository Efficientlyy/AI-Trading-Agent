"""
Performance Attribution Module

This module implements detailed performance analysis and attribution for trading strategies.
It analyzes the contribution of each strategy to overall performance, identifies strengths
and weaknesses, and provides self-assessment reports to guide strategy improvements.

Key features:
1. Performance tracking by strategy, symbol, and time period
2. Contribution analysis to identify key performance drivers
3. Strategy effectiveness evaluation across different market regimes
4. Self-assessment reporting with actionable insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict

from ai_trading_agent.utils.logging import get_logger
from ai_trading_agent.agent.market_regime import MarketRegimeClassifier

# Configure logger
logger = get_logger(__name__)


class PerformanceAttributor:
    """
    Analyzes and attributes trading performance across strategies and market regimes,
    providing detailed insights into strategy effectiveness and potential improvements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Performance Attributor.
        
        Args:
            config: Configuration dictionary with parameters
                - strategies: List of strategies to analyze
                - metrics: List of performance metrics to track
                - attribution_window: Number of periods for rolling attribution
                - output_path: Path to save attribution reports
                - min_data_points: Minimum data points required for analysis
                - regime_classifier: Optional market regime classifier
                - benchmark: Optional benchmark for relative performance
        """
        self.strategies = config.get("strategies", [])
        self.metrics = config.get("metrics", ["returns", "sharpe_ratio", "max_drawdown", "win_rate"])
        self.attribution_window = config.get("attribution_window", 50)
        self.output_path = config.get("output_path", "./performance_reports")
        self.min_data_points = config.get("min_data_points", 20)
        self.benchmark = config.get("benchmark", None)
        
        # Initialize regime classifier if provided
        regime_config = config.get("regime_config", {})
        if regime_config:
            self.regime_classifier = MarketRegimeClassifier(regime_config)
        else:
            self.regime_classifier = None
        
        # Performance data storage
        self.strategy_performance = {}  # strategy -> symbol -> list of performance records
        self.combined_performance = {}  # symbol -> list of performance records
        self.attribution_results = {}   # strategy -> attribution metrics
        
        # Ensure output directory exists
        if self.output_path and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        logger.info(f"Performance Attributor initialized with {len(self.strategies)} strategies")
    
    def record_performance(self, 
                          strategy: str, 
                          symbol: str, 
                          timestamp: str,
                          metrics: Dict[str, float],
                          is_combined: bool = False) -> None:
        """
        Record performance metrics for a strategy.
        
        Args:
            strategy: Strategy name
            symbol: Trading symbol
            timestamp: Performance timestamp
            metrics: Dictionary of performance metrics
            is_combined: Whether this is for the combined strategy
        """
        # Add timestamp to metrics
        metrics["timestamp"] = timestamp
        
        # Determine market regime if available
        if self.regime_classifier is not None and "market_data" in metrics:
            try:
                regime = self.regime_classifier.classify_regime(metrics["market_data"])
                metrics["market_regime"] = regime
            except Exception as e:
                logger.error(f"Error classifying regime: {e}")
                metrics["market_regime"] = "unknown"
        
        if is_combined:
            # Record for combined performance
            if symbol not in self.combined_performance:
                self.combined_performance[symbol] = []
            self.combined_performance[symbol].append(metrics)
        else:
            # Record for individual strategy
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {}
                
            if symbol not in self.strategy_performance[strategy]:
                self.strategy_performance[strategy][symbol] = []
                
            self.strategy_performance[strategy][symbol].append(metrics)
        
        # Trim history if needed
        max_history = self.attribution_window * 2
        if is_combined and symbol in self.combined_performance:
            if len(self.combined_performance[symbol]) > max_history:
                self.combined_performance[symbol] = self.combined_performance[symbol][-max_history:]
        elif not is_combined and strategy in self.strategy_performance and symbol in self.strategy_performance[strategy]:
            if len(self.strategy_performance[strategy][symbol]) > max_history:
                self.strategy_performance[strategy][symbol] = self.strategy_performance[strategy][symbol][-max_history:]
    
    def analyze_contributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the contribution of each strategy to overall performance.
        
        Returns:
            Dictionary with contribution analysis results
        """
        if not self.strategy_performance or not self.combined_performance:
            logger.warning("Not enough data for contribution analysis")
            return {}
        
        results = {}
        
        # Get common symbols (those with both strategy and combined performance)
        common_symbols = set()
        for strategy, symbols in self.strategy_performance.items():
            for symbol in symbols:
                if symbol in self.combined_performance:
                    if (len(self.strategy_performance[strategy][symbol]) >= self.min_data_points and
                        len(self.combined_performance[symbol]) >= self.min_data_points):
                        common_symbols.add(symbol)
        
        if not common_symbols:
            logger.warning("No common symbols with sufficient data for attribution analysis")
            return {}
        
        # Analyze each strategy's contribution
        for strategy in self.strategies:
            if strategy not in self.strategy_performance:
                continue
                
            strategy_results = {
                "overall_contribution": {},
                "symbol_contribution": {},
                "regime_contribution": {},
                "strengths_weaknesses": {},
                "improvement_areas": []
            }
            
            # Overall contribution analysis
            total_returns = 0
            total_contribution = 0
            total_trades = 0
            
            for symbol in common_symbols:
                if symbol not in self.strategy_performance[strategy]:
                    continue
                    
                # Get matched timestamps for this symbol
                strategy_data = self.strategy_performance[strategy][symbol]
                combined_data = self.combined_performance[symbol]
                
                # Match records by timestamp
                matched_records = self._match_performance_records(strategy_data, combined_data)
                
                if not matched_records:
                    continue
                
                symbol_returns = 0
                symbol_contribution = 0
                symbol_trades = 0
                
                for strategy_record, combined_record in matched_records:
                    # Calculate contribution metrics
                    strategy_return = strategy_record.get("returns", 0)
                    combined_return = combined_record.get("returns", 0)
                    
                    # Attribution calculation
                    if combined_return != 0:
                        contribution = strategy_return / combined_return
                    else:
                        contribution = 0 if strategy_return == 0 else 1
                    
                    # Accumulate metrics
                    symbol_returns += strategy_return
                    symbol_contribution += contribution
                    symbol_trades += strategy_record.get("trade_count", 0)
                    
                    # Regime-specific attribution
                    regime = strategy_record.get("market_regime", "unknown")
                    if regime not in strategy_results["regime_contribution"]:
                        strategy_results["regime_contribution"][regime] = {
                            "returns": 0,
                            "contribution": 0,
                            "records": 0
                        }
                    
                    strategy_results["regime_contribution"][regime]["returns"] += strategy_return
                    strategy_results["regime_contribution"][regime]["contribution"] += contribution
                    strategy_results["regime_contribution"][regime]["records"] += 1
                
                # Calculate average contribution for this symbol
                avg_contribution = symbol_contribution / len(matched_records) if matched_records else 0
                
                # Store symbol-specific results
                strategy_results["symbol_contribution"][symbol] = {
                    "total_returns": symbol_returns,
                    "avg_contribution": avg_contribution,
                    "total_trades": symbol_trades,
                    "records_analyzed": len(matched_records)
                }
                
                # Accumulate overall metrics
                total_returns += symbol_returns
                total_contribution += symbol_contribution
                total_trades += symbol_trades
            
            # Calculate overall average contribution
            total_records = sum(len(matched_records) for matched_records in strategy_results["symbol_contribution"].values())
            overall_contribution = total_contribution / total_records if total_records else 0
            
            # Store overall results
            strategy_results["overall_contribution"] = {
                "total_returns": total_returns,
                "avg_contribution": overall_contribution,
                "total_trades": total_trades,
                "records_analyzed": total_records
            }
            
            # Calculate regime averages
            for regime in strategy_results["regime_contribution"]:
                regime_data = strategy_results["regime_contribution"][regime]
                if regime_data["records"] > 0:
                    regime_data["avg_contribution"] = regime_data["contribution"] / regime_data["records"]
                    regime_data["avg_returns"] = regime_data["returns"] / regime_data["records"]
            
            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []
            
            # By symbol
            symbol_contributions = [(symbol, data["avg_contribution"]) 
                                   for symbol, data in strategy_results["symbol_contribution"].items()]
            symbol_contributions.sort(key=lambda x: x[1], reverse=True)
            
            # Top 3 symbols are strengths
            for symbol, contribution in symbol_contributions[:3]:
                if contribution > 0:
                    strengths.append(f"Strong performance in {symbol} (contribution: {contribution:.2f})")
            
            # Bottom 3 symbols are weaknesses
            for symbol, contribution in symbol_contributions[-3:]:
                if contribution < 0:
                    weaknesses.append(f"Underperformance in {symbol} (contribution: {contribution:.2f})")
            
            # By regime
            regime_contributions = [(regime, data["avg_contribution"]) 
                                   for regime, data in strategy_results["regime_contribution"].items()]
            regime_contributions.sort(key=lambda x: x[1], reverse=True)
            
            # Best regime is a strength
            if regime_contributions and regime_contributions[0][1] > 0:
                strengths.append(f"Excellent in {regime_contributions[0][0]} regime (contribution: {regime_contributions[0][1]:.2f})")
            
            # Worst regime is a weakness
            if regime_contributions and regime_contributions[-1][1] < 0:
                weaknesses.append(f"Struggles in {regime_contributions[-1][0]} regime (contribution: {regime_contributions[-1][1]:.2f})")
            
            strategy_results["strengths_weaknesses"] = {
                "strengths": strengths,
                "weaknesses": weaknesses
            }
            
            # Generate improvement recommendations
            improvements = []
            
            # Based on weaknesses
            for weakness in weaknesses:
                if "regime" in weakness:
                    improvements.append(f"Adapt parameters for {weakness.split(' in ')[1].split(' regime')[0]} market conditions")
                elif "performance in" in weakness:
                    symbol = weakness.split("performance in ")[1].split(" ")[0]
                    improvements.append(f"Review signal generation for {symbol}")
            
            # General improvements based on contribution
            if overall_contribution < 0.1:
                improvements.append("Consider reducing capital allocation to improve overall portfolio efficiency")
            
            if total_trades < 10 and total_records > 20:
                improvements.append("Increase trading frequency or adjust signal thresholds")
            
            strategy_results["improvement_areas"] = improvements
            
            # Store results for this strategy
            results[strategy] = strategy_results
            
        # Update attribution results
        self.attribution_results = results
        
        return results
    
    def _match_performance_records(self, 
                                  strategy_records: List[Dict[str, Any]],
                                  combined_records: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Match strategy and combined performance records by timestamp.
        
        Args:
            strategy_records: List of strategy performance records
            combined_records: List of combined performance records
            
        Returns:
            List of matched (strategy_record, combined_record) tuples
        """
        matched = []
        
        # Create lookup dict for combined records
        combined_lookup = {record["timestamp"]: record for record in combined_records}
        
        # Match strategy records to combined records
        for strategy_record in strategy_records:
            timestamp = strategy_record["timestamp"]
            if timestamp in combined_lookup:
                matched.append((strategy_record, combined_lookup[timestamp]))
        
        return matched
    
    def generate_attribution_report(self, output_format: str = "json") -> str:
        """
        Generate a complete performance attribution report.
        
        Args:
            output_format: Report format ('json', 'csv', or 'text')
            
        Returns:
            Report path or content string
        """
        # Analyze contributions if not already done
        if not self.attribution_results:
            self.analyze_contributions()
            
        if not self.attribution_results:
            logger.warning("No attribution results available for report")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attribution_report_{timestamp}"
        
        if output_format == "json":
            # Generate JSON report
            report_path = os.path.join(self.output_path, f"{filename}.json")
            with open(report_path, 'w') as f:
                json.dump(self.attribution_results, f, indent=2)
            return report_path
            
        elif output_format == "csv":
            # Generate CSV report
            report_path = os.path.join(self.output_path, f"{filename}.csv")
            
            # Convert nested dictionaries to flat DataFrame
            rows = []
            for strategy, results in self.attribution_results.items():
                # Overall metrics
                overall = results["overall_contribution"]
                row = {
                    "Strategy": strategy,
                    "Level": "Overall",
                    "Category": "All",
                    "Total Returns": overall["total_returns"],
                    "Avg Contribution": overall["avg_contribution"],
                    "Total Trades": overall["total_trades"],
                    "Records": overall["records_analyzed"]
                }
                rows.append(row)
                
                # Symbol metrics
                for symbol, data in results["symbol_contribution"].items():
                    row = {
                        "Strategy": strategy,
                        "Level": "Symbol",
                        "Category": symbol,
                        "Total Returns": data["total_returns"],
                        "Avg Contribution": data["avg_contribution"],
                        "Total Trades": data["total_trades"],
                        "Records": data["records_analyzed"]
                    }
                    rows.append(row)
                
                # Regime metrics
                for regime, data in results["regime_contribution"].items():
                    row = {
                        "Strategy": strategy,
                        "Level": "Regime",
                        "Category": regime,
                        "Total Returns": data["returns"],
                        "Avg Contribution": data.get("avg_contribution", 0),
                        "Avg Returns": data.get("avg_returns", 0),
                        "Records": data["records"]
                    }
                    rows.append(row)
            
            # Create and save DataFrame
            df = pd.DataFrame(rows)
            df.to_csv(report_path, index=False)
            return report_path
            
        else:  # text format
            # Generate text report
            report_path = os.path.join(self.output_path, f"{filename}.txt")
            
            with open(report_path, 'w') as f:
                f.write(f"Performance Attribution Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                for strategy, results in self.attribution_results.items():
                    f.write(f"Strategy: {strategy}\n")
                    f.write("-"*50 + "\n")
                    
                    # Overall contribution
                    overall = results["overall_contribution"]
                    f.write("Overall Performance:\n")
                    f.write(f"  Total Returns: {overall['total_returns']:.4f}\n")
                    f.write(f"  Average Contribution: {overall['avg_contribution']:.4f}\n")
                    f.write(f"  Total Trades: {overall['total_trades']}\n")
                    f.write(f"  Records Analyzed: {overall['records_analyzed']}\n\n")
                    
                    # Symbol contribution
                    f.write("Performance by Symbol:\n")
                    for symbol, data in results["symbol_contribution"].items():
                        f.write(f"  {symbol}:\n")
                        f.write(f"    Total Returns: {data['total_returns']:.4f}\n")
                        f.write(f"    Average Contribution: {data['avg_contribution']:.4f}\n")
                        f.write(f"    Total Trades: {data['total_trades']}\n")
                    f.write("\n")
                    
                    # Regime contribution
                    f.write("Performance by Market Regime:\n")
                    for regime, data in results["regime_contribution"].items():
                        avg_contrib = data.get("avg_contribution", 0)
                        avg_returns = data.get("avg_returns", 0)
                        f.write(f"  {regime}:\n")
                        f.write(f"    Total Returns: {data['returns']:.4f}\n")
                        f.write(f"    Average Contribution: {avg_contrib:.4f}\n")
                        f.write(f"    Average Returns: {avg_returns:.4f}\n")
                        f.write(f"    Records: {data['records']}\n")
                    f.write("\n")
                    
                    # Strengths and weaknesses
                    f.write("Strengths:\n")
                    for strength in results["strengths_weaknesses"]["strengths"]:
                        f.write(f"  - {strength}\n")
                    f.write("\n")
                    
                    f.write("Weaknesses:\n")
                    for weakness in results["strengths_weaknesses"]["weaknesses"]:
                        f.write(f"  - {weakness}\n")
                    f.write("\n")
                    
                    # Improvement areas
                    f.write("Areas for Improvement:\n")
                    for improvement in results["improvement_areas"]:
                        f.write(f"  - {improvement}\n")
                    f.write("\n")
                    
                    f.write("="*80 + "\n\n")
            
            return report_path
    
    def visualize_attribution(self, strategy: Optional[str] = None) -> str:
        """
        Generate visualizations of attribution analysis.
        
        Args:
            strategy: Specific strategy to visualize (if None, visualize all)
            
        Returns:
            Path to saved visualization file
        """
        if not self.attribution_results:
            self.analyze_contributions()
            
        if not self.attribution_results:
            logger.warning("No attribution results available for visualization")
            return ""
        
        # Determine strategies to visualize
        strategies_to_plot = [strategy] if strategy else list(self.attribution_results.keys())
        
        # Create output directory for plots if needed
        plots_dir = os.path.join(self.output_path, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(strategies_to_plot) == 1:
            # Single strategy visualization
            strategy = strategies_to_plot[0]
            if strategy not in self.attribution_results:
                logger.warning(f"No results for strategy {strategy}")
                return ""
                
            results = self.attribution_results[strategy]
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Performance Attribution Analysis - {strategy}", fontsize=16)
            
            # Plot 1: Symbol contribution
            symbols = list(results["symbol_contribution"].keys())
            contributions = [results["symbol_contribution"][s]["avg_contribution"] for s in symbols]
            returns = [results["symbol_contribution"][s]["total_returns"] for s in symbols]
            
            axs[0, 0].bar(symbols, contributions, alpha=0.7)
            axs[0, 0].set_title("Average Contribution by Symbol")
            axs[0, 0].set_ylabel("Contribution")
            axs[0, 0].set_xticklabels(symbols, rotation=45, ha="right")
            axs[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axs[0, 0].grid(axis='y', linestyle='--', alpha=0.3)
            
            # Plot 2: Regime contribution
            regimes = list(results["regime_contribution"].keys())
            regime_contrib = [results["regime_contribution"][r].get("avg_contribution", 0) for r in regimes]
            
            axs[0, 1].bar(regimes, regime_contrib, alpha=0.7, color='green')
            axs[0, 1].set_title("Average Contribution by Market Regime")
            axs[0, 1].set_ylabel("Contribution")
            axs[0, 1].set_xticklabels(regimes, rotation=45, ha="right")
            axs[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axs[0, 1].grid(axis='y', linestyle='--', alpha=0.3)
            
            # Plot 3: Returns vs Contribution
            axs[1, 0].scatter(returns, contributions, alpha=0.7)
            for i, symbol in enumerate(symbols):
                axs[1, 0].annotate(symbol, (returns[i], contributions[i]))
            axs[1, 0].set_title("Returns vs Contribution")
            axs[1, 0].set_xlabel("Total Returns")
            axs[1, 0].set_ylabel("Average Contribution")
            axs[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axs[1, 0].axvline(x=0, color='r', linestyle='-', alpha=0.3)
            axs[1, 0].grid(linestyle='--', alpha=0.3)
            
            # Plot 4: Regime returns
            regime_returns = [results["regime_contribution"][r].get("avg_returns", 0) for r in regimes]
            
            axs[1, 1].bar(regimes, regime_returns, alpha=0.7, color='purple')
            axs[1, 1].set_title("Average Returns by Market Regime")
            axs[1, 1].set_ylabel("Returns")
            axs[1, 1].set_xticklabels(regimes, rotation=45, ha="right")
            axs[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axs[1, 1].grid(axis='y', linestyle='--', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure
            plot_path = os.path.join(plots_dir, f"{strategy}_attribution_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            return plot_path
            
        else:
            # Multi-strategy comparison
            # Create figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=(15, 12))
            fig.suptitle("Multi-Strategy Performance Attribution Comparison", fontsize=16)
            
            # Plot 1: Overall contribution comparison
            overall_contrib = [self.attribution_results[s]["overall_contribution"]["avg_contribution"] 
                              for s in strategies_to_plot]
            
            axs[0].bar(strategies_to_plot, overall_contrib, alpha=0.7)
            axs[0].set_title("Overall Average Contribution by Strategy")
            axs[0].set_ylabel("Contribution")
            axs[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axs[0].grid(axis='y', linestyle='--', alpha=0.3)
            
            # Plot 2: Returns comparison
            overall_returns = [self.attribution_results[s]["overall_contribution"]["total_returns"] 
                              for s in strategies_to_plot]
            
            axs[1].bar(strategies_to_plot, overall_returns, alpha=0.7, color='green')
            axs[1].set_title("Total Returns by Strategy")
            axs[1].set_ylabel("Returns")
            axs[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axs[1].grid(axis='y', linestyle='--', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure
            plot_path = os.path.join(plots_dir, f"multi_strategy_comparison_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            return plot_path
