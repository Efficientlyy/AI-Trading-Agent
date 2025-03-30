"""
Performance Tracking Module

This module provides functionality for tracking and analyzing trading performance.
It calculates key performance metrics such as returns, win rates, and drawdowns.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = Path("data")
PERFORMANCE_DIR = DATA_DIR / "performance"
TRADES_DIR = DATA_DIR / "trades"

# Ensure directories exist
PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)


class PerformanceTracker:
    """
    Tracks and analyzes trading performance.
    
    This class provides methods to calculate and retrieve performance metrics
    such as returns, win rates, drawdowns, and other key indicators.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the performance tracker.
        
        Args:
            data_dir: Optional custom data directory
        """
        self.data_dir = data_dir or PERFORMANCE_DIR
        self.trades_dir = data_dir / "trades" if data_dir is not None else TRADES_DIR
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for performance data
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes default cache duration
    
    def get_performance_summary(self, period: str = "daily") -> Dict[str, Any]:
        """
        Get a summary of performance metrics for a specific time period.
        
        Args:
            period: Time period for the summary (daily, weekly, monthly, quarterly, yearly, all_time)
            
        Returns:
            Dictionary containing performance summary metrics
        """
        # Check cache first
        cache_key = f"summary_{period}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Load trades data
            trades = self._load_trades(period)
            
            # If no trades, return empty summary
            if not trades:
                summary = self._create_empty_summary()
                self._cache_result(cache_key, summary)
                return summary
            
            # Calculate summary metrics
            summary = self._calculate_summary_metrics(trades, period)
            
            # Cache the result
            self._cache_result(cache_key, summary)
            
            return summary
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            # Return empty summary on error
            return self._create_empty_summary()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics.
        
        Returns:
            Dictionary containing detailed performance metrics
        """
        # Check cache first
        cache_key = "detailed_metrics"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Load all trades
            trades = self._load_trades("all_time")
            
            # If no trades, return empty metrics
            if not trades:
                metrics = self._create_empty_metrics()
                self._cache_result(cache_key, metrics)
                return metrics
            
            # Calculate detailed metrics
            metrics = self._calculate_detailed_metrics(trades)
            
            # Cache the result
            self._cache_result(cache_key, metrics)
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating detailed performance metrics: {e}")
            # Return empty metrics on error
            return self._create_empty_metrics()
    
    def record_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Record a new trade.
        
        Args:
            trade_data: Dictionary containing trade data
            
        Returns:
            True if the trade was recorded successfully
        """
        try:
            # Ensure required fields are present
            required_fields = ["asset", "side", "quantity", "entry_price", "exit_price", 
                              "entry_time", "exit_time"]
            
            for field in required_fields:
                if field not in trade_data:
                    logger.error(f"Missing required field in trade data: {field}")
                    return False
            
            # Add timestamp if not present
            if "timestamp" not in trade_data:
                trade_data["timestamp"] = datetime.now().isoformat()
            
            # Calculate profit/loss if not provided
            if "profit_loss" not in trade_data:
                entry_price = float(trade_data["entry_price"])
                exit_price = float(trade_data["exit_price"])
                quantity = float(trade_data["quantity"])
                
                if trade_data["side"].lower() == "buy":
                    profit_loss = (exit_price - entry_price) * quantity
                else:  # sell
                    profit_loss = (entry_price - exit_price) * quantity
                
                trade_data["profit_loss"] = profit_loss
                
                # Calculate percentage profit/loss
                if entry_price > 0:
                    if trade_data["side"].lower() == "buy":
                        profit_loss_pct = ((exit_price / entry_price) - 1) * 100
                    else:  # sell
                        profit_loss_pct = ((entry_price / exit_price) - 1) * 100
                    
                    trade_data["profit_loss_pct"] = profit_loss_pct
            
            # Generate trade ID if not provided
            if "id" not in trade_data:
                trade_data["id"] = f"T-{int(time.time())}-{hash(str(trade_data)) % 10000:04d}"
            
            # Save trade to file
            trade_id = trade_data["id"]
            trade_file = self.trades_dir / f"{trade_id}.json"
            
            with open(trade_file, "w") as f:
                json.dump(trade_data, f, indent=2)
            
            # Clear cache to force recalculation of metrics
            self._clear_cache()
            
            logger.info(f"Trade recorded: {trade_id}")
            return True
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
    
    def get_trades(self, 
                  period: str = "all_time", 
                  limit: int = 100, 
                  offset: int = 0,
                  sort_by: str = "exit_time",
                  sort_order: str = "desc") -> List[Dict[str, Any]]:
        """
        Get a list of trades for a specific time period.
        
        Args:
            period: Time period (daily, weekly, monthly, quarterly, yearly, all_time)
            limit: Maximum number of trades to return
            offset: Number of trades to skip
            sort_by: Field to sort by
            sort_order: Sort order (asc or desc)
            
        Returns:
            List of trade dictionaries
        """
        try:
            # Load trades for the period
            trades = self._load_trades(period)
            
            # Sort trades
            reverse = sort_order.lower() == "desc"
            trades.sort(key=lambda t: t.get(sort_by, ""), reverse=reverse)
            
            # Apply limit and offset
            trades = trades[offset:offset+limit]
            
            return trades
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def get_equity_curve(self, period: str = "all_time") -> Dict[str, List[Any]]:
        """
        Get equity curve data for a specific time period.
        
        Args:
            period: Time period (daily, weekly, monthly, quarterly, yearly, all_time)
            
        Returns:
            Dictionary with dates and equity values
        """
        # Check cache first
        cache_key = f"equity_curve_{period}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Load trades for the period
            trades = self._load_trades(period)
            
            # If no trades, return empty equity curve
            if not trades:
                curve = {"dates": [], "equity": []}
                self._cache_result(cache_key, curve)
                return curve
            
            # Sort trades by exit time
            trades.sort(key=lambda t: t.get("exit_time", ""))
            
            # Calculate equity curve
            initial_equity = 100000  # Starting equity
            dates = []
            equity = [initial_equity]
            current_equity = initial_equity
            
            for trade in trades:
                # Get exit time and profit/loss
                exit_time = trade.get("exit_time")
                profit_loss = trade.get("profit_loss", 0)
                
                if exit_time and profit_loss is not None:
                    # Add to equity curve
                    current_equity += profit_loss
                    dates.append(exit_time)
                    equity.append(current_equity)
            
            # Create result
            curve = {
                "dates": dates,
                "equity": equity
            }
            
            # Cache the result
            self._cache_result(cache_key, curve)
            
            return curve
        except Exception as e:
            logger.error(f"Error calculating equity curve: {e}")
            return {"dates": [], "equity": []}
    
    def get_drawdowns(self, period: str = "all_time") -> Dict[str, Any]:
        """
        Get drawdown analysis for a specific time period.
        
        Args:
            period: Time period (daily, weekly, monthly, quarterly, yearly, all_time)
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Check cache first
        cache_key = f"drawdowns_{period}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Get equity curve
            curve = self.get_equity_curve(period)
            
            # If no equity data, return empty drawdowns
            if not curve["equity"]:
                drawdowns = {
                    "max_drawdown": 0,
                    "max_drawdown_pct": 0,
                    "current_drawdown": 0,
                    "current_drawdown_pct": 0,
                    "drawdown_periods": []
                }
                self._cache_result(cache_key, drawdowns)
                return drawdowns
            
            # Calculate drawdowns
            equity = curve["equity"]
            dates = curve["dates"]
            
            # Initialize variables
            peak = equity[0]
            max_drawdown = 0
            max_drawdown_pct = 0
            drawdown_periods = []
            current_peak = peak
            current_valley = peak
            
            for i, value in enumerate(equity):
                # Update peak
                if value > peak:
                    peak = value
                
                # Calculate drawdown
                drawdown = peak - value
                drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
                
                # Update max drawdown
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_pct = drawdown_pct
                
                # Track drawdown periods
                if value > current_peak:
                    # New peak, end any current drawdown period
                    if current_valley < current_peak:
                        # We had a drawdown period
                        dd_pct = ((current_peak - current_valley) / current_peak) * 100
                        if dd_pct >= 5:  # Only track significant drawdowns (>= 5%)
                            drawdown_periods.append({
                                "start_date": dates[i-1] if i > 0 else dates[0],
                                "end_date": dates[i] if i < len(dates) else dates[-1],
                                "peak": current_peak,
                                "valley": current_valley,
                                "drawdown": current_peak - current_valley,
                                "drawdown_pct": dd_pct
                            })
                    
                    # Reset peak and valley
                    current_peak = value
                    current_valley = value
                elif value < current_valley:
                    # New valley
                    current_valley = value
            
            # Calculate current drawdown
            current_drawdown = peak - equity[-1]
            current_drawdown_pct = (current_drawdown / peak) * 100 if peak > 0 else 0
            
            # Create result
            drawdowns = {
                "max_drawdown": max_drawdown,
                "max_drawdown_pct": max_drawdown_pct,
                "current_drawdown": current_drawdown,
                "current_drawdown_pct": current_drawdown_pct,
                "drawdown_periods": drawdown_periods
            }
            
            # Cache the result
            self._cache_result(cache_key, drawdowns)
            
            return drawdowns
        except Exception as e:
            logger.error(f"Error calculating drawdowns: {e}")
            return {
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "current_drawdown": 0,
                "current_drawdown_pct": 0,
                "drawdown_periods": []
            }
    
    def get_strategy_performance(self) -> List[Dict[str, Any]]:
        """
        Get performance metrics broken down by strategy.
        
        Returns:
            List of strategy performance dictionaries
        """
        # Check cache first
        cache_key = "strategy_performance"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Load all trades
            trades = self._load_trades("all_time")
            
            # If no trades, return empty list
            if not trades:
                self._cache_result(cache_key, [])
                return []
            
            # Group trades by strategy
            strategy_trades = {}
            for trade in trades:
                strategy = trade.get("strategy", "Unknown")
                if strategy not in strategy_trades:
                    strategy_trades[strategy] = []
                strategy_trades[strategy].append(trade)
            
            # Calculate metrics for each strategy
            strategy_performance = []
            for strategy, strategy_trades_list in strategy_trades.items():
                # Skip strategies with no trades
                if not strategy_trades_list:
                    continue
                
                # Calculate metrics
                total_trades = len(strategy_trades_list)
                winning_trades = sum(1 for t in strategy_trades_list if t.get("profit_loss", 0) > 0)
                losing_trades = sum(1 for t in strategy_trades_list if t.get("profit_loss", 0) < 0)
                
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                total_profit = sum(t.get("profit_loss", 0) for t in strategy_trades_list if t.get("profit_loss", 0) > 0)
                total_loss = sum(t.get("profit_loss", 0) for t in strategy_trades_list if t.get("profit_loss", 0) < 0)
                
                profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
                
                avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
                avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
                
                # Calculate expectancy
                expectancy = (win_rate / 100 * avg_profit) + ((1 - win_rate / 100) * avg_loss)
                
                # Add to results
                strategy_performance.append({
                    "strategy": strategy,
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "expectancy": expectancy,
                    "total_profit": total_profit,
                    "total_loss": total_loss,
                    "avg_profit": avg_profit,
                    "avg_loss": avg_loss
                })
            
            # Sort by total profit
            strategy_performance.sort(key=lambda s: s["total_profit"], reverse=True)
            
            # Cache the result
            self._cache_result(cache_key, strategy_performance)
            
            return strategy_performance
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {e}")
            return []
    
    def get_asset_performance(self) -> List[Dict[str, Any]]:
        """
        Get performance metrics broken down by asset.
        
        Returns:
            List of asset performance dictionaries
        """
        # Check cache first
        cache_key = "asset_performance"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Load all trades
            trades = self._load_trades("all_time")
            
            # If no trades, return empty list
            if not trades:
                self._cache_result(cache_key, [])
                return []
            
            # Group trades by asset
            asset_trades = {}
            for trade in trades:
                asset = trade.get("asset", "Unknown")
                if asset not in asset_trades:
                    asset_trades[asset] = []
                asset_trades[asset].append(trade)
            
            # Calculate metrics for each asset
            asset_performance = []
            for asset, asset_trades_list in asset_trades.items():
                # Skip assets with no trades
                if not asset_trades_list:
                    continue
                
                # Calculate metrics
                total_trades = len(asset_trades_list)
                winning_trades = sum(1 for t in asset_trades_list if t.get("profit_loss", 0) > 0)
                losing_trades = sum(1 for t in asset_trades_list if t.get("profit_loss", 0) < 0)
                
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                total_profit = sum(t.get("profit_loss", 0) for t in asset_trades_list if t.get("profit_loss", 0) > 0)
                total_loss = sum(t.get("profit_loss", 0) for t in asset_trades_list if t.get("profit_loss", 0) < 0)
                
                profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
                
                avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
                avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
                
                # Add to results
                asset_performance.append({
                    "asset": asset,
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "total_profit": total_profit,
                    "total_loss": total_loss,
                    "avg_profit": avg_profit,
                    "avg_loss": avg_loss
                })
            
            # Sort by total profit
            asset_performance.sort(key=lambda a: a["total_profit"], reverse=True)
            
            # Cache the result
            self._cache_result(cache_key, asset_performance)
            
            return asset_performance
        except Exception as e:
            logger.error(f"Error calculating asset performance: {e}")
            return []
    
    def _load_trades(self, period: str = "all_time") -> List[Dict[str, Any]]:
        """
        Load trades for a specific time period.
        
        Args:
            period: Time period (daily, weekly, monthly, quarterly, yearly, all_time)
            
        Returns:
            List of trade dictionaries
        """
        # Get start date based on period
        start_date = self._get_period_start_date(period)
        
        # Get all trade files
        trade_files = list(self.trades_dir.glob("*.json"))
        
        # Load trades
        trades = []
        for file_path in trade_files:
            try:
                with open(file_path, "r") as f:
                    trade = json.load(f)
                
                # Filter by period if needed
                if period != "all_time":
                    # Get exit time
                    exit_time_str = trade.get("exit_time")
                    if not exit_time_str:
                        continue
                    
                    # Parse exit time
                    try:
                        exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
                        # Convert to naive datetime for comparison
                        exit_time = exit_time.replace(tzinfo=None)
                    except ValueError:
                        # Try different format
                        exit_time = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
                    
                    # Skip trades before start date
                    if exit_time < start_date:
                        continue
                
                trades.append(trade)
            except Exception as e:
                logger.error(f"Error loading trade file {file_path}: {e}")
        
        return trades
    
    def _get_period_start_date(self, period: str) -> datetime:
        """
        Get the start date for a specific time period.
        
        Args:
            period: Time period (daily, weekly, monthly, quarterly, yearly, all_time)
            
        Returns:
            Start date as datetime
        """
        now = datetime.now()
        
        if period == "daily":
            return datetime(now.year, now.month, now.day)
        elif period == "weekly":
            # Start of current week (Monday)
            return now - timedelta(days=now.weekday())
        elif period == "monthly":
            # Start of current month
            return datetime(now.year, now.month, 1)
        elif period == "quarterly":
            # Start of current quarter
            quarter = (now.month - 1) // 3
            return datetime(now.year, quarter * 3 + 1, 1)
        elif period == "yearly":
            # Start of current year
            return datetime(now.year, 1, 1)
        else:  # all_time
            return datetime(1970, 1, 1)  # Unix epoch
    
    def _calculate_summary_metrics(self, trades: List[Dict[str, Any]], period: str) -> Dict[str, Any]:
        """
        Calculate summary metrics from a list of trades.
        
        Args:
            trades: List of trade dictionaries
            period: Time period
            
        Returns:
            Dictionary with summary metrics
        """
        # Count trades
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get("profit_loss", 0) > 0)
        losing_trades = sum(1 for t in trades if t.get("profit_loss", 0) < 0)
        
        # Calculate win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate profit/loss
        total_profit = sum(t.get("profit_loss", 0) for t in trades if t.get("profit_loss", 0) > 0)
        total_loss = sum(t.get("profit_loss", 0) for t in trades if t.get("profit_loss", 0) < 0)
        net_profit = total_profit + total_loss
        
        # Calculate profit factor
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate average profit/loss
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        # Calculate expectancy
        expectancy = (win_rate / 100 * avg_profit) + ((1 - win_rate / 100) * avg_loss)
        
        # Get drawdown data
        drawdowns = self.get_drawdowns(period)
        
        # Create summary
        summary = {
            "period": period,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "profit_factor": profit_factor,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "max_drawdown": drawdowns["max_drawdown"],
            "max_drawdown_pct": drawdowns["max_drawdown_pct"],
            "current_drawdown": drawdowns["current_drawdown"],
            "current_drawdown_pct": drawdowns["current_drawdown_pct"]
        }
        
        return summary
    
    def _calculate_detailed_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate detailed metrics from a list of trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with detailed metrics
        """
        # Get strategy and asset performance
        strategy_performance = self.get_strategy_performance()
        asset_performance = self.get_asset_performance()
        
        # Get recent trades (last 20)
        trades_sorted = sorted(trades, key=lambda t: t.get("exit_time", ""), reverse=True)
        recent_trades = trades_sorted[:20]
        
        # Get equity curve
        equity_curve = self.get_equity_curve("all_time")
        
        # Create detailed metrics
        metrics = {
            "strategy_performance": strategy_performance,
            "asset_performance": asset_performance,
            "recent_trades": recent_trades,
            "equity_curve": equity_curve,
            "trade_analytics": self._calculate_trade_analytics(trades)
        }
        
        return metrics
    
    def _calculate_trade_analytics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate advanced trade analytics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with trade analytics
        """
        # Skip if no trades
        if not trades:
            return {}
        
        # Calculate trade distribution by time of day
        time_of_day = {
            "00:00-04:00": 0,
            "04:00-08:00": 0,
            "08:00-12:00": 0,
            "12:00-16:00": 0,
            "16:00-20:00": 0,
            "20:00-24:00": 0
        }
        
        # Calculate trade distribution by day of week
        day_of_week = {
            "Monday": 0,
            "Tuesday": 0,
            "Wednesday": 0,
            "Thursday": 0,
            "Friday": 0,
            "Saturday": 0,
            "Sunday": 0
        }
        
        # Calculate trade distribution by size
        trade_size = {
            "Small (< $1K)": 0,
            "Medium ($1K-$5K)": 0,
            "Large ($5K-$20K)": 0,
            "Very Large (> $20K)": 0
        }
        
        # Calculate trade distribution by holding time
        holding_time = {
            "< 1 hour": 0,
            "1-6 hours": 0,
            "6-24 hours": 0,
            "1-3 days": 0,
            "> 3 days": 0
        }
        
        # Process each trade
        for trade in trades:
            # Get exit time
            exit_time_str = trade.get("exit_time")
            if not exit_time_str:
                continue
            
            # Parse exit time
            try:
                exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
                # Convert to naive datetime for comparison
                exit_time = exit_time.replace(tzinfo=None)
            except ValueError:
                try:
                    # Try different format
                    exit_time = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # Skip if can't parse
                    continue
            
            # Get entry time
            entry_time_str = trade.get("entry_time")
            if not entry_time_str:
                continue
            
            # Parse entry time
            try:
                entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                # Convert to naive datetime for comparison
                entry_time = entry_time.replace(tzinfo=None)
            except ValueError:
                try:
                    # Try different format
                    entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # Skip if can't parse
                    continue
            
            # Calculate time of day
            hour = exit_time.hour
            if 0 <= hour < 4:
                time_of_day["00:00-04:00"] += 1
            elif 4 <= hour < 8:
                time_of_day["04:00-08:00"] += 1
            elif 8 <= hour < 12:
                time_of_day["08:00-12:00"] += 1
            elif 12 <= hour < 16:
                time_of_day["12:00-16:00"] += 1
            elif 16 <= hour < 20:
                time_of_day["16:00-20:00"] += 1
            else:
                time_of_day["20:00-24:00"] += 1
            
            # Calculate day of week
            day = exit_time.strftime("%A")
            day_of_week[day] += 1
            
            # Calculate trade size
            value = abs(trade.get("value", 0))
            if value < 1000:
                trade_size["Small (< $1K)"] += 1
            elif value < 5000:
                trade_size["Medium ($1K-$5K)"] += 1
            elif value < 20000:
                trade_size["Large ($5K-$20K)"] += 1
            else:
                trade_size["Very Large (> $20K)"] += 1
            
            # Calculate holding time
            hold_time = exit_time - entry_time
            hours = hold_time.total_seconds() / 3600
            
            if hours < 1:
                holding_time["< 1 hour"] += 1
            elif hours < 6:
                holding_time["1-6 hours"] += 1
            elif hours < 24:
                holding_time["6-24 hours"] += 1
            elif hours < 72:
                holding_time["1-3 days"] += 1
            else:
                holding_time["> 3 days"] += 1
        
        # Convert counts to percentages
        total_trades = len(trades)
        
        # Convert to percentages
        time_of_day_pct = {k: (v / total_trades) * 100 if total_trades > 0 else 0 for k, v in time_of_day.items()}
        day_of_week_pct = {k: (v / total_trades) * 100 if total_trades > 0 else 0 for k, v in day_of_week.items()}
        trade_size_pct = {k: (v / total_trades) * 100 if total_trades > 0 else 0 for k, v in trade_size.items()}
        holding_time_pct = {k: (v / total_trades) * 100 if total_trades > 0 else 0 for k, v in holding_time.items()}
        
        # Create result
        analytics = {
            "trade_distribution": {
                "time_of_day": time_of_day_pct,
                "day_of_week": day_of_week_pct,
                "trade_size": trade_size_pct,
                "holding_time": holding_time_pct
            },
            "performance_factors": {
                "volatility_impact": self._calculate_factor_impact(trades, "volatility"),
                "volume_impact": self._calculate_factor_impact(trades, "volume"),
                "market_hour_impact": self._calculate_factor_impact(trades, "market_hour"),
                "news_sentiment_impact": self._calculate_factor_impact(trades, "news_sentiment")
            }
        }
        
        return analytics
    
    def _calculate_factor_impact(self, trades: List[Dict[str, Any]], factor: str) -> float:
        """
        Calculate the impact of a factor on trade performance.
        
        This is a simplified implementation that returns a random value.
        In a real implementation, this would analyze the correlation between
        the factor and trade performance.
        
        Args:
            trades: List of trade dictionaries
            factor: Factor to analyze
            
        Returns:
            Impact value (-100 to 100)
        """
        # In a real implementation, this would analyze the correlation between
        # the factor and trade performance. For now, return a random value.
        import random
        return random.uniform(-30, 70)
    
    def _create_empty_summary(self) -> Dict[str, Any]:
        """
        Create an empty performance summary.
        
        Returns:
            Empty performance summary dictionary
        """
        return {
            "period": "unknown",
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_profit": 0,
            "total_loss": 0,
            "net_profit": 0,
            "profit_factor": 0,
            "avg_profit": 0,
            "avg_loss": 0,
            "expectancy": 0,
            "max_drawdown": 0,
            "max_drawdown_pct": 0,
            "current_drawdown": 0,
            "current_drawdown_pct": 0
        }
    
    def _create_empty_metrics(self) -> Dict[str, Any]:
        """
        Create empty detailed metrics.
        
        Returns:
            Empty detailed metrics dictionary
        """
        return {
            "strategy_performance": [],
            "asset_performance": [],
            "recent_trades": [],
            "equity_curve": {"dates": [], "equity": []},
            "trade_analytics": {}
        }
    
    def _is_cache_valid(self, key: str) -> bool:
        """
        Check if a cached item is still valid.
        
        Args:
            key: Cache key
            
        Returns:
            True if the cached item is valid
        """
        if key not in self.cache or key not in self.cache_expiry:
            return False
            
        now = time.time()
        return now - self.cache_expiry[key] < self.cache_duration
    
    def _cache_result(self, key: str, value: Any) -> None:
        """
        Cache a result.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value
        self.cache_expiry[key] = time.time()
    
    def _clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.cache_expiry = {}


# Singleton instance
_PERFORMANCE_TRACKER: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """
    Get the singleton PerformanceTracker instance.
    
    Returns:
        The PerformanceTracker instance
    """
    global _PERFORMANCE_TRACKER
    if _PERFORMANCE_TRACKER is None:
        _PERFORMANCE_TRACKER = PerformanceTracker()
    return _PERFORMANCE_TRACKER
