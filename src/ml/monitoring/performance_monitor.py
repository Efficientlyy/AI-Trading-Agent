"""Real-time performance monitoring system."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
import logging
from ..backtesting.backtester import TradeResult

DateType = Union[datetime, date]

@dataclass
class MonitoringThresholds:
    """Thresholds for monitoring alerts."""
    max_drawdown: float = 0.1  # 10% maximum drawdown
    daily_loss_limit: float = 0.02  # 2% daily loss limit
    position_size_limit: float = 0.2  # 20% position size limit
    concentration_limit: float = 0.3  # 30% concentration limit
    correlation_threshold: float = 0.7  # High correlation threshold
    volatility_threshold: float = 0.02  # 2% daily volatility threshold
    trade_frequency_limit: int = 100  # Maximum trades per day
    win_rate_threshold: float = 0.4  # Minimum win rate
    cost_ratio_threshold: float = 0.002  # Maximum cost ratio (0.2%)

@dataclass
class Alert:
    """Alert information."""
    timestamp: datetime
    level: str  # 'INFO', 'WARNING', 'ERROR'
    category: str
    message: str
    metrics: Dict[str, Union[float, str]]

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(
        self,
        initial_capital: float,
        thresholds: Optional[MonitoringThresholds] = None
    ):
        """Initialize monitor with parameters."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.thresholds = thresholds or MonitoringThresholds()
        
        # Initialize state
        self.positions: Dict[str, float] = {}  # symbol -> position size
        self.trades: List[TradeResult] = []
        self.alerts: List[Alert] = []
        self.daily_stats: Dict[DateType, Dict[str, float]] = {}
        self.metrics = self._init_metrics()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_metrics(self) -> Dict[str, float]:
        """Initialize monitoring metrics."""
        return {
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "drawdown": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
            "trade_count": 0,
            "total_costs": 0.0,
            "cost_ratio": 0.0,
            "exposure": 0.0,
            "concentration": 0.0
        }
    
    def update_position(
        self,
        symbol: str,
        size: float,
        price: float,
        timestamp: datetime
    ) -> None:
        """Update position and check risk limits.
        
        Args:
            symbol: Trading symbol
            size: New position size
            price: Current price
            timestamp: Update timestamp
        """
        old_size = self.positions.get(symbol, 0.0)
        self.positions[symbol] = size
        
        # Calculate position value and exposure
        position_value = abs(size * price)
        total_exposure = sum(
            abs(pos * price) for pos in self.positions.values()
        )
        exposure_ratio = total_exposure / self.current_capital
        
        # Check position size limit
        if position_value / self.current_capital > self.thresholds.position_size_limit:
            self._create_alert(
                timestamp,
                "WARNING",
                "Position Size",
                f"Position size for {symbol} exceeds limit",
                {
                    "symbol": symbol,
                    "size_ratio": float(position_value / self.current_capital),
                    "limit": float(self.thresholds.position_size_limit)
                }
            )
        
        # Check concentration limit
        max_position = max(
            abs(pos * price) for pos in self.positions.values()
        )
        concentration = max_position / total_exposure if total_exposure > 0 else 0
        
        if concentration > self.thresholds.concentration_limit:
            self._create_alert(
                timestamp,
                "WARNING",
                "Concentration",
                "Portfolio concentration exceeds limit",
                {
                    "concentration": float(concentration),
                    "limit": float(self.thresholds.concentration_limit)
                }
            )
        
        # Update metrics
        self.metrics["exposure"] = exposure_ratio
        self.metrics["concentration"] = concentration
    
    def update_trade(self, trade: TradeResult) -> None:
        """Update metrics with new trade.
        
        Args:
            trade: Completed trade information
        """
        self.trades.append(trade)
        
        # Update capital
        self.current_capital += trade["pnl"] - trade["costs"]
        
        # Update daily statistics
        trade_date = trade["exit_time"].date()
        if trade_date not in self.daily_stats:
            self.daily_stats[trade_date] = {
                "pnl": 0.0,
                "costs": 0.0,
                "trade_count": 0.0,
                "win_count": 0.0
            }
        
        daily_stats = self.daily_stats[trade_date]
        daily_stats["pnl"] += trade["pnl"]
        daily_stats["costs"] += trade["costs"]
        daily_stats["trade_count"] += 1.0
        if trade["pnl"] > 0:
            daily_stats["win_count"] += 1.0
        
        # Check daily loss limit
        daily_return = daily_stats["pnl"] / self.current_capital
        if daily_return < -self.thresholds.daily_loss_limit:
            self._create_alert(
                trade["exit_time"],
                "ERROR",
                "Daily Loss",
                "Daily loss exceeds limit",
                {
                    "daily_return": float(daily_return),
                    "limit": float(-self.thresholds.daily_loss_limit)
                }
            )
        
        # Check trade frequency
        if daily_stats["trade_count"] > self.thresholds.trade_frequency_limit:
            self._create_alert(
                trade["exit_time"],
                "WARNING",
                "Trade Frequency",
                "Daily trade count exceeds limit",
                {
                    "trade_count": float(daily_stats["trade_count"]),
                    "limit": float(self.thresholds.trade_frequency_limit)
                }
            )
        
        # Update metrics
        self._update_metrics(trade["exit_time"])
    
    def _update_metrics(self, timestamp: datetime) -> None:
        """Update monitoring metrics.
        
        Args:
            timestamp: Current timestamp
        """
        # Calculate PnL metrics
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_costs = sum(t["costs"] for t in self.trades)
        
        # Calculate drawdown
        peak_capital = max(
            self.initial_capital,
            max(
                self.initial_capital + sum(
                    t["pnl"] - t["costs"] for t in self.trades[:i+1]
                )
                for i in range(len(self.trades))
            )
        )
        drawdown = (peak_capital - self.current_capital) / peak_capital
        
        # Calculate win rate
        win_count = sum(1 for t in self.trades if t["pnl"] > 0)
        win_rate = win_count / len(self.trades) if self.trades else 0
        
        # Calculate returns and volatility
        returns = [t["return_pct"] for t in self.trades]
        if returns:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            avg_return = np.mean(returns)
            sharpe = (
                (avg_return * 252) / volatility  # Annualized Sharpe
                if volatility > 0 else 0
            )
        else:
            volatility = 0.0
            sharpe = 0.0
        
        # Get daily PnL
        current_date = timestamp.date()
        daily_pnl = self.daily_stats.get(current_date, {}).get("pnl", 0.0)
        
        # Update metrics
        self.metrics.update({
            "total_pnl": total_pnl,
            "daily_pnl": daily_pnl,
            "drawdown": drawdown,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "volatility": volatility,
            "trade_count": float(len(self.trades)),
            "total_costs": total_costs,
            "cost_ratio": total_costs / self.initial_capital
        })
        
        # Check metric thresholds
        if drawdown > self.thresholds.max_drawdown:
            self._create_alert(
                timestamp,
                "ERROR",
                "Drawdown",
                "Maximum drawdown exceeded",
                {
                    "drawdown": float(drawdown),
                    "limit": float(self.thresholds.max_drawdown)
                }
            )
        
        if volatility > self.thresholds.volatility_threshold:
            self._create_alert(
                timestamp,
                "WARNING",
                "Volatility",
                "Portfolio volatility exceeds threshold",
                {
                    "volatility": float(volatility),
                    "limit": float(self.thresholds.volatility_threshold)
                }
            )
        
        if (len(self.trades) >= 20 and  # Minimum trades for win rate
            win_rate < self.thresholds.win_rate_threshold):
            self._create_alert(
                timestamp,
                "WARNING",
                "Win Rate",
                "Win rate below threshold",
                {
                    "win_rate": float(win_rate),
                    "limit": float(self.thresholds.win_rate_threshold)
                }
            )
        
        cost_ratio = total_costs / self.initial_capital
        if cost_ratio > self.thresholds.cost_ratio_threshold:
            self._create_alert(
                timestamp,
                "WARNING",
                "Costs",
                "Cost ratio exceeds threshold",
                {
                    "cost_ratio": float(cost_ratio),
                    "limit": float(self.thresholds.cost_ratio_threshold)
                }
            )
    
    def _create_alert(
        self,
        timestamp: datetime,
        level: str,
        category: str,
        message: str,
        metrics: Dict[str, Union[float, str]]
    ) -> None:
        """Create and log monitoring alert.
        
        Args:
            timestamp: Alert timestamp
            level: Alert level
            category: Alert category
            message: Alert message
            metrics: Alert metrics
        """
        alert = Alert(
            timestamp=timestamp,
            level=level,
            category=category,
            message=message,
            metrics=metrics
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_message = (
            f"{category}: {message} "
            f"[{', '.join(f'{k}={v}' for k, v in metrics.items())}]"
        )
        
        if level == "ERROR":
            self.logger.error(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_daily_report(self, date: DateType) -> Dict[str, float]:
        """Get performance report for a specific day.
        
        Args:
            date: Report date
        
        Returns:
            Dictionary of daily statistics
        """
        if date not in self.daily_stats:
            return {}
        
        stats = self.daily_stats[date]
        
        return {
            "pnl": stats["pnl"],
            "return": stats["pnl"] / self.current_capital,
            "costs": stats["costs"],
            "trade_count": stats["trade_count"],
            "win_rate": (
                stats["win_count"] / stats["trade_count"]
                if stats["trade_count"] > 0 else 0
            ),
            "cost_ratio": stats["costs"] / stats["pnl"]
            if stats["pnl"] > 0 else float('inf')
        }
    
    def get_alerts_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        levels: Optional[Set[str]] = None,
        categories: Optional[Set[str]] = None
    ) -> List[Alert]:
        """Get filtered alerts summary.
        
        Args:
            start_time: Filter start time
            end_time: Filter end time
            levels: Filter alert levels
            categories: Filter alert categories
        
        Returns:
            List of filtered alerts
        """
        filtered_alerts = self.alerts
        
        if start_time:
            filtered_alerts = [
                a for a in filtered_alerts
                if a.timestamp >= start_time
            ]
        
        if end_time:
            filtered_alerts = [
                a for a in filtered_alerts
                if a.timestamp <= end_time
            ]
        
        if levels:
            filtered_alerts = [
                a for a in filtered_alerts
                if a.level in levels
            ]
        
        if categories:
            filtered_alerts = [
                a for a in filtered_alerts
                if a.category in categories
            ]
        
        return filtered_alerts
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get current monitoring metrics summary.
        
        Returns:
            Dictionary of current metrics
        """
        return self.metrics.copy()
    
    def get_position_summary(self) -> Dict[str, Dict[str, float]]:
        """Get current positions summary.
        
        Returns:
            Dictionary of position information
        """
        total_abs_size = sum(abs(p) for p in self.positions.values())
        return {
            symbol: {
                "size": size,
                "exposure": abs(size) / total_abs_size if total_abs_size > 0 else 0.0
            }
            for symbol, size in self.positions.items()
        } 