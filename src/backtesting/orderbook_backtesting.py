"""
Order book backtesting module for the AI Crypto Trading System.

This module provides backtesting capabilities specifically for order book strategies,
allowing them to be tested against historical order book snapshots or replayed market data.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from src.common.logging import get_logger
from src.models.market_data import OrderBookData, TimeFrame
from src.models.signals import Signal, SignalType
from src.rust_bridge import OrderBookProcessor, create_order_book_processor
from src.strategy.orderbook_strategy import OrderBookStrategy
from src.backtesting import BacktestStats, BacktestMode


class OrderBookSnapshot:
    """A snapshot of an order book at a specific time."""
    
    def __init__(
        self,
        symbol: str,
        exchange: str,
        timestamp: datetime,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ):
        """Initialize an order book snapshot.
        
        Args:
            symbol: The trading pair symbol
            exchange: The exchange identifier
            timestamp: The timestamp of the snapshot
            bids: List of (price, size) tuples for bids
            asks: List of (price, size) tuples for asks
        """
        self.symbol = symbol
        self.exchange = exchange
        self.timestamp = timestamp
        self.bids = bids
        self.asks = asks
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderBookSnapshot":
        """Create an OrderBookSnapshot from a dictionary.
        
        Args:
            data: Dictionary representation of an OrderBookSnapshot
            
        Returns:
            An OrderBookSnapshot instance
        """
        # Parse timestamp
        if isinstance(data["timestamp"], str):
            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        elif isinstance(data["timestamp"], (int, float)):
            timestamp = datetime.fromtimestamp(data["timestamp"] / 1000.0)
        else:
            timestamp = data["timestamp"]
        
        # Parse bids and asks
        bids = [(float(b[0]), float(b[1])) for b in data["bids"]]
        asks = [(float(a[0]), float(a[1])) for a in data["asks"]]
        
        return cls(
            symbol=data["symbol"],
            exchange=data.get("exchange", "unknown"),
            timestamp=timestamp,
            bids=bids,
            asks=asks
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the snapshot to a dictionary.
        
        Returns:
            Dictionary representation of the snapshot
        """
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat(),
            "bids": self.bids,
            "asks": self.asks
        }
    
    def to_updates(self) -> List[Dict[str, Any]]:
        """Convert the snapshot to a list of order book updates.
        
        Returns:
            A list of updates for OrderBookProcessor
        """
        updates = []
        
        # Add bids
        for i, (price, size) in enumerate(self.bids):
            updates.append({
                "type": "bid",
                "price": price,
                "size": size
            })
        
        # Add asks
        for i, (price, size) in enumerate(self.asks):
            updates.append({
                "type": "ask",
                "price": price,
                "size": size
            })
        
        return updates
    
    def to_orderbook_data(self) -> OrderBookData:
        """Convert the snapshot to OrderBookData.
        
        Returns:
            An OrderBookData instance
        """
        return OrderBookData(
            symbol=self.symbol,
            exchange=self.exchange,
            timestamp=self.timestamp,
            bids=[{"price": p, "size": s} for p, s in self.bids],
            asks=[{"price": p, "size": s} for p, s in self.asks]
        )


class OrderBookDataset:
    """A dataset of historical order book snapshots for backtesting."""
    
    def __init__(self, symbol: str, exchange: str = "unknown"):
        """Initialize an order book dataset.
        
        Args:
            symbol: The trading pair symbol
            exchange: The exchange identifier
        """
        self.symbol = symbol
        self.exchange = exchange
        self.snapshots: List[OrderBookSnapshot] = []
        self.sorted = True
        self.logger = get_logger("backtesting", f"{symbol}_dataset")
    
    def add_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add an order book snapshot to the dataset.
        
        Args:
            snapshot: The snapshot to add
        """
        self.snapshots.append(snapshot)
        self.sorted = False
    
    def sort_snapshots(self) -> None:
        """Sort snapshots by timestamp."""
        if not self.sorted:
            self.snapshots.sort(key=lambda x: x.timestamp)
            self.sorted = True
    
    def load_from_file(self, filepath: Union[str, Path]) -> int:
        """Load snapshots from a file.
        
        The file can be:
        - JSON: List of order book snapshots
        - CSV: With timestamp, price, size, and side columns
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Number of snapshots loaded
        """
        filepath = Path(filepath)
        self.logger.info(f"Loading order book data from {filepath}")
        
        # Check file extension
        if filepath.suffix.lower() == ".json":
            return self._load_from_json(filepath)
        elif filepath.suffix.lower() == ".csv":
            return self._load_from_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _load_from_json(self, filepath: Path) -> int:
        """Load snapshots from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Number of snapshots loaded
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            # List of snapshots
            for item in data:
                # If symbol or exchange not specified, use the dataset defaults
                if "symbol" not in item:
                    item["symbol"] = self.symbol
                if "exchange" not in item:
                    item["exchange"] = self.exchange
                
                self.add_snapshot(OrderBookSnapshot.from_dict(item))
        elif isinstance(data, dict):
            # Single snapshot or dictionary with metadata
            if "snapshots" in data:
                # Dictionary with snapshots list
                for item in data["snapshots"]:
                    if "symbol" not in item:
                        item["symbol"] = self.symbol
                    if "exchange" not in item:
                        item["exchange"] = self.exchange
                    
                    self.add_snapshot(OrderBookSnapshot.from_dict(item))
            else:
                # Single snapshot
                if "symbol" not in data:
                    data["symbol"] = self.symbol
                if "exchange" not in data:
                    data["exchange"] = self.exchange
                
                self.add_snapshot(OrderBookSnapshot.from_dict(data))
        
        # Sort snapshots by timestamp
        self.sort_snapshots()
        
        self.logger.info(f"Loaded {len(self.snapshots)} snapshots from {filepath}")
        return len(self.snapshots)
    
    def _load_from_csv(self, filepath: Path) -> int:
        """Load snapshots from a CSV file.
        
        Expected CSV format:
        timestamp,price,size,side
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Number of snapshots loaded
        """
        # Load data from CSV
        df = pd.read_csv(filepath)
        
        # Check required columns
        required_columns = ["timestamp", "price", "size", "side"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Convert timestamp to datetime
        if df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif df["timestamp"].dtype == "int64":
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Group by timestamp to create snapshots
        grouped = df.groupby(pd.Grouper(key="timestamp", freq="S"))
        
        count = 0
        for timestamp, group in grouped:
            # Skip empty groups
            if group.empty:
                continue
            
            # Filter by side
            bids = group[group["side"].isin(["bid", "buy"])][["price", "size"]].values.tolist()
            asks = group[group["side"].isin(["ask", "sell"])][["price", "size"]].values.tolist()
            
            # Skip if no bids or asks
            if not bids or not asks:
                continue
            
            # Convert to tuples
            bids = [(float(p), float(s)) for p, s in bids]
            asks = [(float(p), float(s)) for p, s in asks]
            
            # Sort bids (highest first) and asks (lowest first)
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])
            
            # Create snapshot - convert pandas Timestamp to python datetime
            snapshot = OrderBookSnapshot(
                symbol=self.symbol,
                exchange=self.exchange,
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
                bids=bids,
                asks=asks
            )
            
            self.add_snapshot(snapshot)
            count += 1
        
        # Sort snapshots
        self.sort_snapshots()
        
        self.logger.info(f"Loaded {count} snapshots from {filepath}")
        return count
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save the dataset to a file.
        
        Args:
            filepath: Path to the output file
        """
        filepath = Path(filepath)
        
        # Make sure snapshots are sorted
        self.sort_snapshots()
        
        # Convert snapshots to dictionaries
        data = [snapshot.to_dict() for snapshot in self.snapshots]
        
        # Check file extension
        if filepath.suffix.lower() == ".json":
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        self.logger.info(f"Saved {len(self.snapshots)} snapshots to {filepath}")
    
    def get_timerange(self) -> Tuple[datetime, datetime]:
        """Get the time range of the dataset.
        
        Returns:
            Tuple of (start_time, end_time)
        """
        if not self.snapshots:
            return (datetime.now(), datetime.now())
        
        # Make sure snapshots are sorted
        self.sort_snapshots()
        
        return (self.snapshots[0].timestamp, self.snapshots[-1].timestamp)
    
    def __len__(self) -> int:
        """Return the number of snapshots in the dataset."""
        return len(self.snapshots)
    
    def __getitem__(self, idx: int) -> OrderBookSnapshot:
        """Get a snapshot by index."""
        return self.snapshots[idx]


class OrderBookBacktestResults:
    """Results from an order book strategy backtest."""
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        signals: List[Signal] = None,
        statistics: Dict[str, Any] = None,
        metrics: Dict[str, Any] = None
    ):
        """Initialize backtest results.
        
        Args:
            strategy_id: The strategy identifier
            symbol: The trading pair symbol
            start_time: Start time of the backtest
            end_time: End time of the backtest
            signals: List of signals generated during the backtest
            statistics: Dictionary of strategy-specific statistics
            metrics: Dictionary of performance metrics
        """
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.signals = signals or []
        self.statistics = statistics or {}
        self.metrics = metrics or {}
        self.logger = get_logger("backtesting", f"{strategy_id}_results")
    
    def add_signal(self, signal: Signal) -> None:
        """Add a signal to the results.
        
        Args:
            signal: The signal to add
        """
        self.signals.append(signal)
    
    def add_statistic(self, name: str, value: Any) -> None:
        """Add a statistic to the results.
        
        Args:
            name: Name of the statistic
            value: Value of the statistic
        """
        self.statistics[name] = value
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the results.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        self.metrics[name] = value
    
    def calculate_metrics(self) -> None:
        """Calculate performance metrics from the signals."""
        if not self.signals:
            self.logger.warning("No signals to calculate metrics from")
            return
        
        # Count signals by type and direction
        entry_signals = [s for s in self.signals if s.signal_type == SignalType.ENTRY]
        exit_signals = [s for s in self.signals if s.signal_type == SignalType.EXIT]
        
        long_signals = [s for s in entry_signals if s.direction == "long"]
        short_signals = [s for s in entry_signals if s.direction == "short"]
        
        self.metrics["total_signals"] = len(self.signals)
        self.metrics["entry_signals"] = len(entry_signals)
        self.metrics["exit_signals"] = len(exit_signals)
        self.metrics["long_signals"] = len(long_signals)
        self.metrics["short_signals"] = len(short_signals)
        
        # Calculate average confidence
        if entry_signals:
            self.metrics["avg_entry_confidence"] = sum(s.confidence for s in entry_signals) / len(entry_signals)
        
        # Calculate signal distribution over time
        if self.signals:
            # Group signals by hour
            signal_hours = {}
            for signal in self.signals:
                hour = signal.timestamp.hour
                signal_hours[hour] = signal_hours.get(hour, 0) + 1
            
            self.statistics["signal_distribution"] = signal_hours
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the results to a dictionary.
        
        Returns:
            Dictionary representation of the results
        """
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "signals": [s.to_dict() for s in self.signals],
            "statistics": self.statistics,
            "metrics": self.metrics
        }
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save the results to a file.
        
        Args:
            filepath: Path to the output file
        """
        filepath = Path(filepath)
        
        # Convert to dictionary
        data = self.to_dict()
        
        # Check file extension
        if filepath.suffix.lower() == ".json":
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        self.logger.info(f"Saved backtest results to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "OrderBookBacktestResults":
        """Load results from a file.
        
        Args:
            filepath: Path to the input file
            
        Returns:
            OrderBookBacktestResults instance
        """
        filepath = Path(filepath)
        
        # Check file extension
        if filepath.suffix.lower() == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Parse times
        start_time = datetime.fromisoformat(data["start_time"].replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(data["end_time"].replace("Z", "+00:00"))
        
        # Parse signals
        signals = []
        for signal_data in data["signals"]:
            signals.append(Signal.from_dict(signal_data))
        
        # Create results
        results = cls(
            strategy_id=data["strategy_id"],
            symbol=data["symbol"],
            start_time=start_time,
            end_time=end_time,
            signals=signals,
            statistics=data.get("statistics", {}),
            metrics=data.get("metrics", {})
        )
        
        return results
    
    def plot_signals(self, title: str = None) -> Tuple[Figure, Axes]:
        """Plot the signals over time.
        
        Args:
            title: Title for the plot
            
        Returns:
            Tuple of (figure, axes)
        """
        if not self.signals:
            self.logger.warning("No signals to plot")
            return plt.subplots()
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract signal data
        timestamps = [s.timestamp for s in self.signals]
        confidences = [s.confidence for s in self.signals]
        directions = [s.direction for s in self.signals]
        prices = [s.price for s in self.signals]
        
        # Plot long and short signals separately
        long_indices = [i for i, d in enumerate(directions) if d == "long"]
        short_indices = [i for i, d in enumerate(directions) if d == "short"]
        
        # Plot price as a line
        if prices:
            ax.plot(timestamps, prices, 'k-', alpha=0.3, label="Price")
        
        # Plot signals
        if long_indices:
            ax.scatter(
                [timestamps[i] for i in long_indices],
                [prices[i] for i in long_indices],
                c='green',
                s=[confidences[i] * 100 for i in long_indices],
                alpha=0.7,
                marker='^',
                label="Long"
            )
        
        if short_indices:
            ax.scatter(
                [timestamps[i] for i in short_indices],
                [prices[i] for i in short_indices],
                c='red',
                s=[confidences[i] * 100 for i in short_indices],
                alpha=0.7,
                marker='v',
                label="Short"
            )
        
        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.set_title(title or f"{self.strategy_id} - {self.symbol} - Signals")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        return fig, ax
    
    def plot_summary(self, title: str = None) -> Tuple[Figure, Axes]:
        """Plot a summary of the backtest results.
        
        Args:
            title: Title for the plot
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title or f"{self.strategy_id} - {self.symbol} - Backtest Summary")
        
        # Plot signals
        ax_signals = axes[0, 0]
        if self.signals:
            timestamps = [s.timestamp for s in self.signals]
            prices = [s.price for s in self.signals]
            directions = [s.direction for s in self.signals]
            
            # Plot price
            ax_signals.plot(timestamps, prices, 'k-', alpha=0.3)
            
            # Plot long and short signals
            long_indices = [i for i, d in enumerate(directions) if d == "long"]
            short_indices = [i for i, d in enumerate(directions) if d == "short"]
            
            if long_indices:
                ax_signals.scatter(
                    [timestamps[i] for i in long_indices],
                    [prices[i] for i in long_indices],
                    c='green',
                    alpha=0.7,
                    marker='^',
                    label="Long"
                )
            
            if short_indices:
                ax_signals.scatter(
                    [timestamps[i] for i in short_indices],
                    [prices[i] for i in short_indices],
                    c='red',
                    alpha=0.7,
                    marker='v',
                    label="Short"
                )
            
            ax_signals.set_title("Signals")
            ax_signals.set_xlabel("Time")
            ax_signals.set_ylabel("Price")
            ax_signals.legend()
            ax_signals.grid(True, alpha=0.3)
            fig.autofmt_xdate()
        else:
            ax_signals.text(0.5, 0.5, "No signals", ha='center', va='center')
            ax_signals.set_title("Signals")
        
        # Plot signal distribution by hour
        ax_dist = axes[0, 1]
        if "signal_distribution" in self.statistics:
            hours = sorted(self.statistics["signal_distribution"].keys())
            counts = [self.statistics["signal_distribution"][h] for h in hours]
            
            ax_dist.bar(hours, counts)
            ax_dist.set_title("Signal Distribution by Hour")
            ax_dist.set_xlabel("Hour of Day")
            ax_dist.set_ylabel("Number of Signals")
            ax_dist.set_xticks(range(0, 24, 2))
            ax_dist.grid(True, alpha=0.3)
        else:
            ax_dist.text(0.5, 0.5, "No data", ha='center', va='center')
            ax_dist.set_title("Signal Distribution by Hour")
        
        # Plot signal metrics
        ax_metrics = axes[1, 0]
        if self.metrics:
            metrics = []
            values = []
            
            for k, v in self.metrics.items():
                metrics.append(k)
                values.append(v)
            
            y_pos = range(len(metrics))
            
            ax_metrics.barh(y_pos, values, align='center')
            ax_metrics.set_yticks(y_pos)
            ax_metrics.set_yticklabels(metrics)
            ax_metrics.invert_yaxis()
            ax_metrics.set_title("Metrics")
            ax_metrics.set_xlabel("Value")
            ax_metrics.grid(True, alpha=0.3)
        else:
            ax_metrics.text(0.5, 0.5, "No metrics", ha='center', va='center')
            ax_metrics.set_title("Metrics")
        
        # Plot signal confidence distribution
        ax_conf = axes[1, 1]
        if self.signals:
            confidences = [s.confidence for s in self.signals if s.signal_type == SignalType.ENTRY]
            
            if confidences:
                ax_conf.hist(confidences, bins=10, alpha=0.7)
                ax_conf.set_title("Signal Confidence Distribution")
                ax_conf.set_xlabel("Confidence")
                ax_conf.set_ylabel("Count")
                ax_conf.grid(True, alpha=0.3)
            else:
                ax_conf.text(0.5, 0.5, "No entry signals", ha='center', va='center')
                ax_conf.set_title("Signal Confidence Distribution")
        else:
            ax_conf.text(0.5, 0.5, "No signals", ha='center', va='center')
            ax_conf.set_title("Signal Confidence Distribution")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig, axes


class OrderBookBacktester:
    """Backtester for order book strategies.
    
    This backtester simulates the execution of an order book strategy against
    historical order book data, allowing for the evaluation of strategy performance.
    """
    
    def __init__(
        self,
        strategy: OrderBookStrategy,
        dataset: OrderBookDataset,
        log_level: str = "INFO"
    ):
        """Initialize the order book backtester.
        
        Args:
            strategy: The order book strategy to test
            dataset: The order book dataset to use
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.strategy = strategy
        self.dataset = dataset
        self.logger = get_logger("backtesting", f"{strategy.strategy_id}_backtester", log_level=log_level)
        
        # Initialize results
        start_time, end_time = dataset.get_timerange()
        self.results = OrderBookBacktestResults(
            strategy_id=strategy.strategy_id,
            symbol=dataset.symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        # Signal collection
        self.signals: List[Signal] = []
        
        # Status tracking
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.processed_snapshots = 0
    
    async def run(self) -> OrderBookBacktestResults:
        """Run the backtest.
        
        Returns:
            Results of the backtest
        """
        if self.is_running:
            self.logger.warning("Backtest is already running")
            return self.results
        
        self.is_running = True
        self.signals = []
        self.processed_snapshots = 0
        
        # Record start time
        self.start_time = datetime.now()
        
        try:
            # Initialize the strategy
            self.logger.info(f"Initializing strategy '{self.strategy.strategy_id}'")
            await self.strategy.initialize()
            
            # Register signal handler
            self._register_signal_handler()
            
            # Start the strategy
            self.logger.info(f"Starting strategy '{self.strategy.strategy_id}'")
            await self.strategy.start()
            
            # Process each snapshot
            total_snapshots = len(self.dataset)
            self.logger.info(f"Running backtest with {total_snapshots} snapshots")
            
            for i, snapshot in enumerate(self.dataset.snapshots):
                # Update progress
                if i % max(1, total_snapshots // 100) == 0:
                    progress = (i / total_snapshots) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({i}/{total_snapshots})")
                
                # Process snapshot
                await self._process_snapshot(snapshot)
                self.processed_snapshots += 1
            
            # Stop the strategy
            self.logger.info(f"Stopping strategy '{self.strategy.strategy_id}'")
            await self.strategy.stop()
            
            # Calculate and populate results
            self._calculate_results()
            
            # Record end time
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            
            self.logger.info(f"Backtest completed in {duration:.2f} seconds")
            self.logger.info(f"Processed {self.processed_snapshots} snapshots")
            self.logger.info(f"Generated {len(self.signals)} signals")
            
            # Return results
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error during backtest: {e}")
            raise
        
        finally:
            self.is_running = False
    
    async def _register_signal_handler(self) -> None:
        """Register a handler for signals from the strategy."""
        # Override the strategy's publish_signal method
        original_publish_signal = self.strategy.publish_signal
        
        async def signal_handler(signal: Signal) -> None:
            # Store the signal
            self.signals.append(signal)
            self.results.add_signal(signal)
            
            # Log the signal
            self.logger.info(f"Signal: {signal.symbol} {signal.direction} {signal.signal_type.name} @ {signal.price}")
        
        # Replace the strategy's publish_signal method
        self.strategy.publish_signal = signal_handler
    
    async def _process_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Process an order book snapshot.
        
        Args:
            snapshot: The order book snapshot to process
        """
        try:
            # Get the symbol
            symbol = snapshot.symbol
            
            # Create processor if it doesn't exist
            if symbol not in self.strategy.orderbook_processors:
                self.strategy._create_processor_for_symbol(symbol)
            
            processor = self.strategy.orderbook_processors[symbol]
            
            # Convert snapshot to updates
            updates = snapshot.to_updates()
            
            # Process the updates
            processor.process_updates(updates)
            
            # Analyze the updated order book
            await self.strategy.analyze_orderbook(symbol, processor)
            
        except Exception as e:
            self.logger.error(f"Error processing snapshot: {e}")
    
    def _calculate_results(self) -> None:
        """Calculate and populate the backtest results."""
        self.results.signals = self.signals
        
        # Calculate basic metrics
        self.results.calculate_metrics()
        
        # Add strategy-specific statistics
        if hasattr(self.strategy, "get_statistics"):
            stats = self.strategy.get_statistics()
            for name, value in stats.items():
                self.results.add_statistic(name, value)
        
        # Add backtester statistics
        self.results.add_statistic("processed_snapshots", self.processed_snapshots)
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.results.add_statistic("backtest_duration_seconds", duration) 