#!/usr/bin/env python3
"""
Risk Budget Manager

This module implements a comprehensive risk budget management system that:
1. Allocates risk across strategies, markets, and positions
2. Tracks risk consumption over time
3. Enforces risk limits at multiple levels
4. Adapts risk allocation based on performance
5. Integrates with portfolio management and alert systems

The risk budget concept treats risk as a resource to be optimally allocated
across various trading opportunities to maximize risk-adjusted returns.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class RiskBudgetLevel(Enum):
    """Levels at which risk budgets are defined and managed."""
    SYSTEM = auto()       # Entire trading system
    STRATEGY = auto()     # Strategy level (e.g., trend following, mean reversion)
    MARKET = auto()       # Market level (e.g., crypto, forex, equities)
    ASSET = auto()        # Asset level (e.g., BTC, ETH, SOL)
    EXCHANGE = auto()     # Exchange level (e.g., Binance, Coinbase)
    POSITION = auto()     # Individual position level


class RiskAllocationMethod(Enum):
    """Methods for allocating risk across different elements."""
    EQUAL = auto()                # Equal risk allocation
    VOLATILITY_ADJUSTED = auto()  # More risk to less volatile assets
    PERFORMANCE_BASED = auto()    # More risk to better performing strategies
    OPPORTUNITY_BASED = auto()    # More risk where there are more opportunities
    CUSTOM = auto()               # Custom allocation based on user-defined weights


@dataclass
class RiskBudget:
    """
    Represents a risk budget allocation for a specific entity.
    
    Risk budgets define how much risk can be taken at various levels
    of the trading system hierarchy.
    """
    level: RiskBudgetLevel
    name: str
    max_risk: float                           # Maximum risk allowed (in % of capital)
    current_risk: float = 0.0                 # Current risk being utilized
    risk_history: List[Tuple[datetime, float]] = field(default_factory=list)
    allocation_method: RiskAllocationMethod = RiskAllocationMethod.EQUAL
    children: Dict[str, 'RiskBudget'] = field(default_factory=dict)
    start_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'RiskBudget') -> None:
        """Add a child risk budget."""
        self.children[child.name] = child
    
    def remove_child(self, name: str) -> Optional['RiskBudget']:
        """Remove a child risk budget by name."""
        return self.children.pop(name, None)
    
    def update_risk(self, risk_value: float) -> None:
        """
        Update the current risk and record in history.
        
        Args:
            risk_value: New risk value to set
        """
        self.current_risk = risk_value
        self.last_updated = datetime.now()
        self.risk_history.append((self.last_updated, risk_value))
        
        # Trim history if it gets too long (keep last 1000 entries)
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
    
    @property
    def risk_utilization(self) -> float:
        """Calculate the percentage of risk budget being utilized."""
        if self.max_risk == 0:
            return 0.0
        return (self.current_risk / self.max_risk) * 100.0
    
    @property
    def available_risk(self) -> float:
        """Calculate the amount of risk budget still available."""
        return max(0.0, self.max_risk - self.current_risk)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.name,
            "name": self.name,
            "max_risk": self.max_risk,
            "current_risk": self.current_risk,
            "risk_utilization": self.risk_utilization,
            "available_risk": self.available_risk,
            "allocation_method": self.allocation_method.name,
            "start_date": self.start_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "children": {name: child.to_dict() for name, child in self.children.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskBudget':
        """Create a RiskBudget from a dictionary."""
        # Convert string enum values back to enum types
        level = RiskBudgetLevel[data["level"]]
        allocation_method = RiskAllocationMethod[data["allocation_method"]]
        
        # Parse dates
        start_date = datetime.fromisoformat(data["start_date"])
        last_updated = datetime.fromisoformat(data["last_updated"])
        
        # Create the risk budget without children first
        risk_budget = cls(
            level=level,
            name=data["name"],
            max_risk=data["max_risk"],
            current_risk=data["current_risk"],
            allocation_method=allocation_method,
            start_date=start_date,
            last_updated=last_updated,
            metadata=data.get("metadata", {})
        )
        
        # Add children recursively
        for child_name, child_data in data.get("children", {}).items():
            child = cls.from_dict(child_data)
            risk_budget.add_child(child)
        
        return risk_budget


class RiskBudgetManager:
    """
    Manages the allocation and tracking of risk budgets across the trading system.
    
    This class is responsible for:
    1. Creating and maintaining the risk budget hierarchy
    2. Allocating risk to different parts of the system
    3. Tracking risk consumption and utilization
    4. Enforcing risk limits
    5. Optimizing risk allocation based on performance
    """
    
    def __init__(self, 
                 total_risk_percent: float = 5.0,
                 risk_tracking_days: int = 30,
                 auto_adjust: bool = True):
        """
        Initialize the risk budget manager.
        
        Args:
            total_risk_percent: Maximum portfolio risk as a percentage (e.g., 5.0 for 5%)
            risk_tracking_days: Number of days to keep detailed risk history
            auto_adjust: Whether to automatically adjust risk allocations based on performance
        """
        self.total_risk_percent = total_risk_percent
        self.risk_tracking_days = risk_tracking_days
        self.auto_adjust = auto_adjust
        
        # Create the root risk budget (system level)
        self.root_budget = RiskBudget(
            level=RiskBudgetLevel.SYSTEM,
            name="system",
            max_risk=total_risk_percent
        )
        
        # Track performance metrics for risk allocation optimization
        self.performance_metrics = {}
        
        # Last time the risk allocations were optimized
        self.last_optimization = datetime.now()
        
        # Flag to indicate if the risk hierarchy has been modified
        self.modified = False
        
        logger.info(
            f"Risk Budget Manager initialized with {total_risk_percent}% total risk, "
            f"{risk_tracking_days} days of tracking, auto_adjust={auto_adjust}"
        )
    
    def create_strategy_budget(self, 
                              strategy_name: str, 
                              max_risk_percent: float,
                              allocation_method: RiskAllocationMethod = RiskAllocationMethod.EQUAL) -> RiskBudget:
        """
        Create a risk budget for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            max_risk_percent: Maximum risk allocated to this strategy as a percentage
            allocation_method: Method to allocate risk to children
            
        Returns:
            The created risk budget
        """
        strategy_budget = RiskBudget(
            level=RiskBudgetLevel.STRATEGY,
            name=strategy_name,
            max_risk=max_risk_percent,
            allocation_method=allocation_method
        )
        
        self.root_budget.add_child(strategy_budget)
        self.modified = True
        
        logger.info(f"Created strategy budget for {strategy_name} with {max_risk_percent}% max risk")
        return strategy_budget
    
    def create_market_budget(self,
                           strategy_name: str,
                           market_name: str,
                           max_risk_percent: float,
                           allocation_method: RiskAllocationMethod = RiskAllocationMethod.EQUAL) -> Optional[RiskBudget]:
        """
        Create a risk budget for a market within a strategy.
        
        Args:
            strategy_name: Name of the parent strategy
            market_name: Name of the market (e.g., "crypto", "forex")
            max_risk_percent: Maximum risk allocated to this market
            allocation_method: Method to allocate risk to children
            
        Returns:
            The created risk budget or None if the strategy doesn't exist
        """
        if strategy_name not in self.root_budget.children:
            logger.warning(f"Cannot create market budget: Strategy {strategy_name} not found")
            return None
        
        strategy_budget = self.root_budget.children[strategy_name]
        
        market_budget = RiskBudget(
            level=RiskBudgetLevel.MARKET,
            name=market_name,
            max_risk=max_risk_percent,
            allocation_method=allocation_method
        )
        
        strategy_budget.add_child(market_budget)
        self.modified = True
        
        logger.info(f"Created market budget for {market_name} under {strategy_name} with {max_risk_percent}% max risk")
        return market_budget
    
    def create_asset_budget(self,
                          strategy_name: str,
                          market_name: str,
                          asset_name: str,
                          max_risk_percent: float) -> Optional[RiskBudget]:
        """
        Create a risk budget for an asset within a market and strategy.
        
        Args:
            strategy_name: Name of the parent strategy
            market_name: Name of the parent market
            asset_name: Name of the asset (e.g., "BTC", "ETH")
            max_risk_percent: Maximum risk allocated to this asset
            
        Returns:
            The created risk budget or None if the parent budgets don't exist
        """
        if strategy_name not in self.root_budget.children:
            logger.warning(f"Cannot create asset budget: Strategy {strategy_name} not found")
            return None
        
        strategy_budget = self.root_budget.children[strategy_name]
        
        if market_name not in strategy_budget.children:
            logger.warning(f"Cannot create asset budget: Market {market_name} not found under {strategy_name}")
            return None
        
        market_budget = strategy_budget.children[market_name]
        
        asset_budget = RiskBudget(
            level=RiskBudgetLevel.ASSET,
            name=asset_name,
            max_risk=max_risk_percent
        )
        
        market_budget.add_child(asset_budget)
        self.modified = True
        
        logger.info(f"Created asset budget for {asset_name} under {strategy_name}/{market_name} with {max_risk_percent}% max risk")
        return asset_budget
    
    def get_budget(self, path: List[str]) -> Optional[RiskBudget]:
        """
        Get a risk budget by its path in the hierarchy.
        
        Args:
            path: List of names defining the path to the budget
                 e.g., ["trend_following", "crypto", "BTC"]
                 
        Returns:
            The risk budget or None if not found
        """
        if not path:
            return self.root_budget
        
        current = self.root_budget
        for name in path:
            if name not in current.children:
                logger.warning(f"Budget path not found: {'/'.join(path)}")
                return None
            current = current.children[name]
        
        return current
    
    def update_risk_utilization(self, path: List[str], risk_value: float) -> bool:
        """
        Update the risk utilization for a specific budget.
        
        Args:
            path: Path to the budget in the hierarchy
            risk_value: Current risk value to set
            
        Returns:
            True if successful, False if the budget wasn't found
        """
        budget = self.get_budget(path)
        if budget is None:
            return False
        
        budget.update_risk(risk_value)
        self._propagate_risk_up(path)
        return True
    
    def _propagate_risk_up(self, path: List[str]) -> None:
        """
        Propagate risk values up the hierarchy from children to parents.
        
        Args:
            path: Path to the budget that was updated
        """
        for i in range(len(path), 0, -1):
            parent_path = path[:i-1]
            parent = self.get_budget(parent_path)
            
            if parent:
                # Sum up the risk from all children
                total_risk = sum(child.current_risk for child in parent.children.values())
                parent.update_risk(total_risk)
    
    def check_risk_breach(self, path: List[str], proposed_risk: float) -> Tuple[bool, float]:
        """
        Check if a proposed risk value would breach the budget.
        
        Args:
            path: Path to the budget to check
            proposed_risk: Proposed risk value to check
            
        Returns:
            (is_breached, available_risk) tuple
        """
        budget = self.get_budget(path)
        if budget is None:
            return True, 0.0
        
        available = budget.max_risk - budget.current_risk
        is_breached = proposed_risk > available
        
        return is_breached, available
    
    def optimize_allocations(self, 
                           performance_data: Dict[str, float],
                           lookback_days: int = 30) -> None:
        """
        Optimize risk allocations based on performance metrics.
        
        Args:
            performance_data: Dictionary mapping strategy paths to performance metrics
                              (e.g., Sharpe ratios)
            lookback_days: Number of days of performance history to consider
        """
        if not self.auto_adjust:
            logger.info("Automatic risk adjustment is disabled, skipping optimization")
            return
        
        logger.info("Optimizing risk allocations based on performance")
        
        # Update performance metrics
        for path_str, performance in performance_data.items():
            path = path_str.split("/")
            self.performance_metrics[path_str] = performance
        
        # Group by parent path
        parent_groups = {}
        for path_str, performance in self.performance_metrics.items():
            path = path_str.split("/")
            if len(path) <= 1:
                continue
                
            parent_path = "/".join(path[:-1])
            if parent_path not in parent_groups:
                parent_groups[parent_path] = []
            
            parent_groups[parent_path].append((path[-1], performance))
        
        # Optimize each group
        for parent_path, children_data in parent_groups.items():
            parent = self.get_budget(parent_path.split("/"))
            if parent is None or parent.allocation_method != RiskAllocationMethod.PERFORMANCE_BASED:
                continue
            
            # Only optimize if we have data for all children
            if len(children_data) != len(parent.children):
                logger.warning(f"Incomplete performance data for {parent_path}, skipping optimization")
                continue
            
            # Calculate new allocations based on relative performance
            total_performance = sum(max(0.1, perf) for _, perf in children_data)
            
            for child_name, performance in children_data:
                if child_name not in parent.children:
                    continue
                
                child = parent.children[child_name]
                perf_ratio = max(0.1, performance) / total_performance
                
                # Allocate risk proportionally to performance with some limits
                child.max_risk = max(0.1, min(parent.max_risk * 0.8, parent.max_risk * perf_ratio))
                
                logger.info(f"Adjusted {parent_path}/{child_name} risk to {child.max_risk:.2f}% based on performance")
        
        self.last_optimization = datetime.now()
        self.modified = True
    
    def risk_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report.
        
        Returns:
            Dictionary with risk statistics and utilization data
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_risk_percent": self.total_risk_percent,
            "current_system_risk": self.root_budget.current_risk,
            "risk_utilization_percent": self.root_budget.risk_utilization,
            "available_risk": self.root_budget.available_risk,
            "strategies": {}
        }
        
        # Add details for each strategy
        for strategy_name, strategy_budget in self.root_budget.children.items():
            strategy_report = {
                "max_risk": strategy_budget.max_risk,
                "current_risk": strategy_budget.current_risk,
                "risk_utilization_percent": strategy_budget.risk_utilization,
                "available_risk": strategy_budget.available_risk,
                "markets": {}
            }
            
            # Add details for each market within the strategy
            for market_name, market_budget in strategy_budget.children.items():
                market_report = {
                    "max_risk": market_budget.max_risk,
                    "current_risk": market_budget.current_risk,
                    "risk_utilization_percent": market_budget.risk_utilization,
                    "available_risk": market_budget.available_risk,
                    "assets": {}
                }
                
                # Add details for each asset within the market
                for asset_name, asset_budget in market_budget.children.items():
                    market_report["assets"][asset_name] = {
                        "max_risk": asset_budget.max_risk,
                        "current_risk": asset_budget.current_risk,
                        "risk_utilization_percent": asset_budget.risk_utilization,
                        "available_risk": asset_budget.available_risk
                    }
                
                strategy_report["markets"][market_name] = market_report
            
            report["strategies"][strategy_name] = strategy_report
        
        return report
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save the risk budget hierarchy to a JSON file.
        
        Args:
            file_path: Path to the file to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "total_risk_percent": self.total_risk_percent,
                "risk_tracking_days": self.risk_tracking_days,
                "auto_adjust": self.auto_adjust,
                "last_optimization": self.last_optimization.isoformat(),
                "root_budget": self.root_budget.to_dict(),
                "performance_metrics": self.performance_metrics
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.modified = False
            logger.info(f"Risk budget hierarchy saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save risk budget hierarchy: {str(e)}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Optional['RiskBudgetManager']:
        """
        Load a risk budget hierarchy from a JSON file.
        
        Args:
            file_path: Path to the file to load from
            
        Returns:
            RiskBudgetManager instance or None if loading failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            manager = cls(
                total_risk_percent=data["total_risk_percent"],
                risk_tracking_days=data["risk_tracking_days"],
                auto_adjust=data["auto_adjust"]
            )
            
            # Replace the root budget with the loaded one
            manager.root_budget = RiskBudget.from_dict(data["root_budget"])
            
            # Load performance metrics
            manager.performance_metrics = data["performance_metrics"]
            
            # Load last optimization time
            manager.last_optimization = datetime.fromisoformat(data["last_optimization"])
            
            logger.info(f"Risk budget hierarchy loaded from {file_path}")
            return manager
            
        except Exception as e:
            logger.error(f"Failed to load risk budget hierarchy: {str(e)}")
            return None
    
    def visualize_risk_allocation(self, save_path: Optional[str] = None):
        """
        Visualize the risk allocation as a treemap.
        
        Args:
            save_path: Path to save the visualization (if None, display only)
        """
        try:
            import matplotlib.pyplot as plt
            import squarify
        except ImportError:
            logger.error("Cannot visualize risk allocation: missing required libraries")
            logger.error("Please install: pip install matplotlib squarify")
            return
        
        # Collect data for visualization
        strategy_sizes = []
        strategy_labels = []
        strategy_colors = []
        
        # Color scale
        cmap = plt.cm.viridis
        
        for strategy_name, strategy_budget in self.root_budget.children.items():
            utilization = strategy_budget.risk_utilization / 100
            strategy_sizes.append(strategy_budget.max_risk)
            strategy_labels.append(f"{strategy_name}\n{strategy_budget.current_risk:.2f}%/{strategy_budget.max_risk:.2f}%")
            strategy_colors.append(cmap(0.2 + utilization * 0.6))
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        squarify.plot(sizes=strategy_sizes, label=strategy_labels, color=strategy_colors,
                     alpha=0.8, pad=True, text_kwargs={'fontsize':12})
        
        plt.axis('off')
        plt.title(f'Risk Allocation - Total: {self.root_budget.current_risk:.2f}%/{self.total_risk_percent:.2f}%',
                 fontsize=16)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Risk allocation visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close() 