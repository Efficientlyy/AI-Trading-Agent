"""Portfolio models.

This module defines data models for portfolio management, including:
- Asset allocation models
- Portfolio allocation configuration
- Rebalancing parameters and thresholds
"""

from enum import Enum
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from src.common.datetime_utils import utc_now, format_iso


class AllocationMethod(Enum):
    """Method used for calculating asset allocations."""
    
    EQUAL_WEIGHT = "equal_weight"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    CUSTOM = "custom"
    RISK_PARITY = "risk_parity"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    PERFORMANCE_WEIGHTED = "performance_weighted"


class RebalanceTrigger(Enum):
    """Trigger method for portfolio rebalancing."""
    
    TIME_BASED = "time_based"  # Rebalance at fixed time intervals
    THRESHOLD_BASED = "threshold_based"  # Rebalance when allocation drifts beyond threshold
    COMBINED = "combined"  # Combination of time and threshold based
    MANUAL = "manual"  # Only rebalance when explicitly triggered


class AssetAllocation:
    """Individual asset allocation in a portfolio."""
    
    def __init__(
        self,
        symbol: str,
        target_percentage: Decimal,
        min_percentage: Optional[Decimal] = None,
        max_percentage: Optional[Decimal] = None
    ):
        """
        Initialize an asset allocation.
        
        Args:
            symbol: Asset symbol
            target_percentage: Target allocation percentage (0-1)
            min_percentage: Minimum allocation percentage (0-1)
            max_percentage: Maximum allocation percentage (0-1)
        """
        self.symbol = symbol
        self.target_percentage = target_percentage
        self.min_percentage = min_percentage or Decimal("0")
        self.max_percentage = max_percentage or Decimal("1")
        self.current_percentage: Optional[Decimal] = None
        self.drift: Optional[Decimal] = None
    
    @property
    def is_within_bounds(self) -> bool:
        """Check if current allocation is within min/max bounds."""
        if self.current_percentage is None:
            return True
        
        return (
            self.current_percentage >= self.min_percentage and
            self.current_percentage <= self.max_percentage
        )
    
    def update_current(self, current_percentage: Decimal) -> None:
        """
        Update the current allocation percentage.
        
        Args:
            current_percentage: Current actual percentage allocation (0-1)
        """
        self.current_percentage = current_percentage
        self.drift = current_percentage - self.target_percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "target_percentage": float(self.target_percentage),
            "min_percentage": float(self.min_percentage),
            "max_percentage": float(self.max_percentage),
            "current_percentage": float(self.current_percentage) if self.current_percentage is not None else None,
            "drift": float(self.drift) if self.drift is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssetAllocation':
        """Create from dictionary."""
        allocation = cls(
            symbol=data["symbol"],
            target_percentage=Decimal(str(data["target_percentage"])),
            min_percentage=Decimal(str(data["min_percentage"])) if "min_percentage" in data else None,
            max_percentage=Decimal(str(data["max_percentage"])) if "max_percentage" in data else None
        )
        
        if "current_percentage" in data and data["current_percentage"] is not None:
            allocation.current_percentage = Decimal(str(data["current_percentage"]))
        
        if "drift" in data and data["drift"] is not None:
            allocation.drift = Decimal(str(data["drift"]))
        
        return allocation


class RebalanceConfig:
    """Configuration for portfolio rebalancing."""
    
    def __init__(
        self,
        allocation_method: AllocationMethod,
        trigger_method: RebalanceTrigger,
        drift_threshold: Decimal = Decimal("0.05"),  # 5% drift threshold
        time_interval_days: int = 30,
        custom_weights: Optional[Dict[str, Decimal]] = None,
        asset_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        min_trade_amount: Decimal = Decimal("100"),  # Minimum trade amount in USD
        minimize_fees: bool = True
    ):
        """
        Initialize rebalance configuration.
        
        Args:
            allocation_method: Method to use for calculating allocations
            trigger_method: Method to determine when to rebalance
            drift_threshold: Allocation drift percentage to trigger rebalance
            time_interval_days: Days between time-based rebalances
            custom_weights: Custom weight allocations for assets
            asset_constraints: Constraints for specific assets
            min_trade_amount: Minimum trade amount to execute
            minimize_fees: Whether to optimize trades to minimize fees
        """
        self.allocation_method = allocation_method
        self.trigger_method = trigger_method
        self.drift_threshold = drift_threshold
        self.time_interval_days = time_interval_days
        self.custom_weights = custom_weights or {}
        self.asset_constraints = asset_constraints or {}
        self.min_trade_amount = min_trade_amount
        self.minimize_fees = minimize_fees
        self.last_rebalance_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allocation_method": self.allocation_method.value,
            "trigger_method": self.trigger_method.value,
            "drift_threshold": float(self.drift_threshold),
            "time_interval_days": self.time_interval_days,
            "custom_weights": {k: float(v) for k, v in self.custom_weights.items()},
            "asset_constraints": self.asset_constraints,
            "min_trade_amount": float(self.min_trade_amount),
            "minimize_fees": self.minimize_fees,
            "last_rebalance_time": self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RebalanceConfig':
        """Create from dictionary."""
        config = cls(
            allocation_method=AllocationMethod(data["allocation_method"]),
            trigger_method=RebalanceTrigger(data["trigger_method"]),
            drift_threshold=Decimal(str(data["drift_threshold"])) if "drift_threshold" in data else Decimal("0.05"),
            time_interval_days=data.get("time_interval_days", 30),
            asset_constraints=data.get("asset_constraints", {}),
            min_trade_amount=Decimal(str(data["min_trade_amount"])) if "min_trade_amount" in data else Decimal("100"),
            minimize_fees=data.get("minimize_fees", True)
        )
        
        # Convert custom weights to Decimal
        if "custom_weights" in data:
            config.custom_weights = {k: Decimal(str(v)) for k, v in data["custom_weights"].items()}
        
        # Parse last rebalance time
        if "last_rebalance_time" in data and data["last_rebalance_time"]:
            config.last_rebalance_time = datetime.fromisoformat(data["last_rebalance_time"])
        
        return config


class PortfolioAllocation:
    """
    Portfolio allocation plan containing asset allocations and rebalance settings.
    
    This class manages a set of asset allocations, tracks their current state,
    and provides methods to check if rebalancing is needed and to calculate
    the trades required to achieve target allocations.
    """
    
    def __init__(
        self,
        name: str,
        config: RebalanceConfig,
        allocations: List[AssetAllocation]
    ):
        """
        Initialize a portfolio allocation.
        
        Args:
            name: Allocation plan name
            config: Rebalance configuration
            allocations: List of asset allocations
        """
        self.name = name
        self.config = config
        self.allocations = allocations
        self.historical_snapshots: List[Dict[str, Any]] = []
    
    def needs_rebalancing(self) -> bool:
        """Determine if portfolio needs rebalancing based on configured triggers."""
        trigger_method = self.config.trigger_method
        
        if trigger_method == RebalanceTrigger.TIME_BASED or trigger_method == RebalanceTrigger.COMBINED:
            # Check time-based trigger
            if self.config.last_rebalance_time is None:
                return True
            
            days_since_last = (utc_now() - self.config.last_rebalance_time).days
            if days_since_last >= self.config.time_interval_days:
                return True
        
        if trigger_method == RebalanceTrigger.THRESHOLD_BASED or trigger_method == RebalanceTrigger.COMBINED:
            # Check drift-based trigger
            for allocation in self.allocations:
                if allocation.drift is not None and abs(allocation.drift) > self.config.drift_threshold:
                    return True
        
        return False
    
    def add_historical_snapshot(self) -> None:
        """Add current allocation state to historical snapshots."""
        snapshot = {
            "timestamp": format_iso(),
            "allocations": [allocation.to_dict() for allocation in self.allocations]
        }
        self.historical_snapshots.append(snapshot)
    
    def calculate_rebalance_trades(
        self,
        current_values: Dict[str, Decimal],
        total_value: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate trades needed to rebalance the portfolio.
        
        Args:
            current_values: Current value of each asset
            total_value: Total portfolio value
            
        Returns:
            Dict mapping symbols to trade amounts (positive = buy, negative = sell)
        """
        trades = {}
        
        # Calculate target values
        for allocation in self.allocations:
            symbol = allocation.symbol
            target_value = allocation.target_percentage * total_value
            current_value = current_values.get(symbol, Decimal("0"))
            
            # Calculate trade amount
            trade_amount = target_value - current_value
            
            # Add to trades if significant
            if abs(trade_amount) > self.config.min_trade_amount:
                trades[symbol] = trade_amount
        
        return trades
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "allocations": [allocation.to_dict() for allocation in self.allocations],
            "historical_snapshots": self.historical_snapshots
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioAllocation':
        """Create from dictionary."""
        config = RebalanceConfig.from_dict(data["config"])
        allocations = [AssetAllocation.from_dict(a) for a in data["allocations"]]
        
        allocation = cls(
            name=data["name"],
            config=config,
            allocations=allocations
        )
        
        if "historical_snapshots" in data:
            allocation.historical_snapshots = data["historical_snapshots"]
        
        return allocation