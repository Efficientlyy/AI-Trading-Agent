"""
Portfolio Rebalancing Demo (Standalone Version)

This script demonstrates portfolio rebalancing functionality with different allocation methods:
1. Equal Weight Allocation
2. Volatility Weighted Allocation
3. Custom Weight Allocation

The demo includes visualization of allocations before and after rebalancing.
"""

import os
import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from tabulate import tabulate
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Visualization requires matplotlib, numpy, and tabulate. Install with:")
    print("pip install matplotlib numpy tabulate")
    VISUALIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("portfolio_rebalance_demo")

# ===== Portfolio Models =====

class AllocationMethod(Enum):
    """Method used for calculating asset allocations."""
    EQUAL_WEIGHT = "equal_weight"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    CUSTOM = "custom"


class RebalanceTrigger(Enum):
    """Trigger method for portfolio rebalancing."""
    THRESHOLD_BASED = "threshold_based"
    TIME_BASED = "time_based"
    COMBINED = "combined"
    MANUAL = "manual"


class PositionType(Enum):
    """Type of trading position."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Trading position."""
    symbol: str
    position_type: PositionType
    size: Decimal
    entry_price: Decimal
    position_id: Optional[str] = None
    last_price: Optional[Decimal] = None


class AssetAllocation:
    """Individual asset allocation in a portfolio."""
    
    def __init__(
        self,
        symbol: str,
        target_percentage: Decimal,
        min_percentage: Optional[Decimal] = None,
        max_percentage: Optional[Decimal] = None
    ):
        self.symbol = symbol
        self.target_percentage = target_percentage
        self.min_percentage = min_percentage or Decimal("0")
        self.max_percentage = max_percentage or Decimal("1")
        self.current_percentage: Optional[Decimal] = None
        self.drift: Optional[Decimal] = None
    
    def update_current(self, current_percentage: Decimal) -> None:
        """Update current allocation percentage."""
        self.current_percentage = current_percentage
        self.drift = current_percentage - self.target_percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "target_percentage": float(self.target_percentage),
            "current_percentage": float(self.current_percentage) if self.current_percentage is not None else None,
            "drift": float(self.drift) if self.drift is not None else None
        }


class RebalanceConfig:
    """Configuration for portfolio rebalancing."""
    
    def __init__(
        self,
        allocation_method: AllocationMethod,
        trigger_method: RebalanceTrigger,
        drift_threshold: Decimal = Decimal("0.05"),
        custom_weights: Optional[Dict[str, Decimal]] = None,
        min_trade_amount: Decimal = Decimal("100")
    ):
        self.allocation_method = allocation_method
        self.trigger_method = trigger_method
        self.drift_threshold = drift_threshold
        self.custom_weights = custom_weights or {}
        self.min_trade_amount = min_trade_amount
        self.last_rebalance_time: Optional[datetime] = None


class PortfolioAllocation:
    """Portfolio allocation plan."""
    
    def __init__(
        self,
        name: str,
        config: RebalanceConfig,
        allocations: List[AssetAllocation]
    ):
        self.name = name
        self.config = config
        self.allocations = allocations
    
    def is_rebalance_needed(self) -> bool:
        """Check if rebalancing is needed."""
        for allocation in self.allocations:
            if allocation.drift is not None and abs(allocation.drift) > self.config.drift_threshold:
                return True
        return False
    
    def calculate_rebalance_trades(
        self,
        current_values: Dict[str, Decimal],
        total_value: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate trades needed to rebalance."""
        trades = {}
        for allocation in self.allocations:
            symbol = allocation.symbol
            target_value = allocation.target_percentage * total_value
            current_value = current_values.get(symbol, Decimal("0"))
            trade_amount = target_value - current_value
            if abs(trade_amount) > self.config.min_trade_amount:
                trades[symbol] = trade_amount
        return trades


class SimplePortfolioManager:
    """Simplified portfolio manager for demo purposes."""
    
    def __init__(self, initial_balance: Decimal):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.open_positions: Set[str] = set()
        self.open_positions_by_symbol: Dict[str, Set[str]] = {}
        self.current_prices: Dict[str, Decimal] = {}
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [self.positions[pos_id] for pos_id in self.open_positions]
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        total_value = sum(
            pos.size * self.current_prices.get(pos.symbol, Decimal("0"))
            for pos in self.get_open_positions()
        )
        return {
            "total_value": total_value,
            "current_balance": self.current_balance
        }


class PortfolioRebalancer:
    """Portfolio rebalancing engine."""
    
    def __init__(self, portfolio_manager: SimplePortfolioManager):
        self.portfolio_manager = portfolio_manager
        self.allocation_plans: Dict[str, PortfolioAllocation] = {}
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
    
    def create_allocation_plan(
        self,
        name: str,
        allocation_method: AllocationMethod,
        symbols: List[str],
        config: Optional[RebalanceConfig] = None
    ) -> PortfolioAllocation:
        """Create a new allocation plan."""
        if config is None:
            config = RebalanceConfig(
                allocation_method=allocation_method,
                trigger_method=RebalanceTrigger.THRESHOLD_BASED
            )
        
        allocations = []
        if allocation_method == AllocationMethod.EQUAL_WEIGHT:
            weight = Decimal("1") / Decimal(str(len(symbols)))
            for symbol in symbols:
                allocations.append(AssetAllocation(symbol=symbol, target_percentage=weight))
        
        elif allocation_method == AllocationMethod.VOLATILITY_WEIGHTED:
            total_inverse_vol = Decimal("0")
            volatilities = {}
            
            for symbol in symbols:
                vol = Decimal(str(self.market_data_cache.get(symbol, {}).get("volatility", 1)))
                inverse_vol = Decimal("1") / vol
                volatilities[symbol] = inverse_vol
                total_inverse_vol += inverse_vol
            
            for symbol in symbols:
                weight = volatilities[symbol] / total_inverse_vol
                allocations.append(AssetAllocation(symbol=symbol, target_percentage=weight))
        
        elif allocation_method == AllocationMethod.CUSTOM:
            for symbol in symbols:
                weight = config.custom_weights.get(symbol, Decimal("0"))
                allocations.append(AssetAllocation(symbol=symbol, target_percentage=weight))
        
        plan = PortfolioAllocation(name=name, config=config, allocations=allocations)
        self.allocation_plans[name] = plan
        return plan
    
    def update_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update market data for calculations."""
        self.market_data_cache[symbol] = data
    
    async def check_rebalance_needed(self, plan_name: str) -> bool:
        """Check if rebalancing is needed."""
        if plan_name not in self.allocation_plans:
            return False
        
        plan = self.allocation_plans[plan_name]
        await self._update_current_allocations(plan)
        return plan.is_rebalance_needed()
    
    async def _update_current_allocations(self, plan: PortfolioAllocation) -> None:
        """Update current allocation percentages."""
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        total_value = portfolio_state["total_value"]
        
        if total_value <= Decimal("0"):
            return
        
        positions = self.portfolio_manager.get_open_positions()
        position_values: Dict[str, Decimal] = {}
        
        for position in positions:
            symbol = position.symbol
            current_price = self.portfolio_manager.current_prices.get(symbol, Decimal("0"))
            if current_price > Decimal("0"):
                position_value = position.size * current_price
                position_values[symbol] = position_values.get(symbol, Decimal("0")) + position_value
        
        for allocation in plan.allocations:
            symbol = allocation.symbol
            current_value = position_values.get(symbol, Decimal("0"))
            allocation.update_current(current_value / total_value)
    
    async def get_allocation_status(self, plan_name: str) -> Dict[str, Any]:
        """Get current allocation status."""
        if plan_name not in self.allocation_plans:
            return {"error": "Plan not found"}
        
        plan = self.allocation_plans[plan_name]
        await self._update_current_allocations(plan)
        
        return {
            "plan_name": plan_name,
            "allocations": [a.to_dict() for a in plan.allocations],
            "needs_rebalance": plan.is_rebalance_needed(),
            "last_rebalance": plan.config.last_rebalance_time
        }
    
    async def generate_rebalance_plan(self, plan_name: str) -> Dict[str, Decimal]:
        """Generate rebalancing trades."""
        if plan_name not in self.allocation_plans:
            return {}
        
        plan = self.allocation_plans[plan_name]
        await self._update_current_allocations(plan)
        
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        total_value = portfolio_state["total_value"]
        
        if total_value <= Decimal("0"):
            return {}
        
        positions = self.portfolio_manager.get_open_positions()
        position_values = {}
        
        for position in positions:
            symbol = position.symbol
            current_price = self.portfolio_manager.current_prices.get(symbol, Decimal("0"))
            if current_price > Decimal("0"):
                position_value = position.size * current_price
                position_values[symbol] = position_values.get(symbol, Decimal("0")) + position_value
        
        return plan.calculate_rebalance_trades(position_values, total_value)
    
    async def execute_rebalance(self, plan_name: str) -> bool:
        """Execute rebalancing trades."""
        if plan_name not in self.allocation_plans:
            return False
        
        trades = await self.generate_rebalance_plan(plan_name)
        if not trades:
            return True
        
        # In a real implementation, this would execute the trades
        # For demo purposes, we just log them
        logger.info("Executing trades:")
        for symbol, amount in trades.items():
            action = "BUY" if amount > 0 else "SELL"
            logger.info(f"  {action} {abs(amount)} of {symbol}")
        
        plan = self.allocation_plans[plan_name]
        plan.config.last_rebalance_time = datetime.utcnow()
        return True


# ===== Demo Functions =====

def setup_test_portfolio(portfolio_manager: SimplePortfolioManager) -> None:
    """Set up a test portfolio."""
    positions = [
        Position(
            symbol="BTC/USD",
            position_type=PositionType.LONG,
            size=Decimal("2"),
            entry_price=Decimal("50000")
        ),
        Position(
            symbol="ETH/USD",
            position_type=PositionType.LONG,
            size=Decimal("40"),
            entry_price=Decimal("3000")
        ),
        Position(
            symbol="SOL/USD",
            position_type=PositionType.LONG,
            size=Decimal("500"),
            entry_price=Decimal("100")
        ),
        Position(
            symbol="AVAX/USD",
            position_type=PositionType.LONG,
            size=Decimal("100"),
            entry_price=Decimal("20")
        ),
    ]
    
    for position in positions:
        position.position_id = f"pos_{position.symbol.split('/')[0].lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        portfolio_manager.positions[position.position_id] = position
        portfolio_manager.open_positions.add(position.position_id)
        
        if position.symbol not in portfolio_manager.open_positions_by_symbol:
            portfolio_manager.open_positions_by_symbol[position.symbol] = set()
        portfolio_manager.open_positions_by_symbol[position.symbol].add(position.position_id)
    
    portfolio_manager.current_prices = {
        "BTC/USD": Decimal("55000"),
        "ETH/USD": Decimal("2900"),
        "SOL/USD": Decimal("110"),
        "AVAX/USD": Decimal("21"),
    }


def display_allocation_status(status: Dict[str, Any]) -> None:
    """Display allocation status."""
    if "error" in status:
        print(f"Error: {status['error']}")
        return
    
    headers = ["Symbol", "Target %", "Current %", "Drift %"]
    rows = []
    
    for allocation in status["allocations"]:
        target_pct = f"{allocation['target_percentage'] * 100:.2f}%"
        current_pct = f"{allocation.get('current_percentage', 0) * 100:.2f}%"
        
        drift = allocation.get('drift')
        if drift is not None:
            drift_pct = f"{drift * 100:+.2f}%"
            if abs(drift) > 0.05:
                drift_pct = f"\033[91m{drift_pct}\033[0m"  # Red
            elif abs(drift) > 0.02:
                drift_pct = f"\033[93m{drift_pct}\033[0m"  # Yellow
        else:
            drift_pct = "N/A"
        
        rows.append([allocation["symbol"], target_pct, current_pct, drift_pct])
    
    if "tabulate" in globals():
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("\t".join(headers))
        print("-" * 60)
        for row in rows:
            print("\t".join(str(cell) for cell in row))


def plot_allocation(status: Dict[str, Any], title: str) -> None:
    """Plot allocation as pie charts."""
    if not VISUALIZATION_AVAILABLE:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    labels = []
    target_sizes = []
    current_sizes = []
    
    for allocation in status["allocations"]:
        labels.append(allocation["symbol"].split("/")[0])
        target_sizes.append(allocation["target_percentage"])
        current_sizes.append(allocation.get("current_percentage", 0) or 0)
    
    ax1.pie(target_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Target Allocation')
    
    ax2.pie(current_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Current Allocation')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    os.makedirs("data/plots", exist_ok=True)
    filename = f"data/plots/allocation_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    logger.info(f"Saved allocation chart to {filename}")
    
    plt.show()


async def run_demo(rebalancer: PortfolioRebalancer) -> None:
    """Run the rebalancing demonstration."""
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    
    # Equal Weight Demo
    logger.info("\n=== Equal Weight Allocation Demo ===")
    plan = rebalancer.create_allocation_plan(
        name="equal_weight",
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
        symbols=symbols
    )
    
    status = await rebalancer.get_allocation_status("equal_weight")
    logger.info("Current allocation (Equal Weight):")
    display_allocation_status(status)
    if VISUALIZATION_AVAILABLE:
        plot_allocation(status, "Equal Weight Allocation")
    
    if await rebalancer.check_rebalance_needed("equal_weight"):
        await rebalancer.execute_rebalance("equal_weight")
    
    # Volatility Weighted Demo
    logger.info("\n=== Volatility Weighted Allocation Demo ===")
    rebalancer.update_market_data("BTC/USD", {"volatility": 0.80})
    rebalancer.update_market_data("ETH/USD", {"volatility": 0.60})
    rebalancer.update_market_data("SOL/USD", {"volatility": 1.20})
    rebalancer.update_market_data("AVAX/USD", {"volatility": 0.40})
    
    plan = rebalancer.create_allocation_plan(
        name="volatility_weighted",
        allocation_method=AllocationMethod.VOLATILITY_WEIGHTED,
        symbols=symbols
    )
    
    status = await rebalancer.get_allocation_status("volatility_weighted")
    logger.info("Current allocation (Volatility Weighted):")
    display_allocation_status(status)
    if VISUALIZATION_AVAILABLE:
        plot_allocation(status, "Volatility Weighted Allocation")
    
    if await rebalancer.check_rebalance_needed("volatility_weighted"):
        await rebalancer.execute_rebalance("volatility_weighted")
    
    # Custom Weight Demo
    logger.info("\n=== Custom Weight Allocation Demo ===")
    config = RebalanceConfig(
        allocation_method=AllocationMethod.CUSTOM,
        trigger_method=RebalanceTrigger.THRESHOLD_BASED,
        custom_weights={
            "BTC/USD": Decimal("0.40"),
            "ETH/USD": Decimal("0.30"),
            "SOL/USD": Decimal("0.20"),
            "AVAX/USD": Decimal("0.10")
        }
    )
    
    plan = rebalancer.create_allocation_plan(
        name="custom_weight",
        allocation_method=AllocationMethod.CUSTOM,
        symbols=symbols,
        config=config
    )
    
    status = await rebalancer.get_allocation_status("custom_weight")
    logger.info("Current allocation (Custom Weight):")
    display_allocation_status(status)
    if VISUALIZATION_AVAILABLE:
        plot_allocation(status, "Custom Weight Allocation")
    
    if await rebalancer.check_rebalance_needed("custom_weight"):
        await rebalancer.execute_rebalance("custom_weight")


async def main():
    """Main function."""
    logger.info("Starting Portfolio Rebalancing Demo")
    
    portfolio_manager = SimplePortfolioManager(initial_balance=Decimal("300000"))
    setup_test_portfolio(portfolio_manager)
    
    rebalancer = PortfolioRebalancer(portfolio_manager)
    await run_demo(rebalancer)
    
    logger.info("Portfolio Rebalancing Demo completed")


if __name__ == "__main__":
    os.makedirs("data/plots", exist_ok=True)
    asyncio.run(main()) 