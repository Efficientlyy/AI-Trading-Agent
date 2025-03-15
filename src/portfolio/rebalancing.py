"""Portfolio rebalancing module.

This module provides functionality for portfolio rebalancing and asset allocation.
It includes algorithms for determining optimal allocations and generating
the trades needed to achieve target portfolio distributions.
"""

import logging
import asyncio
from datetime import timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
from scipy import optimize

from src.common.datetime_utils import utc_now
from src.common.logging import get_logger
from src.models.portfolio import (
    AllocationMethod, 
    RebalanceTrigger, 
    AssetAllocation,
    RebalanceConfig,
    PortfolioAllocation
)
from src.models.position import Position, PositionSide
from src.portfolio.portfolio_manager import PortfolioManager


logger = get_logger("portfolio", "rebalancing")


class PortfolioRebalancer:
    """
    Portfolio rebalancing engine.
    
    This class manages portfolio rebalancing operations, including:
    - Calculating optimal asset allocations
    - Detecting when rebalancing is needed
    - Generating trade plans to achieve target allocations
    - Executing rebalancing trades with cost minimization
    """
    
    def __init__(self, portfolio_manager: PortfolioManager):
        """
        Initialize the portfolio rebalancer.
        
        Args:
            portfolio_manager: Reference to the portfolio manager
        """
        self.portfolio_manager = portfolio_manager
        self.allocation_plans: Dict[str, PortfolioAllocation] = {}
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        self.last_rebalance_time: Optional[datetime] = None
        self.rebalance_in_progress: bool = False
    
    def create_allocation_plan(
        self, 
        name: str,
        allocation_method: AllocationMethod,
        symbols: List[str],
        config: Optional[RebalanceConfig] = None
    ) -> PortfolioAllocation:
        """
        Create a new portfolio allocation plan.
        
        Args:
            name: Name of the allocation plan
            allocation_method: Method to use for calculating allocations
            symbols: List of asset symbols to include
            config: Optional custom rebalance configuration
            
        Returns:
            Created PortfolioAllocation plan
        """
        if config is None:
            config = RebalanceConfig(
                allocation_method=allocation_method,
                trigger_method=RebalanceTrigger.THRESHOLD_BASED
            )
        
        # Create the allocation plan
        plan = PortfolioAllocation(
            name=name,
            config=config,
            allocations=[]
        )
        
        # Calculate initial allocations based on the selected method
        self._calculate_allocations(plan, symbols)
        
        # Store the plan
        self.allocation_plans[name] = plan
        logger.info(f"Created allocation plan '{name}' with {len(symbols)} assets")
        
        return plan
    
    def _calculate_allocations(self, plan: PortfolioAllocation, symbols: List[str]) -> None:
        """
        Calculate asset allocations based on the plan's allocation method.
        
        Args:
            plan: The portfolio allocation plan
            symbols: List of asset symbols to allocate
        """
        allocation_method = plan.config.allocation_method
        allocations = []
        
        if allocation_method == AllocationMethod.EQUAL_WEIGHT:
            # Equal weight allocation
            weight = Decimal("1") / Decimal(str(len(symbols)))
            for symbol in symbols:
                allocations.append(AssetAllocation(
                    symbol=symbol,
                    target_percentage=weight,
                    min_percentage=Decimal("0"),
                    max_percentage=Decimal("1")
                ))
                
        elif allocation_method == AllocationMethod.CUSTOM:
            # Custom weight allocation
            for symbol in symbols:
                weight = plan.config.custom_weights.get(symbol, Decimal("0"))
                allocations.append(AssetAllocation(
                    symbol=symbol,
                    target_percentage=weight,
                    min_percentage=Decimal("0"),
                    max_percentage=Decimal("1")
                ))
                
        elif allocation_method == AllocationMethod.VOLATILITY_WEIGHTED:
            # Inverse volatility weighting requires market data
            volatilities = {}
            total_inverse_vol = Decimal("0")
            
            # Fallback to equal weight if no market data available
            if not self.market_data_cache:
                logger.warning("No market data available for volatility-weighted allocation, using equal weight")
                weight = Decimal("1") / Decimal(str(len(symbols)))
                for symbol in symbols:
                    allocations.append(AssetAllocation(
                        symbol=symbol,
                        target_percentage=weight,
                        min_percentage=Decimal("0"),
                        max_percentage=Decimal("1")
                    ))
            else:
                # Calculate volatility for each symbol
                for symbol in symbols:
                    if symbol in self.market_data_cache and 'volatility' in self.market_data_cache[symbol]:
                        vol = Decimal(str(self.market_data_cache[symbol]['volatility']))
                        if vol <= Decimal("0"):  # Handle zero or negative volatility
                            vol = Decimal("0.0001")  # Small non-zero value
                        inverse_vol = Decimal("1") / vol
                        volatilities[symbol] = inverse_vol
                        total_inverse_vol += inverse_vol
                    else:
                        # Use a default volatility if data not available
                        logger.warning(f"No volatility data for {symbol}, using default")
                        inverse_vol = Decimal("1")
                        volatilities[symbol] = inverse_vol
                        total_inverse_vol += inverse_vol
                
                # Calculate weights
                for symbol in symbols:
                    weight = volatilities[symbol] / total_inverse_vol if total_inverse_vol > 0 else Decimal("0")
                    allocations.append(AssetAllocation(
                        symbol=symbol,
                        target_percentage=weight,
                        min_percentage=Decimal("0"),
                        max_percentage=Decimal("1")
                    ))
        
        elif allocation_method == AllocationMethod.PERFORMANCE_WEIGHTED:
            # Performance-based weighting
            performances = {}
            total_performance = Decimal("0")
            
            # Fallback to equal weight if no performance data available
            if not self.market_data_cache:
                logger.warning("No performance data available, using equal weight")
                weight = Decimal("1") / Decimal(str(len(symbols)))
                for symbol in symbols:
                    allocations.append(AssetAllocation(
                        symbol=symbol,
                        target_percentage=weight,
                        min_percentage=Decimal("0"),
                        max_percentage=Decimal("1")
                    ))
            else:
                # Calculate performance score for each symbol
                for symbol in symbols:
                    if symbol in self.market_data_cache and 'performance_score' in self.market_data_cache[symbol]:
                        score = Decimal(str(self.market_data_cache[symbol]['performance_score']))
                        if score <= Decimal("0"):  # Handle negative performance
                            score = Decimal("0.0001")  # Small non-zero value
                        performances[symbol] = score
                        total_performance += score
                    else:
                        # Use a default score if data not available
                        logger.warning(f"No performance data for {symbol}, using default")
                        score = Decimal("1")
                        performances[symbol] = score
                        total_performance += score
                
                # Calculate weights
                for symbol in symbols:
                    weight = performances[symbol] / total_performance if total_performance > 0 else Decimal("0")
                    allocations.append(AssetAllocation(
                        symbol=symbol,
                        target_percentage=weight,
                        min_percentage=Decimal("0"),
                        max_percentage=Decimal("1")
                    ))
        
        elif allocation_method == AllocationMethod.RISK_PARITY:
            # Risk parity allocation would ideally use covariance matrix
            # For simplicity, we'll use individual volatilities as approximation
            logger.info("Using simplified risk parity allocation (based on individual volatilities)")
            self._calculate_allocations(plan, symbols)  # Fallback to volatility-weighted
            return
            
        elif allocation_method == AllocationMethod.MARKET_CAP_WEIGHT:
            # Market cap weighted allocation
            market_caps = {}
            total_market_cap = Decimal("0")
            
            # Fallback to equal weight if no market cap data available
            if not self.market_data_cache:
                logger.warning("No market cap data available, using equal weight")
                weight = Decimal("1") / Decimal(str(len(symbols)))
                for symbol in symbols:
                    allocations.append(AssetAllocation(
                        symbol=symbol,
                        target_percentage=weight,
                        min_percentage=Decimal("0"),
                        max_percentage=Decimal("1")
                    ))
            else:
                # Get market cap for each symbol
                for symbol in symbols:
                    if symbol in self.market_data_cache and 'market_cap' in self.market_data_cache[symbol]:
                        cap = Decimal(str(self.market_data_cache[symbol]['market_cap']))
                        market_caps[symbol] = cap
                        total_market_cap += cap
                    else:
                        # Use a default value if market cap not available
                        logger.warning(f"No market cap data for {symbol}, using default")
                        cap = Decimal("1")
                        market_caps[symbol] = cap
                        total_market_cap += cap
                
                # Calculate weights
                for symbol in symbols:
                    weight = market_caps[symbol] / total_market_cap if total_market_cap > 0 else Decimal("0")
                    allocations.append(AssetAllocation(
                        symbol=symbol,
                        target_percentage=weight,
                        min_percentage=Decimal("0"),
                        max_percentage=Decimal("1")
                    ))
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(allocation.target_percentage for allocation in allocations)
        if total_weight > 0:
            for allocation in allocations:
                allocation.target_percentage = allocation.target_percentage / total_weight
        
        # Apply constraints from config
        for allocation in allocations:
            symbol = allocation.symbol
            constraints = plan.config.asset_constraints.get(symbol, {})
            
            if 'min' in constraints:
                allocation.min_percentage = Decimal(str(constraints['min']))
            
            if 'max' in constraints:
                allocation.max_percentage = Decimal(str(constraints['max']))
        
        # Set the allocations on the plan
        plan.allocations = allocations
    
    def update_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Update market data for use in allocation calculations.
        
        Args:
            symbol: Asset symbol
            data: Market data dictionary with metrics like volatility, performance, etc.
        """
        self.market_data_cache[symbol] = data
    
    async def check_rebalance_needed(self, plan_name: str) -> bool:
        """
        Check if rebalancing is needed for a specific allocation plan.
        
        Args:
            plan_name: Name of the allocation plan
            
        Returns:
            True if rebalancing is needed, False otherwise
        """
        if plan_name not in self.allocation_plans:
            logger.warning(f"Allocation plan '{plan_name}' not found")
            return False
        
        plan = self.allocation_plans[plan_name]
        
        # Get current portfolio positions and update current allocations
        await self._update_current_allocations(plan)
        
        # Check if rebalance is needed based on plan configuration
        return plan.is_rebalance_needed()
    
    async def _update_current_allocations(self, plan: PortfolioAllocation) -> None:
        """
        Update current allocation percentages in the plan based on actual positions.
        
        Args:
            plan: Portfolio allocation plan to update
        """
        # Get portfolio state
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        total_value = portfolio_state.get("total_value", Decimal("0"))
        
        if total_value <= Decimal("0"):
            logger.warning("Portfolio total value is zero or negative, cannot calculate allocations")
            return
        
        # Get current positions and their values
        positions = self.portfolio_manager.get_open_positions()
        position_values: Dict[str, Decimal] = {}
        
        # Group positions by symbol and sum their values
        for position in positions:
            symbol = position.symbol
            current_price = self.portfolio_manager.current_prices.get(symbol, Decimal("0"))
            
            if current_price > Decimal("0"):
                position_value = position.size * current_price
                position_values[symbol] = position_values.get(symbol, Decimal("0")) + position_value
        
        # Update current allocations
        for allocation in plan.allocations:
            symbol = allocation.symbol
            current_value = position_values.get(symbol, Decimal("0"))
            allocation.current_percentage = current_value / total_value
    
    async def generate_rebalance_plan(self, plan_name: str) -> Dict[str, Decimal]:
        """
        Generate a rebalancing plan with trades needed to achieve target allocation.
        
        Args:
            plan_name: Name of the allocation plan
            
        Returns:
            Dictionary mapping symbols to trade amounts (positive = buy, negative = sell)
        """
        if plan_name not in self.allocation_plans:
            logger.warning(f"Allocation plan '{plan_name}' not found")
            return {}
        
        plan = self.allocation_plans[plan_name]
        
        # Update current allocations
        await self._update_current_allocations(plan)
        
        # Get portfolio state
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        total_value = portfolio_state.get("total_value", Decimal("0"))
        
        if total_value <= Decimal("0"):
            logger.warning("Portfolio total value is zero or negative, cannot generate rebalance plan")
            return {}
        
        # Get current positions grouped by symbol
        positions = self.portfolio_manager.get_open_positions()
        position_values: Dict[str, Decimal] = {}
        
        for position in positions:
            symbol = position.symbol
            current_price = self.portfolio_manager.current_prices.get(symbol, Decimal("0"))
            
            if current_price > Decimal("0"):
                position_value = position.size * current_price
                position_values[symbol] = position_values.get(symbol, Decimal("0")) + position_value
        
        # Calculate trades needed
        trades = plan.calculate_rebalance_trades(position_values, total_value)
        
        # If fee optimization is enabled, try to minimize trading costs
        if plan.config.minimize_fees:
            trades = self._optimize_trades_for_fees(trades, position_values, total_value, plan)
        
        # Log the rebalance plan
        logger.info(f"Generated rebalance plan for '{plan_name}':")
        for symbol, amount in trades.items():
            action = "BUY" if amount > 0 else "SELL"
            logger.info(f"  {action} {abs(amount)} of {symbol}")
        
        return trades
    
    def _optimize_trades_for_fees(
        self,
        initial_trades: Dict[str, Decimal],
        current_positions: Dict[str, Decimal],
        total_value: Decimal,
        plan: PortfolioAllocation
    ) -> Dict[str, Decimal]:
        """
        Optimize trades to minimize fees while achieving allocation targets.
        
        Args:
            initial_trades: Initial trade plan
            current_positions: Current position values by symbol
            total_value: Total portfolio value
            plan: Portfolio allocation plan
            
        Returns:
            Optimized trades dictionary
        """
        # This is a simplified fee optimization
        # In a more sophisticated implementation, this would consider:
        # - Actual fee structures for each exchange
        # - Prioritizing trades that have the most impact on allocation
        # - Potentially omitting small trades that cost more in fees than drift impact
        
        optimized_trades = {}
        min_trade_amount = plan.config.min_trade_amount
        
        # Sort trades by absolute amount, largest first
        sorted_trades = sorted(
            initial_trades.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Skip trades below minimum amount
        for symbol, amount in sorted_trades:
            if abs(amount) < min_trade_amount:
                logger.info(f"Skipping small trade of {amount} {symbol} (below minimum {min_trade_amount})")
                continue
            
            optimized_trades[symbol] = amount
        
        return optimized_trades
    
    async def execute_rebalance(self, plan_name: str) -> bool:
        """
        Execute a portfolio rebalance based on the allocation plan.
        
        Args:
            plan_name: Name of the allocation plan
            
        Returns:
            True if rebalance executed successfully, False otherwise
        """
        if self.rebalance_in_progress:
            logger.warning("Rebalance already in progress, skipping")
            return False
        
        if plan_name not in self.allocation_plans:
            logger.warning(f"Allocation plan '{plan_name}' not found")
            return False
        
        try:
            self.rebalance_in_progress = True
            
            # Generate the rebalance plan
            trades = await self.generate_rebalance_plan(plan_name)
            
            if not trades:
                logger.info(f"No trades required for rebalancing '{plan_name}'")
                return True
            
            # Execute trades
            # In a real implementation, this would call the order management system
            # For now, we'll just log the trades
            logger.info(f"Executing rebalance for plan '{plan_name}':")
            for symbol, amount in trades.items():
                if amount > 0:
                    logger.info(f"  Buying {amount} of {symbol}")
                    # Here we would place buy orders
                else:
                    logger.info(f"  Selling {abs(amount)} of {symbol}")
                    # Here we would place sell orders
            
            # Update the last rebalance time
            self.last_rebalance_time = utc_now()
            
            # Add a historical snapshot
            plan = self.allocation_plans[plan_name]
            plan.add_historical_snapshot()
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing rebalance: {e}")
            return False
        
        finally:
            self.rebalance_in_progress = False
    
    async def get_allocation_status(self, plan_name: str) -> Dict[str, Any]:
        """
        Get the current status of an allocation plan.
        
        Args:
            plan_name: Name of the allocation plan
            
        Returns:
            Dictionary with allocation status information
        """
        if plan_name not in self.allocation_plans:
            logger.warning(f"Allocation plan '{plan_name}' not found")
            return {"error": "Allocation plan not found"}
        
        plan = self.allocation_plans[plan_name]
        
        # Update current allocations
        await self._update_current_allocations(plan)
        
        # Prepare the status report
        allocation_status = []
        for allocation in plan.allocations:
            status = {
                "symbol": allocation.symbol,
                "target_percentage": float(allocation.target_percentage),
                "current_percentage": float(allocation.current_percentage) if allocation.current_percentage is not None else None,
                "drift": float(allocation.drift) if allocation.drift is not None else None
            }
            allocation_status.append(status)
        
        # Calculate max drift
        max_drift = max((abs(a.drift) for a in plan.allocations if a.drift is not None), default=0)
        
        # Calculate overall metrics
        total_current = sum((a.current_percentage for a in plan.allocations if a.current_percentage is not None), 
                           Decimal("0"))
        total_target = sum((a.target_percentage for a in plan.allocations), Decimal("0"))
        
        # Prepare the complete status
        status = {
            "plan_name": plan_name,
            "allocations": allocation_status,
            "max_drift": float(max_drift),
            "total_current_allocation": float(total_current),
            "total_target_allocation": float(total_target),
            "last_rebalance": self.last_rebalance_time.isoformat() if self.last_rebalance_time else None,
            "needs_rebalance": plan.is_rebalance_needed(),
            "rebalance_in_progress": self.rebalance_in_progress
        }
        
        return status 