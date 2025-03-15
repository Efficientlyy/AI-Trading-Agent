#!/usr/bin/env python
"""Smart Order Routing (SOR) Algorithm

This module provides a Smart Order Routing (SOR) algorithm for optimally routing orders
across multiple exchanges. The SOR algorithm analyzes orderbook liquidity and prices
across exchanges to determine the optimal execution strategy for a given order.

Key features:
- Multi-exchange order routing based on best available prices
- Support for partial order execution across exchanges
- Price impact estimation and optimization
- Configurable routing strategies
- Fee-aware routing decisions
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from decimal import Decimal

from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.execution.interface import ExchangeInterface
from src.models.order import OrderSide, OrderType, TimeInForce

logger = get_logger("execution.algorithms.smart_order_routing")


class SmartOrderRouter:
    """Smart Order Router for optimal order execution across multiple exchanges.
    
    This class routes orders to the exchanges offering the best execution conditions,
    considering factors like available liquidity, price, and fees.
    """
    
    def __init__(self, exchange_interface: ExchangeInterface):
        """Initialize the Smart Order Router.
        
        Args:
            exchange_interface: Initialized ExchangeInterface for interacting with exchanges
        """
        self.exchange_interface = exchange_interface
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        
    async def route_order(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        exchanges: Optional[List[str]] = None,
        max_slippage_percent: float = 0.1,
        consider_fees: bool = True,
        execution_style: str = "aggressive",
        use_limit_orders: bool = False,
        limit_price: Optional[float] = None,
        timeout_seconds: float = 30.0,
        client_order_id: Optional[str] = None,
        position_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Route a single order across multiple exchanges for optimal execution.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            side: Order side (buy/sell)
            total_quantity: Total order quantity
            exchanges: List of exchange IDs to consider (if None, uses all available)
            max_slippage_percent: Maximum allowed slippage as percentage of price
            consider_fees: Whether to factor in exchange fees when routing
            execution_style: Execution style - "aggressive" (faster), "passive" (better price)
            use_limit_orders: Whether to use limit orders instead of market orders
            limit_price: Limit price (required if use_limit_orders=True)
            timeout_seconds: Maximum time to wait for execution to complete
            client_order_id: Optional client ID for the parent order
            position_id: Optional position ID for this order
            strategy_id: Optional strategy ID that created this order
            metadata: Optional metadata for the order
            
        Returns:
            Tuple of (success, job_id, error_message)
        """
        # Validate inputs
        if total_quantity <= 0:
            return False, None, "Total quantity must be positive"
        
        if use_limit_orders and limit_price is None:
            return False, None, "Limit price required when using limit orders"
        
        if execution_style not in ["aggressive", "passive"]:
            return False, None, "Execution style must be 'aggressive' or 'passive'"
        
        # Determine which exchanges to consider
        if exchanges is None:
            exchanges = self.exchange_interface.get_available_exchanges()
        
        if not exchanges:
            return False, None, "No exchanges specified"
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Create a job entry
        self.active_jobs[job_id] = {
            "job_id": job_id,
            "symbol": symbol,
            "side": side,
            "total_quantity": total_quantity,
            "exchanges": exchanges,
            "max_slippage_percent": max_slippage_percent,
            "consider_fees": consider_fees,
            "execution_style": execution_style,
            "use_limit_orders": use_limit_orders,
            "limit_price": limit_price,
            "client_order_id": client_order_id,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "metadata": metadata or {},
            "started_at": utc_now(),
            "is_active": True,
            "is_completed": False,
            "orders": [],
            "status": "starting",
            "filled_quantity": 0.0,
            "remaining_quantity": total_quantity,
            "average_execution_price": None,
            "last_update": utc_now(),
            "error": None
        }
        
        # Start the execution in the background
        asyncio.create_task(self._execute_routing(job_id))
        
        return True, job_id, None
    
    async def cancel_routing(self, job_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel a smart order routing job.
        
        Args:
            job_id: The ID of the job to cancel
            
        Returns:
            Tuple of (success, error_message)
        """
        if job_id not in self.active_jobs:
            return False, f"Job ID {job_id} not found"
        
        job = self.active_jobs[job_id]
        if not job["is_active"]:
            return False, f"Job ID {job_id} is no longer active"
        
        # Mark job as inactive
        job["is_active"] = False
        job["status"] = "cancelling"
        job["last_update"] = utc_now()
        
        # Cancel any open orders
        for order in job["orders"]:
            if order.get("status") in ["new", "partially_filled", "pending"]:
                exchange_id = order.get("exchange_id")
                exchange_order_id = order.get("exchange_order_id")
                
                if exchange_id and exchange_order_id:
                    try:
                        await self.exchange_interface.cancel_order(
                            exchange_id=exchange_id,
                            order_id=exchange_order_id,
                            symbol=job["symbol"]
                        )
                    except Exception as e:
                        logger.error(f"Failed to cancel order {exchange_order_id} on {exchange_id}: {str(e)}")
        
        job["status"] = "cancelled"
        job["is_completed"] = True
        job["last_update"] = utc_now()
        
        logger.info(f"SOR job {job_id} cancelled")
        return True, None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a routing job.
        
        Args:
            job_id: The ID of the job
            
        Returns:
            Dict containing job status or None if not found
        """
        job = self.active_jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job["job_id"],
            "symbol": job["symbol"],
            "side": job["side"],
            "total_quantity": job["total_quantity"],
            "filled_quantity": job["filled_quantity"],
            "remaining_quantity": job["remaining_quantity"],
            "percent_complete": (job["filled_quantity"] / job["total_quantity"]) * 100 if job["total_quantity"] > 0 else 0,
            "average_execution_price": job["average_execution_price"],
            "is_active": job["is_active"],
            "is_completed": job["is_completed"],
            "status": job["status"],
            "started_at": job["started_at"],
            "last_update": job["last_update"],
            "order_count": len(job["orders"]),
            "orders": job["orders"],
            "error": job["error"]
        }
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get status information for all routing jobs.
        
        Returns:
            List of job status dictionaries
        """
        return [status for job_id in self.active_jobs if (status := self.get_job_status(job_id)) is not None]
    
    async def _execute_routing(self, job_id: str) -> None:
        """Execute a smart order routing job.
        
        Args:
            job_id: The ID of the routing job
        """
        if job_id not in self.active_jobs:
            logger.error(f"Job ID {job_id} not found")
            return
        
        job = self.active_jobs[job_id]
        logger.info(f"Starting SOR execution for {job['total_quantity']} {job['symbol']} ({job['side']})")
        
        try:
            job["status"] = "analyzing_liquidity"
            job["last_update"] = utc_now()
            
            # Get execution plan
            execution_plan = await self._create_execution_plan(job)
            if not execution_plan:
                job["status"] = "failed"
                job["is_active"] = False
                job["is_completed"] = True
                job["error"] = "Failed to create execution plan - insufficient liquidity"
                job["last_update"] = utc_now()
                logger.error(f"SOR job {job_id} failed: No valid execution plan found")
                return
            
            # Execute the plan
            job["status"] = "executing"
            job["last_update"] = utc_now()
            
            overall_success = True
            for exchange_id, quantity, price in execution_plan:
                # Check if job has been cancelled
                if not job["is_active"]:
                    logger.info(f"SOR job {job_id} cancelled during execution")
                    return
                
                # Execute the order on this exchange
                success, order, error = await self._execute_order_on_exchange(
                    job=job,
                    exchange_id=exchange_id,
                    quantity=quantity,
                    price=price
                )
                
                if not success:
                    logger.error(f"SOR order execution failed on {exchange_id}: {error}")
                    overall_success = False
                    # Continue with other exchanges to fill as much as possible
            
            # Update job status based on execution result
            if job["remaining_quantity"] <= 0.0001:  # Small threshold to account for floating point errors
                job["status"] = "completed"
                job["is_completed"] = True
                job["is_active"] = False
            elif job["filled_quantity"] > 0:
                job["status"] = "partially_filled"
                job["is_completed"] = True
                job["is_active"] = False
            else:
                job["status"] = "failed"
                job["is_completed"] = True
                job["is_active"] = False
                job["error"] = "Failed to execute any orders in the plan"
            
            # Calculate average execution price if we filled anything
            if job["filled_quantity"] > 0:
                total_value = sum(order.get("filled_quantity", 0) * order.get("average_price", 0) 
                                 for order in job["orders"] if order.get("filled_quantity", 0) > 0)
                job["average_execution_price"] = total_value / job["filled_quantity"]
            
            job["last_update"] = utc_now()
            logger.info(f"SOR job {job_id} completed: filled {job['filled_quantity']}/{job['total_quantity']} "
                       f"at avg price {job['average_execution_price']}")
            
        except Exception as e:
            logger.error(f"Error in SOR execution for job {job_id}: {str(e)}")
            job["status"] = "failed"
            job["is_active"] = False
            job["is_completed"] = True
            job["error"] = f"Execution error: {str(e)}"
            job["last_update"] = utc_now()
    
    async def _create_execution_plan(self, job: Dict[str, Any]) -> List[Tuple[str, float, Optional[float]]]:
        """Create an execution plan by analyzing liquidity across exchanges.
        
        Args:
            job: The routing job
            
        Returns:
            List of tuples (exchange_id, quantity, price) representing the execution plan
        """
        symbol = job["symbol"]
        side = job["side"]
        total_quantity = job["total_quantity"]
        exchanges = job["exchanges"]
        max_slippage_percent = job["max_slippage_percent"]
        consider_fees = job["consider_fees"]
        execution_style = job["execution_style"]
        
        # Collect orderbook data from all exchanges
        orderbook_data = {}
        fee_data = {}
        
        for exchange_id in exchanges:
            # Get orderbook
            orderbook = await self.exchange_interface.get_orderbook_snapshot(exchange_id, symbol)
            if not orderbook:
                logger.warning(f"Failed to get orderbook for {symbol} on {exchange_id}")
                continue
                
            orderbook_data[exchange_id] = orderbook
            
            # Get exchange fees if considering them
            if consider_fees:
                connector = self.exchange_interface.get_connector(exchange_id)
                if connector:
                    try:
                        info = await connector.get_exchange_info()
                        fees = info.get("fees", {})
                        fee_data[exchange_id] = {
                            "maker": float(fees.get("maker", 0)),
                            "taker": float(fees.get("taker", 0))
                        }
                    except Exception as e:
                        logger.warning(f"Failed to get fee information for {exchange_id}: {str(e)}")
                        fee_data[exchange_id] = {"maker": 0.001, "taker": 0.001}  # Default to 0.1%
        
        if not orderbook_data:
            logger.error(f"No orderbook data available for any of the specified exchanges")
            return []
        
        # Prepare liquidity map
        # For buys: we need to use the asks
        # For sells: we need to use the bids
        book_side = "asks" if side == OrderSide.BUY else "bids"
        
        # Aggregate all liquidity across exchanges
        all_liquidity = []
        
        for exchange_id, orderbook in orderbook_data.items():
            entries = orderbook.get(book_side, [])
            
            for price_level, size in entries:
                price = float(price_level)
                quantity = float(size)
                
                # Apply fee adjustment if considering fees
                if consider_fees and exchange_id in fee_data:
                    fee_rate = fee_data[exchange_id]["taker"]  # Use taker fee for market orders
                    
                    if side == OrderSide.BUY:
                        # For buys, fees effectively increase the price
                        adjusted_price = price * (1 + fee_rate)
                    else:
                        # For sells, fees effectively decrease the price
                        adjusted_price = price * (1 - fee_rate)
                else:
                    adjusted_price = price
                
                all_liquidity.append((exchange_id, price, adjusted_price, quantity))
        
        # Sort liquidity by adjusted price (best prices first)
        if side == OrderSide.BUY:
            all_liquidity.sort(key=lambda x: x[2])  # Sort by adjusted price ascending
        else:
            all_liquidity.sort(key=lambda x: x[2], reverse=True)  # Sort by adjusted price descending
        
        # Create execution plan
        execution_plan = []
        remaining_quantity = total_quantity
        
        for exchange_id, actual_price, adjusted_price, available_quantity in all_liquidity:
            if remaining_quantity <= 0:
                break
            
            # Calculate price deviation from the best price
            if all_liquidity:
                best_price = all_liquidity[0][1]  # First entry has the best price
                price_deviation = abs(actual_price - best_price) / best_price
                
                # Skip if price deviation exceeds max slippage
                if price_deviation > max_slippage_percent / 100:
                    continue
            
            # Calculate the quantity to execute at this level
            exec_quantity = min(remaining_quantity, available_quantity)
            
            if exec_quantity > 0:
                execution_plan.append((exchange_id, exec_quantity, actual_price))
                remaining_quantity -= exec_quantity
        
        # If we're using a passive execution style, we might want to further adjust
        # the execution plan to use more limit orders at better prices
        if execution_style == "passive" and job["use_limit_orders"]:
            # For now, we keep this simple. In a real production system,
            # the passive strategy would be more sophisticated.
            pass
        
        return execution_plan
    
    async def _execute_order_on_exchange(
        self, 
        job: Dict[str, Any],
        exchange_id: str,
        quantity: float,
        price: Optional[float]
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Execute an order on a specific exchange.
        
        Args:
            job: The routing job
            exchange_id: Exchange ID
            quantity: Order quantity
            price: Price to use for limit orders (or None for market orders)
            
        Returns:
            Tuple of (success, order info, error message)
        """
        symbol = job["symbol"]
        side = job["side"]
        use_limit_orders = job["use_limit_orders"]
        position_id = job["position_id"]
        strategy_id = job["strategy_id"]
        metadata = job["metadata"].copy()
        
        # Add routing job ID to metadata
        metadata["routing_job_id"] = job["job_id"]
        
        # Generate a client order ID if needed
        client_order_id = f"{job.get('client_order_id', 'sor')}_{exchange_id}_{int(time.time())}"
        
        try:
            # Execute the order
            success = False
            order_info = None
            error = None
            
            if use_limit_orders and price is not None:
                success, order_obj, error = await self.exchange_interface.create_limit_order(
                    exchange_id=exchange_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    client_order_id=client_order_id,
                    time_in_force=TimeInForce.IOC,  # Immediate or Cancel for SOR
                    position_id=position_id,
                    strategy_id=strategy_id,
                    metadata=metadata
                )
            else:
                success, order_obj, error = await self.exchange_interface.create_market_order(
                    exchange_id=exchange_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    client_order_id=client_order_id,
                    position_id=position_id,
                    strategy_id=strategy_id,
                    metadata=metadata
                )
            
            if not success or order_obj is None:
                return False, None, error or "Failed to create order"
            
            # Initial order info
            order_info = {
                "exchange_id": exchange_id,
                "exchange_order_id": order_obj.exchange_order_id,
                "client_order_id": client_order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "status": "pending",
                "filled_quantity": 0.0,
                "average_price": 0.0,
                "created_at": utc_now()
            }
            
            # Add to orders list
            job["orders"].append(order_info)
            
            # Wait for order status
            await asyncio.sleep(0.5)  # Small delay to let the order be processed
            
            # Get order status
            if order_obj.exchange_order_id is None:
                return False, order_info, "Order ID is None, cannot get status"
                
            status = await self.exchange_interface.get_order_status(
                exchange_id=exchange_id,
                order_id=order_obj.exchange_order_id,
                symbol=symbol
            )
            
            if status is None:
                return False, order_info, "Failed to get order status"
            
            # Update order info with status
            order_info["status"] = status.get("status", "unknown")
            order_info["filled_quantity"] = status.get("filled_quantity", 0.0)
            order_info["average_price"] = status.get("average_price", 0.0)
            
            # Update job counters
            job["filled_quantity"] += order_info["filled_quantity"]
            job["remaining_quantity"] = max(0.0, job["total_quantity"] - job["filled_quantity"])
            job["last_update"] = utc_now()
            
            return True, order_info, None
            
        except Exception as e:
            logger.error(f"Error executing order on {exchange_id}: {str(e)}")
            return False, None, str(e) 