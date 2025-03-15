"""Iceberg Order Execution Algorithm

This module implements the Iceberg order execution algorithm, which executes
large orders by showing only a small portion of the total order quantity at a time,
thus concealing the true order size from the market.
"""

import asyncio
import uuid
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any

from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.execution.interface import ExchangeInterface
from src.models.order import Order, OrderType, OrderSide, OrderStatus, TimeInForce

logger = get_logger("execution.algorithms.iceberg")


class IcebergExecutor:
    """Iceberg order execution algorithm.
    
    This algorithm executes large orders by showing only a small portion at a time,
    thus concealing the true order size from the market to minimize market impact.
    Each time a "tip" order is filled, another one is placed until the full quantity
    is executed.
    """
    
    def __init__(self, exchange_interface: ExchangeInterface):
        """Initialize the Iceberg executor.
        
        Args:
            exchange_interface: The exchange interface to use for order execution
        """
        self.exchange_interface = exchange_interface
        self.active_jobs: Dict[str, Dict[str, Any]] = {}  # Map of job_id to job details
    
    async def start_iceberg(
        self,
        exchange_id: str,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        visible_quantity: float,
        price: Optional[float] = None,
        max_price_deviation_percent: float = 0.2,
        min_execution_interval_seconds: float = 1.0,
        max_retry_attempts: int = 10,
        position_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Start a new Iceberg execution job.
        
        Args:
            exchange_id: Exchange ID
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            total_quantity: Total quantity to execute
            visible_quantity: Visible quantity for each slice (the "tip" of the iceberg)
            price: Optional limit price (if None, use market price plus offset)
            max_price_deviation_percent: Maximum allowed price deviation as percentage
            min_execution_interval_seconds: Minimum interval between slice executions
            max_retry_attempts: Maximum retry attempts per slice
            position_id: Optional position ID
            strategy_id: Optional strategy ID
            metadata: Optional metadata
            
        Returns:
            Tuple of (success, job_id if successful, error message if failed)
        """
        if total_quantity <= 0:
            return False, None, "Total quantity must be positive"
        
        if visible_quantity <= 0:
            return False, None, "Visible quantity must be positive"
        
        if visible_quantity > total_quantity:
            return False, None, "Visible quantity cannot exceed total quantity"
        
        # Check if visible quantity is too small (exchange may have minimum order size)
        if visible_quantity < 0.0001:  # This is a common minimum size, but should be configurable
            return False, None, f"Visible quantity too small: {visible_quantity} (minimum is 0.0001)"
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Get current market price if no price specified
        current_price = None
        if price is None:
            current_price = await self.exchange_interface.get_market_price(exchange_id, symbol)
            if current_price is None:
                return False, None, f"Failed to get current market price for {symbol} on {exchange_id}"
            
            # For buys, use slightly higher price, for sells use slightly lower price
            # to increase chances of fills
            offset_percent = 0.05  # 0.05% offset
            offset_factor = 1 + (offset_percent / 100) if side == OrderSide.BUY else 1 - (offset_percent / 100)
            price = current_price * offset_factor
            
            # Round price to appropriate precision
            price = round(price, 2)  # Assumes 2 decimal places for price
        
        # Create job details
        job = {
            "id": job_id,
            "exchange_id": exchange_id,
            "symbol": symbol,
            "side": side,
            "total_quantity": total_quantity,
            "remaining_quantity": total_quantity,
            "visible_quantity": visible_quantity,
            "price": price,
            "reference_price": current_price or price,
            "max_price_deviation_percent": max_price_deviation_percent,
            "min_execution_interval_seconds": min_execution_interval_seconds,
            "max_retry_attempts": max_retry_attempts,
            "start_time": utc_now(),
            "is_active": True,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "metadata": metadata or {},
            "slices_executed": 0,
            "slices_successful": 0,
            "orders": [],
            "errors": [],
            "execution_task": None,
        }
        
        # Store the job
        self.active_jobs[job_id] = job
        
        # Start the execution task
        task = asyncio.create_task(self._execute_iceberg(job_id))
        job["execution_task"] = task
        
        logger.info(f"Started Iceberg execution", 
                   job_id=job_id, 
                   symbol=symbol, 
                   exchange=exchange_id,
                   total_quantity=total_quantity,
                   visible_quantity=visible_quantity,
                   price=price)
        
        return True, job_id, None
    
    async def cancel_iceberg(self, job_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel an active Iceberg execution job.
        
        Args:
            job_id: ID of the Iceberg job to cancel
            
        Returns:
            Tuple of (success, error message if failed)
        """
        if job_id not in self.active_jobs:
            return False, f"Iceberg job not found: {job_id}"
        
        job = self.active_jobs[job_id]
        
        if not job["is_active"]:
            return False, f"Iceberg job already completed or cancelled: {job_id}"
        
        # Cancel execution task
        if job["execution_task"] and not job["execution_task"].done():
            job["execution_task"].cancel()
        
        # Mark job as inactive
        job["is_active"] = False
        
        # Cancel any open orders
        open_orders = []
        for order in job["orders"]:
            if order["status"] in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                open_orders.append(order)
        
        # Cancel all open orders
        for order in open_orders:
            exchange_id = job["exchange_id"]
            order_id = order["exchange_order_id"]
            symbol = job["symbol"]
            
            if order_id:
                success, error = await self.exchange_interface.cancel_order(
                    exchange_id, order_id, symbol
                )
                
                if not success:
                    logger.error(f"Failed to cancel order", 
                               job_id=job_id, 
                               order_id=order_id, 
                               error=error)
        
        logger.info(f"Cancelled Iceberg execution", 
                   job_id=job_id, 
                   remaining_quantity=job["remaining_quantity"],
                   slices_executed=job["slices_executed"],
                   slices_successful=job["slices_successful"])
        
        return True, None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an Iceberg job.
        
        Args:
            job_id: ID of the Iceberg job
            
        Returns:
            Dict containing job status or None if job not found
        """
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        
        # Create a copy of the job status without the execution task
        status = {k: v for k, v in job.items() if k != "execution_task"}
        
        # Add some derived fields for convenience
        status["percent_complete"] = ((job["total_quantity"] - job["remaining_quantity"]) / job["total_quantity"]) * 100
        status["average_execution_price"] = self._calculate_average_price(job)
        
        return status
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get the status of all Iceberg jobs.
        
        Returns:
            List of job status dictionaries
        """
        return [status for job_id in self.active_jobs 
                if (status := self.get_job_status(job_id)) is not None]
    
    async def _execute_iceberg(self, job_id: str) -> None:
        """Execute an Iceberg job.
        
        This method runs as a background task and is responsible for
        executing the slices of an Iceberg job until the entire order is filled.
        
        Args:
            job_id: ID of the Iceberg job to execute
        """
        job = self.active_jobs[job_id]
        
        try:
            logger.info(f"Beginning Iceberg execution", 
                       job_id=job_id, 
                       total_quantity=job["total_quantity"], 
                       visible_quantity=job["visible_quantity"])
            
            retry_attempts = 0
            
            # Keep placing orders until the entire quantity is filled
            while job["remaining_quantity"] > 0 and job["is_active"]:
                # Check if we've exceeded max retry attempts
                if retry_attempts >= job["max_retry_attempts"]:
                    logger.error(f"Max retry attempts exceeded", job_id=job_id)
                    job["errors"].append("Max retry attempts exceeded")
                    break
                
                # Get current market price
                current_price = await self.exchange_interface.get_market_price(
                    job["exchange_id"], job["symbol"]
                )
                
                if current_price is None:
                    logger.error(f"Failed to get market price", job_id=job_id)
                    job["errors"].append("Failed to get market price")
                    retry_attempts += 1
                    await asyncio.sleep(1)  # Wait before retrying
                    continue
                
                # Check if price has deviated too much from reference price
                price_deviation = abs(current_price - job["reference_price"]) / job["reference_price"]
                if price_deviation > job["max_price_deviation_percent"] / 100:
                    logger.warning(f"Price deviation exceeds maximum", 
                                 job_id=job_id,
                                 current_price=current_price, 
                                 reference_price=job["reference_price"], 
                                 deviation_percent=price_deviation * 100)
                    job["errors"].append("Price deviation exceeds maximum")
                    retry_attempts += 1
                    await asyncio.sleep(1)  # Wait before retrying
                    continue
                
                # Calculate the quantity for this slice
                visible_quantity = min(job["visible_quantity"], job["remaining_quantity"])
                
                # Round quantity to 8 decimal places (common precision for crypto)
                visible_quantity = round(visible_quantity, 8)
                
                # Execute a slice
                success = await self._execute_slice(job_id, visible_quantity)
                
                if success:
                    retry_attempts = 0  # Reset retry counter on success
                else:
                    retry_attempts += 1
                
                # Wait a bit before the next slice
                await asyncio.sleep(job["min_execution_interval_seconds"])
            
            # Mark job as completed
            job["is_active"] = False
            
            logger.info(f"Iceberg execution completed", 
                       job_id=job_id, 
                       total_executed=job["total_quantity"] - job["remaining_quantity"],
                       slices_executed=job["slices_executed"],
                       slices_successful=job["slices_successful"],
                       average_price=self._calculate_average_price(job))
            
        except asyncio.CancelledError:
            logger.info(f"Iceberg execution cancelled", job_id=job_id)
            raise
            
        except Exception as e:
            logger.error(f"Error in Iceberg execution", job_id=job_id, error=str(e))
            job["is_active"] = False
            job["errors"].append(str(e))
    
    async def _execute_slice(self, job_id: str, quantity: float) -> bool:
        """Execute a single slice of an Iceberg job.
        
        Args:
            job_id: ID of the Iceberg job
            quantity: Quantity to execute in this slice
            
        Returns:
            True if slice was successfully executed, False otherwise
        """
        job = self.active_jobs[job_id]
        
        # Skip if quantity is too small
        if quantity < 0.0001:
            logger.warning(f"Skipping slice due to small quantity", 
                         job_id=job_id, 
                         quantity=quantity)
            return False
        
        # Create client order ID for this slice
        client_order_id = f"iceberg_{job_id}_{job['slices_executed']}"
        
        # Create limit order for this slice
        success, order, error = await self.exchange_interface.create_limit_order(
            exchange_id=job["exchange_id"],
            symbol=job["symbol"],
            side=job["side"],
            quantity=quantity,
            price=job["price"],
            client_order_id=client_order_id,
            time_in_force=TimeInForce.GTC,
            position_id=job["position_id"],
            strategy_id=job["strategy_id"],
            metadata={
                "iceberg_job_id": job_id,
                "slice_num": job["slices_executed"],
                **job["metadata"]
            }
        )
        
        # Update job status
        job["slices_executed"] += 1
        
        if success and order:
            # Store the order in the job
            order_info = {
                "order": order,
                "exchange_order_id": order.exchange_order_id,
                "slice_num": job["slices_executed"] - 1,
                "quantity": quantity,
                "price": job["price"],
                "status": order.status,
                "filled_quantity": 0.0,
                "average_price": None,
                "created_at": utc_now()
            }
            job["orders"].append(order_info)
            
            # Wait for the order to fill or get cancelled
            fill_success = await self._wait_for_order_fill(job_id, len(job["orders"]) - 1)
            
            if fill_success:
                # Update job status
                job["slices_successful"] += 1
                logger.info(f"Executed Iceberg slice", 
                           job_id=job_id, 
                           slice=job["slices_executed"] - 1, 
                           quantity=quantity,
                           remaining=job["remaining_quantity"])
                return True
            else:
                logger.warning(f"Iceberg slice not fully filled", 
                             job_id=job_id, 
                             slice=job["slices_executed"] - 1)
                return False
        else:
            # Log the error
            logger.error(f"Failed to execute Iceberg slice", 
                       job_id=job_id, 
                       slice=job["slices_executed"] - 1, 
                       error=error)
            job["errors"].append(f"Failed to execute slice {job['slices_executed'] - 1}: {error}")
            return False
    
    async def _wait_for_order_fill(self, job_id: str, order_index: int) -> bool:
        """Wait for a limit order to fill.
        
        This method periodically checks the status of a limit order
        and updates the job status accordingly.
        
        Args:
            job_id: ID of the Iceberg job
            order_index: Index of the order in the job's orders list
            
        Returns:
            True if order was filled, False otherwise
        """
        job = self.active_jobs[job_id]
        order_info = job["orders"][order_index]
        order = order_info["order"]
        
        # Maximum time to wait for the order to fill (in seconds)
        # For Iceberg, we typically want to wait longer for fills
        max_wait_time = min(60, job["min_execution_interval_seconds"] * 0.8)
        start_time = utc_now()
        
        while (utc_now() - start_time).total_seconds() < max_wait_time:
            # Check if job is still active
            if not job["is_active"]:
                return False
            
            # Get the current order status
            order_status = await self.exchange_interface.get_order_status(
                job["exchange_id"], order.exchange_order_id, job["symbol"]
            )
            
            if order_status is None:
                logger.warning(f"Failed to get order status", 
                             job_id=job_id, 
                             order_id=order.exchange_order_id)
                await asyncio.sleep(1)
                continue
            
            # Update order info
            current_status = order_status.get("status")
            filled_quantity = float(order_status.get("filled", 0))
            average_price = float(order_status.get("price", order_info["price"]))
            
            # Update the filled quantity only if it has increased
            # to avoid counting the same fills twice
            if filled_quantity > order_info["filled_quantity"]:
                job["remaining_quantity"] -= (filled_quantity - order_info["filled_quantity"])
                order_info["filled_quantity"] = filled_quantity
            
            order_info["status"] = current_status
            order_info["average_price"] = average_price
            
            # If the order is filled, we're done with this slice
            if current_status == OrderStatus.FILLED:
                logger.info(f"Order filled", 
                           job_id=job_id, 
                           order_id=order.exchange_order_id,
                           filled_quantity=filled_quantity,
                           average_price=average_price)
                return True
            
            # If the order is partially filled, we'll wait for more fills
            if current_status == OrderStatus.PARTIALLY_FILLED:
                logger.info(f"Order partially filled", 
                           job_id=job_id, 
                           order_id=order.exchange_order_id,
                           filled_quantity=filled_quantity,
                           average_price=average_price)
            
            # If the order is cancelled, rejected, or expired, we'll move on
            if current_status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                logger.warning(f"Order not filled", 
                             job_id=job_id, 
                             order_id=order.exchange_order_id,
                             status=current_status)
                return False
            
            # Wait a second before checking again
            await asyncio.sleep(1)
        
        # If we've reached here, the order hasn't filled within the maximum wait time
        # We'll cancel it and move on
        logger.warning(f"Order not filled within maximum wait time, cancelling", 
                     job_id=job_id, 
                     order_id=order.exchange_order_id)
        
        await self.exchange_interface.cancel_order(
            job["exchange_id"], order.exchange_order_id, job["symbol"]
        )
        
        return False
    
    def _calculate_average_price(self, job: Dict[str, Any]) -> Optional[float]:
        """Calculate the average execution price for a job.
        
        Args:
            job: Iceberg job dictionary
            
        Returns:
            Average execution price or None if no orders filled
        """
        total_filled = 0.0
        total_value = 0.0
        
        for order_info in job["orders"]:
            if order_info["average_price"] is not None and order_info["filled_quantity"] > 0:
                total_filled += order_info["filled_quantity"]
                total_value += order_info["filled_quantity"] * order_info["average_price"]
        
        if total_filled > 0:
            return total_value / total_filled
        
        return None 