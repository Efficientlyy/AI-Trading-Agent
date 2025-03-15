"""VWAP (Volume-Weighted Average Price) Execution Algorithm

This module implements the VWAP execution algorithm, which splits a large order
into smaller orders executed according to historical volume profiles.
"""

import asyncio
import math
import uuid
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any

from src.common.logging import get_logger
from src.common.datetime_utils import utc_now
from src.execution.interface import ExchangeInterface
from src.models.order import Order, OrderType, OrderSide, OrderStatus, TimeInForce

logger = get_logger("execution.algorithms.vwap")


class VWAPExecutor:
    """VWAP (Volume-Weighted Average Price) execution algorithm.
    
    This algorithm splits a large order into smaller orders executed according
    to historical volume profiles to match the expected market volume distribution.
    """
    
    def __init__(self, exchange_interface: ExchangeInterface):
        """Initialize the VWAP executor.
        
        Args:
            exchange_interface: The exchange interface to use for order execution
        """
        self.exchange_interface = exchange_interface
        self.active_jobs: Dict[str, Dict[str, Any]] = {}  # Map of job_id to job details
        
        # Default volume profiles for different time frames
        # These represent the percentage of daily volume expected in each hour/minute
        # For a real implementation, these should be derived from historical data
        self._hourly_volume_profile = self._default_hourly_volume_profile()
    
    def _default_hourly_volume_profile(self) -> List[float]:
        """Create a default hourly volume profile.
        
        This is a simplified model of typical daily volume distribution
        across 24 hours. In a real implementation, this would be derived
        from historical data and potentially customized per asset.
        
        Returns:
            List of 24 values representing percentage of daily volume in each hour
        """
        # This is a very simplified model that assumes:
        # - Higher volume at market open and close
        # - Lower volume during off-hours
        return [
            0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 00:00 - 05:59
            0.04, 0.06, 0.08, 0.07, 0.06, 0.05,  # 06:00 - 11:59
            0.04, 0.04, 0.05, 0.06, 0.07, 0.08,  # 12:00 - 17:59
            0.07, 0.05, 0.04, 0.03, 0.02, 0.01   # 18:00 - 23:59
        ]
    
    def _get_volume_profile(self, start_time: datetime, end_time: datetime, num_slices: int) -> List[float]:
        """Generate a volume profile for the given time period.
        
        This method takes the start and end times and creates a volume profile
        with the specified number of slices. It uses the default hourly volume
        profile and interpolates as needed.
        
        Args:
            start_time: Start time for the execution
            end_time: End time for the execution
            num_slices: Number of slices to divide the execution into
            
        Returns:
            List of values representing percentage of volume in each slice
        """
        # Calculate duration in hours
        duration = (end_time - start_time).total_seconds() / 3600
        
        # If duration is less than 24 hours, use a subset of the profile
        if duration <= 24:
            start_hour = start_time.hour
            slice_duration = duration / num_slices
            profile = []
            
            for i in range(num_slices):
                slice_start_hour = (start_hour + i * slice_duration) % 24
                slice_end_hour = (start_hour + (i + 1) * slice_duration) % 24
                
                # Handle case where slice spans midnight
                if slice_end_hour < slice_start_hour:
                    slice_end_hour += 24
                
                # Calculate volume for this slice
                slice_volume = 0.0
                current_hour = math.floor(slice_start_hour)
                while current_hour < math.ceil(slice_end_hour):
                    hour_idx = current_hour % 24
                    next_hour = min(current_hour + 1, math.ceil(slice_end_hour))
                    
                    # Calculate the portion of this hour that falls within the slice
                    hour_portion = min(next_hour, slice_end_hour) - max(current_hour, slice_start_hour)
                    slice_volume += self._hourly_volume_profile[hour_idx] * hour_portion
                    
                    current_hour += 1
                
                profile.append(slice_volume)
            
            # Normalize the profile
            total = sum(profile)
            if total > 0:
                profile = [v / total for v in profile]
            else:
                # Fallback to equal distribution
                profile = [1.0 / num_slices] * num_slices
                
            return profile
        else:
            # For longer durations, fallback to equal distribution
            return [1.0 / num_slices] * num_slices
                
    async def start_vwap(
        self,
        exchange_id: str,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        duration_minutes: int,
        num_slices: int,
        max_price_deviation_percent: float = 0.3,
        use_limit_orders: bool = True,
        limit_price_offset_percent: float = 0.05,
        position_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Start a new VWAP execution job.
        
        Args:
            exchange_id: Exchange ID
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            total_quantity: Total quantity to execute
            duration_minutes: Duration in minutes over which to spread the execution
            num_slices: Number of slices to split the order into
            max_price_deviation_percent: Maximum allowed price deviation from expected as percentage
            use_limit_orders: Whether to use limit orders (true) or market orders (false)
            limit_price_offset_percent: Offset from current price for limit orders as percentage
            position_id: Optional position ID
            strategy_id: Optional strategy ID
            metadata: Optional metadata
            
        Returns:
            Tuple of (success, job_id if successful, error message if failed)
        """
        if num_slices <= 0:
            return False, None, "Number of slices must be positive"
        
        if duration_minutes <= 0:
            return False, None, "Duration must be positive"
        
        if total_quantity <= 0:
            return False, None, "Total quantity must be positive"
        
        # Set up the time window
        start_time = utc_now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Generate volume profile for this time window
        volume_profile = self._get_volume_profile(start_time, end_time, num_slices)
        
        # Calculate quantities for each slice based on volume profile
        slice_quantities = [total_quantity * ratio for ratio in volume_profile]
        
        # Calculate interval between slices
        interval_seconds = (duration_minutes * 60) / num_slices
        
        # Check if any slice is too small (exchange may have minimum order size)
        min_quantity = min(slice_quantities)
        if min_quantity < 0.0001:  # This is a common minimum size, but should be configurable
            return False, None, f"Smallest slice too small: {min_quantity} (minimum is 0.0001)"
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Get current market price to use as reference
        current_price = await self.exchange_interface.get_market_price(exchange_id, symbol)
        if current_price is None:
            return False, None, f"Failed to get current market price for {symbol} on {exchange_id}"
        
        # Create job details
        job = {
            "id": job_id,
            "exchange_id": exchange_id,
            "symbol": symbol,
            "side": side,
            "total_quantity": total_quantity,
            "remaining_quantity": total_quantity,
            "slice_quantities": slice_quantities,
            "slices_total": num_slices,
            "slices_executed": 0,
            "slices_successful": 0,
            "interval_seconds": interval_seconds,
            "use_limit_orders": use_limit_orders,
            "limit_price_offset_percent": limit_price_offset_percent,
            "max_price_deviation_percent": max_price_deviation_percent,
            "reference_price": current_price,
            "start_time": start_time,
            "end_time": end_time,
            "is_active": True,
            "position_id": position_id,
            "strategy_id": strategy_id,
            "metadata": metadata or {},
            "orders": [],
            "errors": [],
            "execution_task": None,
        }
        
        # Store the job
        self.active_jobs[job_id] = job
        
        # Start the execution task
        task = asyncio.create_task(self._execute_vwap(job_id))
        job["execution_task"] = task
        
        logger.info(f"Started VWAP execution", 
                   job_id=job_id, 
                   symbol=symbol, 
                   exchange=exchange_id,
                   total_quantity=total_quantity,
                   duration_minutes=duration_minutes,
                   slices=num_slices)
        
        return True, job_id, None
    
    async def cancel_vwap(self, job_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel an active VWAP execution job.
        
        Args:
            job_id: ID of the VWAP job to cancel
            
        Returns:
            Tuple of (success, error message if failed)
        """
        if job_id not in self.active_jobs:
            return False, f"VWAP job not found: {job_id}"
        
        job = self.active_jobs[job_id]
        
        if not job["is_active"]:
            return False, f"VWAP job already completed or cancelled: {job_id}"
        
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
        
        logger.info(f"Cancelled VWAP execution", 
                   job_id=job_id, 
                   remaining_quantity=job["remaining_quantity"],
                   slices_executed=job["slices_executed"],
                   slices_successful=job["slices_successful"])
        
        return True, None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a VWAP job.
        
        Args:
            job_id: ID of the VWAP job
            
        Returns:
            Dict containing job status or None if job not found
        """
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        
        # Create a copy of the job status without the execution task
        status = {k: v for k, v in job.items() if k != "execution_task"}
        
        # Add some derived fields for convenience
        status["percent_complete"] = (job["slices_executed"] / job["slices_total"]) * 100
        status["average_execution_price"] = self._calculate_average_price(job)
        
        return status
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get the status of all VWAP jobs.
        
        Returns:
            List of job status dictionaries
        """
        return [status for job_id in self.active_jobs 
                if (status := self.get_job_status(job_id)) is not None]
    
    async def _execute_vwap(self, job_id: str) -> None:
        """Execute a VWAP job.
        
        This method runs as a background task and is responsible for
        executing the slices of a VWAP job at regular intervals.
        
        Args:
            job_id: ID of the VWAP job to execute
        """
        job = self.active_jobs[job_id]
        
        try:
            logger.info(f"Beginning VWAP execution", 
                       job_id=job_id, 
                       slices=job["slices_total"], 
                       interval_seconds=job["interval_seconds"])
            
            # Execute slices at regular intervals
            for slice_num in range(job["slices_total"]):
                if not job["is_active"]:
                    logger.info(f"VWAP job cancelled during execution", job_id=job_id)
                    break
                
                # Execute this slice
                await self._execute_slice(job_id, slice_num)
                
                # Update job status
                job["slices_executed"] += 1
                
                # If this is the last slice, we're done
                if slice_num == job["slices_total"] - 1:
                    break
                
                # Wait for the next interval
                await asyncio.sleep(job["interval_seconds"])
            
            # Mark job as completed
            job["is_active"] = False
            
            logger.info(f"VWAP execution completed", 
                       job_id=job_id, 
                       slices_executed=job["slices_executed"],
                       slices_successful=job["slices_successful"],
                       average_price=self._calculate_average_price(job))
            
        except asyncio.CancelledError:
            logger.info(f"VWAP execution cancelled", job_id=job_id)
            raise
            
        except Exception as e:
            logger.error(f"Error in VWAP execution", job_id=job_id, error=str(e))
            job["is_active"] = False
            job["errors"].append(str(e))
    
    async def _execute_slice(self, job_id: str, slice_num: int) -> None:
        """Execute a single slice of a VWAP job.
        
        Args:
            job_id: ID of the VWAP job
            slice_num: Slice number (0-based)
        """
        job = self.active_jobs[job_id]
        
        # Get the quantity for this slice based on the volume profile
        quantity = job["slice_quantities"][slice_num]
        
        # Round quantity to 8 decimal places (common precision for crypto)
        quantity = round(quantity, 8)
        
        # Skip if quantity is too small
        if quantity < 0.0001:
            logger.warning(f"Skipping slice due to small quantity", 
                         job_id=job_id, 
                         slice=slice_num, 
                         quantity=quantity)
            return
        
        # Get current market price
        current_price = await self.exchange_interface.get_market_price(
            job["exchange_id"], job["symbol"]
        )
        
        if current_price is None:
            logger.error(f"Failed to get market price for slice", 
                       job_id=job_id, 
                       slice=slice_num)
            job["errors"].append(f"Failed to get market price for slice {slice_num}")
            return
        
        # Check if price has deviated too much from reference price
        price_deviation = abs(current_price - job["reference_price"]) / job["reference_price"]
        if price_deviation > job["max_price_deviation_percent"] / 100:
            logger.warning(f"Price deviation exceeds maximum", 
                         job_id=job_id, 
                         slice=slice_num,
                         current_price=current_price, 
                         reference_price=job["reference_price"], 
                         deviation_percent=price_deviation * 100)
            job["errors"].append(f"Price deviation exceeds maximum for slice {slice_num}")
            return
        
        # Create client order ID for this slice
        client_order_id = f"vwap_{job_id}_{slice_num}"
        
        # Create order for this slice
        if job["use_limit_orders"]:
            # Calculate limit price with offset
            offset_factor = 1 - (job["limit_price_offset_percent"] / 100) if job["side"] == OrderSide.BUY else 1 + (job["limit_price_offset_percent"] / 100)
            limit_price = current_price * offset_factor
            
            # Round limit price to appropriate precision
            limit_price = round(limit_price, 2)  # Assumes 2 decimal places for price
            
            success, order, error = await self.exchange_interface.create_limit_order(
                exchange_id=job["exchange_id"],
                symbol=job["symbol"],
                side=job["side"],
                quantity=quantity,
                price=limit_price,
                client_order_id=client_order_id,
                time_in_force=TimeInForce.GTC,
                position_id=job["position_id"],
                strategy_id=job["strategy_id"],
                metadata={
                    "vwap_job_id": job_id,
                    "slice_num": slice_num,
                    **job["metadata"]
                }
            )
        else:
            # Use market order
            success, order, error = await self.exchange_interface.create_market_order(
                exchange_id=job["exchange_id"],
                symbol=job["symbol"],
                side=job["side"],
                quantity=quantity,
                client_order_id=client_order_id,
                position_id=job["position_id"],
                strategy_id=job["strategy_id"],
                metadata={
                    "vwap_job_id": job_id,
                    "slice_num": slice_num,
                    **job["metadata"]
                }
            )
        
        if success and order:
            # Store the order in the job
            order_info = {
                "order": order,
                "exchange_order_id": order.exchange_order_id,
                "slice_num": slice_num,
                "quantity": quantity,
                "price": limit_price if job["use_limit_orders"] else current_price,
                "status": order.status,
                "filled_quantity": 0.0,
                "average_price": None,
                "created_at": utc_now()
            }
            job["orders"].append(order_info)
            
            # Update job status for limit orders
            if job["use_limit_orders"]:
                # For limit orders, we'll wait for the order to fill
                await self._wait_for_order_fill(job_id, len(job["orders"]) - 1)
            else:
                # For market orders, we assume they fill immediately
                job["remaining_quantity"] -= quantity
                job["slices_successful"] += 1
                
                # Update order info
                order_info["status"] = OrderStatus.FILLED
                order_info["filled_quantity"] = quantity
                order_info["average_price"] = current_price
            
            logger.info(f"Executed VWAP slice", 
                       job_id=job_id, 
                       slice=slice_num, 
                       quantity=quantity,
                       order_type="limit" if job["use_limit_orders"] else "market",
                       price=limit_price if job["use_limit_orders"] else current_price)
        else:
            # Log the error
            logger.error(f"Failed to execute VWAP slice", 
                       job_id=job_id, 
                       slice=slice_num, 
                       error=error)
            job["errors"].append(f"Failed to execute slice {slice_num}: {error}")
    
    async def _wait_for_order_fill(self, job_id: str, order_index: int) -> None:
        """Wait for a limit order to fill.
        
        This method periodically checks the status of a limit order
        and updates the job status accordingly.
        
        Args:
            job_id: ID of the VWAP job
            order_index: Index of the order in the job's orders list
        """
        job = self.active_jobs[job_id]
        order_info = job["orders"][order_index]
        order = order_info["order"]
        
        # Maximum time to wait for the order to fill (in seconds)
        # For VWAP, we typically don't want to wait too long
        max_wait_time = min(60, job["interval_seconds"] * 0.8)
        start_time = utc_now()
        
        while (utc_now() - start_time).total_seconds() < max_wait_time:
            # Check if job is still active
            if not job["is_active"]:
                return
            
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
            
            order_info["status"] = current_status
            order_info["filled_quantity"] = filled_quantity
            order_info["average_price"] = average_price
            
            # If the order is filled or partially filled, update job status
            if current_status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                # Update remaining quantity
                job["remaining_quantity"] -= filled_quantity
                job["slices_successful"] += 1
                
                if current_status == OrderStatus.FILLED:
                    logger.info(f"Order filled", 
                               job_id=job_id, 
                               order_id=order.exchange_order_id,
                               filled_quantity=filled_quantity,
                               average_price=average_price)
                    return
                
                logger.info(f"Order partially filled", 
                           job_id=job_id, 
                           order_id=order.exchange_order_id,
                           filled_quantity=filled_quantity,
                           average_price=average_price)
            
            # If the order is cancelled, rejected, or expired, log it
            if current_status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                logger.warning(f"Order not filled", 
                             job_id=job_id, 
                             order_id=order.exchange_order_id,
                             status=current_status)
                return
            
            # Wait a second before checking again
            await asyncio.sleep(1)
        
        # If we've reached here, the order hasn't filled within the maximum wait time
        # We'll cancel it and continue with the next slice
        logger.warning(f"Order not filled within maximum wait time, cancelling", 
                     job_id=job_id, 
                     order_id=order.exchange_order_id)
        
        await self.exchange_interface.cancel_order(
            job["exchange_id"], order.exchange_order_id, job["symbol"]
        )
    
    def _calculate_average_price(self, job: Dict[str, Any]) -> Optional[float]:
        """Calculate the average execution price for a job.
        
        Args:
            job: VWAP job dictionary
            
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