"""
Fee Management API.

This module provides the API endpoints for the fee management system,
allowing other components of the trading system to interact with it.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union, Any

from src.fees.models import (
    FeeCalculationType,
    FeeDiscount,
    FeeEstimate,
    FeeSchedule,
    FeeSummary,
    FeeTier,
    FeeType,
    TransactionFee,
)
from src.fees.service import FeeManager

logger = logging.getLogger(__name__)


class FeeManagementAPI:
    """
    API for the Fee Management System.
    
    This class provides a simplified interface for other components to interact
    with the fee management system.
    """
    
    def __init__(self):
        """Initialize the Fee Management API."""
        self.fee_manager = FeeManager()
    
    # Fee Schedule Management
    
    def get_fee_schedule(self, exchange_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current fee schedule for an exchange.
        
        Args:
            exchange_id: ID of the exchange
            
        Returns:
            Dictionary representation of the fee schedule, or None if not found
        """
        if exchange_id not in self.fee_manager.fee_schedules:
            return None
        
        schedule = self.fee_manager.fee_schedules[exchange_id]
        
        # Convert to a simple dictionary for API response
        tiers = []
        for tier in schedule.tiers:
            # Create a dictionary with proper type annotations
            tier_dict: Dict[str, Any] = {
                "min_volume": tier.min_volume,
                "maker_fee": tier.maker_fee,
                "taker_fee": tier.taker_fee
            }
            if tier.max_volume is not None:
                tier_dict["max_volume"] = tier.max_volume
            if tier.description:
                # Add description as a string value
                tier_dict["description"] = tier.description
            tiers.append(tier_dict)
        
        return {
            "exchange_id": schedule.exchange_id,
            "default_maker_fee": schedule.default_maker_fee,
            "default_taker_fee": schedule.default_taker_fee,
            "calculation_type": schedule.calculation_type.value,
            "tiers": tiers,
            "withdrawal_fees": schedule.withdrawal_fees,
            "network_fees": schedule.network_fees,
            "updated_at": schedule.updated_at.isoformat()
        }
    
    def update_fee_schedule(self, exchange_id: str, schedule_data: Dict[str, Any]) -> bool:
        """
        Update the fee schedule for an exchange.
        
        Args:
            exchange_id: ID of the exchange
            schedule_data: Dictionary with fee schedule data
            
        Returns:
            True if the update was successful, False otherwise
        """
        try:
            # Convert tiers from dict to FeeTier objects
            tiers = []
            for tier_data in schedule_data.get("tiers", []):
                tiers.append(FeeTier(
                    min_volume=float(tier_data["min_volume"]),
                    max_volume=float(tier_data["max_volume"]) if tier_data.get("max_volume") is not None else None,
                    maker_fee=float(tier_data["maker_fee"]),
                    taker_fee=float(tier_data["taker_fee"]),
                    description=tier_data.get("description")
                ))
            
            # Create the FeeSchedule object
            fee_schedule = FeeSchedule(
                exchange_id=exchange_id,
                default_maker_fee=float(schedule_data["default_maker_fee"]),
                default_taker_fee=float(schedule_data["default_taker_fee"]),
                calculation_type=FeeCalculationType(schedule_data["calculation_type"]),
                tiers=tiers,
                withdrawal_fees={k: float(v) for k, v in schedule_data.get("withdrawal_fees", {}).items()},
                network_fees={k: float(v) for k, v in schedule_data.get("network_fees", {}).items()},
                updated_at=datetime.now()
            )
            
            # Update the fee schedule
            self.fee_manager.update_fee_schedule(fee_schedule)
            return True
        
        except Exception as e:
            logger.error(f"Error updating fee schedule for {exchange_id}: {e}")
            return False
    
    # Fee Discount Management
    
    def get_fee_discounts(self, exchange_id: str) -> List[Dict[str, Any]]:
        """
        Get active fee discounts for an exchange.
        
        Args:
            exchange_id: ID of the exchange
            
        Returns:
            List of dictionaries representing active fee discounts
        """
        if exchange_id not in self.fee_manager.fee_discounts:
            return []
        
        now = datetime.now()
        active_discounts = []
        
        for discount in self.fee_manager.fee_discounts[exchange_id]:
            if discount.expiry is None or discount.expiry > now:
                discount_dict = {
                    "exchange_id": discount.exchange_id,
                    "discount_percentage": discount.discount_percentage,
                    "applies_to": [fee_type.value for fee_type in discount.applies_to],
                    "reason": discount.reason
                }
                if discount.expiry:
                    discount_dict["expiry"] = discount.expiry.isoformat()
                active_discounts.append(discount_dict)
        
        return active_discounts
    
    def add_fee_discount(self, discount_data: Dict[str, Any]) -> bool:
        """
        Add a new fee discount.
        
        Args:
            discount_data: Dictionary with fee discount data
            
        Returns:
            True if the discount was added successfully, False otherwise
        """
        try:
            exchange_id = discount_data["exchange_id"]
            
            # Parse expiry date if provided
            expiry = None
            if discount_data.get("expiry"):
                expiry = datetime.fromisoformat(discount_data["expiry"])
            
            # Create the FeeDiscount object
            fee_discount = FeeDiscount(
                exchange_id=exchange_id,
                discount_percentage=float(discount_data["discount_percentage"]),
                applies_to=[FeeType(ft) for ft in discount_data["applies_to"]],
                reason=discount_data["reason"],
                expiry=expiry
            )
            
            # Add the fee discount
            self.fee_manager.add_fee_discount(fee_discount)
            return True
        
        except Exception as e:
            logger.error(f"Error adding fee discount: {e}")
            return False
    
    # Fee Rate and Estimation
    
    def get_fee_rate(
        self, 
        exchange_id: str, 
        fee_type: str, 
        volume_30d: Optional[float] = None
    ) -> float:
        """
        Get the current fee rate for a specific exchange and fee type.
        
        Args:
            exchange_id: ID of the exchange
            fee_type: Type of fee ("maker" or "taker")
            volume_30d: 30-day trading volume for tier calculation
            
        Returns:
            The fee rate as a decimal (e.g., 0.001 for 0.1%)
        """
        try:
            return self.fee_manager.get_fee_rate(
                exchange_id=exchange_id,
                fee_type=FeeType(fee_type),
                volume_30d=volume_30d
            )
        except Exception as e:
            logger.error(f"Error getting fee rate for {exchange_id}/{fee_type}: {e}")
            return 0.0
    
    def estimate_transaction_fee(
        self,
        exchange_id: str,
        fee_type: str,
        asset: str,
        amount: float,
        price: float,
        volume_30d: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Estimate the fee for a planned transaction.
        
        Args:
            exchange_id: ID of the exchange
            fee_type: Type of fee ("maker", "taker", etc.)
            asset: Asset in which the fee will be paid
            amount: Amount of the base asset in the transaction
            price: Price of the asset in quote currency
            volume_30d: 30-day trading volume for tier calculation
            
        Returns:
            Dictionary with fee estimate details
        """
        try:
            estimate = self.fee_manager.estimate_fee(
                exchange_id=exchange_id,
                fee_type=FeeType(fee_type),
                asset=asset,
                amount=amount,
                price=price,
                volume_30d=volume_30d
            )
            
            # Convert to a dictionary for API response
            result = {
                "exchange_id": estimate.exchange_id,
                "fee_type": estimate.fee_type.value,
                "asset": estimate.asset,
                "estimated_amount": estimate.estimated_amount,
                "calculation_details": estimate.calculation_details
            }
            
            if estimate.usd_value is not None:
                result["usd_value"] = estimate.usd_value
            
            return result
        
        except Exception as e:
            logger.error(f"Error estimating fee for {exchange_id}/{fee_type}: {e}")
            return {
                "exchange_id": exchange_id,
                "fee_type": fee_type,
                "asset": asset,
                "estimated_amount": 0.0,
                "error": str(e)
            }
    
    # Transaction Fee Recording
    
    def record_fee(self, fee_data: Dict[str, Any]) -> bool:
        """
        Record a fee paid for a transaction.
        
        Args:
            fee_data: Dictionary with transaction fee data
            
        Returns:
            True if the fee was recorded successfully, False otherwise
        """
        try:
            # Set transaction time if not provided
            if "transaction_time" not in fee_data:
                fee_data["transaction_time"] = datetime.now()
            else:
                fee_data["transaction_time"] = datetime.fromisoformat(fee_data["transaction_time"])
            
            # Create the TransactionFee object
            transaction_fee = TransactionFee(
                transaction_id=fee_data["transaction_id"],
                exchange_id=fee_data["exchange_id"],
                fee_type=FeeType(fee_data["fee_type"]),
                asset=fee_data["asset"],
                amount=float(fee_data["amount"]),
                usd_value=float(fee_data["usd_value"]) if "usd_value" in fee_data else None,
                transaction_time=fee_data["transaction_time"],
                related_to=fee_data.get("related_to"),
                details=fee_data.get("details", {})
            )
            
            # Record the transaction fee
            self.fee_manager.record_transaction_fee(transaction_fee)
            return True
        
        except Exception as e:
            logger.error(f"Error recording transaction fee: {e}")
            return False
    
    # Fee Analytics
    
    def get_fee_summary(
        self,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        exchange_ids: Optional[List[str]] = None,
        fee_types: Optional[List[str]] = None,
        assets: Optional[List[str]] = None,
        related_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary of fees paid over a time period.
        
        Args:
            start_time: Start time of the period (ISO format string or datetime)
            end_time: End time of the period (ISO format string or datetime)
            exchange_ids: Only include these exchanges (or all if None)
            fee_types: Only include these fee types (or all if None)
            assets: Only include these assets (or all if None)
            related_to: Only include fees related to this entity (e.g., strategy ID)
            
        Returns:
            Dictionary with fee summary details
        """
        try:
            # Convert string times to datetime if needed
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
            
            # Convert fee types to enum values if provided
            fee_type_enums = None
            if fee_types:
                fee_type_enums = {FeeType(ft) for ft in fee_types}
            
            # Convert exchange IDs to a set if provided
            exchange_id_set = None
            if exchange_ids:
                exchange_id_set = set(exchange_ids)
            
            # Convert assets to a set if provided
            asset_set = None
            if assets:
                asset_set = set(assets)
            
            # Get the fee summary
            summary = self.fee_manager.get_fee_summary(
                start_time=start_time,
                end_time=end_time,
                exchange_ids=exchange_id_set,
                fee_types=fee_type_enums,
                assets=asset_set,
                related_to=related_to
            )
            
            # Convert to a dictionary for API response
            result = {
                "start_time": summary.start_time.isoformat(),
                "end_time": summary.end_time.isoformat(),
                "total_fees_usd": summary.total_fees_usd,
                "by_exchange": summary.by_exchange,
                "by_type": {ft.value: amount for ft, amount in summary.by_type.items()},
                "by_asset": summary.by_asset,
                "by_related": summary.by_related
            }
            
            if summary.trade_volume_usd is not None:
                result["trade_volume_usd"] = summary.trade_volume_usd
            if summary.effective_fee_rate is not None:
                result["effective_fee_rate"] = summary.effective_fee_rate
            if summary.details:
                result["details"] = summary.details
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating fee summary: {e}")
            return {
                "error": str(e),
                "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                "end_time": end_time.isoformat() if isinstance(end_time, datetime) else end_time,
                "total_fees_usd": 0.0
            }
    
    # Fee Optimization
    
    def optimize_exchange_allocation(
        self,
        strategies: List[Dict[str, Any]],
        exchanges: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize exchange allocation for strategies based on fee structures.
        
        Args:
            strategies: List of trading strategies with volume and asset details
            exchanges: List of available exchanges to consider
            constraints: Constraints on exchange usage
            
        Returns:
            Dictionary with optimized allocation plan
        """
        try:
            return self.fee_manager.optimize_fees(
                strategies=strategies,
                exchanges=exchanges,
                constraints=constraints
            )
        except Exception as e:
            logger.error(f"Error optimizing exchange allocation: {e}")
            return {
                "error": str(e),
                "allocation": {}
            } 