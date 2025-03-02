"""
Fee Management Service.

This module provides the core functionality for tracking, calculating, and analyzing
trading fees across different exchanges and trading strategies.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

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

logger = logging.getLogger(__name__)


class FeeManager:
    """
    Manages the calculation, tracking, and optimization of trading fees.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the FeeManager.
        
        Args:
            data_dir: Directory for storing fee data. If None, a default location is used.
        """
        self.data_dir = data_dir or Path("./data/fees")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load fee schedules from exchanges
        self.fee_schedules: Dict[str, FeeSchedule] = {}
        self.fee_discounts: Dict[str, List[FeeDiscount]] = {}  # Keyed by exchange_id
        self.transaction_fees: List[TransactionFee] = []
        
        # Initialize structures
        self._load_fee_schedules()
        self._load_fee_discounts()
        self._load_transaction_history()
    
    def _load_fee_schedules(self) -> None:
        """Load fee schedules from storage."""
        schedule_file = self.data_dir / "fee_schedules.json"
        if schedule_file.exists():
            try:
                with open(schedule_file, "r") as f:
                    schedules_data = json.load(f)
                
                for exchange_id, schedule_data in schedules_data.items():
                    # Convert the raw data back to FeeSchedule objects
                    tiers = []
                    for tier_data in schedule_data.get("tiers", []):
                        tiers.append(FeeTier(
                            min_volume=float(tier_data["min_volume"]),
                            max_volume=float(tier_data["max_volume"]) if tier_data.get("max_volume") is not None else None,
                            maker_fee=float(tier_data["maker_fee"]),
                            taker_fee=float(tier_data["taker_fee"]),
                            description=tier_data.get("description")
                        ))
                    
                    # Convert string values to float in withdrawal_fees and network_fees
                    withdrawal_fees = {}
                    for asset, fee in schedule_data.get("withdrawal_fees", {}).items():
                        withdrawal_fees[asset] = float(fee)
                    
                    network_fees = {}
                    for asset, fee in schedule_data.get("network_fees", {}).items():
                        network_fees[asset] = float(fee)
                    
                    self.fee_schedules[exchange_id] = FeeSchedule(
                        exchange_id=exchange_id,
                        default_maker_fee=float(schedule_data["default_maker_fee"]),
                        default_taker_fee=float(schedule_data["default_taker_fee"]),
                        calculation_type=FeeCalculationType(schedule_data["calculation_type"]),
                        tiers=tiers,
                        withdrawal_fees=withdrawal_fees,
                        network_fees=network_fees,
                        updated_at=datetime.fromisoformat(schedule_data["updated_at"])
                    )
                
                logger.info(f"Loaded fee schedules for {len(self.fee_schedules)} exchanges")
            except Exception as e:
                logger.error(f"Error loading fee schedules: {e}")
    
    def _load_fee_discounts(self) -> None:
        """Load fee discounts from storage."""
        discount_file = self.data_dir / "fee_discounts.json"
        if discount_file.exists():
            try:
                with open(discount_file, "r") as f:
                    discounts_data = json.load(f)
                
                for exchange_id, discounts_list in discounts_data.items():
                    self.fee_discounts[exchange_id] = []
                    for discount_data in discounts_list:
                        expiry = None
                        if discount_data.get("expiry"):
                            expiry = datetime.fromisoformat(discount_data["expiry"])
                        
                        self.fee_discounts[exchange_id].append(FeeDiscount(
                            exchange_id=exchange_id,
                            discount_percentage=float(discount_data["discount_percentage"]),
                            applies_to=[FeeType(ft) for ft in discount_data["applies_to"]],
                            reason=discount_data["reason"],
                            expiry=expiry
                        ))
                
                logger.info(f"Loaded fee discounts for {len(self.fee_discounts)} exchanges")
            except Exception as e:
                logger.error(f"Error loading fee discounts: {e}")
    
    def _load_transaction_history(self) -> None:
        """Load transaction fee history from storage."""
        history_file = self.data_dir / "transaction_fees.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    transactions_data = json.load(f)
                
                self.transaction_fees = []
                for tx_data in transactions_data:
                    self.transaction_fees.append(TransactionFee(
                        transaction_id=tx_data["transaction_id"],
                        exchange_id=tx_data["exchange_id"],
                        fee_type=FeeType(tx_data["fee_type"]),
                        asset=tx_data["asset"],
                        amount=float(tx_data["amount"]),
                        usd_value=float(tx_data["usd_value"]) if tx_data.get("usd_value") is not None else None,
                        transaction_time=datetime.fromisoformat(tx_data["transaction_time"]),
                        related_to=tx_data.get("related_to"),
                        details=tx_data.get("details", {})
                    ))
                
                logger.info(f"Loaded {len(self.transaction_fees)} transaction fee records")
            except Exception as e:
                logger.error(f"Error loading transaction fee history: {e}")
    
    def save_data(self) -> None:
        """Save all fee data to disk."""
        self._save_fee_schedules()
        self._save_fee_discounts()
        self._save_transaction_history()
    
    def _save_fee_schedules(self) -> None:
        """Save fee schedules to storage."""
        schedule_file = self.data_dir / "fee_schedules.json"
        schedules_data = {}
        
        for exchange_id, schedule in self.fee_schedules.items():
            tiers_data = []
            for tier in schedule.tiers:
                # Create a dictionary with numeric fields
                tier_data: Dict[str, Any] = {
                    "min_volume": tier.min_volume,
                    "maker_fee": tier.maker_fee,
                    "taker_fee": tier.taker_fee
                }
                if tier.max_volume is not None:
                    tier_data["max_volume"] = tier.max_volume
                if tier.description:
                    # Add description separately as Any type
                    tier_data["description"] = tier.description
                tiers_data.append(tier_data)
            
            # Make sure all values in the dictionaries are convertible to JSON
            withdrawal_fees = {}
            for asset, fee in schedule.withdrawal_fees.items():
                withdrawal_fees[asset] = fee
            
            network_fees = {}
            for asset, fee in schedule.network_fees.items():
                network_fees[asset] = fee
            
            schedules_data[exchange_id] = {
                "default_maker_fee": schedule.default_maker_fee,
                "default_taker_fee": schedule.default_taker_fee,
                "calculation_type": schedule.calculation_type.value,
                "tiers": tiers_data,
                "withdrawal_fees": withdrawal_fees,
                "network_fees": network_fees,
                "updated_at": schedule.updated_at.isoformat()
            }
        
        with open(schedule_file, "w") as f:
            json.dump(schedules_data, f, indent=2)
        
        logger.info(f"Saved fee schedules for {len(schedules_data)} exchanges")
    
    def _save_fee_discounts(self) -> None:
        """Save fee discounts to storage."""
        discount_file = self.data_dir / "fee_discounts.json"
        discounts_data = {}
        
        for exchange_id, discounts in self.fee_discounts.items():
            discounts_data[exchange_id] = []
            for discount in discounts:
                discount_data = {
                    "discount_percentage": discount.discount_percentage,
                    "applies_to": [ft.value for ft in discount.applies_to],
                    "reason": discount.reason
                }
                if discount.expiry:
                    discount_data["expiry"] = discount.expiry.isoformat()
                discounts_data[exchange_id].append(discount_data)
        
        with open(discount_file, "w") as f:
            json.dump(discounts_data, f, indent=2)
        
        logger.info(f"Saved fee discounts for {len(discounts_data)} exchanges")
    
    def _save_transaction_history(self) -> None:
        """Save transaction fee history to storage."""
        history_file = self.data_dir / "transaction_fees.json"
        transactions_data = []
        
        for tx in self.transaction_fees:
            tx_data = {
                "transaction_id": tx.transaction_id,
                "exchange_id": tx.exchange_id,
                "fee_type": tx.fee_type.value,
                "asset": tx.asset,
                "amount": tx.amount,
                "transaction_time": tx.transaction_time.isoformat()
            }
            
            if tx.usd_value is not None:
                tx_data["usd_value"] = tx.usd_value
            if tx.related_to:
                tx_data["related_to"] = tx.related_to
            if tx.details:
                tx_data["details"] = tx.details
            
            transactions_data.append(tx_data)
        
        with open(history_file, "w") as f:
            json.dump(transactions_data, f, indent=2)
        
        logger.info(f"Saved {len(transactions_data)} transaction fee records")
    
    def update_fee_schedule(self, fee_schedule: FeeSchedule) -> None:
        """
        Update the fee schedule for an exchange.
        
        Args:
            fee_schedule: The updated fee schedule
        """
        self.fee_schedules[fee_schedule.exchange_id] = fee_schedule
        logger.info(f"Updated fee schedule for {fee_schedule.exchange_id}")
        self._save_fee_schedules()
    
    def add_fee_discount(self, fee_discount: FeeDiscount) -> None:
        """
        Add a fee discount for an exchange.
        
        Args:
            fee_discount: The fee discount to add
        """
        if fee_discount.exchange_id not in self.fee_discounts:
            self.fee_discounts[fee_discount.exchange_id] = []
        
        self.fee_discounts[fee_discount.exchange_id].append(fee_discount)
        logger.info(f"Added fee discount for {fee_discount.exchange_id}: {fee_discount.discount_percentage}%")
        self._save_fee_discounts()
    
    def record_transaction_fee(self, transaction_fee: TransactionFee) -> None:
        """
        Record a fee paid for a transaction.
        
        Args:
            transaction_fee: The transaction fee to record
        """
        self.transaction_fees.append(transaction_fee)
        logger.info(
            f"Recorded {transaction_fee.fee_type.value} fee of {transaction_fee.amount} {transaction_fee.asset} "
            f"for transaction {transaction_fee.transaction_id} on {transaction_fee.exchange_id}"
        )
        
        # Periodically save to disk (could be optimized to batch save)
        if len(self.transaction_fees) % 10 == 0:
            self._save_transaction_history()
    
    def get_fee_rate(
        self, 
        exchange_id: str, 
        fee_type: FeeType, 
        volume_30d: Optional[float] = None
    ) -> float:
        """
        Get the current fee rate for a specific exchange and fee type.
        
        Args:
            exchange_id: ID of the exchange
            fee_type: Type of fee (MAKER or TAKER)
            volume_30d: 30-day trading volume for tier calculation
        
        Returns:
            The fee rate as a decimal (e.g., 0.001 for 0.1%)
        """
        if exchange_id not in self.fee_schedules:
            logger.warning(f"No fee schedule found for {exchange_id}, using default values")
            return 0.001 if fee_type == FeeType.MAKER else 0.002
        
        schedule = self.fee_schedules[exchange_id]
        
        # Default rate based on fee type
        if fee_type == FeeType.MAKER:
            rate = schedule.default_maker_fee
        elif fee_type == FeeType.TAKER:
            rate = schedule.default_taker_fee
        else:
            # For other fee types, we don't have standard rates
            return 0.0
        
        # Apply volume-based tier if available
        if volume_30d is not None and schedule.tiers:
            for tier in sorted(schedule.tiers, key=lambda t: t.min_volume, reverse=True):
                if volume_30d >= tier.min_volume and (tier.max_volume is None or volume_30d < tier.max_volume):
                    rate = tier.maker_fee if fee_type == FeeType.MAKER else tier.taker_fee
                    break
        
        # Apply discounts if available
        if exchange_id in self.fee_discounts:
            now = datetime.now()
            for discount in self.fee_discounts[exchange_id]:
                if fee_type in discount.applies_to and (discount.expiry is None or discount.expiry > now):
                    # Apply discount percentage
                    rate *= (1 - discount.discount_percentage / 100)
        
        return rate
    
    def estimate_fee(
        self,
        exchange_id: str,
        fee_type: FeeType,
        asset: str,
        amount: float,
        price: float,
        volume_30d: Optional[float] = None
    ) -> FeeEstimate:
        """
        Estimate the fee for a planned transaction.
        
        Args:
            exchange_id: ID of the exchange
            fee_type: Type of fee
            asset: Asset in which the fee will be paid
            amount: Amount of the base asset in the transaction
            price: Price of the asset in quote currency
            volume_30d: 30-day trading volume for tier calculation
        
        Returns:
            Estimated fee details
        """
        trade_value = amount * price
        fee_rate = self.get_fee_rate(exchange_id, fee_type, volume_30d)
        
        # Calculate the estimated fee amount
        if fee_type in [FeeType.MAKER, FeeType.TAKER]:
            # Most exchanges charge trading fees as a percentage of the trade value
            estimated_amount = trade_value * fee_rate
            
            # Some exchanges charge fees in the base asset instead of quote
            if exchange_id in ["binance", "kucoin"]:  # Example exchanges
                # For simplicity, we're assuming fees are charged in the quote currency
                # In reality, this depends on the exchange's fee structure
                pass
            
            # Calculate USD value if the fee is not already in USD
            usd_value = estimated_amount  # Simplified assumption
            
            calculation_details = {
                "rate": fee_rate,
                "trade_value": trade_value,
                "calculation_method": "percentage"
            }
            
            return FeeEstimate(
                exchange_id=exchange_id,
                fee_type=fee_type,
                asset=asset,
                estimated_amount=estimated_amount,
                usd_value=usd_value,
                calculation_details=calculation_details
            )
        
        elif fee_type == FeeType.WITHDRAWAL:
            if exchange_id in self.fee_schedules and asset in self.fee_schedules[exchange_id].withdrawal_fees:
                estimated_amount = self.fee_schedules[exchange_id].withdrawal_fees[asset]
                
                # We'd need asset price data to convert to USD
                usd_value = None  # In a real system, we'd look up the USD value
                
                return FeeEstimate(
                    exchange_id=exchange_id,
                    fee_type=fee_type,
                    asset=asset,
                    estimated_amount=estimated_amount,
                    usd_value=usd_value,
                    calculation_details={"calculation_method": "fixed"}
                )
            else:
                logger.warning(f"No withdrawal fee data for {asset} on {exchange_id}")
                return FeeEstimate(
                    exchange_id=exchange_id,
                    fee_type=fee_type,
                    asset=asset,
                    estimated_amount=0.0,
                    calculation_details={"calculation_method": "unknown"}
                )
        
        # Other fee types would be handled similarly
        logger.warning(f"Fee estimation not implemented for {fee_type.value} fees")
        return FeeEstimate(
            exchange_id=exchange_id,
            fee_type=fee_type,
            asset=asset,
            estimated_amount=0.0,
            calculation_details={"calculation_method": "unknown"}
        )
    
    def get_fee_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        exchange_ids: Optional[Set[str]] = None,
        fee_types: Optional[Set[FeeType]] = None,
        assets: Optional[Set[str]] = None,
        related_to: Optional[str] = None
    ) -> FeeSummary:
        """
        Generate a summary of fees paid over a time period.
        
        Args:
            start_time: Start time of the period
            end_time: End time of the period
            exchange_ids: Only include these exchanges (or all if None)
            fee_types: Only include these fee types (or all if None)
            assets: Only include these assets (or all if None)
            related_to: Only include fees related to this entity (e.g., strategy ID)
        
        Returns:
            Summary of fees paid during the period
        """
        # Filter transaction fees by criteria
        filtered_fees = []
        for fee in self.transaction_fees:
            if fee.transaction_time < start_time or fee.transaction_time > end_time:
                continue
            
            if exchange_ids and fee.exchange_id not in exchange_ids:
                continue
            
            if fee_types and fee.fee_type not in fee_types:
                continue
            
            if assets and fee.asset not in assets:
                continue
            
            if related_to and fee.related_to != related_to:
                continue
            
            filtered_fees.append(fee)
        
        # Calculate summaries
        total_fees_usd = 0.0
        by_exchange: Dict[str, float] = {}
        by_type: Dict[FeeType, float] = {}
        by_asset: Dict[str, float] = {}
        by_related: Dict[str, float] = {}
        
        for fee in filtered_fees:
            # Skip fees without USD value
            if fee.usd_value is None:
                continue
            
            total_fees_usd += fee.usd_value
            
            # Update breakdowns
            by_exchange[fee.exchange_id] = by_exchange.get(fee.exchange_id, 0.0) + fee.usd_value
            by_type[fee.fee_type] = by_type.get(fee.fee_type, 0.0) + fee.usd_value
            by_asset[fee.asset] = by_asset.get(fee.asset, 0.0) + fee.usd_value
            
            if fee.related_to:
                by_related[fee.related_to] = by_related.get(fee.related_to, 0.0) + fee.usd_value
        
        # Create the summary
        summary = FeeSummary(
            start_time=start_time,
            end_time=end_time,
            total_fees_usd=total_fees_usd,
            by_exchange=by_exchange,
            by_type=by_type,
            by_asset=by_asset,
            by_related=by_related,
        )
        
        return summary
    
    def optimize_fees(
        self, 
        strategies: List[Dict[str, Any]],
        exchanges: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize fee allocation across exchanges based on strategies and constraints.
        
        Args:
            strategies: List of trading strategies with volume and asset details
            exchanges: List of available exchanges
            constraints: Constraints on exchange usage
        
        Returns:
            Optimized allocation plan
        """
        # This is a placeholder for a more sophisticated optimization algorithm
        # In a real system, this would consider:
        # - Fee tiers across exchanges
        # - Volume discounts
        # - Token incentives
        # - Cross-margin benefits
        # - Liquidity considerations
        
        # Simple allocation plan based on fees
        allocation = {}
        
        for strategy in strategies:
            strategy_id = strategy.get("id")
            asset_pair = strategy.get("asset_pair")
            monthly_volume = strategy.get("monthly_volume", 0)
            
            best_exchange = None
            lowest_fee_rate = float('inf')
            
            for exchange in exchanges:
                maker_fee = self.get_fee_rate(exchange, FeeType.MAKER, monthly_volume)
                taker_fee = self.get_fee_rate(exchange, FeeType.TAKER, monthly_volume)
                
                # Assume a mix of 70% maker, 30% taker orders
                effective_fee = (0.7 * maker_fee) + (0.3 * taker_fee)
                
                if effective_fee < lowest_fee_rate:
                    lowest_fee_rate = effective_fee
                    best_exchange = exchange
            
            if best_exchange:
                allocation[strategy_id] = {
                    "exchange": best_exchange,
                    "asset_pair": asset_pair,
                    "estimated_fee_rate": lowest_fee_rate,
                    "estimated_monthly_fee": monthly_volume * lowest_fee_rate
                }
        
        return {
            "allocation": allocation,
            "estimated_total_monthly_fee": sum(a["estimated_monthly_fee"] for a in allocation.values())
        } 