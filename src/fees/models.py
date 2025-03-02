"""
Fee management system models.

This module defines the data structures for tracking and calculating trading fees
across different exchanges and asset types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any


class FeeType(str, Enum):
    """Types of fees charged by exchanges."""
    MAKER = "maker"              # Fee for providing liquidity (limit orders that don't cross the spread)
    TAKER = "taker"              # Fee for taking liquidity (market orders or limit orders that cross the spread)
    WITHDRAWAL = "withdrawal"    # Fee for withdrawing assets from an exchange
    DEPOSIT = "deposit"          # Fee for depositing assets to an exchange (rare)
    NETWORK = "network"          # Blockchain network fees for crypto transfers
    FUNDING = "funding"          # Funding fees for perpetual contracts
    OTHER = "other"              # Other miscellaneous fees


class FeeCalculationType(str, Enum):
    """Methods for calculating fees."""
    PERCENTAGE = "percentage"    # Fee calculated as a percentage of the trade value
    FIXED = "fixed"              # Fixed fee regardless of trade size
    TIERED = "tiered"            # Fee depends on trading volume tier
    HYBRID = "hybrid"            # Combination of percentage and fixed fee


@dataclass
class FeeTier:
    """A volume-based fee tier used by exchanges."""
    min_volume: float            # Minimum volume required for this tier
    max_volume: Optional[float]  # Maximum volume for this tier (None if unlimited)
    maker_fee: float             # Maker fee rate (percentage as decimal)
    taker_fee: float             # Taker fee rate (percentage as decimal)
    description: Optional[str] = None  # Human-readable description of the tier


@dataclass
class FeeSchedule:
    """Fee schedule for an exchange."""
    exchange_id: str                           # ID of the exchange
    default_maker_fee: float                   # Default maker fee (percentage as decimal)
    default_taker_fee: float                   # Default taker fee (percentage as decimal)
    calculation_type: FeeCalculationType       # How fees are calculated
    tiers: List[FeeTier] = field(default_factory=list)  # Volume-based tiers if applicable
    withdrawal_fees: Dict[str, float] = field(default_factory=dict)  # Asset-specific withdrawal fees
    network_fees: Dict[str, float] = field(default_factory=dict)     # Asset-specific network fees
    updated_at: datetime = field(default_factory=datetime.now)       # When this schedule was last updated


@dataclass
class FeeDiscount:
    """A fee discount applied to an account."""
    exchange_id: str                 # Exchange where the discount applies
    discount_percentage: float       # Discount amount as a percentage
    applies_to: List[FeeType]        # Which fee types this discount applies to
    reason: str                      # Reason for the discount (e.g., "VIP tier", "Token holding")
    expiry: Optional[datetime] = None  # When the discount expires (None if permanent)


@dataclass
class TransactionFee:
    """A record of a fee paid for a transaction."""
    transaction_id: str              # ID of the transaction (order ID, withdrawal ID, etc.)
    exchange_id: str                 # Exchange where the fee was paid
    fee_type: FeeType                # Type of fee
    asset: str                       # Asset in which the fee was paid
    amount: float                    # Amount of fee paid
    usd_value: Optional[float] = None  # USD value of the fee at the time it was paid
    transaction_time: datetime = field(default_factory=datetime.now)  # When the transaction occurred
    related_to: Optional[str] = None   # ID of related entity (e.g., strategy, portfolio)
    details: Dict[str, Any] = field(default_factory=dict)  # Additional details about the fee


@dataclass
class FeeEstimate:
    """Estimated fee for a planned transaction."""
    exchange_id: str                 # Exchange where the transaction will occur
    fee_type: FeeType                # Type of fee
    asset: str                       # Asset in which the fee will be paid
    estimated_amount: float          # Estimated fee amount
    usd_value: Optional[float] = None  # Estimated USD value of the fee
    calculation_details: Dict[str, Any] = field(default_factory=dict)  # How the estimate was calculated


@dataclass
class FeeSummary:
    """Summary of fees paid over a time period."""
    start_time: datetime              # Start of the period
    end_time: datetime                # End of the period
    total_fees_usd: float             # Total fees in USD value
    by_exchange: Dict[str, float]     # Fees broken down by exchange
    by_type: Dict[FeeType, float]     # Fees broken down by type
    by_asset: Dict[str, float]        # Fees broken down by asset
    by_related: Dict[str, float]      # Fees broken down by related entity
    
    # Optional fields for more detailed analysis
    trade_volume_usd: Optional[float] = None  # Total trading volume
    effective_fee_rate: Optional[float] = None  # Total fees / total volume
    details: Dict[str, Any] = field(default_factory=dict)  # Additional analysis details 