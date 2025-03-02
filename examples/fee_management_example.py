#!/usr/bin/env python
"""
AI Trading Agent - Fee Management Example

This script demonstrates how to use the Fee Management System to:
1. Setup fee schedules for different exchanges
2. Configure fee discounts
3. Record transaction fees
4. Estimate fees for planned transactions
5. Generate fee summaries
6. Optimize trading allocation across exchanges
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import fee management modules
from src.fees.models import FeeType, FeeTier, FeeSchedule, FeeDiscount, TransactionFee, FeeCalculationType
from src.fees.service import FeeManager
from src.fees.api import FeeManagementAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_fee_schedules(fee_manager):
    """Set up fee schedules for different exchanges."""
    logger.info("Setting up fee schedules...")
    
    # Binance fee schedule
    binance_tiers = [
        FeeTier(
            min_volume=0,
            max_volume=1_000_000,
            maker_fee=0.001,
            taker_fee=0.001,
            description="Default tier"
        ),
        FeeTier(
            min_volume=1_000_000,
            max_volume=5_000_000,
            maker_fee=0.0008,
            taker_fee=0.0009,
            description="VIP 1"
        ),
        FeeTier(
            min_volume=5_000_000,
            max_volume=10_000_000,
            maker_fee=0.0006,
            taker_fee=0.0008,
            description="VIP 2"
        ),
    ]
    
    binance_schedule = FeeSchedule(
        exchange_id="Binance",
        default_maker_fee=0.001,
        default_taker_fee=0.001,
        calculation_type=FeeCalculationType.TIERED,
        tiers=binance_tiers
    )
    
    # Coinbase fee schedule
    coinbase_tiers = [
        FeeTier(
            min_volume=0,
            max_volume=10_000,
            maker_fee=0.006,
            taker_fee=0.006,
            description="Default tier"
        ),
        FeeTier(
            min_volume=10_000,
            max_volume=50_000,
            maker_fee=0.004,
            taker_fee=0.006,
            description="Level 2"
        ),
        FeeTier(
            min_volume=50_000,
            max_volume=100_000,
            maker_fee=0.0025,
            taker_fee=0.005,
            description="Level 3"
        ),
    ]
    
    coinbase_schedule = FeeSchedule(
        exchange_id="Coinbase",
        default_maker_fee=0.006,
        default_taker_fee=0.006,
        calculation_type=FeeCalculationType.TIERED,
        tiers=coinbase_tiers
    )
    
    # Kraken fee schedule
    kraken_tiers = [
        FeeTier(
            min_volume=0,
            max_volume=50_000,
            maker_fee=0.0016,
            taker_fee=0.0026,
            description="Default tier"
        ),
        FeeTier(
            min_volume=50_000,
            max_volume=100_000,
            maker_fee=0.0014,
            taker_fee=0.0024,
            description="Intermediate"
        ),
        FeeTier(
            min_volume=100_000,
            max_volume=250_000,
            maker_fee=0.0012,
            taker_fee=0.0022,
            description="Advanced"
        ),
    ]
    
    kraken_schedule = FeeSchedule(
        exchange_id="Kraken",
        default_maker_fee=0.0016,
        default_taker_fee=0.0026,
        calculation_type=FeeCalculationType.TIERED,
        tiers=kraken_tiers
    )
    
    # Add all fee schedules to the service using update_fee_schedule method
    fee_manager.update_fee_schedule(binance_schedule)
    fee_manager.update_fee_schedule(coinbase_schedule)
    fee_manager.update_fee_schedule(kraken_schedule)
    
    logger.info("Fee schedules set up for Binance, Coinbase, and Kraken")

def setup_fee_discounts(fee_manager):
    """Set up fee discounts for different exchanges."""
    logger.info("Setting up fee discounts...")
    
    # Binance BNB discount
    bnb_discount = FeeDiscount(
        exchange_id="Binance",
        discount_percentage=25,
        applies_to=[FeeType.MAKER, FeeType.TAKER],
        reason="25% discount when paying fees with BNB",
        expiry=datetime.now() + timedelta(days=365)
    )
    
    # Coinbase USDC discount
    usdc_discount = FeeDiscount(
        exchange_id="Coinbase",
        discount_percentage=15,
        applies_to=[FeeType.MAKER, FeeType.TAKER],
        reason="15% discount for USDC holders",
        expiry=datetime.now() + timedelta(days=180)
    )
    
    # Add all discounts to the service
    fee_manager.add_fee_discount(bnb_discount)
    fee_manager.add_fee_discount(usdc_discount)
    
    logger.info("Fee discounts set up for Binance and Coinbase")

def record_sample_transaction_fees(fee_manager):
    """Record sample transaction fees."""
    logger.info("Recording sample transaction fees...")
    
    # Record a Binance trading fee (maker order)
    binance_trading_fee = TransactionFee(
        transaction_id="tx_binance_1",
        exchange_id="Binance",
        fee_type=FeeType.MAKER,
        asset="BTC",
        amount=0.01,
        usd_value=400,  # Assuming BTC price of $40,000
        transaction_time=datetime.now() - timedelta(hours=2),
        related_to="btc_grid_strategy",
        details={"order_size": 1.0, "order_price": 40000}
    )
    
    # Record a Coinbase trading fee (taker order)
    coinbase_trading_fee = TransactionFee(
        transaction_id="tx_coinbase_1",
        exchange_id="Coinbase",
        fee_type=FeeType.TAKER,
        asset="USD",
        amount=30.0,
        usd_value=30.0,
        transaction_time=datetime.now() - timedelta(hours=5),
        related_to="eth_momentum_strategy",
        details={"order_size": 5.0, "order_price": 2000}
    )
    
    # Record a Kraken withdrawal fee
    kraken_withdrawal_fee = TransactionFee(
        transaction_id="tx_kraken_1",
        exchange_id="Kraken",
        fee_type=FeeType.WITHDRAWAL,
        asset="BTC",
        amount=0.0005,
        usd_value=20.0,  # Assuming BTC price of $40,000
        transaction_time=datetime.now() - timedelta(days=1),
        related_to=None,
        details={"withdrawal_amount": 0.1, "network": "Bitcoin"}
    )
    
    # Add all fees to the service using record_transaction_fee method
    fee_manager.record_transaction_fee(binance_trading_fee)
    fee_manager.record_transaction_fee(coinbase_trading_fee)
    fee_manager.record_transaction_fee(kraken_withdrawal_fee)
    
    # Record a few more trading fees over different timeframes
    for i in range(10):
        days_ago = i % 7
        exchange = ["Binance", "Coinbase", "Kraken"][i % 3]
        asset = ["BTC", "ETH", "USD"][i % 3]
        fee_type = FeeType.MAKER if i % 2 == 0 else FeeType.TAKER
        amount = 5.0 + i
        
        # Get the exchange's current fee rate
        if fee_type == FeeType.MAKER:
            rate = fee_manager.get_fee_rate(exchange, FeeType.MAKER)
        else:
            rate = fee_manager.get_fee_rate(exchange, FeeType.TAKER)
        
        # Calculate fee amount
        fee_amount = amount * rate
        
        # Estimate USD value
        if asset == "BTC":
            usd_value = fee_amount * 40000
        elif asset == "ETH":
            usd_value = fee_amount * 2000
        else:
            usd_value = fee_amount
        
        # Record the fee
        fee = TransactionFee(
            transaction_id=f"tx_{exchange.lower()}_{i+2}",
            exchange_id=exchange,
            fee_type=fee_type,
            asset=asset,
            amount=fee_amount,
            usd_value=usd_value,
            transaction_time=datetime.now() - timedelta(days=days_ago),
            related_to="sample_strategy",
            details={"sample": True}
        )
        
        fee_manager.record_transaction_fee(fee)
    
    logger.info("Recorded 13 sample transaction fees")

def estimate_transaction_fees(fee_manager):
    """Estimate fees for planned transactions."""
    logger.info("Estimating fees for planned transactions...")
    
    # Estimate trading fee on Binance for a maker order
    binance_estimate = fee_manager.estimate_fee(
        exchange_id="Binance",
        fee_type=FeeType.MAKER,
        asset="BTC",
        amount=1.0,  # 1 BTC
        price=40000  # $40,000 per BTC
    )
    
    logger.info(f"Binance maker fee estimate: {binance_estimate.estimated_amount} {binance_estimate.asset}")
    logger.info(f"Binance estimated USD value: ${binance_estimate.usd_value:.2f}")
    
    # Estimate trading fee on Coinbase for a taker order
    coinbase_estimate = fee_manager.estimate_fee(
        exchange_id="Coinbase",
        fee_type=FeeType.TAKER,
        asset="USD",
        amount=5.0,  # 5 ETH
        price=2000  # $2,000 per ETH
    )
    
    logger.info(f"Coinbase taker fee estimate: {coinbase_estimate.estimated_amount} {coinbase_estimate.asset}")
    logger.info(f"Coinbase estimated USD value: ${coinbase_estimate.usd_value:.2f}")
    
    # Estimate trading fee on Kraken for a maker order
    kraken_estimate = fee_manager.estimate_fee(
        exchange_id="Kraken",
        fee_type=FeeType.MAKER,
        asset="ETH",
        amount=2.5,  # 2.5 ETH
        price=2000  # $2,000 per ETH
    )
    
    logger.info(f"Kraken maker fee estimate: {kraken_estimate.estimated_amount} {kraken_estimate.asset}")
    logger.info(f"Kraken estimated USD value: ${kraken_estimate.usd_value:.2f}")

def generate_fee_summary(fee_manager):
    """Generate and display a fee summary."""
    logger.info("Generating fee summary...")
    
    # Generate a summary for all fees
    summary = fee_manager.get_fee_summary(
        start_time=datetime.now() - timedelta(days=30),
        end_time=datetime.now()
    )
    
    logger.info(f"Total fees (past 30 days): ${summary.total_fees_usd:.2f}")
    
    # Print exchange breakdowns
    for exchange, amount in summary.by_exchange.items():
        logger.info(f"{exchange} fees: ${amount:.2f}")
    
    # Print fee type breakdowns
    for fee_type, amount in summary.by_type.items():
        logger.info(f"{fee_type} fees: ${amount:.2f}")
    
    # Print asset breakdowns
    for asset, amount in summary.by_asset.items():
        logger.info(f"{asset} fees: ${amount:.2f}")
    
    # Print strategy breakdowns
    for strategy, amount in summary.by_related.items():
        strategy_name = strategy if strategy else "Unassigned"
        logger.info(f"{strategy_name} fees: ${amount:.2f}")

def optimize_exchange_allocation(fee_manager):
    """Optimize exchange allocation for multiple trading pairs."""
    logger.info("Optimizing exchange allocation...")
    
    # Define strategies with expected volumes
    strategies = [
        {
            "id": "btc_grid_strategy",
            "asset_pair": "BTC/USDT",
            "monthly_volume": 500000,
            "description": "BTC Grid Trading"
        },
        {
            "id": "eth_momentum_strategy",
            "asset_pair": "ETH/USDT",
            "monthly_volume": 250000,
            "description": "ETH Momentum"
        },
        {
            "id": "altcoin_strategy",
            "asset_pair": "ALT/USDT",
            "monthly_volume": 100000,
            "description": "Altcoin Basket"
        }
    ]
    
    # Define available exchanges
    exchanges = ["Binance", "Coinbase", "Kraken"]
    
    # Define constraints
    constraints = {
        "max_exchanges_per_strategy": 1,
        "required_exchanges": {},
        "excluded_exchanges": {}
    }
    
    # Get optimized allocation using optimize_fees method
    allocation = fee_manager.optimize_fees(
        strategies=strategies,
        exchanges=exchanges,
        constraints=constraints
    )
    
    logger.info(f"Optimization completed with estimated monthly savings: ${allocation.get('estimated_monthly_savings', 0):.2f}")
    
    # Display the optimized allocation
    if "allocation" in allocation:
        for strategy_id, details in allocation["allocation"].items():
            strategy_details = next((s for s in strategies if s["id"] == strategy_id), None)
            if strategy_details:
                logger.info(f"\nAllocation for {strategy_details['description']}:")
                exchange = details.get("exchange", "Unknown")
                volume = strategy_details["monthly_volume"]
                fee_rate = details.get("estimated_fee_rate", 0)
                monthly_fee = details.get("estimated_monthly_fee", 0)
                logger.info(f"  Exchange: {exchange}")
                logger.info(f"  Volume: ${volume:,.2f}")
                logger.info(f"  Fee rate: {fee_rate*100:.4f}%")
                logger.info(f"  Monthly fee: ${monthly_fee:.2f}")

def main():
    """Main function to run the fee management example."""
    logger.info("Starting Fee Management Example...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/fees")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the service and API
    fee_manager = FeeManager(data_dir=data_dir)
    # Initialize API separately - it creates its own FeeManager
    api = FeeManagementAPI()
    
    # Set up fee schedules and discounts
    setup_fee_schedules(fee_manager)
    setup_fee_discounts(fee_manager)
    
    # Record sample transaction fees
    record_sample_transaction_fees(fee_manager)
    
    # Estimate fees for planned transactions
    estimate_transaction_fees(fee_manager)
    
    # Generate and display fee summary
    generate_fee_summary(fee_manager)
    
    # Optimize exchange allocation
    optimize_exchange_allocation(fee_manager)
    
    # Save the fee data
    fee_manager.save_data()
    
    logger.info("Fee Management Example completed")

if __name__ == "__main__":
    main() 