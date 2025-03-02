#!/usr/bin/env python
"""
AI Trading Agent - Fee Visualization Example

This script demonstrates how to visualize various fee metrics and analyses
using the Fee Management System. It includes examples of:
- Exchange fee comparison
- Fee breakdown by exchange, symbol, and fee type
- Fee trends over time
- Fee tier visualization
- Fee optimization suggestions
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colormaps
from pathlib import Path

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import fee management modules
from src.fees.models import FeeType, FeeTier, FeeSchedule, FeeDiscount, TransactionFee, FeeCalculationType
from src.fees.service import FeeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directory for saving visualizations
VISUAL_DIR = Path("examples/visualizations")
VISUAL_DIR.mkdir(exist_ok=True, parents=True)

def setup_sample_data():
    """Create sample fee data for visualization purposes."""
    logger.info("Setting up sample fee data...")
    
    # Sample exchanges
    exchanges = ["Binance", "Coinbase", "Kraken", "FTX", "Bybit"]
    
    # Sample symbols
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
    
    # Create a fee manager instance
    fee_manager = FeeManager()
    
    # Create fee schedules for each exchange
    for exchange in exchanges:
        # Create tiers with slightly different rates for each exchange
        base_maker_rate = max(0.0001, round(random.uniform(0.0001, 0.001), 6))
        base_taker_rate = max(0.0002, round(random.uniform(0.0002, 0.0015), 6))
        
        tiers = [
            FeeTier(
                min_volume=0,
                max_volume=100_000,
                maker_fee=base_maker_rate,
                taker_fee=base_taker_rate,
                description=f"{exchange} Default Tier"
            ),
            FeeTier(
                min_volume=100_000,
                max_volume=1_000_000,
                maker_fee=max(0.00005, round(base_maker_rate * 0.8, 6)),
                taker_fee=max(0.0001, round(base_taker_rate * 0.8, 6)),
                description=f"{exchange} Silver Tier"
            ),
            FeeTier(
                min_volume=1_000_000,
                max_volume=10_000_000,
                maker_fee=max(0.00002, round(base_maker_rate * 0.5, 6)),
                taker_fee=max(0.00005, round(base_taker_rate * 0.5, 6)),
                description=f"{exchange} Gold Tier"
            ),
            FeeTier(
                min_volume=10_000_000,
                max_volume=float('inf'),
                maker_fee=max(0.00001, round(base_maker_rate * 0.2, 6)),
                taker_fee=max(0.00002, round(base_taker_rate * 0.2, 6)),
                description=f"{exchange} Platinum Tier"
            ),
        ]
        
        # Create a fee schedule
        schedule = FeeSchedule(
            exchange_id=exchange,
            default_maker_fee=base_maker_rate,
            default_taker_fee=base_taker_rate,
            calculation_type=FeeCalculationType.TIERED,
            tiers=tiers
        )
        
        # Add schedule to fee manager
        fee_manager.fee_schedules[exchange] = schedule
        
        # Add some discounts - Properly handle the dict structure
        if random.random() > 0.5:
            discount = FeeDiscount(
                exchange_id=exchange,
                discount_percentage=random.randint(10, 25),
                applies_to=[FeeType.MAKER, FeeType.TAKER],
                reason=f"{exchange} Token Discount",
                expiry=datetime.now() + timedelta(days=60)
            )
            
            # Initialize the list if needed
            if exchange not in fee_manager.fee_discounts:
                fee_manager.fee_discounts[exchange] = []
                
            # Now append to the list
            fee_manager.fee_discounts[exchange].append(discount)
    
    # Generate random transaction fees over the past 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    current_date = start_date
    
    # Generate approximately 1000 fee records
    while current_date < end_date:
        # More transactions on weekdays, fewer on weekends
        day_of_week = current_date.weekday()
        daily_transactions = 15 if day_of_week < 5 else 5
        
        for _ in range(random.randint(daily_transactions - 3, daily_transactions + 3)):
            exchange = random.choice(exchanges)
            symbol = random.choice(symbols)
            fee_type = random.choice([FeeType.MAKER, FeeType.TAKER, FeeType.WITHDRAWAL, FeeType.DEPOSIT])
            
            # For trading fees (maker/taker)
            if fee_type in [FeeType.MAKER, FeeType.TAKER]:
                # Determine the asset pair
                base_asset = symbol.split('/')[0]
                quote_asset = symbol.split('/')[1]
                
                # Get appropriate fee rate from the schedule
                if fee_type == FeeType.MAKER:
                    rate = fee_manager.fee_schedules[exchange].tiers[0].maker_fee
                else:
                    rate = fee_manager.fee_schedules[exchange].tiers[0].taker_fee
                
                # Generate random trade amount
                trade_amount = random.uniform(100, 10000)
                
                # Calculate fee (usually paid in quote currency)
                fee_amount = trade_amount * rate
                
                # Determine USD value (assuming USDT is USD)
                if quote_asset == "USDT":
                    usd_value = fee_amount
                else:
                    # Some mock conversion rate
                    usd_value = fee_amount * random.uniform(0.8, 1.2)
                
                fee_manager.transaction_fees.append(TransactionFee(
                    transaction_id=f"tx_{current_date.strftime('%Y%m%d')}_{_}",
                    exchange_id=exchange,
                    fee_type=fee_type,
                    asset=quote_asset,
                    amount=fee_amount,
                    usd_value=usd_value,
                    transaction_time=current_date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59)),
                    related_to=f"{base_asset.lower()}_strategy" if random.random() > 0.3 else None,
                    details={"trade_size": trade_amount, "pair": symbol}
                ))
            
            # For withdrawal fees
            elif fee_type == FeeType.WITHDRAWAL and random.random() < 0.7:  # 70% chance of recording a withdrawal
                base_asset = symbol.split('/')[0]
                
                # Withdrawal fees are usually a fixed amount of the asset being withdrawn
                if base_asset == "BTC":
                    fee_amount = random.uniform(0.0001, 0.0005)
                    usd_value = fee_amount * random.uniform(35000, 45000)  # Mock BTC price
                elif base_asset == "ETH":
                    fee_amount = random.uniform(0.001, 0.005)
                    usd_value = fee_amount * random.uniform(1800, 2500)  # Mock ETH price
                else:
                    fee_amount = random.uniform(0.01, 1.0)
                    usd_value = fee_amount * random.uniform(10, 200)  # Mock altcoin price
                
                fee_manager.transaction_fees.append(TransactionFee(
                    transaction_id=f"wd_{current_date.strftime('%Y%m%d')}_{_}",
                    exchange_id=exchange,
                    fee_type=FeeType.WITHDRAWAL,
                    asset=base_asset,
                    amount=fee_amount,
                    usd_value=usd_value,
                    transaction_time=current_date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59)),
                    related_to=None,
                    details={"withdrawal_amount": fee_amount * random.randint(10, 100), "network": "Mainnet"}
                ))
            
            # For deposit fees (rare)
            elif fee_type == FeeType.DEPOSIT and random.random() < 0.1:  # 10% chance of a deposit fee
                base_asset = symbol.split('/')[0]
                
                # Deposit fees are usually smaller
                if base_asset == "BTC":
                    fee_amount = random.uniform(0.00005, 0.0001)
                    usd_value = fee_amount * random.uniform(35000, 45000)  # Mock BTC price
                elif base_asset == "ETH":
                    fee_amount = random.uniform(0.0005, 0.001)
                    usd_value = fee_amount * random.uniform(1800, 2500)  # Mock ETH price
                else:
                    fee_amount = random.uniform(0.005, 0.1)
                    usd_value = fee_amount * random.uniform(10, 200)  # Mock altcoin price
                
                fee_manager.transaction_fees.append(TransactionFee(
                    transaction_id=f"dep_{current_date.strftime('%Y%m%d')}_{_}",
                    exchange_id=exchange,
                    fee_type=FeeType.DEPOSIT,
                    asset=base_asset,
                    amount=fee_amount,
                    usd_value=usd_value,
                    transaction_time=current_date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59)),
                    related_to=None,
                    details={"deposit_amount": fee_amount * random.randint(10, 100), "network": "Mainnet"}
                ))
        
        current_date += timedelta(days=1)
    
    logger.info(f"Created sample fee data across {len(exchanges)} exchanges")
    return fee_manager

def visualize_exchange_comparison(fee_manager):
    """Create a comparison of fees across different exchanges."""
    logger.info("Visualizing exchange fee comparison...")
    
    # Get fee summary for all exchanges
    start_time = datetime.now() - timedelta(days=90)
    end_time = datetime.now()
    summary = fee_manager.get_fee_summary(start_time=start_time, end_time=end_time)
    
    if not summary or not summary.by_exchange:
        logger.warning("No exchange summary data available")
        return
    
    # Extract data for plotting
    exchanges = []
    total_fees = []
    
    # Get fee rates for comparison
    maker_rates = []
    taker_rates = []
    
    for exchange in fee_manager.fee_schedules:
        exchanges.append(exchange)
        total_fees.append(summary.by_exchange.get(exchange, 0))
        
        # Get current fee rates
        maker_rate = fee_manager.fee_schedules[exchange].tiers[0].maker_fee * 100  # Convert to percentage
        taker_rate = fee_manager.fee_schedules[exchange].tiers[0].taker_fee * 100  # Convert to percentage
        
        maker_rates.append(maker_rate)
        taker_rates.append(taker_rate)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot total fees by exchange
    bars = ax1.bar(exchanges, total_fees, color='skyblue')
    ax1.set_title('Total Fees by Exchange (USD)')
    ax1.set_xlabel('Exchange')
    ax1.set_ylabel('Total Fees (USD)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${height:.2f}', ha='center', va='bottom')
    
    # Plot fee rates comparison
    x = np.arange(len(exchanges))
    width = 0.35
    
    ax2.bar(x - width/2, maker_rates, width, label='Maker', color='lightgreen')
    ax2.bar(x + width/2, taker_rates, width, label='Taker', color='lightcoral')
    
    ax2.set_title('Fee Rates by Exchange')
    ax2.set_xlabel('Exchange')
    ax2.set_ylabel('Fee Rate (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(exchanges, rotation=45)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "exchange_fee_comparison.png")
    plt.close()
    logger.info(f"Saved exchange fee comparison visualization to {VISUAL_DIR / 'exchange_fee_comparison.png'}")

def visualize_fee_breakdown(fee_manager):
    """Create a breakdown of fees by type, exchange, and asset."""
    logger.info("Visualizing fee breakdowns...")
    
    # Get fee summary
    start_time = datetime.now() - timedelta(days=90)
    end_time = datetime.now()
    summary = fee_manager.get_fee_summary(start_time=start_time, end_time=end_time)
    
    if not summary:
        logger.warning("No fee data available for breakdown")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Fee breakdown by type
    if summary.by_type:
        labels = list(summary.by_type.keys())
        values = list(summary.by_type.values())
        # Create color list for the pie chart
        paired_cmap = plt.get_cmap('Paired')
        colors = [paired_cmap(i) for i in np.linspace(0, 1, len(labels))]
        axes[0, 0].pie(values, labels=labels, autopct='%1.1f%%', 
                      shadow=True, startangle=90, colors=colors)
        axes[0, 0].set_title('Fee Distribution by Type')
    else:
        axes[0, 0].text(0.5, 0.5, 'No fee type data available', ha='center', va='center')
        axes[0, 0].set_title('Fee Distribution by Type')
    
    # 2. Fee breakdown by exchange
    if summary.by_exchange:
        labels = list(summary.by_exchange.keys())
        values = list(summary.by_exchange.values())
        # Create color list for the pie chart
        paired_cmap = plt.get_cmap('Paired')
        colors = [paired_cmap(i) for i in np.linspace(0, 1, len(labels))]
        axes[0, 1].pie(values, labels=labels, autopct='%1.1f%%', 
                      shadow=True, startangle=90, colors=colors)
        axes[0, 1].set_title('Fee Distribution by Exchange')
    else:
        axes[0, 1].text(0.5, 0.5, 'No exchange data available', ha='center', va='center')
        axes[0, 1].set_title('Fee Distribution by Exchange')
    
    # 3. Fee breakdown by asset
    if summary.by_asset:
        # Get top 5 assets by fee amount
        assets = dict(sorted(summary.by_asset.items(), key=lambda x: x[1], reverse=True)[:5])
        labels = list(assets.keys())
        values = list(assets.values())
        
        axes[1, 0].bar(labels, values, color='skyblue')
        axes[1, 0].set_title('Top 5 Assets by Fee Amount')
        axes[1, 0].set_xlabel('Asset')
        axes[1, 0].set_ylabel('Total Fees (USD)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No asset data available', ha='center', va='center')
        axes[1, 0].set_title('Top Assets by Fee Amount')
    
    # 4. Fee breakdown by related entity (strategy)
    if summary.by_related:
        # Clean up the data - replace None with "Unassigned"
        related = {k if k else "Unassigned": v for k, v in summary.by_related.items()}
        
        # Get top 5 related entities by fee amount
        related = dict(sorted(related.items(), key=lambda x: x[1], reverse=True)[:5])
        labels = list(related.keys())
        values = list(related.values())
        
        axes[1, 1].bar(labels, values, color='lightgreen')
        axes[1, 1].set_title('Top 5 Strategies by Fee Amount')
        axes[1, 1].set_xlabel('Strategy')
        axes[1, 1].set_ylabel('Total Fees (USD)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No strategy data available', ha='center', va='center')
        axes[1, 1].set_title('Top Strategies by Fee Amount')
    
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "fee_breakdown.png")
    plt.close()
    logger.info(f"Saved fee breakdown visualization to {VISUAL_DIR / 'fee_breakdown.png'}")

def visualize_fee_trends(fee_manager):
    """Visualize fee trends over time."""
    logger.info("Visualizing fee trends over time...")
    
    # Get all fee records
    all_fees = fee_manager.transaction_fees
    
    if not all_fees:
        logger.warning("No fee data available for trend analysis")
        return
    
    # Convert to DataFrame for easier manipulation
    fee_data = []
    for fee in all_fees:
        fee_data.append({
            'exchange': fee.exchange_id,
            'fee_type': fee.fee_type,
            'amount': fee.amount,
            'usd_value': fee.usd_value if fee.usd_value is not None else 0,
            'date': fee.transaction_time.date()  # Group by date
        })
    
    df = pd.DataFrame(fee_data)
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    try:
        # 1. Daily fees over time
        daily_fees = df.groupby('date')['usd_value'].sum()
        daily_fees.plot(ax=ax1, marker='o', linestyle='-', color='blue', alpha=0.7)
        ax1.set_title('Daily Fees Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Fees (USD)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Fees by type over time
        fees_by_type = df.groupby(['date', 'fee_type'])['usd_value'].sum().unstack()
        if not fees_by_type.empty:
            fees_by_type.plot(ax=ax2, marker='.', linestyle='-')
            ax2.set_title('Fees by Type Over Time')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Fees (USD)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(title='Fee Type')
        else:
            ax2.text(0.5, 0.5, 'Not enough data for fee type trends', ha='center', va='center')
            ax2.set_title('Fees by Type Over Time')
        
        plt.tight_layout()
        plt.savefig(VISUAL_DIR / "fee_trends.png")
        plt.close()
        logger.info(f"Saved fee trend visualization to {VISUAL_DIR / 'fee_trends.png'}")
    
    except Exception as e:
        logger.error(f"Error creating fee trend visualization: {e}")
        plt.close()

def visualize_fee_tiers(fee_manager):
    """Visualize fee tiers across exchanges."""
    logger.info("Visualizing fee tier structures...")
    
    # Get fee schedules for all exchanges
    exchanges = []
    tier_levels = []
    maker_rates = []
    taker_rates = []
    volume_thresholds = []
    
    for exchange_id, schedule in fee_manager.fee_schedules.items():
        if not schedule.tiers:
            continue
            
        for i, tier in enumerate(schedule.tiers):
            exchanges.append(exchange_id)
            tier_levels.append(i + 1)  # 1-based tier level
            maker_rates.append(tier.maker_fee * 100)  # Convert to percentage
            taker_rates.append(tier.taker_fee * 100)  # Convert to percentage
            volume_thresholds.append(tier.min_volume)
    
    if not exchanges:
        logger.warning("No fee tier data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Exchange': exchanges,
        'Tier': tier_levels,
        'Maker Rate (%)': maker_rates,
        'Taker Rate (%)': taker_rates,
        'Volume Threshold': volume_thresholds
    })
    
    # Create a figure for the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get unique exchanges for consistent colors
    unique_exchanges = list(set(exchanges))
    # Use proper colormap access pattern
    tab10_cmap = plt.get_cmap('tab10')
    colors = [tab10_cmap(i) for i in np.linspace(0, 1, len(unique_exchanges))]
    exchange_colors = {exchange: color for exchange, color in zip(unique_exchanges, colors)}
    
    # Plot maker fee rates by tier and exchange
    for exchange in unique_exchanges:
        exchange_data = df[df['Exchange'] == exchange]
        ax1.plot(exchange_data['Tier'], exchange_data['Maker Rate (%)'], 
                'o-', label=exchange, color=exchange_colors[exchange])
    
    ax1.set_title('Maker Fee Rates by Tier and Exchange')
    ax1.set_xlabel('Tier Level')
    ax1.set_ylabel('Maker Fee Rate (%)')
    ax1.set_xticks(range(1, max(tier_levels) + 1))
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title='Exchange')
    
    # Plot taker fee rates by tier and exchange
    for exchange in unique_exchanges:
        exchange_data = df[df['Exchange'] == exchange]
        ax2.plot(exchange_data['Tier'], exchange_data['Taker Rate (%)'], 
                'o-', label=exchange, color=exchange_colors[exchange])
    
    ax2.set_title('Taker Fee Rates by Tier and Exchange')
    ax2.set_xlabel('Tier Level')
    ax2.set_ylabel('Taker Fee Rate (%)')
    ax2.set_xticks(range(1, max(tier_levels) + 1))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(title='Exchange')
    
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "fee_tier_structure.png")
    plt.close()
    logger.info(f"Saved fee tier visualization to {VISUAL_DIR / 'fee_tier_structure.png'}")

def visualize_optimization_suggestions(fee_manager):
    """Visualize fee optimization suggestions."""
    logger.info("Visualizing fee optimization suggestions...")
    
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
            "id": "sol_strategy",
            "asset_pair": "SOL/USDT",
            "monthly_volume": 100000,
            "description": "SOL Strategy"
        }
    ]
    
    # Get exchanges from fee manager
    exchanges = list(fee_manager.fee_schedules.keys())
    
    # Define constraints
    constraints = {
        "max_exchanges_per_strategy": 1,
        "required_exchanges": {},
        "excluded_exchanges": {}
    }
    
    # Get optimized allocation - use optimize_fees instead of optimize_exchange_allocation
    optimization = fee_manager.optimize_fees(
        strategies=strategies,
        exchanges=exchanges,
        constraints=constraints
    )
    
    if not optimization or 'allocation' not in optimization:
        logger.warning("No optimization data available")
        return
    
    # Prepare data for visualization
    strategy_names = []
    volumes = []
    assigned_exchanges = []
    estimated_fees = []
    fee_rates = []
    
    for strategy_id, details in optimization['allocation'].items():
        # Find strategy details
        strategy = next((s for s in strategies if s['id'] == strategy_id), None)
        if not strategy:
            continue
            
        strategy_names.append(strategy['description'])
        volumes.append(strategy['monthly_volume'])
        assigned_exchanges.append(details.get('exchange', 'Unknown'))
        estimated_fees.append(details.get('estimated_monthly_fee', 0))
        fee_rates.append(details.get('estimated_fee_rate', 0) * 100)  # Convert to percentage
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Strategy allocation by exchange
    ax1.bar(strategy_names, volumes, color='skyblue')
    ax1.set_title('Strategy Volume by Exchange Allocation')
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Monthly Volume (USD)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add exchange annotations
    for i, (volume, exchange) in enumerate(zip(volumes, assigned_exchanges)):
        ax1.annotate(
            exchange,
            xy=(i, volume),
            xytext=(0, 10),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # 2. Estimated fees and rates
    ax2_twin = ax2.twinx()
    
    # Plot estimated fees
    bars = ax2.bar(strategy_names, estimated_fees, color='lightgreen')
    ax2.set_title('Estimated Monthly Fees by Strategy')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Estimated Monthly Fee (USD)', color='green')
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Plot fee rates
    ax2_twin.plot(strategy_names, fee_rates, 'ro-', linewidth=2)
    ax2_twin.set_ylabel('Fee Rate (%)', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # Add fee annotations
    for bar, rate in zip(bars, fee_rates):
        height = bar.get_height()
        ax2.annotate(
            f'${height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom'
        )
    
    # Add total savings annotation
    if 'estimated_monthly_savings' in optimization:
        savings = optimization['estimated_monthly_savings']
        fig.text(
            0.5, 0.01,
            f"Estimated Monthly Savings: ${savings:.2f}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="orange", alpha=0.8)
        )
    
    # Use tuple for rect parameter
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Make room for the savings annotation
    plt.savefig(VISUAL_DIR / "fee_optimization.png")
    plt.close()
    logger.info(f"Saved fee optimization visualization to {VISUAL_DIR / 'fee_optimization.png'}")

def main():
    """Main function to run all visualizations."""
    logger.info("Starting fee visualization example...")
    
    # Setup sample data
    fee_manager = setup_sample_data()
    
    # Run all visualizations
    visualize_exchange_comparison(fee_manager)
    visualize_fee_breakdown(fee_manager)
    visualize_fee_trends(fee_manager)
    visualize_fee_tiers(fee_manager)
    visualize_optimization_suggestions(fee_manager)
    
    logger.info(f"All visualizations saved to {VISUAL_DIR}")
    logger.info("Fee visualization example completed")

if __name__ == "__main__":
    main() 