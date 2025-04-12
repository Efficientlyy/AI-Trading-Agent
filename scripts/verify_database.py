#!/usr/bin/env python
"""
Database verification script for the AI Trading Agent.

This script verifies the database setup and checks the data that has been migrated.
"""

import os
import sys
import logging
from tabulate import tabulate

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.database import SessionLocal
from backend.database.models import (
    User, Strategy, Backtest, Asset, OHLCV, SentimentData,
    Trade, PortfolioSnapshot, Optimization
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def count_records(db, model):
    """Count records in a table."""
    return db.query(model).count()


def print_table_counts(db):
    """Print record counts for all tables."""
    tables = [
        ("Users", User),
        ("Strategies", Strategy),
        ("Optimizations", Optimization),
        ("Backtests", Backtest),
        ("Trades", Trade),
        ("Portfolio Snapshots", PortfolioSnapshot),
        ("Assets", Asset),
        ("OHLCV Data", OHLCV),
        ("Sentiment Data", SentimentData),
    ]
    
    counts = [(name, count_records(db, model)) for name, model in tables]
    
    print("\nDatabase Record Counts:")
    print(tabulate(counts, headers=["Table", "Count"], tablefmt="grid"))


def print_users(db):
    """Print user information."""
    users = db.query(User).all()
    
    if not users:
        logger.info("No users found in the database")
        return
    
    user_data = []
    for user in users:
        user_data.append([
            user.id,
            user.username,
            user.email,
            "Yes" if user.is_active else "No",
            "Yes" if user.is_superuser else "No"
        ])
    
    print("\nUsers:")
    print(tabulate(
        user_data,
        headers=["ID", "Username", "Email", "Active", "Admin"],
        tablefmt="grid"
    ))


def print_strategies(db):
    """Print strategy information."""
    strategies = db.query(Strategy).all()
    
    if not strategies:
        logger.info("No strategies found in the database")
        return
    
    strategy_data = []
    for strategy in strategies:
        strategy_data.append([
            strategy.id,
            strategy.name,
            strategy.strategy_type,
            strategy.description[:50] + "..." if strategy.description and len(strategy.description) > 50 else strategy.description,
            strategy.user_id
        ])
    
    print("\nStrategies:")
    print(tabulate(
        strategy_data,
        headers=["ID", "Name", "Type", "Description", "User ID"],
        tablefmt="grid"
    ))


def print_assets(db):
    """Print asset information."""
    assets = db.query(Asset).all()
    
    if not assets:
        logger.info("No assets found in the database")
        return
    
    asset_data = []
    for asset in assets:
        ohlcv_count = db.query(OHLCV).filter(OHLCV.asset_id == asset.id).count()
        asset_data.append([
            asset.id,
            asset.symbol,
            asset.name,
            asset.asset_type,
            ohlcv_count
        ])
    
    print("\nAssets:")
    print(tabulate(
        asset_data,
        headers=["ID", "Symbol", "Name", "Type", "OHLCV Records"],
        tablefmt="grid"
    ))


def print_backtests(db):
    """Print backtest information."""
    backtests = db.query(Backtest).all()
    
    if not backtests:
        logger.info("No backtests found in the database")
        return
    
    backtest_data = []
    for backtest in backtests:
        trade_count = db.query(Trade).filter(Trade.backtest_id == backtest.id).count()
        snapshot_count = db.query(PortfolioSnapshot).filter(PortfolioSnapshot.backtest_id == backtest.id).count()
        
        backtest_data.append([
            backtest.id,
            backtest.name,
            backtest.strategy_id,
            backtest.start_date.strftime("%Y-%m-%d") if backtest.start_date else "N/A",
            backtest.end_date.strftime("%Y-%m-%d") if backtest.end_date else "N/A",
            f"{backtest.final_equity:.2f}" if backtest.final_equity else "N/A",
            trade_count,
            snapshot_count
        ])
    
    print("\nBacktests:")
    print(tabulate(
        backtest_data,
        headers=["ID", "Name", "Strategy ID", "Start Date", "End Date", "Final Equity", "Trades", "Snapshots"],
        tablefmt="grid"
    ))


def main():
    """Verify database setup."""
    logger.info("Verifying database setup...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Print table counts
        print_table_counts(db)
        
        # Print detailed information
        print_users(db)
        print_strategies(db)
        print_assets(db)
        print_backtests(db)
        
        logger.info("Database verification completed successfully")
    
    except Exception as e:
        logger.error(f"Error verifying database: {e}")
    
    finally:
        db.close()


if __name__ == "__main__":
    main()
