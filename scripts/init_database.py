#!/usr/bin/env python
"""
Database initialization script for the AI Trading Agent.

This script initializes the database by creating all tables and setting up initial data.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.database import Base, engine, SessionLocal, init_db
from backend.database.models import (
    User, Strategy, Backtest, Asset, OHLCV, SentimentData
)
from backend.database.repositories import (
    UserRepository, StrategyRepository, BacktestRepository,
    AssetRepository, OHLCVRepository, SentimentRepository
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repositories
        user_repo = UserRepository()
        strategy_repo = StrategyRepository()
        backtest_repo = BacktestRepository()
        asset_repo = AssetRepository()
        
        # Create test user if it doesn't exist
        test_user = user_repo.get_by_username(db, "test_user")
        if not test_user:
            logger.info("Creating test user...")
            test_user = user_repo.create_user(
                db=db,
                username="test_user",
                email="test@example.com",
                password="test_password",
                is_superuser=False
            )
        
        # Create sample assets
        assets = [
            {"symbol": "BTC/USD", "name": "Bitcoin", "asset_type": "crypto"},
            {"symbol": "ETH/USD", "name": "Ethereum", "asset_type": "crypto"},
            {"symbol": "SOL/USD", "name": "Solana", "asset_type": "crypto"},
            {"symbol": "AAPL", "name": "Apple Inc.", "asset_type": "stock"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "asset_type": "stock"},
            {"symbol": "AMZN", "name": "Amazon.com, Inc.", "asset_type": "stock"}
        ]
        
        for asset_data in assets:
            existing = asset_repo.get_by_symbol(db, asset_data["symbol"])
            if not existing:
                logger.info(f"Creating asset {asset_data['symbol']}...")
                asset_repo.create_asset(
                    db=db,
                    symbol=asset_data["symbol"],
                    name=asset_data["name"],
                    asset_type=asset_data["asset_type"]
                )
        
        # Create sample strategies
        strategies = [
            {
                "name": "Moving Average Crossover",
                "strategy_type": "MovingAverageCrossover",
                "config": {
                    "fast_period": 10,
                    "slow_period": 50,
                    "threshold": 0.0,
                    "position_size_pct": 0.1
                },
                "description": "A simple moving average crossover strategy"
            },
            {
                "name": "RSI Strategy",
                "strategy_type": "RSIStrategy",
                "config": {
                    "rsi_period": 14,
                    "overbought_threshold": 70,
                    "oversold_threshold": 30,
                    "position_size_pct": 0.1,
                    "exit_rsi": 50
                },
                "description": "A relative strength index strategy"
            },
            {
                "name": "Sentiment Strategy",
                "strategy_type": "SentimentStrategy",
                "config": {
                    "sentiment_threshold": 0.3,
                    "position_size_pct": 0.1,
                    "exit_threshold": 0.1,
                    "lookback_period": 3
                },
                "description": "A strategy based on sentiment analysis"
            }
        ]
        
        for strategy_data in strategies:
            existing = strategy_repo.get_strategy_by_name(db, test_user.id, strategy_data["name"])
            if not existing:
                logger.info(f"Creating strategy {strategy_data['name']}...")
                strategy_repo.create_strategy(
                    db=db,
                    user_id=test_user.id,
                    name=strategy_data["name"],
                    strategy_type=strategy_data["strategy_type"],
                    config=strategy_data["config"],
                    description=strategy_data["description"]
                )
        
        logger.info("Sample data created successfully")
    
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        db.rollback()
    
    finally:
        db.close()


def main():
    """Initialize the database."""
    logger.info("Initializing database...")
    
    # Create tables
    if init_db():
        logger.info("Database tables created successfully")
        
        # Create sample data
        create_sample_data()
        
        logger.info("Database initialization completed successfully")
    else:
        logger.error("Error initializing database")
        sys.exit(1)


if __name__ == "__main__":
    main()
