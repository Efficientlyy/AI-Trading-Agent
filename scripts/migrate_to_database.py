#!/usr/bin/env python
"""
Data migration script for the AI Trading Agent.

This script migrates data from in-memory storage to the database.
"""

import os
import sys
import logging
import json
import yaml
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.database import Base, engine, SessionLocal
from backend.database.repositories import (
    UserRepository, StrategyRepository, OptimizationRepository, 
    BacktestRepository, AssetRepository, OHLCVRepository, SentimentRepository
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def migrate_users():
    """Migrate user data to the database."""
    logger.info("Migrating user data...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repository
        user_repo = UserRepository()
        
        # Check if fake_users_db exists in the API module
        try:
            from backend.api import fake_users_db
            
            # Migrate each user
            for username, user_data in fake_users_db.items():
                # Check if user already exists
                existing = user_repo.get_by_username(db, username)
                if not existing:
                    logger.info(f"Migrating user {username}...")
                    
                    # Create user
                    user = user_repo.create(db, {
                        "username": username,
                        "email": f"{username}@example.com",  # Default email
                        "hashed_password": user_data["hashed_password"],
                        "is_active": not user_data.get("disabled", False),
                        "is_superuser": False
                    })
                    
                    logger.info(f"User {username} migrated successfully")
                else:
                    logger.info(f"User {username} already exists, skipping")
            
            logger.info("User migration completed successfully")
            
        except ImportError:
            logger.warning("No in-memory user data found, skipping user migration")
    
    except Exception as e:
        logger.error(f"Error migrating user data: {e}")
        db.rollback()
    
    finally:
        db.close()


def migrate_strategies():
    """Migrate strategy data to the database."""
    logger.info("Migrating strategy data...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repositories
        user_repo = UserRepository()
        strategy_repo = StrategyRepository()
        
        # Get default user
        default_user = user_repo.get_by_username(db, "test_user")
        if not default_user:
            logger.warning("Default user not found, creating test_user...")
            default_user = user_repo.create_user(
                db=db,
                username="test_user",
                email="test@example.com",
                password="test_password",
                is_superuser=False
            )
        
        # Check for strategies.json file
        strategies_file = os.path.join(project_root, "config", "strategies.json")
        if os.path.exists(strategies_file):
            logger.info(f"Found strategies file: {strategies_file}")
            
            # Load strategies
            with open(strategies_file, "r") as f:
                strategies = json.load(f)
            
            # Migrate each strategy
            for strategy_name, strategy_data in strategies.items():
                # Check if strategy already exists
                existing = strategy_repo.get_strategy_by_name(db, default_user.id, strategy_name)
                if not existing:
                    logger.info(f"Migrating strategy {strategy_name}...")
                    
                    # Create strategy
                    strategy = strategy_repo.create_strategy(
                        db=db,
                        user_id=default_user.id,
                        name=strategy_name,
                        strategy_type=strategy_data["type"],
                        config=strategy_data["config"],
                        description=f"Migrated from strategies.json"
                    )
                    
                    logger.info(f"Strategy {strategy_name} migrated successfully")
                else:
                    logger.info(f"Strategy {strategy_name} already exists, skipping")
            
            logger.info("Strategy migration completed successfully")
        else:
            logger.warning("No strategies.json file found, skipping strategy migration")
    
    except Exception as e:
        logger.error(f"Error migrating strategy data: {e}")
        db.rollback()
    
    finally:
        db.close()


def migrate_backtest_results():
    """Migrate backtest results to the database."""
    logger.info("Migrating backtest results...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repositories
        user_repo = UserRepository()
        strategy_repo = StrategyRepository()
        backtest_repo = BacktestRepository()
        
        # Get default user
        default_user = user_repo.get_by_username(db, "test_user")
        if not default_user:
            logger.warning("Default user not found, skipping backtest migration")
            return
        
        # Check for backtest results directory
        results_dir = os.path.join(project_root, "results")
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            logger.info(f"Found results directory: {results_dir}")
            
            # Find backtest result files
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    if file.endswith(".json") and "backtest" in file.lower():
                        result_file = os.path.join(root, file)
                        logger.info(f"Found backtest result file: {result_file}")
                        
                        try:
                            # Load backtest results
                            with open(result_file, "r") as f:
                                backtest_data = json.load(f)
                            
                            # Extract strategy name
                            strategy_name = backtest_data.get("strategy_name", "Unknown Strategy")
                            
                            # Get strategy
                            strategy = strategy_repo.get_strategy_by_name(db, default_user.id, strategy_name)
                            if not strategy:
                                logger.warning(f"Strategy {strategy_name} not found, creating placeholder...")
                                strategy = strategy_repo.create_strategy(
                                    db=db,
                                    user_id=default_user.id,
                                    name=strategy_name,
                                    strategy_type="Unknown",
                                    config={},
                                    description=f"Placeholder for migrated backtest"
                                )
                            
                            # Create backtest
                            backtest_name = os.path.splitext(file)[0]
                            
                            # Extract dates
                            start_date = backtest_data.get("start_date")
                            end_date = backtest_data.get("end_date")
                            
                            # Convert string dates to datetime if needed
                            if isinstance(start_date, str):
                                try:
                                    start_date = datetime.fromisoformat(start_date)
                                except ValueError:
                                    start_date = datetime.now()
                            
                            if isinstance(end_date, str):
                                try:
                                    end_date = datetime.fromisoformat(end_date)
                                except ValueError:
                                    end_date = datetime.now()
                            
                            # Default dates if not available
                            if not start_date:
                                start_date = datetime.now()
                            if not end_date:
                                end_date = datetime.now()
                            
                            # Extract parameters
                            parameters = backtest_data.get("parameters", {})
                            if not parameters:
                                # Try to extract from strategy config
                                parameters = backtest_data.get("strategy_config", {})
                            
                            # Create backtest
                            backtest = backtest_repo.create_backtest(
                                db=db,
                                user_id=default_user.id,
                                strategy_id=strategy.id,
                                name=backtest_name,
                                parameters=parameters,
                                start_date=start_date,
                                end_date=end_date,
                                initial_capital=backtest_data.get("initial_capital", 10000.0),
                                description=f"Migrated from {result_file}"
                            )
                            
                            # Update backtest status and results
                            backtest_repo.update_backtest_status(
                                db=db,
                                backtest_id=backtest.id,
                                user_id=default_user.id,
                                status="completed",
                                results=backtest_data.get("performance_metrics", {})
                            )
                            
                            # Migrate trades if available
                            trades = backtest_data.get("trades", [])
                            if trades:
                                logger.info(f"Migrating {len(trades)} trades...")
                                backtest_repo.add_trades(db, backtest.id, trades)
                            
                            logger.info(f"Backtest {backtest_name} migrated successfully")
                        
                        except Exception as e:
                            logger.error(f"Error migrating backtest from {result_file}: {e}")
                            continue
            
            logger.info("Backtest migration completed successfully")
        else:
            logger.warning("No results directory found, skipping backtest migration")
    
    except Exception as e:
        logger.error(f"Error migrating backtest data: {e}")
        db.rollback()
    
    finally:
        db.close()


def migrate_market_data():
    """Migrate market data to the database."""
    logger.info("Migrating market data...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repositories
        asset_repo = AssetRepository()
        ohlcv_repo = OHLCVRepository()
        
        # Check for market data directory
        data_dir = os.path.join(project_root, "data")
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            logger.info(f"Found data directory: {data_dir}")
            
            # Find market data files
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith((".csv", ".json")) and "ohlcv" in file.lower():
                        data_file = os.path.join(root, file)
                        logger.info(f"Found market data file: {data_file}")
                        
                        try:
                            # Extract symbol from filename
                            symbol = file.split("_")[0] if "_" in file else os.path.splitext(file)[0]
                            
                            # Get or create asset
                            asset = asset_repo.get_by_symbol(db, symbol)
                            if not asset:
                                logger.info(f"Creating asset for {symbol}...")
                                asset = asset_repo.create_asset(
                                    db=db,
                                    symbol=symbol,
                                    name=symbol,
                                    asset_type="unknown"
                                )
                            
                            # Load market data
                            if file.endswith(".csv"):
                                import pandas as pd
                                df = pd.read_csv(data_file)
                                
                                # Convert to OHLCV format
                                ohlcv_data = []
                                for _, row in df.iterrows():
                                    # Extract timestamp
                                    timestamp = None
                                    for col in ["timestamp", "date", "time", "datetime"]:
                                        if col in row:
                                            try:
                                                timestamp = pd.to_datetime(row[col])
                                                break
                                            except:
                                                pass
                                    
                                    if timestamp is None:
                                        continue
                                    
                                    # Extract OHLCV data
                                    item = {
                                        "timestamp": timestamp,
                                        "open": row.get("open", 0.0),
                                        "high": row.get("high", 0.0),
                                        "low": row.get("low", 0.0),
                                        "close": row.get("close", 0.0),
                                        "volume": row.get("volume", 0.0)
                                    }
                                    
                                    ohlcv_data.append(item)
                                
                                # Insert OHLCV data
                                if ohlcv_data:
                                    logger.info(f"Migrating {len(ohlcv_data)} OHLCV records for {symbol}...")
                                    ohlcv_repo.bulk_insert_ohlcv(db, asset.id, "1d", ohlcv_data)
                            
                            elif file.endswith(".json"):
                                with open(data_file, "r") as f:
                                    data = json.load(f)
                                
                                # Check if it's an array of OHLCV data
                                if isinstance(data, list):
                                    ohlcv_data = []
                                    for item in data:
                                        if isinstance(item, dict):
                                            # Extract timestamp
                                            timestamp = None
                                            for key in ["timestamp", "date", "time", "datetime"]:
                                                if key in item:
                                                    try:
                                                        if isinstance(item[key], (int, float)):
                                                            # Assume Unix timestamp in milliseconds
                                                            timestamp = datetime.fromtimestamp(item[key] / 1000)
                                                        else:
                                                            timestamp = datetime.fromisoformat(item[key])
                                                        break
                                                    except:
                                                        pass
                                            
                                            if timestamp is None:
                                                continue
                                            
                                            # Extract OHLCV data
                                            ohlcv_item = {
                                                "timestamp": timestamp,
                                                "open": item.get("open", 0.0),
                                                "high": item.get("high", 0.0),
                                                "low": item.get("low", 0.0),
                                                "close": item.get("close", 0.0),
                                                "volume": item.get("volume", 0.0)
                                            }
                                            
                                            ohlcv_data.append(ohlcv_item)
                                    
                                    # Insert OHLCV data
                                    if ohlcv_data:
                                        logger.info(f"Migrating {len(ohlcv_data)} OHLCV records for {symbol}...")
                                        ohlcv_repo.bulk_insert_ohlcv(db, asset.id, "1d", ohlcv_data)
                            
                            logger.info(f"Market data for {symbol} migrated successfully")
                        
                        except Exception as e:
                            logger.error(f"Error migrating market data from {data_file}: {e}")
                            continue
            
            logger.info("Market data migration completed successfully")
        else:
            logger.warning("No data directory found, skipping market data migration")
    
    except Exception as e:
        logger.error(f"Error migrating market data: {e}")
        db.rollback()
    
    finally:
        db.close()


def main():
    """Migrate data to the database."""
    logger.info("Starting data migration...")
    
    # Migrate users
    migrate_users()
    
    # Migrate strategies
    migrate_strategies()
    
    # Migrate backtest results
    migrate_backtest_results()
    
    # Migrate market data
    migrate_market_data()
    
    logger.info("Data migration completed successfully")


if __name__ == "__main__":
    main()
