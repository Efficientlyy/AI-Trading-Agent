#!/usr/bin/env python
"""
Database operations test script for the AI Trading Agent.

This script tests all database operations to ensure repositories and models are working correctly.
"""

import os
import sys
import logging
import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.database import SessionLocal
from backend.database.repositories import (
    UserRepository, StrategyRepository, OptimizationRepository,
    BacktestRepository, AssetRepository, OHLCVRepository, SentimentRepository
)
from backend.database.models import (
    User, Strategy, Optimization, Backtest, Trade, PortfolioSnapshot,
    Asset, OHLCV, SentimentData
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def test_user_repository():
    """Test UserRepository operations."""
    logger.info("Testing UserRepository operations...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repository
        user_repo = UserRepository()
        
        # Test user creation
        test_username = f"test_user_{uuid.uuid4().hex[:8]}"
        test_email = f"{test_username}@example.com"
        
        logger.info(f"Creating test user: {test_username}")
        user = user_repo.create_user(
            db=db,
            username=test_username,
            email=test_email,
            password="test_password",
            is_superuser=False
        )
        
        assert user is not None, "User creation failed"
        assert user.username == test_username, "Username mismatch"
        
        # Test user retrieval
        retrieved_user = user_repo.get_by_username(db, test_username)
        assert retrieved_user is not None, "User retrieval failed"
        assert retrieved_user.id == user.id, "User ID mismatch"
        
        # Test user authentication
        is_authenticated = user_repo.authenticate(
            db=db,
            username=test_username,
            password="test_password"
        )
        assert is_authenticated is not None, "User authentication failed"
        
        # Test user update
        db_user = user_repo.get(db, user.id)
        updated_user = user_repo.update(
            db=db,
            db_obj=db_user,
            obj_in={"email": f"updated_{test_email}"}
        )
        assert updated_user is not None, "User update failed"
        assert updated_user.email == f"updated_{test_email}", "Email update failed"
        
        # Test user deletion
        user_repo.delete(db, user.id)
        deleted_user = user_repo.get(db, user.id)
        assert deleted_user is None, "User deletion failed"
        
        logger.info("UserRepository operations tested successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing UserRepository operations: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


def test_strategy_repository():
    """Test StrategyRepository operations."""
    logger.info("Testing StrategyRepository operations...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repositories
        user_repo = UserRepository()
        strategy_repo = StrategyRepository()
        optimization_repo = OptimizationRepository()
        
        # Create test user
        test_username = f"strategy_test_user_{uuid.uuid4().hex[:8]}"
        user = user_repo.create_user(
            db=db,
            username=test_username,
            email=f"{test_username}@example.com",
            password="test_password",
            is_superuser=False
        )
        
        # Test strategy creation
        strategy_name = f"Test Strategy {uuid.uuid4().hex[:8]}"
        strategy_config = {
            "fast_period": 10,
            "slow_period": 50,
            "threshold": 0.0,
            "position_size_pct": 0.1
        }
        
        logger.info(f"Creating test strategy: {strategy_name}")
        strategy = strategy_repo.create_strategy(
            db=db,
            user_id=user.id,
            name=strategy_name,
            strategy_type="MovingAverageCrossover",
            config=strategy_config,
            description="A test strategy"
        )
        
        assert strategy is not None, "Strategy creation failed"
        assert strategy.name == strategy_name, "Strategy name mismatch"
        
        # Test strategy retrieval
        retrieved_strategy = strategy_repo.get_strategy_by_name(
            db=db,
            user_id=user.id,
            name=strategy_name
        )
        assert retrieved_strategy is not None, "Strategy retrieval failed"
        assert retrieved_strategy.id == strategy.id, "Strategy ID mismatch"
        
        # Test strategy update
        updated_config = strategy_config.copy()
        updated_config["fast_period"] = 20
        
        db_strategy = strategy_repo.get(db, strategy.id)
        updated_strategy = strategy_repo.update(
            db=db,
            db_obj=db_strategy,
            obj_in={"config": updated_config}
        )
        assert updated_strategy is not None, "Strategy update failed"
        assert updated_strategy.config["fast_period"] == 20, "Strategy config update failed"
        
        # Test optimization creation
        optimization_params = {
            "fast_period": {"min": 5, "max": 20, "step": 5},
            "slow_period": {"min": 20, "max": 100, "step": 20}
        }
        
        logger.info(f"Creating test optimization for strategy: {strategy_name}")
        optimization = optimization_repo.create_optimization(
            db=db,
            user_id=user.id,
            strategy_id=strategy.id,
            name=f"Optimization for {strategy_name}",
            optimization_type="grid_search",
            parameters=optimization_params,
            description="Test optimization"
        )
        
        assert optimization is not None, "Optimization creation failed"
        assert optimization.strategy_id == strategy.id, "Optimization strategy ID mismatch"
        
        # Test optimization update
        updated_optimization = optimization_repo.update_optimization_status(
            db=db,
            optimization_id=optimization.id,
            user_id=user.id,
            status="completed",
            results={"best_params": {"fast_period": 10, "slow_period": 40}},
            best_parameters={"fast_period": 10, "slow_period": 40}
        )
        assert updated_optimization is not None, "Optimization update failed"
        assert updated_optimization.status == "completed", "Optimization status update failed"
        
        # Test strategy deletion
        strategy_repo.delete(db, strategy.id)
        deleted_strategy = strategy_repo.get(db, strategy.id)
        assert deleted_strategy is None, "Strategy deletion failed"
        
        # Clean up test user
        user_repo.delete(db, user.id)
        
        logger.info("StrategyRepository operations tested successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing StrategyRepository operations: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


def test_asset_repository():
    """Test AssetRepository operations."""
    logger.info("Testing AssetRepository operations...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repository
        asset_repo = AssetRepository()
        
        # Test asset creation
        test_symbol = f"TEST_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Creating test asset: {test_symbol}")
        asset_data = {
            "symbol": test_symbol,
            "name": f"Test Asset {test_symbol}",
            "asset_type": "crypto",
            "is_active": True
        }
        
        asset = asset_repo.create(db, asset_data)
        
        assert asset is not None, "Asset creation failed"
        assert asset.symbol == test_symbol, "Asset symbol mismatch"
        
        # Test asset retrieval
        retrieved_asset = asset_repo.get_by_symbol(db, test_symbol)
        assert retrieved_asset is not None, "Asset retrieval failed"
        assert retrieved_asset.id == asset.id, "Asset ID mismatch"
        
        # Test asset update
        updated_asset = asset_repo.update_asset(
            db=db,
            asset_id=asset.id,
            name=f"Updated {test_symbol}"
        )
        assert updated_asset is not None, "Asset update failed"
        assert updated_asset.name == f"Updated {test_symbol}", "Asset name update failed"
        
        # Test asset listing
        active_assets = asset_repo.get_active_assets(db)
        assert len(active_assets) > 0, "Active assets listing failed"
        
        # Test asset deletion
        asset_repo.delete(db, asset.id)
        deleted_asset = asset_repo.get(db, asset.id)
        assert deleted_asset is None, "Asset deletion failed"
        
        logger.info("AssetRepository operations tested successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing AssetRepository operations: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


def test_market_data_repository():
    """Test market data repository operations."""
    logger.info("Testing market data repository operations...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repositories
        asset_repo = AssetRepository()
        ohlcv_repo = OHLCVRepository()
        sentiment_repo = SentimentRepository()
        
        # Create test asset
        test_symbol = f"MARKET_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Creating test asset: {test_symbol}")
        asset_data = {
            "symbol": test_symbol,
            "name": f"Test Asset {test_symbol}",
            "asset_type": "crypto",
            "is_active": True
        }
        
        asset = asset_repo.create(db, asset_data)
        
        # Test OHLCV data creation
        logger.info("Creating test OHLCV data")
        start_date = datetime.now() - timedelta(days=30)
        ohlcv_data = []
        
        for i in range(30):
            data_date = start_date + timedelta(days=i)
            ohlcv_data.append({
                "asset_id": asset.id,
                "timeframe": "1d",
                "timestamp": data_date,
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 1000.0 + i * 10
            })
        
        # Bulk insert OHLCV data
        inserted_count = ohlcv_repo.bulk_insert_ohlcv(db, ohlcv_data)
        assert inserted_count == 30, "OHLCV data insertion failed"
        
        # Test OHLCV data retrieval
        retrieved_data = ohlcv_repo.get_ohlcv_data(
            db=db,
            asset_id=asset.id,
            timeframe="1d",
            start_date=start_date,
            end_date=datetime.now()
        )
        assert len(retrieved_data) == 30, "OHLCV data retrieval failed"
        
        # Test latest OHLCV data retrieval
        latest_data = ohlcv_repo.get_latest_ohlcv(
            db=db,
            asset_id=asset.id,
            timeframe="1d",
            limit=5
        )
        assert len(latest_data) == 5, "Latest OHLCV data retrieval failed"
        
        # Test sentiment data creation
        logger.info("Creating test sentiment data")
        sentiment_data = []
        
        for i in range(30):
            data_date = start_date + timedelta(days=i)
            sentiment_data.append({
                "asset_id": asset.id,
                "source": "twitter" if i % 2 == 0 else "news",
                "timestamp": data_date,
                "sentiment_score": -0.5 + i * 0.03,
                "volume": 100 + i * 5
            })
        
        # Bulk insert sentiment data
        inserted_count = sentiment_repo.bulk_insert_sentiment(db, sentiment_data)
        assert inserted_count == 30, "Sentiment data insertion failed"
        
        # Test sentiment data retrieval
        retrieved_sentiment = sentiment_repo.get_sentiment_data(
            db=db,
            asset_id=asset.id,
            start_date=start_date,
            end_date=datetime.now()
        )
        assert len(retrieved_sentiment) == 30, "Sentiment data retrieval failed"
        
        # Test sentiment data retrieval with source filter
        twitter_sentiment = sentiment_repo.get_sentiment_data(
            db=db,
            asset_id=asset.id,
            source="twitter",
            start_date=start_date,
            end_date=datetime.now()
        )
        assert len(twitter_sentiment) == 15, "Sentiment data retrieval with source filter failed"
        
        # Test average sentiment calculation
        avg_sentiment = sentiment_repo.get_average_sentiment(
            db=db,
            asset_id=asset.id,
            days=30
        )
        assert isinstance(avg_sentiment, float), "Average sentiment calculation failed"
        
        # Clean up test data
        db.query(OHLCV).filter(OHLCV.asset_id == asset.id).delete()
        db.query(SentimentData).filter(SentimentData.asset_id == asset.id).delete()
        db.query(Asset).filter(Asset.id == asset.id).delete()
        db.commit()
        
        logger.info("Market data repository operations tested successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing market data repository operations: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


def test_backtest_repository():
    """Test BacktestRepository operations."""
    logger.info("Testing BacktestRepository operations...")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create repositories
        user_repo = UserRepository()
        strategy_repo = StrategyRepository()
        backtest_repo = BacktestRepository()
        
        # Create test user
        test_username = f"backtest_test_user_{uuid.uuid4().hex[:8]}"
        user = user_repo.create_user(
            db=db,
            username=test_username,
            email=f"{test_username}@example.com",
            password="test_password",
            is_superuser=False
        )
        
        # Create test strategy
        strategy_name = f"Backtest Test Strategy {uuid.uuid4().hex[:8]}"
        strategy = strategy_repo.create_strategy(
            db=db,
            user_id=user.id,
            name=strategy_name,
            strategy_type="MovingAverageCrossover",
            config={"fast_period": 10, "slow_period": 50},
            description="A test strategy for backtesting"
        )
        
        # Test backtest creation
        backtest_name = f"Test Backtest {uuid.uuid4().hex[:8]}"
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        logger.info(f"Creating test backtest: {backtest_name}")
        backtest = backtest_repo.create_backtest(
            db=db,
            user_id=user.id,
            strategy_id=strategy.id,
            name=backtest_name,
            parameters={"test_param": "test_value"},
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            description="Test backtest description"
        )
        
        assert backtest is not None, "Backtest creation failed"
        assert backtest.name == backtest_name, "Backtest name mismatch"
        
        # Test backtest status update
        updated_backtest = backtest_repo.update_backtest_status(
            db=db,
            backtest_id=backtest.id,
            user_id=user.id,
            status="completed",
            results={
                "final_equity": 13000.0,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "total_trades": 10,
                "win_rate": 0.7
            }
        )
        
        assert updated_backtest is not None, "Backtest update failed"
        assert updated_backtest.status == "completed", "Backtest status update failed"
        
        # Test trade creation
        logger.info("Creating test trades")
        trades_data = []
        
        for i in range(10):
            trade_date = start_date + timedelta(days=i)
            trades_data.append({
                "symbol": "TEST/USD",
                "order_type": "market",
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": 1.0,
                "price": 100.0 + i,
                "timestamp": trade_date,
                "commission": 0.1,
                "slippage": 0.05
            })
        
        trades = backtest_repo.add_trades(db, backtest.id, trades_data)
        assert len(trades) == 10, "Trade creation failed"
        
        # Test portfolio snapshot creation
        logger.info("Creating test portfolio snapshots")
        snapshots_data = []
        
        for i in range(30):
            snapshot_date = start_date + timedelta(days=i)
            snapshots_data.append({
                "timestamp": snapshot_date,
                "equity": 10000.0 + i * 100,
                "cash": 5000.0 + i * 50,
                "positions": {"TEST/USD": 1.0}
            })
        
        snapshots = backtest_repo.add_portfolio_snapshots(db, backtest.id, snapshots_data)
        assert len(snapshots) == 30, "Portfolio snapshot creation failed"
        
        # Test backtest retrieval
        retrieved_backtests = backtest_repo.get_strategy_backtests(
            db=db,
            strategy_id=strategy.id,
            user_id=user.id
        )
        
        assert len(retrieved_backtests) > 0, "Backtest retrieval failed"
        assert retrieved_backtests[0].id == backtest.id, "Backtest ID mismatch"
        
        # Test trade retrieval
        retrieved_trades = backtest_repo.get_backtest_trades(
            db=db,
            backtest_id=backtest.id,
            user_id=user.id
        )
        assert len(retrieved_trades) == 10, "Trade retrieval failed"
        
        # Test portfolio snapshot retrieval
        retrieved_snapshots = backtest_repo.get_backtest_portfolio_snapshots(
            db=db,
            backtest_id=backtest.id,
            user_id=user.id
        )
        assert len(retrieved_snapshots) == 30, "Portfolio snapshot retrieval failed"
        
        # Clean up test data
        db.query(Trade).filter(Trade.backtest_id == backtest.id).delete()
        db.query(PortfolioSnapshot).filter(PortfolioSnapshot.backtest_id == backtest.id).delete()
        db.query(Backtest).filter(Backtest.id == backtest.id).delete()
        db.query(Strategy).filter(Strategy.id == strategy.id).delete()
        db.query(User).filter(User.id == user.id).delete()
        db.commit()
        
        logger.info("BacktestRepository operations tested successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing BacktestRepository operations: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


def main():
    """Run all database operation tests."""
    logger.info("Starting database operation tests...")
    
    # Run tests
    tests = [
        ("UserRepository", test_user_repository),
        ("StrategyRepository", test_strategy_repository),
        ("AssetRepository", test_asset_repository),
        ("Market Data Repositories", test_market_data_repository),
        ("BacktestRepository", test_backtest_repository)
    ]
    
    results = []
    
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        success = test_func()
        results.append((name, success))
    
    # Print results
    logger.info("\nTest Results:")
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name}: {status}")
    
    # Check if all tests passed
    all_passed = all(success for _, success in results)
    
    if all_passed:
        logger.info("\nAll database operation tests passed successfully!")
    else:
        logger.error("\nSome database operation tests failed!")
    
    return all_passed


if __name__ == "__main__":
    main()
