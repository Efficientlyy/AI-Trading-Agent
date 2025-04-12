"""
Performance tests for API endpoints with large datasets.
"""

import unittest
import time
import json
import random
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.api import app
from backend.database.config import get_db, SessionLocal
from backend.database.models.user import User
from backend.database.models.market_data import Asset, OHLCV
from backend.database.models.strategy import Strategy
from backend.database.models.backtest import Backtest

# Create test client
client = TestClient(app)

class TestAPIPerformance(unittest.TestCase):
    """Test API performance with large datasets."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        cls.db = SessionLocal()
        
        # Create test user
        cls.create_test_user(cls.db)
        
        # Create test assets
        cls.create_test_assets(cls.db)
        
        # Create test market data
        cls.create_test_market_data(cls.db)
        
        # Create test strategies
        cls.create_test_strategies(cls.db)
        
        # Create test backtests
        cls.create_test_backtests(cls.db)
        
        # Get authentication token
        cls.auth_token = cls.get_auth_token()
        cls.auth_headers = {"Authorization": f"Bearer {cls.auth_token}"}
        
        cls.db.close()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        db = SessionLocal()
        
        # Clean up backtests
        db.query(Backtest).filter(Backtest.user_id == cls.test_user.id).delete()
        
        # Clean up strategies
        db.query(Strategy).filter(Strategy.user_id == cls.test_user.id).delete()
        
        # Clean up market data
        for asset in cls.test_assets:
            db.query(OHLCV).filter(OHLCV.symbol == asset.symbol).delete()
        
        # Clean up assets
        for asset in cls.test_assets:
            db.query(Asset).filter(Asset.symbol == asset.symbol).delete()
        
        # Clean up user
        db.query(User).filter(User.id == cls.test_user.id).delete()
        
        db.commit()
        db.close()
    
    @classmethod
    def create_test_user(cls, db):
        """Create a test user."""
        # Check if test user already exists
        test_user = db.query(User).filter(User.username == "performance_test_user").first()
        
        if not test_user:
            # Create test user
            test_user = User(
                username="performance_test_user",
                email="performance_test@example.com",
                hashed_password="$2b$12$IrOt5sBJMD9mE6MJTQZrhOQFEf/CpXCaO3nrgvJWcFJZgZC5pOlUy"  # hashed 'test_password'
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)
        
        cls.test_user = test_user
    
    @classmethod
    def create_test_assets(cls, db):
        """Create test assets."""
        cls.test_assets = []
        
        # Create 10 test assets
        for i in range(10):
            symbol = f"PERF_ASSET_{i}"
            
            # Check if asset already exists
            asset = db.query(Asset).filter(Asset.symbol == symbol).first()
            
            if not asset:
                # Create asset
                asset = Asset(
                    symbol=symbol,
                    name=f"Performance Test Asset {i}",
                    asset_type="crypto",
                    is_active=True
                )
                db.add(asset)
                db.commit()
                db.refresh(asset)
            
            cls.test_assets.append(asset)
    
    @classmethod
    def create_test_market_data(cls, db):
        """Create test market data."""
        from datetime import datetime, timedelta
        import numpy as np
        
        # Generate 1 year of daily data for each asset
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)
        
        for asset in cls.test_assets:
            # Check if data already exists
            existing_data = db.query(OHLCV).filter(
                OHLCV.symbol == asset.symbol,
                OHLCV.timestamp >= start_date,
                OHLCV.timestamp <= end_date
            ).count()
            
            if existing_data > 0:
                # Skip if data already exists
                continue
            
            # Generate random price data
            np.random.seed(42 + cls.test_assets.index(asset))  # Different seed for each asset
            
            # Start with a base price
            base_price = 100.0 * (1 + cls.test_assets.index(asset) / 10)
            
            # Generate dates
            dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]
            
            # Generate random price movements
            price_changes = np.random.normal(0, 0.02, len(dates))
            
            # Calculate cumulative price changes
            cumulative_changes = np.cumsum(price_changes)
            
            # Calculate close prices
            close_prices = base_price * (1 + cumulative_changes)
            
            # Generate OHLCV data
            ohlcv_data = []
            for i, date in enumerate(dates):
                close = close_prices[i]
                ohlcv = OHLCV(
                    symbol=asset.symbol,
                    timestamp=date,
                    open=close_prices[i-1] if i > 0 else close,
                    high=close * (1 + np.random.uniform(0, 0.01)),
                    low=close * (1 - np.random.uniform(0, 0.01)),
                    close=close,
                    volume=np.random.uniform(1000, 10000)
                )
                ohlcv_data.append(ohlcv)
            
            # Add to database in chunks to avoid memory issues
            chunk_size = 50
            for i in range(0, len(ohlcv_data), chunk_size):
                chunk = ohlcv_data[i:i+chunk_size]
                db.add_all(chunk)
                db.commit()
    
    @classmethod
    def create_test_strategies(cls, db):
        """Create test strategies."""
        cls.test_strategies = []
        
        # Create 5 test strategies
        for i in range(5):
            name = f"Performance_Strategy_{i}"
            
            # Check if strategy already exists
            strategy = db.query(Strategy).filter(
                Strategy.name == name,
                Strategy.user_id == cls.test_user.id
            ).first()
            
            if not strategy:
                # Create strategy
                strategy = Strategy(
                    name=name,
                    strategy_type="momentum" if i % 2 == 0 else "mean_reversion",
                    config={
                        "param1": 10 + i,
                        "param2": 20 + i * 2
                    },
                    description=f"Performance test strategy {i}",
                    is_public=False,
                    user_id=cls.test_user.id
                )
                db.add(strategy)
                db.commit()
                db.refresh(strategy)
            
            cls.test_strategies.append(strategy)
    
    @classmethod
    def create_test_backtests(cls, db):
        """Create test backtests."""
        cls.test_backtests = []
        
        # Create 20 test backtests (4 for each strategy)
        for strategy in cls.test_strategies:
            for asset in cls.test_assets[:4]:  # Use first 4 assets for each strategy
                # Check if backtest already exists
                backtest = db.query(Backtest).filter(
                    Backtest.strategy_id == strategy.id,
                    Backtest.asset_symbol == asset.symbol,
                    Backtest.user_id == cls.test_user.id
                ).first()
                
                if not backtest:
                    # Create backtest
                    backtest = Backtest(
                        strategy_id=strategy.id,
                        user_id=cls.test_user.id,
                        asset_symbol=asset.symbol,
                        start_date="2023-01-01",
                        end_date="2023-12-31",
                        initial_capital=10000,
                        parameters={
                            "param1": strategy.config["param1"],
                            "param2": strategy.config["param2"]
                        },
                        status="completed",
                        metrics={
                            "total_return": random.uniform(5, 25),
                            "sharpe_ratio": random.uniform(0.8, 2.5),
                            "max_drawdown": random.uniform(5, 15),
                            "win_rate": random.uniform(40, 70)
                        }
                    )
                    db.add(backtest)
                    db.commit()
                    db.refresh(backtest)
                
                cls.test_backtests.append(backtest)
    
    @classmethod
    def get_auth_token(cls):
        """Get authentication token for test user."""
        response = client.post(
            "/auth/login",
            data={"username": "performance_test_user", "password": "test_password"}
        )
        
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            raise Exception(f"Failed to get auth token: {response.text}")
    
    def test_get_assets_performance(self):
        """Test performance of getting all assets."""
        # Measure response time
        start_time = time.time()
        response = client.get("/assets", headers=self.auth_headers)
        end_time = time.time()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        
        # Log performance
        response_time = end_time - start_time
        print(f"Get assets response time: {response_time:.4f} seconds")
        
        # Response should be reasonably fast
        self.assertLess(response_time, 1.0)
    
    def test_get_market_data_performance(self):
        """Test performance of getting market data."""
        # Get first asset
        asset = self.test_assets[0]
        
        # Measure response time
        start_time = time.time()
        response = client.get(
            f"/market-data/ohlcv/{asset.symbol}",
            params={"start_date": "2023-01-01", "end_date": "2023-12-31"},
            headers=self.auth_headers
        )
        end_time = time.time()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        
        # Log performance
        response_time = end_time - start_time
        print(f"Get market data response time: {response_time:.4f} seconds")
        
        # Response should be reasonably fast
        self.assertLess(response_time, 2.0)
    
    def test_get_strategies_performance(self):
        """Test performance of getting all strategies."""
        # Measure response time
        start_time = time.time()
        response = client.get("/strategies", headers=self.auth_headers)
        end_time = time.time()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        
        # Log performance
        response_time = end_time - start_time
        print(f"Get strategies response time: {response_time:.4f} seconds")
        
        # Response should be reasonably fast
        self.assertLess(response_time, 1.0)
    
    def test_get_backtests_performance(self):
        """Test performance of getting all backtests."""
        # Measure response time
        start_time = time.time()
        response = client.get("/backtests", headers=self.auth_headers)
        end_time = time.time()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        
        # Log performance
        response_time = end_time - start_time
        print(f"Get backtests response time: {response_time:.4f} seconds")
        
        # Response should be reasonably fast
        self.assertLess(response_time, 1.0)
    
    def test_filter_backtests_performance(self):
        """Test performance of filtering backtests."""
        # Get first strategy
        strategy = self.test_strategies[0]
        
        # Measure response time
        start_time = time.time()
        response = client.get(
            f"/backtests?strategy_id={strategy.id}",
            headers=self.auth_headers
        )
        end_time = time.time()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        
        # All backtests should be for the specified strategy
        for backtest in data:
            self.assertEqual(backtest["strategy_id"], strategy.id)
        
        # Log performance
        response_time = end_time - start_time
        print(f"Filter backtests response time: {response_time:.4f} seconds")
        
        # Response should be reasonably fast
        self.assertLess(response_time, 1.0)
    
    def test_backtest_metrics_aggregation_performance(self):
        """Test performance of aggregating backtest metrics."""
        # Measure response time
        start_time = time.time()
        response = client.get(
            "/backtests/metrics/summary",
            headers=self.auth_headers
        )
        end_time = time.time()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)
        
        # Log performance
        response_time = end_time - start_time
        print(f"Backtest metrics aggregation response time: {response_time:.4f} seconds")
        
        # Response should be reasonably fast
        self.assertLess(response_time, 2.0)


if __name__ == "__main__":
    unittest.main()
