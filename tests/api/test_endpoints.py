"""
Comprehensive unit and integration tests for API endpoints.
"""

import pytest
import json
import uuid
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.api import app
from backend.database.models.user import User
from backend.database.models.market_data import Asset
from backend.database.models.strategy import Strategy
from backend.database.models.backtest import Backtest
from backend.database.config import get_db, SessionLocal

client = TestClient(app)

# Fixture for database session
@pytest.fixture
def db_session():
    """Create a fresh database session for each test."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Fixture for test user
@pytest.fixture
def test_user(db_session):
    """Create a test user for authentication tests."""
    unique_id = uuid.uuid4().hex[:8]
    username = f"test_user_{unique_id}"
    email = f"{username}@example.com"
    
    # Create user directly in the database
    user = User(
        username=username,
        email=email,
        hashed_password="$2b$12$IrOt5sBJMD9mE6MJTQZrhOQFEf/CpXCaO3nrgvJWcFJZgZC5pOlUy"  # hashed 'test_password'
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    yield user
    
    # Clean up
    db_session.delete(user)
    db_session.commit()

# Fixture for authentication token
@pytest.fixture
def auth_token(test_user):
    """Get authentication token for test user."""
    response = client.post(
        "/auth/login",
        data={"username": test_user.username, "password": "test_password"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]

# Fixture for auth headers
@pytest.fixture
def auth_headers(auth_token):
    """Create authorization headers with token."""
    return {"Authorization": f"Bearer {auth_token}"}

# Fixture for test asset
@pytest.fixture
def test_asset(db_session):
    """Create a test asset."""
    unique_id = uuid.uuid4().hex[:8]
    symbol = f"TEST_{unique_id}"
    
    asset = Asset(
        symbol=symbol,
        name=f"Test Asset {symbol}",
        asset_type="crypto",
        is_active=True
    )
    db_session.add(asset)
    db_session.commit()
    db_session.refresh(asset)
    
    yield asset
    
    # Clean up
    db_session.delete(asset)
    db_session.commit()

# Fixture for test strategy
@pytest.fixture
def test_strategy(db_session, test_user):
    """Create a test strategy."""
    unique_id = uuid.uuid4().hex[:8]
    name = f"Test Strategy {unique_id}"
    
    strategy = Strategy(
        name=name,
        strategy_type="momentum",
        config={"param1": 10, "param2": 20},
        description="Test strategy for API tests",
        is_public=False,
        user_id=test_user.id
    )
    db_session.add(strategy)
    db_session.commit()
    db_session.refresh(strategy)
    
    yield strategy
    
    # Clean up
    db_session.delete(strategy)
    db_session.commit()

# Authentication Tests
class TestAuthentication:
    def test_register(self):
        """Test user registration endpoint."""
        unique_id = uuid.uuid4().hex[:8]
        username = f"new_user_{unique_id}"
        
        response = client.post(
            "/auth/register",
            json={
                "username": username,
                "email": f"{username}@example.com",
                "password": "secure_password"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["username"] == username
        
        # Clean up - delete the created user
        db = SessionLocal()
        user = db.query(User).filter(User.username == username).first()
        if user:
            db.delete(user)
            db.commit()
        db.close()
    
    def test_login_success(self, test_user):
        """Test successful login."""
        response = client.post(
            "/auth/login",
            data={"username": test_user.username, "password": "test_password"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, test_user):
        """Test login with invalid credentials."""
        response = client.post(
            "/auth/login",
            data={"username": test_user.username, "password": "wrong_password"}
        )
        
        assert response.status_code == 401
        assert "detail" in response.json()
    
    def test_get_current_user(self, auth_headers):
        """Test getting current user information."""
        response = client.get("/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data

# Asset Tests
class TestAssets:
    def test_get_assets(self, auth_headers):
        """Test getting all assets."""
        response = client.get("/assets", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_asset_by_symbol(self, auth_headers, test_asset):
        """Test getting an asset by symbol."""
        response = client.get(f"/assets/{test_asset.symbol}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == test_asset.symbol
        assert data["name"] == test_asset.name
        assert data["asset_type"] == test_asset.asset_type
    
    def test_get_nonexistent_asset(self, auth_headers):
        """Test getting a nonexistent asset."""
        response = client.get("/assets/NONEXISTENT", headers=auth_headers)
        
        assert response.status_code == 404
    
    def test_create_asset(self, auth_headers):
        """Test creating a new asset."""
        unique_id = uuid.uuid4().hex[:8]
        symbol = f"NEW_{unique_id}"
        
        asset_data = {
            "symbol": symbol,
            "name": f"New Asset {symbol}",
            "asset_type": "stock",
            "is_active": True
        }
        
        response = client.post("/assets", json=asset_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == symbol
        assert data["name"] == asset_data["name"]
        assert data["asset_type"] == asset_data["asset_type"]
        
        # Clean up - delete the created asset
        db = SessionLocal()
        asset = db.query(Asset).filter(Asset.symbol == symbol).first()
        if asset:
            db.delete(asset)
            db.commit()
        db.close()

# Strategy Tests
class TestStrategies:
    def test_get_strategies(self, auth_headers):
        """Test getting all strategies."""
        response = client.get("/strategies", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_strategy_by_id(self, auth_headers, test_strategy):
        """Test getting a strategy by ID."""
        response = client.get(f"/strategies/{test_strategy.id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_strategy.id
        assert data["name"] == test_strategy.name
        assert data["strategy_type"] == test_strategy.strategy_type
    
    def test_create_strategy(self, auth_headers):
        """Test creating a new strategy."""
        unique_id = uuid.uuid4().hex[:8]
        name = f"New Strategy {unique_id}"
        
        strategy_data = {
            "name": name,
            "strategy_type": "mean_reversion",
            "config": {"param1": 15, "param2": 30},
            "description": "New test strategy",
            "is_public": True
        }
        
        response = client.post("/strategies", json=strategy_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == name
        assert data["strategy_type"] == strategy_data["strategy_type"]
        assert data["description"] == strategy_data["description"]
        
        # Clean up - delete the created strategy
        db = SessionLocal()
        strategy = db.query(Strategy).filter(Strategy.name == name).first()
        if strategy:
            db.delete(strategy)
            db.commit()
        db.close()
    
    def test_update_strategy(self, auth_headers, test_strategy):
        """Test updating a strategy."""
        update_data = {
            "description": "Updated description",
            "config": {"param1": 25, "param2": 40}
        }
        
        response = client.put(
            f"/strategies/{test_strategy.id}", 
            json=update_data, 
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_strategy.id
        assert data["description"] == update_data["description"]
        assert data["config"] == update_data["config"]
    
    def test_delete_strategy(self, auth_headers, db_session, test_user):
        """Test deleting a strategy."""
        # Create a strategy to delete
        unique_id = uuid.uuid4().hex[:8]
        name = f"Delete Strategy {unique_id}"
        
        strategy = Strategy(
            name=name,
            strategy_type="momentum",
            config={"param1": 10, "param2": 20},
            description="Strategy to delete",
            is_public=False,
            user_id=test_user.id
        )
        db_session.add(strategy)
        db_session.commit()
        db_session.refresh(strategy)
        
        # Delete the strategy
        response = client.delete(f"/strategies/{strategy.id}", headers=auth_headers)
        
        assert response.status_code == 204
        
        # Verify it's deleted
        db_session.expire_all()
        deleted_strategy = db_session.query(Strategy).filter(Strategy.id == strategy.id).first()
        assert deleted_strategy is None

# Backtest Tests
class TestBacktests:
    def test_start_backtest(self, auth_headers, test_strategy, test_asset):
        """Test starting a backtest."""
        backtest_data = {
            "strategy_id": test_strategy.id,
            "asset_symbol": test_asset.symbol,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 10000,
            "parameters": {"param1": 15, "param2": 25}
        }
        
        response = client.post("/backtests", json=backtest_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["strategy_id"] == test_strategy.id
        assert data["asset_symbol"] == test_asset.symbol
        
        # Clean up - delete the created backtest
        db = SessionLocal()
        backtest = db.query(Backtest).filter(Backtest.id == data["id"]).first()
        if backtest:
            db.delete(backtest)
            db.commit()
        db.close()
    
    def test_get_backtests(self, auth_headers):
        """Test getting all backtests."""
        response = client.get("/backtests", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_backtest_by_id(self, auth_headers, db_session, test_user, test_strategy, test_asset):
        """Test getting a backtest by ID."""
        # Create a backtest
        backtest = Backtest(
            strategy_id=test_strategy.id,
            user_id=test_user.id,
            asset_symbol=test_asset.symbol,
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=10000,
            parameters={"param1": 15, "param2": 25},
            status="completed",
            metrics={
                "total_return": 15.5,
                "sharpe_ratio": 1.8,
                "max_drawdown": 8.5
            }
        )
        db_session.add(backtest)
        db_session.commit()
        db_session.refresh(backtest)
        
        # Get the backtest
        response = client.get(f"/backtests/{backtest.id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == backtest.id
        assert data["strategy_id"] == test_strategy.id
        assert data["asset_symbol"] == test_asset.symbol
        assert data["metrics"]["total_return"] == 15.5
        
        # Clean up
        db_session.delete(backtest)
        db_session.commit()

# Market Data Tests
class TestMarketData:
    def test_get_ohlcv_data(self, auth_headers, test_asset):
        """Test getting OHLCV data for an asset."""
        response = client.get(
            f"/market-data/ohlcv/{test_asset.symbol}",
            params={"start_date": "2023-01-01", "end_date": "2023-01-31"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_sentiment_data(self, auth_headers, test_asset):
        """Test getting sentiment data for an asset."""
        response = client.get(
            f"/market-data/sentiment/{test_asset.symbol}",
            params={"start_date": "2023-01-01", "end_date": "2023-01-31"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

# Portfolio Tests
class TestPortfolio:
    def test_get_portfolio(self, auth_headers):
        """Test getting user portfolio."""
        response = client.get("/portfolio", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "holdings" in data
        assert "cash" in data
        assert "total_value" in data
    
    def test_get_portfolio_history(self, auth_headers):
        """Test getting portfolio history."""
        response = client.get(
            "/portfolio/history",
            params={"start_date": "2023-01-01", "end_date": "2023-01-31"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

# Trading Tests
class TestTrading:
    def test_place_order(self, auth_headers, test_asset):
        """Test placing a trading order."""
        order_data = {
            "symbol": test_asset.symbol,
            "side": "buy",
            "order_type": "market",
            "quantity": 1.0,
            "price": None  # Market order
        }
        
        response = client.post("/orders", json=order_data, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "order_id" in data
        assert data["symbol"] == test_asset.symbol
        assert data["side"] == "buy"
        assert data["status"] in ["pending", "filled"]
    
    def test_get_orders(self, auth_headers):
        """Test getting user orders."""
        response = client.get("/orders", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_cancel_order(self, auth_headers):
        """Test canceling an order."""
        # First place an order
        order_data = {
            "symbol": "BTC/USD",  # Using a common symbol
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.1,
            "price": 20000  # Limit price
        }
        
        place_response = client.post("/orders", json=order_data, headers=auth_headers)
        assert place_response.status_code == 200
        order_id = place_response.json()["order_id"]
        
        # Then cancel it
        cancel_response = client.delete(f"/orders/{order_id}", headers=auth_headers)
        
        # Either it's successfully canceled or already filled/canceled
        assert cancel_response.status_code in [200, 404, 409]
        if cancel_response.status_code == 200:
            data = cancel_response.json()
            assert data["order_id"] == order_id
            assert data["status"] in ["canceled", "canceling"]
