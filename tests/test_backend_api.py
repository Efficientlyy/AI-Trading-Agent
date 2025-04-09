"""
Tests for backend API endpoints.
"""

from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_register_and_login():
    response = client.post("/auth/register", params={"username": "user1", "password": "pass1"})
    assert response.status_code == 200

    response = client.post("/auth/login", data={"username": "user1", "password": "pass1"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    assert token

def get_auth_headers():
    response = client.post("/auth/login", data={"username": "test_user", "password": "test_password"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_portfolio_endpoint():
    headers = get_auth_headers()
    response = client.get("/portfolio", headers=headers)
    assert response.status_code == 200
    assert "portfolio" in response.json()

def test_orders_endpoint():
    headers = get_auth_headers()
    order_data = {
        "symbol": "BTC/USD",
        "side": "buy",
        "order_type": "market",
        "quantity": 0.1
    }
    response = client.post("/orders", json=order_data, headers=headers)
    assert response.status_code == 200
    assert "order" in response.json()

def test_get_assets():
    headers = get_auth_headers()
    response = client.get("/assets", headers=headers)
    assert response.status_code == 200
    assert "assets" in response.json()

def test_sentiment_endpoint():
    headers = get_auth_headers()
    response = client.get("/sentiment", headers=headers)
    assert response.status_code == 200
    assert "sentiment_signal" in response.json()

def test_backtest_endpoints():
    headers = get_auth_headers()
    params = {"strategy_name": "ma_crossover", "parameters": {"fast": 10, "slow": 30}}
    response = client.post("/backtest/start", json=params, headers=headers)
    assert response.status_code == 200

    response = client.get("/backtest/status", headers=headers)
    assert response.status_code == 200

def test_strategy_crud():
    headers = get_auth_headers()
    strategy = {"name": "test_strategy", "params": {"param1": 1}}

    # Create
    response = client.post("/strategies", json=strategy, headers=headers)
    assert response.status_code == 200

    # List
    response = client.get("/strategies", headers=headers)
    assert response.status_code == 200

    # Update
    response = client.put("/strategies/1", json=strategy, headers=headers)
    assert response.status_code == 200

    # Delete
    response = client.delete("/strategies/1", headers=headers)
    assert response.status_code == 200