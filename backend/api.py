"""
Backend API for AI Trading Agent dashboard.
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import json
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import asyncio

from fastapi.responses import JSONResponse
from fastapi.requests import Request
import logging

from sqlalchemy.orm import Session

from backend.database import get_db
from backend.database.repositories import (
    UserRepository, StrategyRepository, OptimizationRepository,
    BacktestRepository, AssetRepository, OHLCVRepository, SentimentRepository
)

# Import CcxtProvider for live OHLCV streaming
from ai_trading_agent.data_acquisition.ccxt_provider import CcxtProvider

# Load environment variables
load_dotenv()

# Create repositories
user_repository = UserRepository()
strategy_repository = StrategyRepository()
optimization_repository = OptimizationRepository()
backtest_repository = BacktestRepository()
asset_repository = AssetRepository()
ohlcv_repository = OHLCVRepository()
sentiment_repository = SentimentRepository()

app = FastAPI(title="AI Trading Agent API")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT secret and algorithm
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# --- Existing code omitted for brevity ---

# --- WebSocket Connection Manager (existing code) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_text(message)
    async def broadcast(self, message: str):
        for user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_text(message)

manager = ConnectionManager()

# --- WebSocket Endpoint with Live OHLCV Streaming ---
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    # Authenticate user (existing logic)
    token = websocket.query_params.get("token")
    user_id = None
    try:
        if token:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub")
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    if not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Agent status store (existing logic)
    if not hasattr(websocket_endpoint, "_agent_status_store"):
        websocket_endpoint._agent_status_store = {}
    agent_status_store = websocket_endpoint._agent_status_store
    if user_id not in agent_status_store:
        agent_status_store[user_id] = {
            "status": "stopped",
            "reasoning": "Agent is idle.",
            "timestamp": datetime.now().isoformat()
        }

    await manager.connect(websocket, user_id)
    subscriptions = set()
    ohlcv_tasks = {}
    ccxt_provider = None
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                action = message.get("action")
                if action == "ping":
                    await websocket.send_text(json.dumps({"action": "pong", "timestamp": datetime.now().isoformat()}))
                elif action == "subscribe":
                    topic = message.get("topic")
                    if topic:
                        subscriptions.add(topic)
                        await websocket.send_text(json.dumps({"action": "subscribed", "topic": topic}))
                        # --- Live OHLCV subscription ---
                        if topic == "ohlcv":
                            symbol = message.get("symbol")
                            timeframe = message.get("timeframe", "1m")
                            if symbol:
                                if not ccxt_provider:
                                    ccxt_provider = CcxtProvider()
                                    await ccxt_provider.connect_realtime()
                                await ccxt_provider.subscribe_to_symbols([symbol])
                                async def stream_ohlcv():
                                    while True:
                                        ohlcv_data = await ccxt_provider.get_realtime_data()
                                        if ohlcv_data:
                                            await websocket.send_text(json.dumps({
                                                "topic": "ohlcv",
                                                "symbol": symbol,
                                                "timeframe": timeframe,
                                                "data": ohlcv_data,
                                                "timestamp": datetime.now().isoformat()
                                            }))
                                        await asyncio.sleep(1)  # Adjust polling interval as needed
                                ohlcv_tasks[symbol] = asyncio.create_task(stream_ohlcv())
                elif action == "unsubscribe":
                    topic = message.get("topic")
                    if topic:
                        subscriptions.discard(topic)
                        await websocket.send_text(json.dumps({"action": "unsubscribed", "topic": topic}))
                        # Cancel OHLCV streaming if needed
                        if topic == "ohlcv":
                            symbol = message.get("symbol")
                            if symbol and symbol in ohlcv_tasks:
                                ohlcv_tasks[symbol].cancel()
                                del ohlcv_tasks[symbol]
                elif action == "start_agent":
                    agent_status_store[user_id] = {
                        "status": "running",
                        "reasoning": "Agent started and monitoring market.",
                        "timestamp": datetime.now().isoformat()
                    }
                    # Broadcast new status to agent_status subscribers
                    status_update = {
                        "topic": "agent_status",
                        "timestamp": agent_status_store[user_id]["timestamp"],
                        "status": agent_status_store[user_id]["status"],
                        "reasoning": agent_status_store[user_id]["reasoning"]
                    }
                    await websocket.send_text(json.dumps(status_update))
                elif action == "stop_agent":
                    agent_status_store[user_id] = {
                        "status": "stopped",
                        "reasoning": "Agent stopped by user.",
                        "timestamp": datetime.now().isoformat()
                    }
                    # Broadcast new status to agent_status subscribers
                    status_update = {
                        "topic": "agent_status",
                        "timestamp": agent_status_store[user_id]["timestamp"],
                        "status": "stopped",
                        "reasoning": agent_status_store[user_id]["reasoning"]
                    }
                    await websocket.send_text(json.dumps(status_update))
                else:
                    await websocket.send_text(json.dumps({"error": "Unknown action"}))
            except asyncio.TimeoutError:
                # Periodically send updates for subscribed topics
                for topic in list(subscriptions):
                    if topic == "portfolio":
                        # Simulate portfolio update
                        portfolio_update = {
                            "topic": "portfolio",
                            "timestamp": datetime.now().isoformat(),
                            "total_value": 100000 + int(datetime.now().second),
                            "cash": 50000,
                            "positions": {"AAPL": {"quantity": 10, "price": 150.0}},
                        }
                        await websocket.send_text(json.dumps(portfolio_update))
                    elif topic == "trades":
                        # Simulate trade update
                        trade_update = {
                            "topic": "trades",
                            "timestamp": datetime.now().isoformat(),
                            "trade": {"symbol": "AAPL", "side": "buy", "quantity": 1, "price": 150.0},
                        }
                        await websocket.send_text(json.dumps(trade_update))
                    elif topic == "agent_status":
                        # Send current agent status
                        status_update = {
                            "topic": "agent_status",
                            "timestamp": agent_status_store[user_id]["timestamp"],
                            "status": agent_status_store[user_id]["status"],
                            "reasoning": agent_status_store[user_id]["reasoning"],
                            "regime_label": agent_status_store[user_id].get("regime_label"),
                            "adaptive_reason": agent_status_store[user_id].get("adaptive_reason")
                        }
                        await websocket.send_text(json.dumps(status_update))
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        # Cancel any running OHLCV tasks
        for task in ohlcv_tasks.values():
            task.cancel()
        if ccxt_provider:
            await ccxt_provider.close()
