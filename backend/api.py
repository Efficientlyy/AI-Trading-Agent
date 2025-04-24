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

    # Initialize global CCXT provider if needed
    if not hasattr(websocket_endpoint, "_ccxt_provider"):
        websocket_endpoint._ccxt_provider = None
    
    # Global mapping of user subscriptions
    if not hasattr(websocket_endpoint, "_user_subscriptions"):
        websocket_endpoint._user_subscriptions = {}

    await manager.connect(websocket, user_id)
    
    # Initialize or get subscriptions for this user
    if user_id not in websocket_endpoint._user_subscriptions:
        websocket_endpoint._user_subscriptions[user_id] = {
            "topics": set(),
            "ohlcv_symbols": {},  # Map symbol to timeframe
            "tasks": {}
        }
    
    user_subs = websocket_endpoint._user_subscriptions[user_id]
    
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
                        user_subs["topics"].add(topic)
                        await websocket.send_text(json.dumps({"action": "subscribed", "topic": topic}))
                        
                        # Handle OHLCV subscription
                        if topic == "ohlcv":
                            symbol = message.get("symbol")
                            timeframe = message.get("timeframe", "1m")
                            
                            if symbol:
                                # Store the symbol-timeframe subscription
                                user_subs["ohlcv_symbols"][symbol] = timeframe
                                
                                # Initialize CCXT provider if needed
                                if websocket_endpoint._ccxt_provider is None:
                                    # Create provider with paper trading config
                                    config = {
                                        "exchange_id": os.getenv("EXCHANGE_ID", "binance"),
                                        "api_key": os.getenv("EXCHANGE_API_KEY", ""),
                                        "secret_key": os.getenv("EXCHANGE_SECRET_KEY", ""),
                                        "options": {
                                            "defaultType": "spot",
                                            "adjustForTimeDifference": True
                                        }
                                    }
                                    websocket_endpoint._ccxt_provider = CcxtProvider(config)
                                    await websocket_endpoint._ccxt_provider.connect_realtime()
                                    
                                # Subscribe to the symbol
                                await websocket_endpoint._ccxt_provider.subscribe_to_symbols([symbol])
                                
                                # Create streaming task if not already running
                                if symbol not in user_subs["tasks"]:
                                    task_name = f"ohlcv_stream_{symbol}_{timeframe}_{user_id}"
                                    
                                    async def stream_ohlcv(s=symbol, tf=timeframe):
                                        logging.info(f"Started OHLCV streaming for {s} ({tf})")
                                        last_data = None
                                        
                                        while True:
                                            try:
                                                # Get latest data from provider
                                                ohlcv_data = await websocket_endpoint._ccxt_provider.get_realtime_data()
                                                
                                                if ohlcv_data and ohlcv_data.get('symbol') == s:
                                                    # Convert raw data to frontend format
                                                    data = ohlcv_data.get('data')
                                                    if isinstance(data, list) and len(data) >= 6:
                                                        # Format: [timestamp, open, high, low, close, volume]
                                                        formatted_data = {
                                                            "timestamp": datetime.fromtimestamp(data[0]/1000).isoformat(),
                                                            "open": data[1],
                                                            "high": data[2],
                                                            "low": data[3],
                                                            "close": data[4],
                                                            "volume": data[5]
                                                        }
                                                        
                                                        # Avoid sending duplicate data
                                                        current_data = json.dumps(formatted_data)
                                                        if current_data != last_data:
                                                            await websocket.send_text(json.dumps({
                                                                "topic": "ohlcv",
                                                                "symbol": s,
                                                                "timeframe": tf,
                                                                "data": formatted_data
                                                            }))
                                                            last_data = current_data
                                            
                                            except Exception as e:
                                                logging.error(f"Error in OHLCV streaming for {s}: {e}")
                                                
                                            # Sleep to avoid excessive polling
                                            await asyncio.sleep(0.5)
                                    
                                    user_subs["tasks"][symbol] = asyncio.create_task(stream_ohlcv())
                                    logging.info(f"Created streaming task for {symbol} ({timeframe}) for user {user_id}")
                
                elif action == "unsubscribe":
                    topic = message.get("topic")
                    symbol = message.get("symbol", "")
                    
                    if topic:
                        user_subs["topics"].discard(topic)
                        await websocket.send_text(json.dumps({"action": "unsubscribed", "topic": topic}))
                        
                        # Handle OHLCV unsubscription
                        if topic == "ohlcv" and symbol:
                            if symbol in user_subs["ohlcv_symbols"]:
                                del user_subs["ohlcv_symbols"][symbol]
                                
                            # Cancel streaming task if it exists
                            if symbol in user_subs["tasks"]:
                                user_subs["tasks"][symbol].cancel()
                                del user_subs["tasks"][symbol]
                                logging.info(f"Cancelled streaming task for {symbol} for user {user_id}")
                
                elif action == "start_agent":
                    # Update agent status and broadcast
                    agent_status_store[user_id] = {
                        "status": "running",
                        "reasoning": "Agent started and monitoring market.",
                        "timestamp": datetime.now().isoformat()
                    }
                    status_update = {
                        "topic": "agent_status",
                        "timestamp": agent_status_store[user_id]["timestamp"],
                        "status": agent_status_store[user_id]["status"],
                        "reasoning": agent_status_store[user_id]["reasoning"]
                    }
                    await websocket.send_text(json.dumps(status_update))
                
                elif action == "stop_agent":
                    # Update agent status and broadcast
                    agent_status_store[user_id] = {
                        "status": "stopped",
                        "reasoning": "Agent stopped by user.",
                        "timestamp": datetime.now().isoformat()
                    }
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
                for topic in list(user_subs["topics"]):
                    if topic == "portfolio":
                        # Get real portfolio data or use mock data
                        is_paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
                        
                        # In a real implementation, this would query the portfolio manager
                        portfolio_update = {
                            "topic": "portfolio",
                            "timestamp": datetime.now().isoformat(),
                            "paper_trading": is_paper_trading,
                            "portfolio": {
                                "total_value": 100000 + int(datetime.now().second),
                                "available_cash": 50000,
                                "total_pnl": 1500,
                                "total_pnl_percentage": 1.5,
                                "positions": {
                                    "BTC/USDT": {"quantity": 0.1, "entry_price": 45000, "current_price": 46000, "pnl": 100},
                                    "ETH/USDT": {"quantity": 2, "entry_price": 3000, "current_price": 3100, "pnl": 200}
                                }
                            }
                        }
                        await websocket.send_text(json.dumps(portfolio_update))
                    
                    elif topic == "trades":
                        # In a real implementation, this would query recent trades
                        trade_update = {
                            "topic": "trades",
                            "timestamp": datetime.now().isoformat(),
                            "trade": {"symbol": "BTC/USDT", "side": "buy", "quantity": 0.01, "price": 46000.0},
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
        
        # Clean up user resources
        if user_id in websocket_endpoint._user_subscriptions:
            # Cancel all tasks
            for task in user_subs["tasks"].values():
                task.cancel()
                
            # Check if this was the last connection for this user
            if user_id not in manager.active_connections:
                # If no more connections for this user, clean up subscriptions
                del websocket_endpoint._user_subscriptions[user_id]
                
        # Clean up global provider if no more users
        if not websocket_endpoint._user_subscriptions and websocket_endpoint._ccxt_provider:
            await websocket_endpoint._ccxt_provider.close()
            websocket_endpoint._ccxt_provider = None
            logging.info("Closed CCXT provider as there are no more active users")
