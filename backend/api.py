"""
Backend API for AI Trading Agent dashboard.
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import json
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from src.trading_engine.portfolio_manager import PortfolioManager
from src.sentiment_analysis.manager import SentimentManager
from src.trading_engine.order_manager import OrderManager
from src.backtesting.performance_metrics import calculate_metrics

from fastapi.responses import JSONResponse
from fastapi.requests import Request
import logging

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
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# In-memory user store (replace with database)
fake_users_db = {
    "test_user": {
        "username": "test_user",
        "hashed_password": pwd_context.hash("test_password"),
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict):
    from datetime import datetime, timedelta
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = fake_users_db.get(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/register")
async def register(username: str, password: str):
    if username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = pwd_context.hash(password)
    fake_users_db[username] = {
        "username": username,
        "hashed_password": hashed_password,
        "disabled": False,
    }
    return {"msg": "User registered successfully"}

@app.post("/auth/refresh")
async def refresh_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        new_token = create_access_token({"sub": username})
        return {"access_token": new_token, "token_type": "bearer"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/auth/reset-password")
async def reset_password(username: str, new_password: str):
    user = fake_users_db.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["hashed_password"] = pwd_context.hash(new_password)
    return {"msg": "Password reset successful"}

@app.post("/auth/disable-user")
async def disable_user(username: str):
    user = fake_users_db.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["disabled"] = True
    return {"msg": f"User {username} disabled"}

@app.post("/auth/delete-user")
async def delete_user(username: str):
    if username in fake_users_db:
        del fake_users_db[username]
        return {"msg": f"User {username} deleted"}
    else:
        raise HTTPException(status_code=404, detail="User not found")

def require_role(role: str):
    def role_checker(user=Depends(get_current_user)):
        # Placeholder: all users are 'user' role
        user_role = "user"
        if user_role != role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user
    return role_checker

class OrderRequest(BaseModel):
    symbol: str = Field(..., example="BTC/USD")
    side: str = Field(..., example="buy")
    order_type: str = Field(..., example="market")
    quantity: float = Field(..., gt=0)
    price: Optional[float] = None

class BacktestParams(BaseModel):
    strategy_name: str
    parameters: Dict[str, Any]

# Global instances (replace with dependency injection or state management as needed)
portfolio_manager = PortfolioManager()
sentiment_manager = SentimentManager()
order_manager = OrderManager(portfolio_manager.portfolio)

# Placeholder for latest performance metrics
latest_metrics = {}

@app.get("/portfolio")
def get_portfolio(user=Depends(get_current_user)):
    portfolio = portfolio_manager.get_portfolio_state()
    return {"portfolio": portfolio}

@app.get("/performance")
def get_performance(user=Depends(get_current_user)):
    return {"performance": latest_metrics}

@app.get("/sentiment")
def get_sentiment(user=Depends(get_current_user)):
    signal = sentiment_manager.get_sentiment_signal()
    return {"sentiment_signal": signal}

@app.post("/orders")
def place_order(order: OrderRequest, user=Depends(get_current_user)):
    new_order = order_manager.create_order(
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        price=order.price,
    )
    return {"status": "order placed", "order": new_order}

@app.get("/orders")
def list_orders(user=Depends(get_current_user)):
    open_orders = order_manager.get_open_orders()
    return {"orders": list(open_orders.values())}

@app.post("/backtest/start")
def start_backtest(params: BacktestParams, user=Depends(get_current_user)):
    global latest_metrics
    latest_metrics = {"status": "running", "params": params.dict()}
    return {"status": "backtest started", "params": params.dict()}

@app.get("/backtest/status")
def get_backtest_status(user=Depends(get_current_user)):
    return latest_metrics

@app.get("/assets")
def get_assets(user=Depends(get_current_user)):
    return {"assets": ["BTC/USD", "ETH/USD", "SOL/USD"]}

@app.get("/history")
def get_history(symbol: str, start: str, end: str, timeframe: str = "1d", user=Depends(get_current_user)):
    return {"symbol": symbol, "start": start, "end": end, "timeframe": timeframe, "data": []}

@app.get("/metrics")
def get_metrics(user=Depends(get_current_user)):
    return {"metrics": latest_metrics}

@app.get("/strategies")
def list_strategies(user=Depends(get_current_user)):
    return {"strategies": []}

@app.post("/strategies")
def create_strategy(strategy: dict, user=Depends(get_current_user)):
    return {"status": "strategy saved", "strategy": strategy}

@app.put("/strategies/{strategy_id}")
def update_strategy(strategy_id: str, strategy: dict, user=Depends(get_current_user)):
    return {"status": "strategy updated", "strategy_id": strategy_id, "strategy": strategy}

@app.delete("/strategies/{strategy_id}")
def delete_strategy(strategy_id: str, user=Depends(get_current_user)):
    return {"status": "strategy deleted", "strategy_id": strategy_id}

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    subscriptions = set()
    try:
        while True:
            message = await websocket.receive_text()
            try:
                msg_data = json.loads(message)
                action = msg_data.get("action")
                topic = msg_data.get("topic")
                if action == "subscribe" and topic:
                    subscriptions.add(topic)
                    await websocket.send_text(json.dumps({"status": f"Subscribed to {topic}"}))
                elif action == "unsubscribe" and topic:
                    subscriptions.discard(topic)
                    await websocket.send_text(json.dumps({"status": f"Unsubscribed from {topic}"}))
            except Exception:
                pass

            update = {}
            if "portfolio" in subscriptions:
                update["portfolio"] = portfolio_manager.get_portfolio_state()
            if "sentiment" in subscriptions:
                update["sentiment_signal"] = sentiment_manager.get_sentiment_signal()
            if "performance" in subscriptions:
                update["performance"] = latest_metrics

            if update:
                await websocket.send_text(json.dumps(update))
    except WebSocketDisconnect:
        pass