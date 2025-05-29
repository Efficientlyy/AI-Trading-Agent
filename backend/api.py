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

# Security imports - use centralized configuration
from backend.security import configure_security
from backend.security.credential_manager import CredentialManager
from backend.security.rate_limiter import rate_limiter, get_rate_limit_middleware, start_rate_limiter_cleanup, stop_rate_limiter_cleanup
from backend.security.auth import get_current_user, require_scope, get_mock_user_override

# WebSocket imports - use new implementation
from backend.websockets import (
    websocket_router, 
    startup_websocket_manager,
    shutdown_websocket_manager,
    startup_market_data_streamer,
    shutdown_market_data_streamer
)

# Import CcxtProvider for live OHLCV streaming
from ai_trading_agent.data_acquisition.ccxt_provider import CcxtProvider

# Load environment variables
load_dotenv()

# Initialize credential manager
credential_manager = CredentialManager()

# Create repositories
user_repository = UserRepository()
strategy_repository = StrategyRepository()
optimization_repository = OptimizationRepository()
backtest_repository = BacktestRepository()
asset_repository = AssetRepository()
ohlcv_repository = OHLCVRepository()
sentiment_repository = SentimentRepository()

app = FastAPI(title="AI Trading Agent API")

# Configure security settings
# Must be called AFTER defining app and BEFORE adding routes that need security
configure_security(app)

# JWT secret and algorithm
# Get secrets from credential manager instead of direct environment variables
SECRET_KEY = credential_manager.get_credential_value("JWT_SECRET_KEY") or os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = credential_manager.get_credential_value("JWT_ALGORITHM") or os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(credential_manager.get_credential_value("ACCESS_TOKEN_EXPIRE_MINUTES") or os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Include WebSocket router
app.include_router(websocket_router)

# Include Paper Trading router
from backend.paper_trading import paper_trading_router
app.include_router(paper_trading_router)

# Include System Control router
from backend.system_control import system_control_router
app.include_router(system_control_router)

# Include Alpha Vantage router
from backend.alpha_vantage import alpha_vantage_router
app.include_router(alpha_vantage_router)

# Include System WebSocket routes
from backend.websockets.system_routes import router as system_websocket_router
app.include_router(system_websocket_router)

# Add startup and shutdown events for WebSocket services
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    # Start API rate limiter cleanup task
    await start_rate_limiter_cleanup()
    
    # Start WebSocket connection manager and heartbeat task
    await startup_websocket_manager()
    
    # Start market data streaming service
    # Use mock mode in development/testing, set to False in production
    use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    await startup_market_data_streamer(mock_mode=use_mock)
    
    logging.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown"""
    # Stop market data streaming service
    await shutdown_market_data_streamer()
    
    # Stop WebSocket connection manager and heartbeat task
    await shutdown_websocket_manager()
    
    # Stop API rate limiter cleanup task
    await stop_rate_limiter_cleanup()
    
    logging.info("Application shutdown complete")

# Add the required get_current_user dependency if not already defined
if 'get_current_user' not in locals() and 'get_current_user' not in globals():
    async def get_current_user(token: str = Depends(oauth2_scheme)):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid authentication credentials")
            return username
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Add the health endpoint after the startup/shutdown events
@app.get("/health", status_code=200)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    
    Returns basic system information and status of dependencies
    """
    import psutil
    import sys
    from datetime import datetime
    
    # Check database connection
    db_status = "healthy"
    try:
        db = next(get_db())
        # Execute a simple query
        db.execute("SELECT 1")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check redis connection
    redis_status = "healthy"
    try:
        from redis import Redis
        import os
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = Redis.from_url(redis_url, socket_timeout=2.0)
        redis_client.ping()
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    # System info
    memory = psutil.virtual_memory()
    
    # Build health response
    health_info = {
        "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",  # Should come from a version file or environment variable
        "dependencies": {
            "database": db_status,
            "redis": redis_status,
        },
        "system": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_total": memory.total,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "python_version": sys.version,
            "uptime_seconds": int(psutil.boot_time()),
        }
    }
    
    return health_info

# --- Backtesting API Endpoints ---

class BacktestConfig(BaseModel):
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    symbols: List[str]
    strategy_id: str
    timeframe: str = "1d"  # 1m, 5m, 15m, 1h, 4h, 1d
    commission_rate: float = 0.001
    slippage: float = 0.0005
    use_sentiment: bool = True
    sentiment_sources: Optional[List[str]] = None
    risk_controls: Optional[Dict[str, Any]] = None
    
class BacktestStartResponse(BaseModel):
    backtest_id: str
    status: str
    message: str
    estimated_completion: datetime
    
class BacktestStatus(BaseModel):
    backtest_id: str
    status: str  # running, completed, failed
    progress: float  # 0.0 to 1.0
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
class TradeMetrics(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_profit: float
    average_loss: float
    largest_profit: float
    largest_loss: float
    
class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    volatility: float
    calmar_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    
class BacktestResult(BaseModel):
    backtest_id: str
    config: BacktestConfig
    status: str
    trade_metrics: TradeMetrics
    performance_metrics: PerformanceMetrics
    portfolio_history: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    
class BacktestResultSummary(BaseModel):
    backtest_id: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    status: str
    created_at: datetime
    
@app.post("/api/backtest/start")
async def start_backtest(config: BacktestConfig, current_user: str = Depends(get_current_user)):
    """Start a new backtest"""
    try:
        import uuid
        import random
        
        # Validate dates
        if config.start_date >= config.end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
            
        # Validate symbols
        if not config.symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required")
        
        # Validate strategy
        if not config.strategy_id:
            raise HTTPException(status_code=400, detail="Strategy ID is required")
            
        # In production, this would initiate a backtest job
        # For now, create a mock backtest ID
        backtest_id = str(uuid.uuid4())
        
        # Estimate completion time (1 second per day in backtest range)
        days_in_range = (config.end_date - config.start_date).days
        estimated_seconds = max(30, days_in_range * 1) # Minimum 30 seconds
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        # Store backtest job in database or memory
        # This is a placeholder for actual implementation
        
        # Create a mock backtest job
        from backend.job_queue import submit_backtest_job
        submit_backtest_job(backtest_id, config.dict())
        
        return BacktestStartResponse(
            backtest_id=backtest_id,
            status="queued",
            message="Backtest has been queued and will start shortly",
            estimated_completion=estimated_completion
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error starting backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

@app.get("/api/backtest/{backtest_id}/status")
async def get_backtest_status(backtest_id: str, current_user: str = Depends(get_current_user)):
    """Get the status of a backtest job"""
    try:
        # In production, this would query the actual backtest job
        # For now, generate a mock status
        import random
        
        # Simulate various states
        hash_value = hash(backtest_id)
        random.seed(hash_value)
        
        # Determine status based on backtest_id
        status_options = ["running", "completed", "failed"]
        status_weights = [0.2, 0.7, 0.1]
        status = random.choices(status_options, weights=status_weights, k=1)[0]
        
        start_time = datetime.now() - timedelta(minutes=random.randint(5, 60))
        
        if status == "running":
            progress = random.uniform(0.1, 0.9)
            end_time = None
            error_message = None
        elif status == "completed":
            progress = 1.0
            end_time = start_time + timedelta(minutes=random.randint(5, 30))
            error_message = None
        else:  # failed
            progress = random.uniform(0.1, 0.9)
            end_time = start_time + timedelta(minutes=random.randint(5, 30))
            error_messages = [
                "Error accessing historical data",
                "Strategy execution error",
                "Out of memory during backtesting",
                "Invalid parameter configuration"
            ]
            error_message = random.choice(error_messages)
            
        return BacktestStatus(
            backtest_id=backtest_id,
            status=status,
            progress=progress,
            start_time=start_time,
            end_time=end_time,
            error_message=error_message
        )
    except Exception as e:
        logging.error(f"Error getting backtest status for {backtest_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest status: {str(e)}")

@app.get("/api/backtest/{backtest_id}/result")
async def get_backtest_result(backtest_id: str, current_user: str = Depends(get_current_user)):
    """Get the result of a completed backtest"""
    try:
        # In production, this would query the actual backtest result from database
        # For now, generate a mock result
        import random
        import uuid
        
        # Check if backtest is completed
        status_response = await get_backtest_status(backtest_id, current_user)
        
        if status_response.status == "running":
            raise HTTPException(status_code=400, detail="Backtest is still running")
        elif status_response.status == "failed":
            raise HTTPException(status_code=400, detail=f"Backtest failed: {status_response.error_message}")
            
        # Generate mock config (in real implementation, this would be stored)
        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now()
        symbols = random.sample(["BTC/USDT", "ETH/USDT", "SOL/USDT", "AAPL", "MSFT", "AMZN", "GOOGL"], 3)
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            symbols=symbols,
            strategy_id="sentiment_strategy_v1",
            timeframe="1d",
            commission_rate=0.001,
            slippage=0.0005,
            use_sentiment=True,
            sentiment_sources=["reddit", "twitter", "news"]
        )
        
        # Generate mock trade metrics
        total_trades = random.randint(50, 200)
        winning_trades = int(total_trades * random.uniform(0.4, 0.7))
        losing_trades = total_trades - winning_trades
        
        trade_metrics = TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=winning_trades / total_trades,
            profit_factor=random.uniform(1.2, 2.5),
            average_profit=random.uniform(200, 1000),
            average_loss=random.uniform(100, 500),
            largest_profit=random.uniform(1000, 5000),
            largest_loss=random.uniform(500, 2000)
        )
        
        # Generate mock performance metrics
        performance_metrics = PerformanceMetrics(
            total_return=random.uniform(0.05, 0.3),
            annualized_return=random.uniform(0.1, 0.5),
            sharpe_ratio=random.uniform(0.8, 2.5),
            sortino_ratio=random.uniform(1.0, 3.0),
            max_drawdown=random.uniform(0.05, 0.2),
            max_drawdown_duration=random.randint(10, 40),
            volatility=random.uniform(0.02, 0.1),
            calmar_ratio=random.uniform(1.2, 3.0),
            beta=random.uniform(0.8, 1.2),
            alpha=random.uniform(0.01, 0.1)
        )
        
        # Generate mock portfolio history
        portfolio_history = []
        current_value = config.initial_capital
        current_date = start_date
        
        while current_date <= end_date:
            # Generate daily change with some randomness
            daily_return = random.normalvariate(0.001, 0.02)  # Mean 0.1%, std dev 2%
            current_value *= (1 + daily_return)
            
            portfolio_history.append({
                "timestamp": current_date,
                "total_value": current_value,
                "cash": current_value * random.uniform(0.3, 0.7),
                "positions_value": current_value * random.uniform(0.3, 0.7)
            })
            
            current_date += timedelta(days=1)
            
        # Generate mock trades
        trades = []
        for i in range(total_trades):
            # Generate a random trade
            symbol = random.choice(symbols)
            side = random.choice(["buy", "sell"])
            
            # Random execution time
            trade_time = start_date + (end_date - start_date) * random.random()
            
            # Random price based on symbol
            if symbol == "BTC/USDT":
                price = random.uniform(40000, 50000)
            elif symbol == "ETH/USDT":
                price = random.uniform(2500, 3500)
            elif symbol == "SOL/USDT":
                price = random.uniform(80, 120)
            else:
                price = random.uniform(100, 3000)
                
            # Random quantity
            quantity = random.uniform(0.1, 10.0) if symbol in ["BTC/USDT", "ETH/USDT"] else random.uniform(1, 100)
            
            trades.append({
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "value": quantity * price,
                "timestamp": trade_time
            })
            
        # Sort trades by timestamp
        trades.sort(key=lambda x: x["timestamp"])
            
        return BacktestResult(
            backtest_id=backtest_id,
            config=config,
            status="completed",
            trade_metrics=trade_metrics,
            performance_metrics=performance_metrics,
            portfolio_history=portfolio_history,
            trades=trades
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error getting backtest result for {backtest_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest result: {str(e)}")

@app.get("/api/backtest/history")
async def get_backtest_history(
    limit: int = 10, 
    offset: int = 0,
    strategy_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get the history of backtests run by the user"""
    try:
        import random
        import uuid
        
        # In production, this would query the database for backtest history
        # For now, generate mock history
        
        # Generate mock strategies
        strategies = [
            {"id": "sentiment_strategy_v1", "name": "Sentiment Strategy V1"},
            {"id": "ma_crossover_strategy", "name": "Moving Average Crossover Strategy"},
            {"id": "rsi_strategy", "name": "RSI Strategy"}
        ]
        
        # Filter by strategy if provided
        available_strategies = [s for s in strategies if not strategy_id or s["id"] == strategy_id]
        
        # Generate random backtest history
        history = []
        for i in range(min(20, limit)):
            strategy = random.choice(available_strategies)
            
            # Random dates in the past 3 months
            end_date = datetime.now() - timedelta(days=random.randint(1, 90))
            start_date = end_date - timedelta(days=random.randint(30, 180))
            
            # Random performance metrics
            initial_capital = 100000.0
            total_return = random.uniform(-0.2, 0.4)
            final_value = initial_capital * (1 + total_return)
            
            # Random status with weights
            status = random.choices(
                ["completed", "failed", "running"],
                weights=[0.8, 0.15, 0.05],
                k=1
            )[0]
            
            history.append(BacktestResultSummary(
                backtest_id=str(uuid.uuid4()),
                strategy_id=strategy["id"],
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_value=final_value,
                total_return=total_return,
                status=status,
                created_at=datetime.now() - timedelta(days=random.randint(1, 90))
            ))
            
        # Sort by created_at descending
        history.sort(key=lambda x: x.created_at, reverse=True)
            
        return {"backtests": history, "count": len(history)}
    except Exception as e:
        logging.error(f"Error getting backtest history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest history: {str(e)}")

@app.get("/api/strategies", response_model=List[Dict[str, Any]])
async def get_strategies(
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get available trading strategies"""
    try:
        # In production, this would query the database for strategies
        # For now, return mock strategies
        strategies = [
            {
                "id": "sentiment_strategy_v1",
                "name": "Sentiment Strategy V1",
                "description": "Trading strategy based on Reddit and Twitter sentiment analysis",
                "parameters": [
                    {"name": "buy_threshold", "type": "float", "default": 0.3, "min": 0.1, "max": 0.9},
                    {"name": "sell_threshold", "type": "float", "default": -0.3, "min": -0.9, "max": -0.1},
                    {"name": "position_size", "type": "float", "default": 0.1, "min": 0.01, "max": 0.5},
                    {"name": "stop_loss", "type": "float", "default": 0.05, "min": 0.01, "max": 0.2}
                ]
            },
            {
                "id": "ma_crossover_strategy",
                "name": "Moving Average Crossover Strategy",
                "description": "Classic strategy using short and long period moving averages",
                "parameters": [
                    {"name": "short_window", "type": "int", "default": 20, "min": 5, "max": 50},
                    {"name": "long_window", "type": "int", "default": 50, "min": 20, "max": 200},
                    {"name": "position_size", "type": "float", "default": 0.1, "min": 0.01, "max": 0.5}
                ]
            },
            {
                "id": "rsi_strategy",
                "name": "RSI Strategy",
                "description": "Strategy based on Relative Strength Index indicator",
                "parameters": [
                    {"name": "rsi_period", "type": "int", "default": 14, "min": 7, "max": 30},
                    {"name": "overbought", "type": "int", "default": 70, "min": 60, "max": 90},
                    {"name": "oversold", "type": "int", "default": 30, "min": 10, "max": 40},
                    {"name": "position_size", "type": "float", "default": 0.1, "min": 0.01, "max": 0.5}
                ]
            },
            {
                "id": "sentiment_ma_hybrid",
                "name": "Sentiment-MA Hybrid Strategy",
                "description": "Combined strategy using both sentiment analysis and technical indicators",
                "parameters": [
                    {"name": "sentiment_weight", "type": "float", "default": 0.6, "min": 0.2, "max": 0.8},
                    {"name": "ma_weight", "type": "float", "default": 0.4, "min": 0.2, "max": 0.8},
                    {"name": "short_window", "type": "int", "default": 20, "min": 5, "max": 50},
                    {"name": "sentiment_threshold", "type": "float", "default": 0.2, "min": 0.1, "max": 0.5}
                ]
            }
        ]
        
        return strategies
    except Exception as e:
        logging.error(f"Error getting strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get strategies: {str(e)}")

# Add API key management endpoints
from fastapi import Depends
from pydantic import BaseModel
from typing import List, Optional
import secrets
from backend.security.rate_limiter import get_api_key_manager, ApiKeyRateLimitManager

class ApiKeyRequest(BaseModel):
    name: str
    tier: str = "basic"
    
class ApiKeyResponse(BaseModel):
    key: str
    name: str
    tier: str
    requests_per_minute: int
    requests_per_day: int
    created_at: str

class ApiKeySummary(BaseModel):
    key_id: str  # Last few characters for identification
    name: str
    tier: str
    created_at: str
    last_used: Optional[str]
    requests_per_minute: int
    requests_per_day: int
    total_requests: int

class ApiKeyUpdateRequest(BaseModel):
    name: Optional[str] = None
    tier: Optional[str] = None
    requests_per_minute: Optional[int] = None
    requests_per_day: Optional[int] = None

@app.post("/api/keys", response_model=ApiKeyResponse)
async def create_api_key(
    request: ApiKeyRequest,
    current_user: str = Depends(get_current_user),
    api_key_manager: ApiKeyRateLimitManager = Depends(get_api_key_manager)
):
    """Generate a new API key for programmatic access"""
    
    # Generate a secure random API key
    api_key = secrets.token_urlsafe(32)
    
    # Set rate limits based on tier
    if request.tier == "basic":
        requests_per_minute = 60
        requests_per_day = 10000
    elif request.tier == "premium":
        requests_per_minute = 180
        requests_per_day = 50000
    elif request.tier == "enterprise":
        requests_per_minute = 600
        requests_per_day = 200000
    else:
        requests_per_minute = 30
        requests_per_day = 5000
    
    # Register the API key
    api_key_manager.register_api_key(
        api_key=api_key,
        owner_id=current_user,
        name=request.name,
        tier=request.tier,
        requests_per_minute=requests_per_minute,
        requests_per_day=requests_per_day
    )
    
    return ApiKeyResponse(
        key=api_key,
        name=request.name,
        tier=request.tier,
        requests_per_minute=requests_per_minute,
        requests_per_day=requests_per_day,
        created_at=datetime.now().isoformat()
    )

@app.get("/api/keys", response_model=List[ApiKeySummary])
async def list_api_keys(
    current_user: str = Depends(get_current_user),
    api_key_manager: ApiKeyRateLimitManager = Depends(get_api_key_manager)
):
    """List all API keys for the current user"""
    keys = api_key_manager.list_api_keys(owner_id=current_user)
    
    # Format the response
    return [
        ApiKeySummary(
            key_id=key.get("key_id", ""),
            name=key.get("name", ""),
            tier=key.get("tier", "basic"),
            created_at=key.get("created_at", ""),
            last_used=key.get("last_used"),
            requests_per_minute=key.get("limits", {}).get("requests_per_minute", 0),
            requests_per_day=key.get("limits", {}).get("requests_per_day", 0),
            total_requests=key.get("total_requests", 0)
        )
        for key in keys
    ]

@app.put("/api/keys/{key_id}")
async def update_api_key(
    key_id: str,
    request: ApiKeyUpdateRequest,
    current_user: str = Depends(get_current_user),
    api_key_manager: ApiKeyRateLimitManager = Depends(get_api_key_manager)
):
    """Update an API key's settings"""
    # Find the full API key from the key_id suffix
    user_keys = api_key_manager.list_api_keys(owner_id=current_user)
    target_key = None
    
    for key in user_keys:
        if key.get("key_id") == key_id:
            target_key = key
            break
    
    if not target_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Set rate limits based on requested tier
    requests_per_minute = None
    requests_per_day = None
    
    if request.tier:
        if request.tier == "basic":
            requests_per_minute = 60
            requests_per_day = 10000
        elif request.tier == "premium":
            requests_per_minute = 180
            requests_per_day = 50000
        elif request.tier == "enterprise":
            requests_per_minute = 600
            requests_per_day = 200000
    
    # Override with specific values if provided
    if request.requests_per_minute is not None:
        requests_per_minute = request.requests_per_minute
    
    if request.requests_per_day is not None:
        requests_per_day = request.requests_per_day
    
    # Update the API key
    # Note: In a real implementation, we would need to map key_id back to the full API key
    success = api_key_manager.update_api_key(
        api_key=target_key["full_key"],  # This would be the actual full API key in a real implementation
        name=request.name,
        tier=request.tier,
        requests_per_minute=requests_per_minute,
        requests_per_day=requests_per_day
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update API key")
    
    return {"status": "success", "message": "API key updated successfully"}

@app.delete("/api/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: str = Depends(get_current_user),
    api_key_manager: ApiKeyRateLimitManager = Depends(get_api_key_manager)
):
    """Revoke an API key"""
    # Find the full API key from the key_id suffix
    user_keys = api_key_manager.list_api_keys(owner_id=current_user)
    target_key = None
    
    for key in user_keys:
        if key.get("key_id") == key_id:
            target_key = key
            break
    
    if not target_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Revoke the API key
    # Note: In a real implementation, we would need to map key_id back to the full API key
    success = api_key_manager.revoke_api_key(
        api_key=target_key["full_key"]  # This would be the actual full API key in a real implementation
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to revoke API key")
    
    return {"status": "success", "message": "API key revoked successfully"}
