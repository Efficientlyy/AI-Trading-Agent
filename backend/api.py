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

from fastapi.responses import JSONResponse
from fastapi.requests import Request
import logging

from sqlalchemy.orm import Session

from backend.database import get_db
from backend.database.repositories import (
    UserRepository, StrategyRepository, OptimizationRepository,
    BacktestRepository, AssetRepository, OHLCVRepository, SentimentRepository
)

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
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# User models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    is_superuser: bool

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    username: Optional[str] = None

class PasswordReset(BaseModel):
    email: str

class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = user_repository.get_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user

async def get_current_active_user(current_user = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_superuser(current_user = Depends(get_current_user)):
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user

# Authentication endpoints
@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        # Authenticate user with repository
        user = user_repository.authenticate_user(
            db, 
            username=form_data.username, 
            password=form_data.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = user_repository.create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )
        
        # Create refresh token
        refresh_token = user_repository.create_refresh_token(db, user.id)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": refresh_token
        }
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    try:
        # Check if username or email already exists
        if user_repository.get_by_username(db, user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if user_repository.get_by_email(db, user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user with repository
        user = user_repository.create_user(db, user_data)
        return user
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/auth/refresh", response_model=Token)
async def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    user = user_repository.verify_refresh_token(db, refresh_token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = user_repository.create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "refresh_token": refresh_token
    }

@app.post("/auth/password-reset")
async def request_password_reset(reset_data: PasswordReset, db: Session = Depends(get_db)):
    user = user_repository.get_by_email(db, reset_data.email)
    if not user:
        # Don't reveal that the email doesn't exist
        return {"message": "If your email is registered, you will receive a password reset link"}
    
    # Create password reset token
    reset_token = user_repository.create_password_reset_token(db, user.id)
    
    # In a real application, send an email with the reset token
    # For now, just return it in the response
    return {
        "message": "Password reset token created",
        "reset_token": reset_token  # Remove this in production
    }

@app.post("/auth/password-reset/{reset_token}")
async def reset_password(reset_token: str, new_password: str, db: Session = Depends(get_db)):
    success = user_repository.reset_password(db, reset_token, new_password)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    return {"message": "Password reset successfully"}

@app.post("/auth/change-password")
async def change_password(
    password_data: PasswordUpdate,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    success = user_repository.change_password(
        db, 
        current_user.id, 
        password_data.current_password, 
        password_data.new_password
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    return {"message": "Password changed successfully"}

@app.post("/auth/logout")
async def logout(refresh_token: str, db: Session = Depends(get_db)):
    success = user_repository.invalidate_refresh_token(db, refresh_token)
    return {"message": "Logged out successfully"}

# Strategy models
class StrategyBase(BaseModel):
    name: str
    strategy_type: str
    config: Dict[str, Any]
    description: Optional[str] = None
    is_public: bool = False

class StrategyCreate(StrategyBase):
    pass

class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    strategy_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None

class StrategyResponse(StrategyBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

# Strategy endpoints
@app.get("/strategies", response_model=List[StrategyResponse])
async def list_strategies(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        strategies = strategy_repository.get_user_strategies(db, current_user.id, skip, limit)
        return strategies
    except Exception as e:
        logging.error(f"Error listing strategies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list strategies: {str(e)}"
        )

@app.get("/strategies/public", response_model=List[StrategyResponse])
async def list_public_strategies(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    try:
        strategies = strategy_repository.get_public_strategies(db, skip, limit)
        return strategies
    except Exception as e:
        logging.error(f"Error listing public strategies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list public strategies: {str(e)}"
        )

@app.get("/strategies/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        strategy = strategy_repository.get(db, strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Check if user has access to the strategy
        if strategy.user_id != current_user.id and not strategy.is_public and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not authorized to access this strategy")
        
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting strategy {strategy_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get strategy: {str(e)}"
        )

@app.post("/strategies", response_model=StrategyResponse)
async def create_strategy(
    strategy: StrategyCreate,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        # Create strategy data with user ID
        strategy_data = strategy.dict()
        strategy_data["user_id"] = current_user.id
        
        # Create strategy with repository
        new_strategy = strategy_repository.create(db, strategy_data)
        return new_strategy
    except Exception as e:
        logging.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create strategy: {str(e)}"
        )

@app.put("/strategies/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: int,
    strategy_update: StrategyUpdate,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        # Get existing strategy
        strategy = strategy_repository.get(db, strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Check if user has access to update the strategy
        if strategy.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not authorized to update this strategy")
        
        # Update strategy with repository
        updated_strategy = strategy_repository.update(db, strategy, strategy_update.dict(exclude_unset=True))
        return updated_strategy
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating strategy {strategy_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update strategy: {str(e)}"
        )

@app.delete("/strategies/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        # Get existing strategy
        strategy = strategy_repository.get(db, strategy_id)
        
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Check if user has access to delete the strategy
        if strategy.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not authorized to delete this strategy")
        
        # Delete strategy with repository
        success = strategy_repository.delete(db, strategy_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete strategy")
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting strategy {strategy_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete strategy: {str(e)}"
        )

# Backtest models
class BacktestBase(BaseModel):
    name: str
    strategy_id: int
    parameters: Dict[str, Any]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    description: Optional[str] = None

class BacktestCreate(BacktestBase):
    pass

class BacktestResponse(BacktestBase):
    id: int
    user_id: int
    status: str
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

# Backtest endpoints
@app.get("/backtests", response_model=List[BacktestResponse])
async def list_backtests(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    backtests = backtest_repository.get_user_backtests(db, current_user.id, skip, limit)
    return backtests

@app.get("/backtests/{backtest_id}", response_model=BacktestResponse)
async def get_backtest(
    backtest_id: int,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    backtest = backtest_repository.get(db, backtest_id)
    
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")
    
    if backtest.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this backtest")
    
    return backtest

@app.post("/backtests", response_model=BacktestResponse)
async def create_backtest(
    backtest: BacktestCreate,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Check if strategy exists and belongs to user
    strategy = strategy_repository.get(db, backtest.strategy_id)
    
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    if strategy.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to use this strategy")
    
    # Create backtest
    new_backtest = backtest_repository.create_backtest(
        db=db,
        user_id=current_user.id,
        strategy_id=backtest.strategy_id,
        name=backtest.name,
        parameters=backtest.parameters,
        start_date=backtest.start_date,
        end_date=backtest.end_date,
        initial_capital=backtest.initial_capital,
        description=backtest.description
    )
    
    # In a real application, you would start the backtest in a background task
    # For now, just update the status to "running"
    backtest_repository.update_backtest_status(
        db=db,
        backtest_id=new_backtest.id,
        user_id=current_user.id,
        status="running"
    )
    
    return new_backtest

@app.get("/backtests/{backtest_id}/trades")
async def get_backtest_trades(
    backtest_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    trades = backtest_repository.get_backtest_trades(db, backtest_id, current_user.id, skip, limit)
    return trades

@app.get("/backtests/{backtest_id}/portfolio")
async def get_backtest_portfolio(
    backtest_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    snapshots = backtest_repository.get_backtest_portfolio_snapshots(db, backtest_id, current_user.id, skip, limit)
    return snapshots

# Asset models
class AssetBase(BaseModel):
    symbol: str
    name: str
    asset_type: str

class AssetCreate(AssetBase):
    pass

class AssetResponse(AssetBase):
    id: int
    is_active: int
    created_at: datetime
    updated_at: Optional[datetime] = None

# Asset endpoints
@app.get("/assets", response_model=List[Dict[str, Any]])
async def list_assets(
    asset_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    try:
        if asset_type:
            assets = asset_repository.get_active_assets(db, asset_type)
        else:
            assets = asset_repository.get_active_assets(db)
        
        # Convert to response format
        response_data = [
            {
                "id": asset.id,
                "symbol": asset.symbol,
                "name": asset.name,
                "asset_type": asset.asset_type,
                "is_active": asset.is_active
            }
            for asset in assets
        ]
        
        return response_data
    except Exception as e:
        logging.error(f"Error listing assets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list assets: {str(e)}"
        )

@app.get("/assets/{symbol}", response_model=Dict[str, Any])
async def get_asset(symbol: str, db: Session = Depends(get_db)):
    try:
        asset = asset_repository.get_by_symbol(db, symbol)
        
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        # Convert to response format
        response_data = {
            "id": asset.id,
            "symbol": asset.symbol,
            "name": asset.name,
            "asset_type": asset.asset_type,
            "is_active": asset.is_active
        }
        
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting asset {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get asset: {str(e)}"
        )

@app.post("/assets", response_model=Dict[str, Any])
async def create_asset(
    asset_data: Dict[str, Any],
    current_user = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    try:
        # Check if asset already exists
        existing_asset = asset_repository.get_by_symbol(db, asset_data["symbol"])
        
        if existing_asset:
            if existing_asset.is_active:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Asset already exists"
                )
            else:
                # Reactivate asset
                existing_asset.is_active = True
                updated_asset = asset_repository.update(db, existing_asset, {"is_active": True})
                
                # Convert to response format
                response_data = {
                    "id": updated_asset.id,
                    "symbol": updated_asset.symbol,
                    "name": updated_asset.name,
                    "asset_type": updated_asset.asset_type,
                    "is_active": updated_asset.is_active
                }
                
                return response_data
        
        # Create new asset
        new_asset = asset_repository.create(db, asset_data)
        
        # Convert to response format
        response_data = {
            "id": new_asset.id,
            "symbol": new_asset.symbol,
            "name": new_asset.name,
            "asset_type": new_asset.asset_type,
            "is_active": new_asset.is_active
        }
        
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating asset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create asset: {str(e)}"
        )

@app.get("/assets/{symbol}/history", response_model=List[Dict[str, Any]])
async def get_asset_history(
    symbol: str,
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        # Get asset
        asset = asset_repository.get_by_symbol(db, symbol)
        
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        # Parse dates
        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                start_date_obj = datetime.now() - timedelta(days=30)
        else:
            start_date_obj = datetime.now() - timedelta(days=30)
        
        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                end_date_obj = datetime.now()
        else:
            end_date_obj = datetime.now()
        
        # Get OHLCV data
        ohlcv_data = ohlcv_repository.get_ohlcv_data(
            db, 
            asset.id, 
            timeframe, 
            start_date_obj, 
            end_date_obj
        )
        
        # Convert to response format
        response_data = [
            {
                "timestamp": item.timestamp.isoformat(),
                "open": item.open,
                "high": item.high,
                "low": item.low,
                "close": item.close,
                "volume": item.volume
            }
            for item in ohlcv_data
        ]
        
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting history for asset {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get asset history: {str(e)}"
        )

@app.get("/assets/{symbol}/sentiment", response_model=List[Dict[str, Any]])
async def get_asset_sentiment(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        # Get asset
        asset = asset_repository.get_by_symbol(db, symbol)
        
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        # Parse dates
        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                start_date_obj = datetime.now() - timedelta(days=30)
        else:
            start_date_obj = datetime.now() - timedelta(days=30)
        
        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                end_date_obj = datetime.now()
        else:
            end_date_obj = datetime.now()
        
        # Get sentiment data
        sentiment_data = sentiment_repository.get_sentiment_data(
            db, 
            asset.id, 
            source, 
            start_date_obj, 
            end_date_obj
        )
        
        # Convert to response format
        response_data = [
            {
                "timestamp": item.timestamp.isoformat(),
                "source": item.source,
                "sentiment_score": item.sentiment_score,
                "volume": item.volume
            }
            for item in sentiment_data
        ]
        
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting sentiment for asset {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get asset sentiment: {str(e)}"
        )

# WebSocket connection manager
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

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, db: Session = Depends(get_db)):
    # In a real application, you would verify the user_id with a token
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                action = message.get("action")
                
                if action == "ping":
                    await websocket.send_text(json.dumps({"action": "pong", "timestamp": datetime.now().isoformat()}))
                
                elif action == "subscribe":
                    # Handle subscription
                    topic = message.get("topic")
                    if topic:
                        await websocket.send_text(json.dumps({"action": "subscribed", "topic": topic}))
                
                elif action == "unsubscribe":
                    # Handle unsubscription
                    topic = message.get("topic")
                    if topic:
                        await websocket.send_text(json.dumps({"action": "unsubscribed", "topic": topic}))
                
                else:
                    await websocket.send_text(json.dumps({"error": "Unknown action"}))
            
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)