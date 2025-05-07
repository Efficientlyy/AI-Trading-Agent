"""
Paper Trading API endpoints for AI Trading Agent.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Import authentication
from backend.security.auth import get_current_user, get_mock_user_override

# Create router
paper_trading_router = APIRouter(prefix="/api/paper-trading", tags=["paper-trading"])

# Models
class PaperTradingConfig(BaseModel):
    config_path: str
    duration_minutes: int
    interval_minutes: int

class PaperTradingStatus(BaseModel):
    status: str
    uptime_seconds: Optional[int] = None
    symbols: Optional[List[str]] = None
    current_portfolio: Optional[Dict[str, Any]] = None
    recent_trades: Optional[List[Dict[str, Any]]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

class PaperTradingSession(BaseModel):
    session_id: str
    status: str
    start_time: Optional[str] = None
    uptime_seconds: Optional[int] = None
    symbols: Optional[List[str]] = None
    current_portfolio: Optional[Dict[str, Any]] = None

class PaperTradingResults(BaseModel):
    portfolio_history: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

# Mock data for testing
MOCK_SESSIONS = [
    {
        "session_id": "7b726ba8-213b-47e0-845f-e07651ae5edd",
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "uptime_seconds": 3600,
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "current_portfolio": {
            "cash": 50000,
            "assets": {
                "BTC/USD": {"quantity": 0.5, "value": 25000},
                "ETH/USD": {"quantity": 5, "value": 15000},
            },
            "total_value": 90000
        }
    }
]

# Routes
@paper_trading_router.post("/start")
async def start_paper_trading(
    config: PaperTradingConfig,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Start a new paper trading session"""
    session_id = str(uuid.uuid4())
    
    # In a real implementation, this would start a background task
    return {
        "status": "success", 
        "session_id": session_id,
        "message": "Paper trading session started successfully"
    }

@paper_trading_router.post("/stop/{session_id}")
async def stop_paper_trading(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Stop a paper trading session"""
    # Check if session exists
    session_exists = any(s["session_id"] == session_id for s in MOCK_SESSIONS)
    if not session_exists:
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    # In a real implementation, this would stop the background task
    return {"status": "success", "message": "Paper trading session stopped successfully"}

@paper_trading_router.get("/status", response_model=PaperTradingStatus)
async def get_status(
    session_id: str = Query(..., description="The ID of the session to get status for"),
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get the current status of a paper trading session"""
    # Find session
    session = next((s for s in MOCK_SESSIONS if s["session_id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    return {
        "status": session["status"],
        "uptime_seconds": session["uptime_seconds"],
        "symbols": session["symbols"],
        "current_portfolio": session["current_portfolio"],
        "recent_trades": [
            {"symbol": "BTC/USD", "side": "buy", "price": 50000, "quantity": 0.1, "timestamp": datetime.now().isoformat()},
            {"symbol": "ETH/USD", "side": "sell", "price": 3000, "quantity": 1.0, "timestamp": datetime.now().isoformat()}
        ],
        "performance_metrics": {
            "total_return": 0.05,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.03,
            "win_rate": 0.65
        }
    }

@paper_trading_router.get("/results/{session_id}", response_model=PaperTradingResults)
async def get_results(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get the results of a completed paper trading session"""
    # Find session
    session = next((s for s in MOCK_SESSIONS if s["session_id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    # Check if session is completed
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Paper trading session {session_id} is not completed")
    
    # Return mock results
    return {
        "portfolio_history": [
            {"timestamp": datetime.now().isoformat(), "total_value": 100000},
            {"timestamp": (datetime.now() + timedelta(hours=1)).isoformat(), "total_value": 102000},
            {"timestamp": (datetime.now() + timedelta(hours=2)).isoformat(), "total_value": 103000}
        ],
        "trades": [
            {"symbol": "BTC/USD", "side": "buy", "price": 50000, "quantity": 0.1, "timestamp": datetime.now().isoformat()},
            {"symbol": "ETH/USD", "side": "sell", "price": 3000, "quantity": 1.0, "timestamp": datetime.now().isoformat()}
        ],
        "performance_metrics": {
            "total_return": 0.05,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.03,
            "win_rate": 0.65
        }
    }

@paper_trading_router.get("/sessions", response_model=Dict[str, List[PaperTradingSession]])
async def get_sessions(
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get all paper trading sessions"""
    return {"sessions": MOCK_SESSIONS}

@paper_trading_router.get("/sessions/{session_id}", response_model=PaperTradingSession)
async def get_session_details(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get details for a specific paper trading session"""
    # Find session
    session = next((s for s in MOCK_SESSIONS if s["session_id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    return session

@paper_trading_router.get("/alerts/{session_id}", response_model=Dict[str, List[Dict[str, Any]]])
async def get_session_alerts(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get alerts for a specific paper trading session"""
    # Find session
    session = next((s for s in MOCK_SESSIONS if s["session_id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    # Return mock alerts
    return {
        "alerts": [
            {"type": "price_alert", "symbol": "BTC/USD", "condition": "above", "price": 55000, "triggered": False},
            {"type": "volatility_alert", "symbol": "ETH/USD", "condition": "above", "value": 0.05, "triggered": True}
        ]
    }

@paper_trading_router.post("/alerts/{session_id}/settings")
async def update_alert_settings(
    session_id: str,
    settings: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Update alert settings for a paper trading session"""
    # Find session
    session = next((s for s in MOCK_SESSIONS if s["session_id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    # In a real implementation, this would update the alert settings
    return {"status": "success", "message": "Alert settings updated successfully"}

@paper_trading_router.post("/alerts/{session_id}")
async def add_alert(
    session_id: str,
    alert: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Add an alert for a paper trading session"""
    # Find session
    session = next((s for s in MOCK_SESSIONS if s["session_id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    # In a real implementation, this would add the alert
    return {"status": "success", "message": "Alert added successfully"}
