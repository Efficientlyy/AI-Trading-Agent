"""
Simplified Paper Trading API endpoints.

This module provides minimal API endpoints for paper trading sessions.
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..common import logger

# Create API router
router = APIRouter(
    prefix="/paper-trading",
    tags=["paper-trading"]
)

# Define models
class PerformanceMetrics(BaseModel):
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: float = 0.0

class PaperTradingSession(BaseModel):
    session_id: str
    name: str
    description: str = ""
    exchange: str
    symbols: List[str]
    strategy: str
    initial_capital: float
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    performance_metrics: Optional[PerformanceMetrics] = None

class SessionCreate(BaseModel):
    name: str
    description: str = ""
    exchange: str
    symbols: List[str]
    strategy: str
    initial_capital: float

# API Endpoints
@router.get("/sessions", response_model=List[PaperTradingSession])
async def get_paper_trading_sessions():
    """Get all paper trading sessions."""
    logger.info("Getting paper trading sessions - returning empty list")
    return []

@router.post("/sessions", response_model=PaperTradingSession)
async def create_paper_trading_session(session: SessionCreate):
    """Create a new paper trading session."""
    logger.info(f"Creating new paper trading session: {session.dict()}")
    
    # Create a new session ID
    import uuid
    session_id = str(uuid.uuid4())
    
    # Return a new session
    return PaperTradingSession(
        session_id=session_id,
        name=session.name,
        description=session.description,
        exchange=session.exchange,
        symbols=session.symbols,
        strategy=session.strategy,
        initial_capital=session.initial_capital,
        status="active",
        start_time=datetime.utcnow(),
        performance_metrics=PerformanceMetrics()
    )

@router.delete("/{session_id}")
async def delete_paper_trading_session(session_id: str):
    """Delete a paper trading session."""
    logger.info(f"Deleting paper trading session {session_id}")
    return {"success": True, "message": f"Paper trading session {session_id} deleted"}
