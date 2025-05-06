"""
Direct Paper Trading API

This module provides API endpoints for paper trading functionality without relying on external imports.
"""

import os
import json
import pickle
import logging
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/paper-trading", tags=["paper-trading"])

# File storage for paper trading sessions
SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct_paper_trading_sessions.pkl")

# In-memory sessions storage
_sessions = {}

# Models
class SessionCreate(BaseModel):
    name: str
    description: str = ""
    exchange: str
    symbols: List[str]
    strategy: str
    initial_capital: float

class Session(BaseModel):
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
    performance_metrics: Optional[dict] = None
    
    class Config:
        # This allows the model to populate from a dict with different field names
        # It helps with compatibility with the frontend
        orm_mode = True

# Load sessions from file
def _load_sessions():
    global _sessions
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, 'rb') as f:
                _sessions = pickle.load(f)
                logger.info(f"Loaded {len(_sessions)} paper trading sessions from {SESSIONS_FILE}")
        else:
            _sessions = {}
            logger.info(f"No sessions file found at {SESSIONS_FILE}, starting with empty sessions")
    except Exception as e:
        logger.error(f"Error loading sessions: {str(e)}")
        _sessions = {}

# Save sessions to file
def _save_sessions():
    try:
        with open(SESSIONS_FILE, 'wb') as f:
            pickle.dump(_sessions, f)
        logger.info(f"Saved {len(_sessions)} paper trading sessions to {SESSIONS_FILE}")
    except Exception as e:
        logger.error(f"Error saving sessions: {str(e)}")

# Initialize sessions
_load_sessions()

# API Endpoints
@router.get("/sessions", response_model=List[Session])
async def get_paper_trading_sessions():
    """Get all paper trading sessions."""
    logger.info(f"Getting paper trading sessions - returning {len(_sessions)} sessions")
    return list(_sessions.values())

# Model for the /start endpoint which has different fields
class PaperTradingConfig(BaseModel):
    config_path: Optional[str] = None
    duration_minutes: Optional[int] = None
    interval_minutes: Optional[int] = None
    # Add these fields to make it compatible with the form data too
    name: Optional[str] = "Paper Trading Session"
    description: Optional[str] = ""
    exchange: Optional[str] = "binance"
    symbols: Optional[List[str]] = ["BTC/USDT"]
    strategy: Optional[str] = "default"
    initial_capital: Optional[float] = 10000.0

@router.post("/start", response_model=dict)
async def start_paper_trading(config: PaperTradingConfig):
    """Start a new paper trading session (frontend uses this endpoint)."""
    logger.info(f"Starting new paper trading session via /start endpoint: {config.dict()}")
    
    # Create a new session ID
    session_id = str(uuid.uuid4())
    
    # Create the session object
    new_session = Session(
        session_id=session_id,
        name=config.name or "Paper Trading Session",
        description=config.description or "",
        exchange=config.exchange or "binance",
        symbols=config.symbols or ["BTC/USDT"],
        strategy=config.strategy or "default",
        initial_capital=config.initial_capital or 10000.0,
        status="active",
        start_time=datetime.utcnow()
    )
    
    # Store the session
    _sessions[session_id] = new_session
    
    # Save to file
    _save_sessions()
    
    logger.info(f"Created new paper trading session with ID: {session_id}")
    
    # Return the response in the format expected by the frontend
    return {
        "status": "success", 
        "session_id": session_id, 
        "message": "Paper trading session started successfully"
    }

@router.post("/sessions", response_model=Session)
async def create_paper_trading_session(session: SessionCreate):
    """Create a new paper trading session."""
    logger.info(f"Creating new paper trading session: {session.dict()}")
    
    # Create a new session ID
    session_id = str(uuid.uuid4())
    
    # Create the session object
    new_session = Session(
        session_id=session_id,
        name=session.name,
        description=session.description,
        exchange=session.exchange,
        symbols=session.symbols,
        strategy=session.strategy,
        initial_capital=session.initial_capital,
        status="active",
        start_time=datetime.utcnow()
    )
    
    # Store the session
    _sessions[session_id] = new_session
    
    # Save to file
    _save_sessions()
    
    logger.info(f"Created new paper trading session with ID: {session_id}")
    
    # Return the new session
    return new_session

@router.delete("/{session_id}")
async def delete_paper_trading_session(session_id: str):
    """Delete a paper trading session."""
    logger.info(f"Deleting paper trading session {session_id}")
    
    # Check if session exists
    if session_id not in _sessions:
        logger.warning(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    # Delete the session
    del _sessions[session_id]
    
    # Save to file
    _save_sessions()
    
    logger.info(f"Deleted paper trading session {session_id}")
    
    return {"success": True, "message": f"Paper trading session {session_id} deleted"}

@router.post("/sessions/{session_id}/stop")
async def stop_paper_trading_session(session_id: str):
    """Stop a paper trading session."""
    logger.info(f"Stopping paper trading session {session_id}")
    
    # Check if session exists
    if session_id not in _sessions:
        logger.warning(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    # Update session status
    _sessions[session_id].status = "completed"
    _sessions[session_id].end_time = datetime.utcnow()
    
    # Save to file
    _save_sessions()
    
    logger.info(f"Stopped paper trading session {session_id}")
    
    return {"success": True, "message": f"Paper trading session {session_id} stopped"}

@router.post("/stop/{session_id}")
async def stop_paper_trading_frontend(session_id: str):
    """Stop a paper trading session using the endpoint expected by the frontend."""
    logger.info(f"Stopping paper trading session via /stop endpoint: {session_id}")
    
    # Check if session exists
    if session_id not in _sessions:
        logger.warning(f"Session {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    # Update session status
    _sessions[session_id].status = "completed"
    _sessions[session_id].end_time = datetime.utcnow()
    
    # Save to file
    _save_sessions()
    
    logger.info(f"Stopped paper trading session {session_id}")
    
    return {"status": "success", "message": f"Paper trading session {session_id} stopped"}

@router.delete("/sessions")
async def delete_all_paper_trading_sessions():
    """Delete all paper trading sessions."""
    global _sessions
    
    # Count sessions before deletion
    session_count = len(_sessions)
    
    # Clear all sessions
    _sessions = {}
    
    # Save to file
    _save_sessions()
    
    logger.info(f"Deleted all {session_count} paper trading sessions")
    
    return {"success": True, "message": f"All {session_count} paper trading sessions deleted"}
