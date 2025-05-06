"""
Paper Trading API Router

This module provides API endpoints for paper trading functionality.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal

from ai_trading_agent.trading_engine.live_trading_bridge import LiveTradingBridge
from ai_trading_agent.trading_engine.models import Position, Order
from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus, TradingMode

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/paper-trading", tags=["paper-trading"])

import os
import json
import pickle

# File storage for paper trading sessions
# In a production app, this would be stored in a database
_SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trading_sessions.pkl")
_paper_trading_sessions = {}

# Load sessions from file if it exists
def _load_sessions():
    global _paper_trading_sessions
    try:
        # Try to load existing sessions from the pickle file
        if os.path.exists(_SESSIONS_FILE):
            with open(_SESSIONS_FILE, 'rb') as f:
                loaded_sessions = pickle.load(f)
                if isinstance(loaded_sessions, dict):
                    _paper_trading_sessions = loaded_sessions
                    logger.info(f"Loaded {len(_paper_trading_sessions)} paper trading sessions from file")
                    return
    except Exception as e:
        logger.error(f"Error loading paper trading sessions: {e}")
    
    # If we get here, either the file doesn't exist or there was an error loading it
    # Initialize with an empty dictionary
    _paper_trading_sessions = {}
    logger.info("Initialized paper trading sessions with an empty dictionary")
    
    # Save the empty sessions to file
    _save_sessions()
    logger.info(f"Initialized with {len(_paper_trading_sessions)} paper trading sessions")

# Save sessions to file
def _save_sessions():
    try:
        # We can't directly pickle the LiveTradingBridge objects, so we'll just save the configs
        session_data = {}
        for session_id, session in _paper_trading_sessions.items():
            session_data[session_id] = {
                "config": session["config"],
                "name": session.get("name", ""),
                "created_at": session["created_at"]
            }
        
        logger.info(f"Saving {len(session_data)} paper trading sessions to {_SESSIONS_FILE}")
        with open(_SESSIONS_FILE, 'wb') as f:
            pickle.dump(session_data, f)
        logger.info("Sessions saved successfully")
    except Exception as e:
        logger.error(f"Error saving paper trading sessions: {e}")

# Load sessions on module import
_load_sessions()

# Models
class PaperTradingSessionCreate(BaseModel):
    """Model for creating a new paper trading session."""
    name: Optional[str] = None
    description: Optional[str] = None
    exchange: Optional[str] = "binance"
    symbols: List[str] = ["BTC/USDT"]
    strategy: Optional[str] = "default"
    initial_capital: float = 10000.0  # Frontend sends this as initial_capital, not initial_balance

class PaperTradingSession(BaseModel):
    """Model for a paper trading session."""
    id: str
    name: str
    initial_balance: Decimal
    current_balance: Decimal
    symbols: List[str]
    created_at: datetime
    status: str

class OrderCreate(BaseModel):
    """Model for creating a new order."""
    symbol: str
    side: str  # "buy" or "sell"
    type: str  # "market" or "limit"
    quantity: Decimal
    price: Optional[Decimal] = None

class OrderResponse(BaseModel):
    """Model for order response."""
    id: str
    symbol: str
    side: str
    type: str
    quantity: Decimal
    price: Optional[Decimal]
    status: str
    created_at: datetime

class PositionResponse(BaseModel):
    """Model for position response."""
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None

# Helper function to get a paper trading bridge
def get_paper_trading_bridge(session_id: str) -> LiveTradingBridge:
    """Get a paper trading bridge for a session."""
    if session_id not in _paper_trading_sessions:
        raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
    
    return _paper_trading_sessions[session_id]["bridge"]

# Endpoints
@router.post("/start")
async def start_paper_trading(session: PaperTradingSessionCreate):
    """Start a new paper trading session."""
    try:
        # Log the incoming request
        logger.info(f"Received request to start paper trading session: {session.dict()}")
        
        # Generate a unique session ID
        import uuid
        session_id = str(uuid.uuid4())
        logger.info(f"Generated session ID: {session_id}")
        
        # Create a name if not provided
        name = session.name or f"Paper Trading Session {len(_paper_trading_sessions) + 1}"
        logger.info(f"Using session name: {name}")
        
        # Create a paper trading bridge with a proper config dictionary
        config = {
            "exchange": session.exchange,
            "symbols": session.symbols,
            "trading_mode": TradingMode.PAPER,
            "initial_balance": session.initial_capital,
            "name": name
        }
        bridge = LiveTradingBridge(config=config)
        
        # Store the session in memory
        _paper_trading_sessions[session_id] = {
            "bridge": bridge,
            "config": session.dict(),
            "name": name,
            "created_at": datetime.now().isoformat()
        }
        
        # Log the creation for debugging
        logger.info(f"Created new paper trading session: {session_id} with name: {name}")
        logger.info(f"Current sessions: {list(_paper_trading_sessions.keys())}")
        
        # Save sessions to file
        _save_sessions()
        
        # Return the session info in the format expected by the frontend
        response = {
            "status": "success",
            "session_id": session_id,
            "message": f"Paper trading session started successfully"
        }
        
        logger.info(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error starting paper trading session: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting paper trading session: {str(e)}")

@router.get("/sessions")
async def get_paper_trading_sessions():
    """Get all paper trading sessions."""
    try:
        logger.info("Getting paper trading sessions")
        
        # Convert the sessions dictionary to a list of session objects
        sessions = []
        for session_id, session_data in _paper_trading_sessions.items():
            # Get the session config
            config = session_data.get("config", {})
            
            # Create a session object matching the frontend's expected format
            session = {
                "session_id": session_id,
                "name": session_data.get("name", "Paper Trading Session"),
                "status": "running",
                "start_time": session_data.get("created_at", datetime.now().isoformat()),
                "uptime_seconds": 0,
                "symbols": config.get("symbols", ["BTC/USDT"]),
                "exchange": config.get("exchange", "binance"),
                "strategy": config.get("strategy", "default"),
                "initial_capital": config.get("initial_capital", 10000.0)
            }
            sessions.append(session)
        
        logger.info(f"Found {len(sessions)} paper trading sessions")
        logger.info(f"Sessions data: {sessions}")
        
        # Return a dictionary with a 'sessions' key to match frontend expectations
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error getting paper trading sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting paper trading sessions: {str(e)}")


@router.get("/sessions/{session_id}")
async def get_paper_trading_session(session_id: str):
    """Get a specific paper trading session by ID."""
    try:
        logger.info(f"Getting paper trading session with ID: {session_id}")
        
        # Check if the session exists
        if session_id not in _paper_trading_sessions:
            logger.error(f"Session not found: {session_id}")
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
        # Get the session data
        session_data = _paper_trading_sessions[session_id]
        
        # Get the session config
        config = session_data.get("config", {})
        
        # Create a session object matching the frontend's expected format
        session = {
            "session_id": session_id,
            "name": session_data.get("name", "Paper Trading Session"),
            "status": "running",
            "start_time": session_data.get("created_at", datetime.now().isoformat()),
            "uptime_seconds": 0,
            "symbols": config.get("symbols", ["BTC/USDT"]),
            "exchange": config.get("exchange", "binance"),
            "strategy": config.get("strategy", "default"),
            "initial_capital": config.get("initial_capital", 10000.0)
        }
        
        logger.info(f"Found session: {session}")
        
        return session
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting paper trading session: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting paper trading session: {str(e)}")

@router.get("/status")
async def get_paper_trading_status(session_id: str):
    """Get the status of a paper trading session."""
    try:
        bridge = get_paper_trading_bridge(session_id)
        status = await bridge.get_trading_status()
        return status
    except Exception as e:
        logger.error(f"Error getting paper trading status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting paper trading status: {str(e)}")

@router.get("/{session_id}/positions", response_model=List[PositionResponse])
async def get_positions(session_id: str):
    """Get all positions for a paper trading session."""
    try:
        bridge = get_paper_trading_bridge(session_id)
        positions = await bridge.get_positions()
        
        # Convert positions to response format
        position_responses = []
        for symbol, position in positions.items():
            position_responses.append({
                "symbol": position.symbol,
                "side": position.side.value,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price if hasattr(position, "current_price") else None,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl
            })
        
        return position_responses
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting positions: {str(e)}")

@router.post("/{session_id}/orders", response_model=OrderResponse)
async def place_order(session_id: str, order: OrderCreate):
    """Place a new order in a paper trading session."""
    try:
        bridge = get_paper_trading_bridge(session_id)
        
        # Convert order side and type to enums
        order_side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        order_type = OrderType.MARKET if order.type.lower() == "market" else OrderType.LIMIT
        
        # Place the order
        order_result = await bridge.place_order(
            symbol=order.symbol,
            side=order_side,
            order_type=order_type,
            quantity=order.quantity,
            price=order.price
        )
        
        # Return the order info
        return {
            "id": order_result.order_id,
            "symbol": order_result.symbol,
            "side": order_result.side.value,
            "type": order_result.type.value,
            "quantity": order_result.quantity,
            "price": order_result.price,
            "status": order_result.status.value,
            "created_at": order_result.created_at
        }
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")

@router.get("/{session_id}/orders", response_model=List[OrderResponse])
async def get_orders(session_id: str):
    """Get all orders for a paper trading session."""
    try:
        bridge = get_paper_trading_bridge(session_id)
        orders = await bridge.get_orders()
        
        # Convert orders to response format
        order_responses = []
        for order_id, order in orders.items():
            order_responses.append({
                "id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "type": order.type.value,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status.value,
                "created_at": order.created_at
            })
        
        return order_responses
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting orders: {str(e)}")

@router.delete("/{session_id}/orders/{order_id}")
async def cancel_order(session_id: str, order_id: str):
    """Cancel an order in a paper trading session."""
    try:
        bridge = get_paper_trading_bridge(session_id)
        result = await bridge.cancel_order(order_id)
        return {"success": result, "message": f"Order {order_id} cancelled"}
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")

@router.post("/stop/{session_id}")
async def stop_paper_trading(session_id: str):
    """Stop a paper trading session."""
    try:
        if session_id not in _paper_trading_sessions:
            raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
        
        # Remove the session
        del _paper_trading_sessions[session_id]
        
        return {"success": True, "message": f"Paper trading session {session_id} stopped"}
    except Exception as e:
        logger.error(f"Error stopping paper trading session: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping paper trading session: {str(e)}")

@router.post("/sessions/{session_id}/stop")
async def stop_paper_trading_session(session_id: str):
    """Stop a paper trading session using the sessions/{session_id}/stop endpoint pattern.
    This matches the URL pattern used by the frontend.
    """
    try:
        if session_id not in _paper_trading_sessions:
            raise HTTPException(status_code=404, detail=f"Paper trading session {session_id} not found")
        
        # Remove the session
        del _paper_trading_sessions[session_id]
        
        return {"success": True, "message": f"Paper trading session {session_id} stopped"}
    except Exception as e:
        logger.error(f"Error stopping paper trading session: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping paper trading session: {str(e)}")
