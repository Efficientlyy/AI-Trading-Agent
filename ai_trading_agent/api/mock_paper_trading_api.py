"""
Mock Paper Trading API endpoints.

This module provides mock API endpoints for paper trading functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query

# Create API router
router = APIRouter()

# Store mock sessions
mock_sessions = {}

# Store mock alerts
mock_alerts = {}

# Test session ID for development
TEST_SESSION_ID = "4fb71ca9-351d-46cb-996f-2c3bc4d90b70"

@router.get("/sessions")
async def get_paper_trading_sessions():
    """
    Get a list of all paper trading sessions.
    
    Returns:
        List of paper trading sessions
    """
    # Create a mock session if none exist
    if not mock_sessions:
        mock_sessions[TEST_SESSION_ID] = {
            "session_id": TEST_SESSION_ID,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "uptime_seconds": 3600,
            "config_path": "configs/default_config.yaml",
            "duration_minutes": 60,
            "interval_minutes": 1
        }
    
    # Return list of sessions
    return list(mock_sessions.values())

@router.get("/status")
async def get_paper_trading_status(session_id: str = Query(..., description="ID of the session to get status for")):
    """
    Get the current status of a specific paper trading session.

    Args:
        session_id: ID of the session to get status for

    Returns:
        Status information for the session
    """
    # For development, create a mock session if it doesn't exist
    if session_id == TEST_SESSION_ID and session_id not in mock_sessions:
        # Create a mock session with mock data
        mock_sessions[session_id] = {
            "session_id": session_id,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "uptime_seconds": 3600,
            "config_path": "configs/default_config.yaml",
            "duration_minutes": 60,
            "interval_minutes": 1,
            "portfolio": {
                "total_value": 10000,
                "cash": 5000,
                "assets": {
                    "BTC": {"quantity": 0.1, "value": 3000},
                    "ETH": {"quantity": 1.0, "value": 2000}
                },
                "initial_value": 10000,
                "peak_value": 10500
            },
            "performance_metrics": {
                "total_return": 0.127,
                "annualized_return": 0.215,
                "sharpe_ratio": 1.85,
                "max_drawdown": -0.089,
                "win_rate": 0.68,
                "profit_factor": 2.3,
                "average_win": 0.042,
                "average_loss": -0.025,
                "risk_reward_ratio": 1.68,
                "recovery_factor": 1.43,
                "volatility": 0.12,
                "sortino_ratio": 2.1,
                "calmar_ratio": 2.42,
                "max_consecutive_wins": 5,
                "max_consecutive_losses": 2,
                "profit_per_day": 0.0032,
                "current_drawdown": -0.015,
                "drawdown_duration": 4
            },
            "recent_trades": [
                {
                    "id": "trade-1",
                    "symbol": "BTC/USDT",
                    "entry_time": int((datetime.now().timestamp() - 3600) * 1000),
                    "exit_time": int((datetime.now().timestamp() - 1800) * 1000),
                    "entry_price": 30000,
                    "exit_price": 30500,
                    "quantity": 0.1,
                    "pnl": 50,
                    "pnl_percent": 0.0167,
                    "duration": 1800,
                    "side": "buy",
                    "status": "closed"
                },
                {
                    "id": "trade-2",
                    "symbol": "ETH/USDT",
                    "entry_time": int((datetime.now().timestamp() - 7200) * 1000),
                    "exit_time": int((datetime.now().timestamp() - 3600) * 1000),
                    "entry_price": 2000,
                    "exit_price": 1950,
                    "quantity": 1.0,
                    "pnl": -50,
                    "pnl_percent": -0.025,
                    "duration": 3600,
                    "side": "buy",
                    "status": "closed"
                }
            ],
            "drawdown_data": {
                "timestamps": [int(t * 1000) for t in range(int(datetime.now().timestamp() - 86400), int(datetime.now().timestamp()), 3600)],
                "equity": [10000, 10100, 10200, 10150, 10050, 9950, 10000, 10100, 10200, 10300, 10400, 10500, 10450, 10400, 10350, 10300, 10250, 10200, 10150, 10100, 10050, 10000, 10050, 10100, 10150],
                "drawdown": [0, 0, 0, -0.0049, -0.0147, -0.0245, -0.0196, -0.0098, 0, 0, 0, 0, -0.0048, -0.0095, -0.0143, -0.019, -0.0238, -0.0286, -0.0333, -0.0381, -0.0429, -0.0476, -0.0429, -0.0381, -0.0333]
            },
            "trade_statistics": {
                "win_rate": 0.68,
                "profit_factor": 2.3,
                "average_win": 0.042,
                "average_loss": -0.025,
                "risk_reward_ratio": 1.68,
                "max_consecutive_wins": 5,
                "max_consecutive_losses": 2,
                "average_duration": 240
            },
            "agent_status": {
                "status": "active",
                "reasoning": "Market conditions are favorable for trading",
                "confidence": 0.85,
                "last_update": datetime.now().isoformat()
            },
            "alerts": [],
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "errors": []
        }

    # Check if session exists
    if session_id not in mock_sessions:
        # Return a mock response for development
        return {
            "session_id": session_id,
            "status": "not_found",
            "error": f"Session {session_id} not found"
        }

    # Return the mock session data
    return mock_sessions[session_id]

@router.post("/start")
async def start_paper_trading(config: Dict[str, Any]):
    """
    Start a new paper trading session.
    
    Args:
        config: Configuration for the paper trading session
        
    Returns:
        Information about the newly created session
    """
    # Generate a new session ID
    session_id = str(uuid.uuid4())
    
    # Create a new mock session
    mock_sessions[session_id] = {
        "session_id": session_id,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "uptime_seconds": 0,
        "config_path": config.get("config_path", "configs/default_config.yaml"),
        "duration_minutes": config.get("duration_minutes", 60),
        "interval_minutes": config.get("interval_minutes", 1)
    }
    
    # Return the session information
    return mock_sessions[session_id]

@router.post("/stop")
async def stop_paper_trading(session_id: str = Query(..., description="ID of the session to stop")):
    """
    Stop a paper trading session.
    
    Args:
        session_id: ID of the session to stop
        
    Returns:
        Status information for the stopped session
    """
    # Check if session exists
    if session_id not in mock_sessions:
        return {
            "session_id": session_id,
            "status": "not_found",
            "error": f"Session {session_id} not found"
        }
    
    # Update session status
    mock_sessions[session_id]["status"] = "stopped"
    
    # Return the updated session information
    return mock_sessions[session_id]

@router.get("/alerts/{session_id}")
async def get_session_alerts(session_id: str):
    """
    Get alerts for a specific paper trading session.
    
    Args:
        session_id: ID of the session to get alerts for
        
    Returns:
        List of alerts for the session
    """
    # Check if session exists
    if session_id not in mock_sessions:
        return {
            "session_id": session_id,
            "status": "not_found",
            "error": f"Session {session_id} not found"
        }
    
    # Return alerts for the session
    return {
        "alerts": mock_alerts.get(session_id, [])
    }

@router.post("/alerts/{session_id}")
async def add_alert(session_id: str, alert: Dict[str, Any]):
    """
    Add an alert for a paper trading session.
    
    Args:
        session_id: ID of the session to add the alert for
        alert: Alert data
        
    Returns:
        Status information for the added alert
    """
    # Check if session exists
    if session_id not in mock_sessions:
        return {
            "status": "error",
            "message": f"Session {session_id} not found"
        }
    
    # Initialize alerts for the session if not exists
    if session_id not in mock_alerts:
        mock_alerts[session_id] = []
    
    # Add ID to the alert
    alert_with_id = {**alert, "id": str(uuid.uuid4())}
    
    # Add alert to the session
    mock_alerts[session_id].append(alert_with_id)
    
    # Return success response
    return {
        "status": "success",
        "message": "Alert added successfully"
    }

@router.post("/alerts/{session_id}/settings")
async def update_alert_settings(session_id: str, settings: Dict[str, Any]):
    """
    Update alert settings for a paper trading session.
    
    Args:
        session_id: ID of the session to update settings for
        settings: Alert settings
        
    Returns:
        Status information for the updated settings
    """
    # Check if session exists
    if session_id not in mock_sessions:
        return {
            "status": "error",
            "message": f"Session {session_id} not found"
        }
    
    # Update alert settings in the session
    if "alert_settings" not in mock_sessions[session_id]:
        mock_sessions[session_id]["alert_settings"] = {}
    
    mock_sessions[session_id]["alert_settings"] = settings
    
    # Return success response
    return {
        "status": "success",
        "message": "Alert settings updated successfully"
    }
