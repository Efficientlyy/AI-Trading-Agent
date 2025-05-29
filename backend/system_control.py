"""
System Control API

This module provides endpoints for controlling the entire paper trading system
and individual agents.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

# Import authentication
from backend.security.auth import get_current_user, get_mock_user_override

# Setup logging
logger = logging.getLogger(__name__)

# Create router
system_control_router = APIRouter(prefix="/api/system", tags=["system-control"])

# Models
class SystemStatus(BaseModel):
    status: str  # "running", "stopped", "starting", "stopping", "error"
    active_agents: int = 0
    total_agents: int = 0
    active_sessions: int = 0
    total_sessions: int = 0
    uptime_seconds: Optional[int] = None
    start_time: Optional[str] = None
    last_update: Optional[str] = None
    health_metrics: Optional[Dict[str, float]] = None

class AgentStatus(BaseModel):
    agent_id: str
    name: str
    type: str
    status: str  # "running", "stopped", "error"
    metrics: Optional[Dict[str, Any]] = None
    last_updated: datetime = datetime.now()

class AgentConfig(BaseModel):
    agent_id: str
    name: str
    type: str
    config: Dict[str, Any]

class SystemHealthMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    api_calls_remaining: Dict[str, int]
    errors_last_hour: int
    warnings_last_hour: int

class AgentLogsResponse(BaseModel):
    agent_id: str
    lines: List[str]  # Changed from 'logs' to 'lines'
    total_lines: Optional[int] = None # Added 'total_lines'
    last_updated: Optional[datetime] = None # Made 'last_updated' optional as frontend doesn't expect it
# Mock data for testing
MOCK_AGENTS = {
    "sentiment-agent-1": {
        "agent_id": "sentiment-agent-1",
        "name": "Sentiment Analysis Agent",
        "type": "sentiment",
        "status": "running",
        "metrics": {
            "sentiment_score": 0.78,
            "sentiment_direction": "bullish",
            "confidence": 0.85,
            "sources_analyzed": 125
        },
        "last_updated": datetime.now()
    },
    "strategy-agent-1": {
        "agent_id": "strategy-agent-1",
        "name": "MA Crossover Strategy Agent",
        "type": "strategy",
        "status": "running",
        "metrics": {
            "active_strategy": "MA Crossover",
            "signals_generated": 8,
            "current_position": "long",
            "strategy_performance": 0.034
        },
        "last_updated": datetime.now()
    },
    "data-agent-1": {
        "agent_id": "data-agent-1",
        "name": "Market Data Agent",
        "type": "data",
        "status": "running",
        "metrics": {
            "data_sources": 3,
            "symbols_tracked": 5,
            "last_update": "5s ago",
            "data_points_processed": 12500
        },
        "last_updated": datetime.now()
    },
    "execution-agent-1": {
        "agent_id": "execution-agent-1",
        "name": "Order Execution Agent",
        "type": "execution",
        "status": "running",
        "metrics": {
            "orders_processed": 12,
            "success_rate": 0.98,
            "avg_execution_time": 0.45,
            "pending_orders": 0
        },
        "last_updated": datetime.now()
    },
    "spec_sentiment_alpha": {
        "agent_id": "spec_sentiment_alpha",
        "name": "Alpha Sentiment Specialist",
        "type": "specialist_sentiment",
        "status": "running",
        "metrics": {"score": 0.65, "confidence": 0.9},
        "last_updated": datetime.now()
    },
    "spec_tech_twelve": {
        "agent_id": "spec_tech_twelve",
        "name": "Twelve Data Technical Specialist",
        "type": "specialist_technical",
        "status": "running",
        "metrics": {"active_indicators": 5, "signals": 2},
        "last_updated": datetime.now()
    },
    "decision_main": {
        "agent_id": "decision_main",
        "name": "Main Decision Agent",
        "type": "decision_engine",
        "status": "running",
        "metrics": {"decisions_made": 50, "accuracy": 0.7},
        "last_updated": datetime.now()
    },
    "exec_broker_alpaca": {
        "agent_id": "exec_broker_alpaca",
        "name": "Alpaca Execution Broker",
        "type": "execution_broker",
        "status": "running",
        "metrics": {"orders_filled": 10, "avg_fill_price_accuracy": 0.99},
        "last_updated": datetime.now()
    },
    "session_mock_generic": {
        "agent_id": "session_mock_generic",
        "name": "Generic Mock Session Agent",
        "type": "session_runner",
        "status": "running",
        "metrics": {"trades_executed": 5, "pnl": 150.75},
        "last_updated": datetime.now()
    },
    "default-agent": {
        "agent_id": "default-agent",
        "name": "Default System Agent",
        "type": "system_utility",
        "status": "running",
        "metrics": {"tasks_processed": 1000, "errors": 0},
        "last_updated": datetime.now()
    }
}

# Global system status
SYSTEM_STATUS = {
    "status": "running",
    "agent_statuses": MOCK_AGENTS,
    "active_sessions": 2,
    "uptime_seconds": 3600,
    "resource_utilization": {
        "cpu": 0.25,
        "memory": 0.35,
        "disk": 0.15,
        "network": 0.20
    },
    "last_updated": datetime.now()
}

# Routes
@system_control_router.get("/status", response_model=SystemStatus)
async def get_system_status(
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get the current status of the entire paper trading system"""
    # Count active agents
    active_agents = sum(1 for agent in MOCK_AGENTS.values() if agent["status"] == "running")
    total_agents = len(MOCK_AGENTS)
    
    # Format the response according to the SystemStatus model
    return {
        "status": SYSTEM_STATUS["status"],
        "active_agents": active_agents,
        "total_agents": total_agents,
        "active_sessions": SYSTEM_STATUS["active_sessions"],
        "total_sessions": 3,  # Mock value
        "uptime_seconds": SYSTEM_STATUS["uptime_seconds"],
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "health_metrics": {
            "cpu_usage": SYSTEM_STATUS["resource_utilization"]["cpu"],
            "memory_usage": SYSTEM_STATUS["resource_utilization"]["memory"],
            "disk_usage": SYSTEM_STATUS["resource_utilization"]["disk"]
        }
    }

@system_control_router.post("/start")
async def start_system(
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Start the entire paper trading system"""
    global SYSTEM_STATUS
    
    # Check if system is already running
    if SYSTEM_STATUS["status"] == "running":
        return {"status": "success", "message": "System is already running"}
    
    # Update system status
    SYSTEM_STATUS["status"] = "starting"
    
    # In a real implementation, this would start all agents
    # For now, we'll simulate starting all agents
    for agent_id in MOCK_AGENTS:
        MOCK_AGENTS[agent_id]["status"] = "running"
    
    # Update system status
    SYSTEM_STATUS["status"] = "running"
    SYSTEM_STATUS["last_updated"] = datetime.now()
    
    return {
        "status": "success",
        "message": "Paper trading system started successfully"
    }

@system_control_router.post("/stop")
async def stop_system(
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Stop the entire paper trading system"""
    global SYSTEM_STATUS
    
    # Check if system is already stopped
    if SYSTEM_STATUS["status"] == "stopped":
        return {"status": "success", "message": "System is already stopped"}
    
    # Update system status
    SYSTEM_STATUS["status"] = "stopping"
    
    # In a real implementation, this would stop all agents
    # For now, we'll simulate stopping all agents
    for agent_id in MOCK_AGENTS:
        MOCK_AGENTS[agent_id]["status"] = "stopped"
    
    # Update system status
    SYSTEM_STATUS["status"] = "stopped"
    SYSTEM_STATUS["last_updated"] = datetime.now()
    
    return {
        "status": "success",
        "message": "Paper trading system stopped successfully"
    }

@system_control_router.get("/health", response_model=SystemHealthMetrics)
async def get_system_health(
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get detailed system health metrics"""
    return {
        "cpu_usage": 0.25,
        "memory_usage": 0.35,
        "disk_usage": 0.15,
        "network_usage": 0.20,
        "api_calls_remaining": {
            "coingecko": 45,
            "cryptocompare": 980,
            "binance": 1200
        },
        "errors_last_hour": 0,
        "warnings_last_hour": 2
    }

# Agent-specific routes
@system_control_router.get("/agents", response_model=Dict[str, AgentStatus])
async def get_agents(
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """List all available agents with their status"""
    return MOCK_AGENTS

@system_control_router.get("/agents/{agent_id}", response_model=AgentStatus)
async def get_agent(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get detailed information about a specific agent"""
    if agent_id not in MOCK_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return MOCK_AGENTS[agent_id]

@system_control_router.post("/agents/{agent_id}/start")
async def start_agent(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Start a specific agent"""
    if agent_id not in MOCK_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Check if agent is already running
    if MOCK_AGENTS[agent_id]["status"] == "running":
        return {"status": "success", "message": f"Agent {agent_id} is already running"}
    
    # Update agent status
    MOCK_AGENTS[agent_id]["status"] = "running"
    MOCK_AGENTS[agent_id]["last_updated"] = datetime.now()
    
    return {
        "status": "success",
        "message": f"Agent {agent_id} started successfully"
    }

@system_control_router.post("/agents/{agent_id}/stop")
async def stop_agent(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Stop a specific agent"""
    if agent_id not in MOCK_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Check if agent is already stopped
    if MOCK_AGENTS[agent_id]["status"] == "stopped":
        return {"status": "success", "message": f"Agent {agent_id} is already stopped"}
    
    # Update agent status
    MOCK_AGENTS[agent_id]["status"] = "stopped"
    MOCK_AGENTS[agent_id]["last_updated"] = datetime.now()
    
    return {
        "status": "success",
        "message": f"Agent {agent_id} stopped successfully"
    }

@system_control_router.get("/agents/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get performance metrics for a specific agent"""
    if agent_id not in MOCK_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return {
        "agent_id": agent_id,
        "metrics": MOCK_AGENTS[agent_id]["metrics"],
        "last_updated": MOCK_AGENTS[agent_id]["last_updated"]
    }

@system_control_router.get("/agents/{agent_id}/logs", response_model=AgentLogsResponse)
async def get_agent_logs(
    agent_id: str,
    lines: Optional[int] = Query(100, description="Number of recent log lines to retrieve"),
    current_user: Dict[str, Any] = Depends(get_mock_user_override)
):
    """Get recent logs for a specific agent"""
    if agent_id not in MOCK_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # In a real implementation, this would fetch logs from a logging system or file
    # For now, generate mock logs
    mock_logs = [
        f"{datetime.now() - timedelta(seconds=i*5)} INFO: Agent {agent_id} processed item {100-i}"
        for i in range(min(lines, 20)) # Limit mock logs for brevity
    ]
    mock_logs.reverse() # Show most recent last

    return AgentLogsResponse(
        agent_id=agent_id,
        lines=mock_logs, # Changed from 'logs' to 'lines'
        total_lines=len(mock_logs), # Added 'total_lines'
        last_updated=datetime.now() # Kept last_updated for now, though frontend doesn't strictly need it
    )
