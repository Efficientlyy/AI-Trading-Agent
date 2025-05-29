"""
System Control API

This module provides endpoints for controlling the entire paper trading system
and individual agents.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import logging
import os
import re
import time
import asyncio
from pathlib import Path

# Import data feed manager
from .data_feed_manager import data_feed_manager

# Import authentication (adjust as needed based on your actual auth implementation)
try:
    from ai_trading_agent.api.auth import get_current_user, get_mock_user
    auth_available = True
except ImportError:
    auth_available = False
    # Create mock auth functions if not available
    def get_mock_user(): # type: ignore
        return {"user_id": "mock-user", "username": "mock-user"}

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/system", tags=["system-control"])

# --- REAL SYSTEM CONTROL LOGIC (SessionManager import) ---
# This should be the actual session manager from your project structure
try:
    from ai_trading_agent.agent.session_manager import session_manager
except ImportError:
    # Fallback for environments where session_manager might not be fully set up
    # This allows the API to still load, but real operations will fail.
    logger.error("Failed to import session_manager. Real operations will not work.")
    class MockSessionManager:
        async def resume_all_sessions(self): return True
        async def stop_all_sessions(self): return True
        def get_all_sessions(self): return []
        def get_session(self, session_id: str): return None
        async def resume_session(self, session_id: str): return True
        async def stop_session(self, session_id: str): return True

    session_manager = MockSessionManager() # type: ignore


# --- Models (Consolidated) ---
class SystemStatus(BaseModel):
    status: str  # "running", "stopped", "starting", "stopping", "error"
    active_agents: int = 0
    total_agents: int = 0
    active_sessions: int = 0
    total_sessions: int = 0
    uptime_seconds: Optional[int] = 0
    start_time: Optional[str] = None
    last_update: Optional[str] = None
    health_metrics: Optional[Dict[str, float]] = None

class AgentStatus(BaseModel):
    agent_id: str
    name: str
    type: str
    status: str  # "running", "stopped", "error", "paused", "initializing", "starting"
    metrics: Optional[Dict[str, Any]] = None
    last_updated: datetime = datetime.now()
    symbols: Optional[List[str]] = None
    strategy: Optional[str] = None
    agent_role: Optional[str] = None    # Added agent_role
    outputs_to: Optional[List[str]] = None # Added outputs_to


# Flag to use mock data (for development/testing)
USE_MOCK_DATA = True  # Force using mock data to fix the startup issue

# --- MOCK DATA DEFINITIONS (for development/testing when USE_MOCK_DATA is true) ---
MOCK_AGENTS_DATA = {
    "spec_sentiment_alpha": {
        "name": "Sentiment Analyzer (AlphaV)",
        "type": "sentiment_alpha_vantage",
        "status": "stopped",
        "agent_role": "specialized_sentiment",
        "outputs_to": ["decision_main"],
        "metrics": { "accuracy": 0.78, "signals_generated": 155, "api_calls": 50 },
        "last_updated": datetime.now(),
        "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
        "strategy": "NewsSentimentWeighted"
    },
    "spec_tech_twelve": {
        "name": "Technical Analyst (TwelveData)",
        "type": "technical_twelve_data",
        "status": "stopped",
        "agent_role": "specialized_technical",
        "outputs_to": ["decision_main"],
        "metrics": { "active_indicators": 5, "last_signal_age_sec": 25, "patterns_detected": 3 },
        "last_updated": datetime.now(),
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "strategy": "MultiTimeframeCrossover"
    },
    "decision_main": {
        "name": "Main Decision Aggregator",
        "type": "weighted_signal_aggregator",
        "status": "stopped",
        "agent_role": "decision_aggregator",
        "outputs_to": ["exec_broker_alpaca"],
        "metrics": { "signals_processed": 320, "decisions_made": 45, "risk_overrides": 2 },
        "last_updated": datetime.now(),
        "symbols": [], # Decision agent might not directly trade symbols but oversee them
        "strategy": "PortfolioRiskBalanced"
    },
    "exec_broker_alpaca": {
        "name": "Execution Handler (Alpaca)",
        "type": "broker_alpaca",
        "status": "stopped",
        "agent_role": "execution_broker",
        "outputs_to": [], # Execution layer is typically the end of this internal flow
        "metrics": { "orders_placed": 45, "orders_filled": 43, "avg_fill_latency_ms": 150 },
        "last_updated": datetime.now(),
        "symbols": [], # Execution layer acts on decisions, doesn't monitor symbols itself
        "strategy": None # Not a strategy agent
    },
    # Keeping one of the old session-style mocks for broader compatibility during transition,
    # but it should ideally be phased out or adapted to the new roles.
    "session_mock_generic": {
        "name": "Generic Paper Trading Session",
        "type": "paper_trading_session", # This type might be too generic now
        "status": "stopped",
        "agent_role": "specialized_standalone", # Example role for a self-contained session
        "outputs_to": ["decision_main"], # Or could output to a decision agent
        "metrics": {
            "win_rate": 0.60, "profit_factor": 1.5, "avg_profit_loss": 10.0, "max_drawdown": 0.15,
            "pnl": -50.25, "trades": 12
        },
        "last_updated": datetime.now(),
        "symbols": ["DOGE/USDT", "SHIB/USDT"],
        "strategy": "MomentumScalperV1"
    }
}

MOCK_SESSIONS_DATA = [
    {
        "session_id": "session-mock-1",
        "name": "Aggressive BTC/ETH Paper Session",
        "status": "stopped",
        "start_time": (datetime.now() - timedelta(minutes=45)).isoformat(),
        "end_time": None,
        "strategy_name": "aggressive_v2",
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "initial_capital": 10000.0,
        "current_capital": 10120.50,
        "profit_loss": 120.50,
        "profit_loss_pct": 1.205,
        "trade_count": 5,
        "log_file": "logs/session-mock-1.log"
    },
    {
        "session_id": "session-mock-2",
        "name": "Conservative SOL Paper Session",
        "status": "paused",
        "start_time": (datetime.now() - timedelta(hours=2)).isoformat(),
        "end_time": None,
        "strategy_name": "conservative_rsi",
        "symbols": ["SOL/USDT"],
        "initial_capital": 5000.0,
        "current_capital": 4980.0,
        "profit_loss": -20.0,
        "profit_loss_pct": -0.4,
        "trade_count": 2,
        "log_file": "logs/session-mock-2.log"
    }
]

GLOBAL_SYSTEM_STATUS_MOCK = {
    "status": "stopped", # "running", "stopped", "starting", "stopping", "error"
    "active_agents": 0,
    "total_agents": len(MOCK_AGENTS_DATA),
    "active_sessions": 0,
    "total_sessions": len(MOCK_SESSIONS_DATA),
    "uptime_seconds": 0,
    "start_time": None,
    "last_update": datetime.now().isoformat(),
    "health_metrics": {"cpu_usage": 15.5, "memory_usage_mb": 256.0, "disk_free_gb": 50.0, "data_feed_connected": False}
}

# --- SYSTEM CONTROL MODE SWITCH ---
# Force mock data to false by default for predictable behavior
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").lower() == "true"  # Default to false

if USE_MOCK_DATA:
    logger.warning("System Control API is running in MOCK DATA mode.")
else:
    logger.info("System Control API is running in REAL DATA mode (interacting with SessionManager).")

# --- SYSTEM CONTROL ENDPOINTS ---

@router.post("/start", summary="Start the entire paper trading system")
async def start_system():
    logger.info("Request to start the entire paper trading system.")
    if USE_MOCK_DATA:
        try:
            logger.info("Starting mock system...")
            if GLOBAL_SYSTEM_STATUS_MOCK["status"] == "running":
                logger.info("Mock system is already running.")
                return {"status": "success", "message": "System is already running (mock)"}
                
            # Set system to starting state
            GLOBAL_SYSTEM_STATUS_MOCK["status"] = "starting"
            
            # Start all mock agents - but initialize them as stopped by default
            # Only set them to running when explicitly requested
            for agent_id in MOCK_AGENTS_DATA:
                # Don't auto-start agents, keep them in stopped state
                MOCK_AGENTS_DATA[agent_id]["status"] = "stopped"  
                MOCK_AGENTS_DATA[agent_id]["last_updated"] = datetime.now()
                
            # Update global status
            GLOBAL_SYSTEM_STATUS_MOCK["status"] = "running"
            GLOBAL_SYSTEM_STATUS_MOCK["active_agents"] = sum(1 for agent in MOCK_AGENTS_DATA.values() if agent["status"] == "running")
            GLOBAL_SYSTEM_STATUS_MOCK["active_sessions"] = sum(1 for sess in MOCK_SESSIONS_DATA if sess["status"] == "running") 
            GLOBAL_SYSTEM_STATUS_MOCK["start_time"] = datetime.now().isoformat()
            GLOBAL_SYSTEM_STATUS_MOCK["last_update"] = datetime.now().isoformat()
            
            # Make sure data feed appears connected
            if data_feed_manager:
                data_feed_manager.force_connected = True
                
            logger.info("Mock system started successfully.")
            return {"status": "success", "message": "Paper trading system started successfully (mock)"}
        except Exception as e:
            logger.error(f"Error starting mock system: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error starting system: {str(e)}")
    else:
        try:
            # Start the data feed manager
            logger.info("Starting data feed manager...")
            data_feed_manager.start()
            
            # Wait briefly for data feed to initialize
            await asyncio.sleep(1)
            
            # Check data feed status
            data_feed_status = data_feed_manager.get_status()
            data_feed_initializing = data_feed_status.get("status") in ["connected", "connecting"]
            
            if not data_feed_initializing:
                logger.warning("Data feed may not be properly connected, but continuing with system startup")
            else:
                logger.info("Data feed manager initialized successfully")
            
            # Start all trading sessions
            logger.info("Attempting to resume all sessions...")
            await session_manager.resume_all_sessions()
            
            logger.info("System started successfully (real mode)")
            return {
                "status": "success", 
                "message": "System started successfully",
                "data_feed_status": data_feed_status.get("status")
            }
        except Exception as e:
            logger.error(f"Error starting system (real mode): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error starting system: {str(e)}")

@router.post("/stop", summary="Stop the entire paper trading system")
async def stop_system():
    logger.info("Request to stop the entire paper trading system.")
    if USE_MOCK_DATA:
        if GLOBAL_SYSTEM_STATUS_MOCK["status"] == "stopped":
            logger.info("Mock system is already stopped.")
            return {"status": "success", "message": "System is already stopped (mock)"}
            
        GLOBAL_SYSTEM_STATUS_MOCK["status"] = "stopping"
        for agent_id in MOCK_AGENTS_DATA:
            MOCK_AGENTS_DATA[agent_id]["status"] = "stopped"
            MOCK_AGENTS_DATA[agent_id]["last_updated"] = datetime.now()
            
        for session in MOCK_SESSIONS_DATA:
            session["status"] = "stopped"
            
        GLOBAL_SYSTEM_STATUS_MOCK["status"] = "stopped"
        GLOBAL_SYSTEM_STATUS_MOCK["active_agents"] = 0
        GLOBAL_SYSTEM_STATUS_MOCK["active_sessions"] = 0
        GLOBAL_SYSTEM_STATUS_MOCK["uptime_seconds"] = 0
        GLOBAL_SYSTEM_STATUS_MOCK["last_update"] = datetime.now().isoformat()
        
        logger.info("Mock system stopped successfully.")
        return {"status": "success", "message": "Paper trading system stopped successfully (mock)"}
    else:
        try:
            # Stop the data feed manager first
            logger.info("Stopping data feed manager...")
            data_feed_manager.stop()
            
            logger.info("Attempting to stop all sessions (real mode)...")
            await session_manager.stop_all_sessions()
            
            logger.info("System stopped (all sessions stopped).")
            return {"status": "success", "message": "System stopped (all sessions stopped)"}
        except Exception as e:
            logger.error(f"Error stopping system (real mode): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error stopping system: {str(e)}")

@router.get("/status", response_model=SystemStatus, summary="Get current system status")
async def get_system_status():
    if USE_MOCK_DATA:
        # Update active counts for mock
        GLOBAL_SYSTEM_STATUS_MOCK["active_agents"] = sum(1 for agent in MOCK_AGENTS_DATA.values() if agent["status"] == "running")
        GLOBAL_SYSTEM_STATUS_MOCK["total_agents"] = len(MOCK_AGENTS_DATA)
        GLOBAL_SYSTEM_STATUS_MOCK["active_sessions"] = sum(1 for sess in MOCK_SESSIONS_DATA if sess["status"] == "running")
        GLOBAL_SYSTEM_STATUS_MOCK["total_sessions"] = len(MOCK_SESSIONS_DATA)
        if GLOBAL_SYSTEM_STATUS_MOCK["status"] == "running" and GLOBAL_SYSTEM_STATUS_MOCK["start_time"]:
            uptime = datetime.now() - datetime.fromisoformat(GLOBAL_SYSTEM_STATUS_MOCK["start_time"])
            GLOBAL_SYSTEM_STATUS_MOCK["uptime_seconds"] = int(uptime.total_seconds())
        else:
            GLOBAL_SYSTEM_STATUS_MOCK["uptime_seconds"] = 0
        GLOBAL_SYSTEM_STATUS_MOCK["last_update"] = datetime.now().isoformat()
        return SystemStatus(**GLOBAL_SYSTEM_STATUS_MOCK)
    else:
        try:
            # Get data feed status
            try:
                data_feed_status = data_feed_manager.get_status()
                data_feed_connected = data_feed_status.get("status") in ["connected", "online", "active"]
            except Exception as e:
                logger.error(f"Error getting data feed status: {e}")
                data_feed_status = {"status": "disconnected", "uptime_seconds": 0}
                data_feed_connected = False
            
            # Get session data
            sessions = session_manager.get_all_sessions()
            active_sessions_list = [s for s in sessions if s.status == "running"] # Assuming 'running' status exists
            
            # Determine system status based on data feed and sessions
            system_overall_status = "stopped"
            
            # If data feed is disconnected, system is in partial state
            if not data_feed_connected:
                system_overall_status = "partial"
                logger.warning("System status set to partial: Data feed disconnected")
            # If all connections are good and sessions are running, system is running
            elif any(s.status == "running" for s in sessions):
                system_overall_status = "running"
            # If sessions are starting, system is starting
            elif any(s.status == "starting" for s in sessions):
                system_overall_status = "starting"

            # Calculate uptime (this is a simplified example, real uptime might be tracked differently)
            calculated_uptime = 0
            sys_start_time = None
            if sessions:
                valid_start_times = [datetime.fromisoformat(s.start_time) for s in sessions if s.start_time and s.status == "running"]
                if valid_start_times:
                    sys_start_time = min(valid_start_times).isoformat()
                    # This uptime is a sum, might not be what's desired.
                    # A more accurate system uptime would be from the earliest running session's start_time.
                    # For now, let's use a placeholder or sum of individual session uptimes if available.
                    # uptime_seconds = sum([s.uptime_seconds for s in active_sessions_list if hasattr(s, 'uptime_seconds')])

            # Create health metrics that include data feed status
            health_metrics = {
                "data_feed_connected": data_feed_connected,
                "data_feed_uptime": data_feed_status.get("uptime_seconds", 0),
                "data_feed_status": data_feed_status.get("status", "unknown"),
                "cpu_usage": 45.2,  # Example values
                "memory_usage": 32.7,
                "disk_usage": 68.3
            }
            
            return SystemStatus(
                status=system_overall_status,
                active_agents=len(active_sessions_list), # Assuming 1 agent per active session
                total_agents=len(sessions),             # Assuming 1 agent per session
                active_sessions=len(active_sessions_list),
                total_sessions=len(sessions),
                uptime_seconds=data_feed_status.get("uptime_seconds", 0),
                start_time=sys_start_time,
                last_update=datetime.now().isoformat(),
                health_metrics=health_metrics
            )
        except Exception as e:
            logger.error(f"Error getting system status (real mode): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@router.get("/agents", response_model=List[AgentStatus], summary="List all agents (maps to sessions)")
async def get_agents():
    if USE_MOCK_DATA:
        agents = []
        for agent_id, agent_data in MOCK_AGENTS_DATA.items():
            agents.append(AgentStatus(agent_id=agent_id, **agent_data))
        return agents
    else:
        try:
            sessions = session_manager.get_all_sessions()
            agent_statuses = []
            for session in sessions:
                # Convert float timestamp to datetime
                last_updated_dt = datetime.fromtimestamp(session.last_updated) if hasattr(session, 'last_updated') and isinstance(session.last_updated, (int, float)) else datetime.now()
                
                # Safely access performance metrics
                performance_metrics = {}
                if hasattr(session, 'results') and isinstance(session.results, dict):
                    performance_metrics = session.results.get('performance_metrics', {})

                agent_statuses.append(
                    AgentStatus(
                        agent_id=session.session_id,
                        name=getattr(session, 'name', f"Session {session.session_id[:8]}"), # Use the default from PaperTradingSession
                        type="paper_trading_session",
                        status=session.status,
                        metrics=performance_metrics,
                        last_updated=last_updated_dt,
                        symbols=getattr(session, 'symbols', []),
                        strategy=getattr(session, 'strategy_name', None),
                        agent_role=getattr(session, 'agent_role', None),
                        outputs_to=getattr(session, 'outputs_to', [])
                    )
                )
            return agent_statuses
        except Exception as e:
            logger.error(f"Error listing agents (real mode): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@router.get("/agents/{agent_id}", response_model=AgentStatus, summary="Get details for a specific agent (session)")
async def get_agent(agent_id: str):
    if USE_MOCK_DATA:
        agent_data = MOCK_AGENTS_DATA.get(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail=f"Mock agent {agent_id} not found")
        return AgentStatus(agent_id=agent_id, **agent_data)
    else:
        try:
            session = session_manager.get_session(agent_id)
            if not session:
                raise HTTPException(status_code=404, detail=f"Agent/Session {agent_id} not found")

            last_updated_dt = datetime.fromtimestamp(session.last_updated) if hasattr(session, 'last_updated') and isinstance(session.last_updated, (int, float)) else datetime.now()
            performance_metrics = {}
            if hasattr(session, 'results') and isinstance(session.results, dict):
                performance_metrics = session.results.get('performance_metrics', {})

            return AgentStatus(
                agent_id=session.session_id,
                name=getattr(session, 'name', f"Session {session.session_id[:8]}"),
                type="paper_trading_session",
                status=session.status,
                metrics=performance_metrics,
                last_updated=last_updated_dt,
                symbols=getattr(session, 'symbols', []),
                strategy=getattr(session, 'strategy_name', None),
                agent_role=getattr(session, 'agent_role', None),
                outputs_to=getattr(session, 'outputs_to', [])
            )
        except Exception as e:
            logger.error(f"Error getting agent {agent_id} (real mode): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error getting agent: {str(e)}")

@router.post("/agents/{agent_id}/start", summary="Start/Resume a specific agent (session)")
async def start_agent(agent_id: str):
    logger.info(f"Request to start/resume agent/session {agent_id}.")
    if USE_MOCK_DATA:
        if agent_id not in MOCK_AGENTS_DATA:
            raise HTTPException(status_code=404, detail=f"Mock agent {agent_id} not found")
        if MOCK_AGENTS_DATA[agent_id]["status"] == "running":
            return {"status": "success", "message": f"Mock agent {agent_id} is already running"}
        MOCK_AGENTS_DATA[agent_id]["status"] = "running"
        MOCK_AGENTS_DATA[agent_id]["last_updated"] = datetime.now()
        logger.info(f"Mock agent {agent_id} started.")
        return {"status": "success", "message": f"Mock agent {agent_id} started successfully"}
    else:
        try:
            logger.info(f"Attempting to resume session {agent_id} (real mode)...")
            # Assuming resume_session returns True on success, False or raises on failure/not found
            success = await session_manager.resume_session(agent_id)
            if not success: # Need to check how session_manager indicates "not found" vs "cannot start"
                # This part might need refinement based on session_manager's actual behavior
                session = session_manager.get_session(agent_id)
                if not session:
                     raise HTTPException(status_code=404, detail=f"Agent/Session {agent_id} not found.")
                # If session exists but couldn't be resumed (e.g. already running, or in a non-resumable state)
                raise HTTPException(status_code=400, detail=f"Agent/Session {agent_id} cannot be started/resumed (current status: {session.status}).")
            logger.info(f"Agent/Session {agent_id} started/resumed successfully.")
            return {"status": "success", "message": f"Agent/Session {agent_id} started/resumed successfully"}
        except HTTPException: # Re-raise known HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error starting agent {agent_id} (real mode): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error starting agent: {str(e)}")

@router.post("/agents/{agent_id}/stop", summary="Stop a specific agent (session)")
async def stop_agent(agent_id: str):
    logger.info(f"Request to stop agent/session {agent_id}.")
    if USE_MOCK_DATA:
        if agent_id not in MOCK_AGENTS_DATA:
            raise HTTPException(status_code=404, detail=f"Mock agent {agent_id} not found")
        if MOCK_AGENTS_DATA[agent_id]["status"] == "stopped":
            return {"status": "success", "message": f"Mock agent {agent_id} is already stopped"}
        MOCK_AGENTS_DATA[agent_id]["status"] = "stopped"
        MOCK_AGENTS_DATA[agent_id]["last_updated"] = datetime.now()
        logger.info(f"Mock agent {agent_id} stopped.")
        return {"status": "success", "message": f"Mock agent {agent_id} stopped successfully"}
    else:
        try:
            logger.info(f"Attempting to stop session {agent_id} (real mode)...")
            success = await session_manager.stop_session(agent_id)
            if not success: # Similar to start, refine based on session_manager behavior
                session = session_manager.get_session(agent_id)
                if not session:
                    raise HTTPException(status_code=404, detail=f"Agent/Session {agent_id} not found.")
                raise HTTPException(status_code=400, detail=f"Agent/Session {agent_id} cannot be stopped (current status: {session.status}).")
            logger.info(f"Agent/Session {agent_id} stopped successfully.")
            return {"status": "success", "message": f"Agent/Session {agent_id} stopped successfully"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error stopping agent {agent_id} (real mode): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error stopping agent: {str(e)}")

@router.get("/agents/{agent_id}/logs", summary="Get logs for a specific agent (session)")
async def get_agent_logs(agent_id: str, lines: int = Query(100, ge=1, le=1000)):
    logger.info(f"Request for logs for agent/session {agent_id}, last {lines} lines.")
    # Validate agent_id: only allow alphanumeric, dash, and underscore
    if not re.fullmatch(r"[\w\-.]+", agent_id): # Added dot for potential session_id formats
        logger.warning(f"Invalid agent_id format received: {agent_id}")
        raise HTTPException(status_code=400, detail="Invalid agent_id format")

    # Construct path more safely, assuming logs_dir is correctly determined
    # In a real app, logs_dir might come from config or be a fixed relative path
    try:
        # Assuming this script is in ai_trading_agent/api/
        # So logs dir is ai_trading_agent/logs/
        base_dir = Path(__file__).resolve().parent.parent 
        logs_dir = base_dir / "logs"
        if not logs_dir.exists():
            logs_dir.mkdir(parents=True, exist_ok=True) # Create if not exists
            logger.info(f"Created logs directory: {logs_dir}")

        log_file = logs_dir / f"{agent_id}.log" # Ensure agent_id is sanitized
        logger.debug(f"Attempting to read log file: {log_file}")

        if not log_file.exists() or not log_file.is_file():
            logger.warning(f"Log file not found for agent {agent_id} at {log_file}")
            raise HTTPException(status_code=404, detail=f"Log file for agent {agent_id} not found")

        log_lines = []
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        log_lines = [line.strip() for line in all_lines[-lines:]] # Strip newlines for cleaner JSON
        
        logger.info(f"Successfully retrieved {len(log_lines)} log lines for agent {agent_id}.")
        return {"agent_id": agent_id, "log_lines": log_lines, "line_count": len(log_lines)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading log file for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")


# This endpoint seems redundant if /system/agents provides session info mapped to agents.
# If it's meant to be distinct, its purpose needs clarification.
# For now, I'll keep its structure similar to the one found in the original file.
@router.get("/paper-trading/sessions", summary="List all paper trading sessions (distinct from agent view)")
async def get_paper_trading_sessions_distinct():
    if USE_MOCK_DATA:
        logger.info("Serving MOCK paper trading sessions data from system_control.py")
        return {"sessions": MOCK_SESSIONS_DATA}
    else:
        try:
            sessions = session_manager.get_all_sessions()
            # Assuming session objects have a to_dict() method or can be easily serialized
            return {"sessions": [s.to_dict() if hasattr(s, 'to_dict') else vars(s) for s in sessions]}
        except Exception as e:
            logger.error(f"Error getting paper trading sessions (real mode): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error getting paper trading sessions: {str(e)}")


# Export the router under the correct name for main.py or other FastAPI app assembly
system_control_router = router
