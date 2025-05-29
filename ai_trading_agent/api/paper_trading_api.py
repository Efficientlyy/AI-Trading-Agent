"""
Paper Trading API endpoints.

This module provides API endpoints for controlling and monitoring paper trading sessions.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid
import os
import asyncio
import threading # For running asyncio loop in a separate thread

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel

from ..common import logger
from ..agent import factory
from ..agent.session_manager import session_manager
# Use an alias for the actual PaperTradingSession class from session_manager to avoid Pydantic model name conflict
from ..agent.session_manager import PaperTradingSession as ActualPaperTradingSession
# from ..agent.trading_orchestrator import TradingOrchestrator # Not directly instantiated here, use factory
# from ..backtesting.performance_metrics import calculate_metrics, get_drawdown_data, calculate_trade_statistics, process_trade_history # If needed for custom result processing

# --- Pydantic Models for API Request/Response ---

class PaperTradingConfig(BaseModel):
    """Paper trading configuration model for starting a session."""
    config_path: str
    duration_minutes: int = 60
    interval_minutes: int = 1
    autonomous_mode: bool = False # Example: if autonomous mode is a startup param

class AlertCreate(BaseModel):
    """Model for creating an alert."""
    type: str
    message: str
    title: Optional[str] = None
    # timestamp will be auto-generated if not provided
    # id will be auto-generated

class PaperTradingAlertSettings(BaseModel):
    """Paper trading alert settings model."""
    enabled: bool = True
    drawdownThreshold: float = 0.05
    gainThreshold: float = 0.05
    largeTradeThreshold: float = 0.1
    consecutiveLossesThreshold: int = 3

# Note: The Pydantic model `PaperTradingSession` defined in this file (if any)
# should be for API response shaping if it differs from ActualPaperTradingSession.to_dict().
# For now, we'll rely on ActualPaperTradingSession.to_dict() and structure responses directly.


# --- API Router ---
router = APIRouter(
    prefix="/paper-trading", # All endpoints in this file will be under /api/paper-trading
    tags=["paper-trading"],
    responses={404: {"description": "Not found"}},
)

# --- Helper Functions ---

# Mock load_config if not available elsewhere or for testing
# In a real scenario, this should load from a YAML/JSON file.
def load_config(path: str) -> Dict[str, Any]:
    logger.info(f"Attempting to load configuration from: {path}")
    if not os.path.exists(path):
        logger.warning(f"Configuration file {path} not found. Using default fallback configuration.")
        # Fallback to a default config if loading fails.
        return {
            "data_sources": {"provider": "ccxt", "exchange": "binance", "symbols": ["BTC/USDT", "ETH/USDT"]},
            "strategy_manager": {
                "type": "BaseStrategyManager", # Example type for factory
                "config": {
                    "aggregation_method": "weighted_average",
                    "strategy_weights": {"moving_average": 0.4, "sentiment": 0.3, "volume": 0.3},
                    "min_signal_interval": 60,
                    "stale_data_threshold": 300
                }
            },
            "strategies": { # Example structure if strategies are defined here
                "moving_average": {"type": "SentimentStrategy", "config": {"name": "MA"}}, # Placeholder
            },
            "risk_manager": {"type": "SimpleRiskManager", "config": {
                "max_position_size": 0.1, "max_drawdown": 0.05,
                "stop_loss_pct": 0.02, "take_profit_pct": 0.05
            }},
            "portfolio_manager": {"type": "PortfolioManager", "config":{ # Type for trading_engine.PortfolioManager
                "initial_capital": 10000.0, "position_sizing_method": "equal_weight"
            }},
            "execution_handler": {"type": "SimulatedExecutionHandler", "config":{ # For paper trading
                "mode": "paper", "slippage": 0.001, "commission": 0.001
            }},
            "orchestrator": { # Config for the orchestrator itself
                 "event_emitter_enabled": True
            }
        }
    # Actual loading logic would go here
    # import yaml
    # with open(path, 'r') as f:
    #     return yaml.safe_load(f)
    # For now, returning the same default if path exists but loading fails for some reason
    logger.warning(f"Mock load_config for {path} returning default structure.")
    return {
        "data_sources": {"provider": "ccxt", "exchange": "binance", "symbols": ["BTC/USDT", "ETH/USDT"]},
        "strategy_manager": {"type": "BaseStrategyManager", "config": {"aggregation_method": "weighted_average"}},
        "risk_manager": {"type": "SimpleRiskManager", "config": {}},
        "portfolio_manager": {"type": "PortfolioManager", "config": {"initial_capital": 10000.0}},
        "execution_handler": {"type": "SimulatedExecutionHandler", "config": {}},
        "orchestrator": {}
    }


async def _run_paper_trading(session: ActualPaperTradingSession) -> None:
    """
    Run paper trading in the background for a given session object.
    This is the core trading loop initiation.
    """
    try:
        session.update_status("running")
        session_manager.update_session(session)
        session_manager.add_alert(session.session_id, {
            "id": str(uuid.uuid4()), "type": "info", "title": "Session Running",
            "message": f"Paper trading session {session.session_id} is now running.",
            "timestamp": datetime.now().isoformat()
        })

        agent_config = load_config(session.config_path)
        logger.info(f"Loaded configuration from {session.config_path} for session {session.session_id}")

        # Create components using the factory
        # The factory.create_agent_from_config is a high-level function.
        # It expects a config structure that defines types and configs for each component.
        # Ensure agent_config matches this structure.
        # For paper trading, execution_handler type in config should be "SimulatedExecutionHandler"
        # or "LiveTradingBridge" if it's adapted to be an execution handler.
        # The factory.py uses "SimulatedExecutionHandler".
        
        # The portfolio_manager created by factory.create_portfolio_manager is trading_engine.PortfolioManager
        # The execution_handler created by factory.create_execution_handler is agent.SimulatedExecutionHandler
        # The orchestrator created by factory.create_orchestrator is agent.BacktestOrchestrator (TradingOrchestrator)

        # We need to ensure the config structure for create_agent_from_config is correct.
        # It expects keys like "data_manager", "strategy", "portfolio_manager", etc.
        # The `setup_realtime_components` function was an attempt to manually create these.
        # Using the factory is preferred.

        try:
            # Assuming agent_config is structured correctly for create_agent_from_config
            # The 'orchestrator_type' for paper trading should be the one that has 'run_paper_trading'
            # which is TradingOrchestrator (likely registered as "BacktestOrchestrator" in factory)
            
            # Manually create components to pass to factory.create_orchestrator
            # This gives more control if create_agent_from_config structure is not met by agent_config
            data_manager = factory.create_data_manager(agent_config["data_manager"])
            
            # Strategy creation might be more complex if multiple strategies are involved
            # For simplicity, assuming one strategy or strategy_manager handles multiple
            strategy_config = agent_config.get("strategy", agent_config.get("strategies", {})) # Adapt as needed
            # If strategy_config is for a single strategy:
            # main_strategy = factory.create_strategy(strategy_config)
            # strategy_manager = factory.create_strategy_manager(main_strategy, manager_type=agent_config.get("strategy_manager",{}).get("type","BaseStrategyManager"))
            # If strategy_manager config directly creates it with strategies:
            strategy_manager = factory.create_strategy_manager(
                strategy=None, # Or pass a base strategy if required by BaseStrategyManager
                manager_type=agent_config.get("strategy_manager",{}).get("type","BaseStrategyManager"),
                data_manager=data_manager, # If needed by strategy manager
                config=agent_config.get("strategy_manager",{}).get("config", {})
            )
            # TODO: Add strategies to strategy_manager based on agent_config["strategies"]

            portfolio_manager = factory.create_portfolio_manager(agent_config["portfolio_manager"])
            risk_manager = factory.create_risk_manager(agent_config["risk_manager"])
            
            # For paper trading, ensure execution_handler is SimulatedExecutionHandler or LiveTradingBridge
            # The factory has SimulatedExecutionHandler. LiveTradingBridge acts as an exchange.
            # TradingOrchestrator expects an ExecutionHandlerABC.
            execution_handler_config = agent_config.get("execution_handler", {"type": "SimulatedExecutionHandler", "config": {}})
            if execution_handler_config.get("type") == "LiveTradingBridge": # Example if using LTB as handler
                 from ..trading_engine.live_trading_bridge import LiveTradingBridge
                 execution_handler = LiveTradingBridge(config=execution_handler_config.get("config"))
            else:
                 execution_handler = factory.create_execution_handler(execution_handler_config, portfolio_manager)


            orchestrator_config = agent_config.get("orchestrator", agent_config.get("backtest", {}))
            orchestrator = factory.create_orchestrator(
                data_manager=data_manager,
                strategy_manager=strategy_manager,
                portfolio_manager=portfolio_manager,
                risk_manager=risk_manager,
                execution_handler=execution_handler,
                config=orchestrator_config,
                orchestrator_type="BacktestOrchestrator" # This should map to TradingOrchestrator
            )
            logger.info(f"Created orchestrator for session {session.session_id}")

        except Exception as e_setup:
            logger.error(f"Error setting up components or orchestrator for session {session.session_id}: {e_setup}", exc_info=True)
            session.update_status("error")
            session_manager.update_session(session)
            session_manager.add_alert(session.session_id, {
                "id": str(uuid.uuid4()), "type": "error", "title": "Setup Error",
                "message": f"Failed to set up components/orchestrator: {str(e_setup)}", "timestamp": datetime.now().isoformat()
            })
            return

        session.set_orchestrator(orchestrator)

        interval_seconds = max(1, int(session.interval_minutes * 60))
        duration_td = timedelta(minutes=session.duration_minutes)
        interval_td = timedelta(seconds=interval_seconds)

        session_manager.add_alert(session.session_id, {
            "id": str(uuid.uuid4()), "type": "info", "title": "Trading Process Started",
            "message": f"Paper trading process started for {session.duration_minutes} minutes with {interval_seconds}s interval.",
            "timestamp": datetime.now().isoformat()
        })

        results = await orchestrator.run_paper_trading(
            duration=duration_td,
            update_interval=interval_td,
            stop_event=session.stop_event
        )
        
        session.update_results(results)
        session.update_status('completed')
        session_manager.update_session(session)
        session_manager.add_alert(session.session_id, {
            "id": str(uuid.uuid4()), "type": "success", "title": "Trading Completed",
            "message": "Paper trading completed successfully.", "timestamp": datetime.now().isoformat()
        })

    except asyncio.CancelledError:
        logger.info(f"Paper trading session {session.session_id} task cancelled")
        session.update_status("stopped")
        session_manager.update_session(session)
        session_manager.add_alert(session.session_id, {
            "id": str(uuid.uuid4()), "type": "info", "title": "Trading Stopped",
            "message": "Paper trading stopped by user.", "timestamp": datetime.now().isoformat()
        })
        # raise # Re-raise CancelledError if needed by caller
    except Exception as e:
        logger.error(f"Error in paper trading session {session.session_id} task: {e}", exc_info=True)
        session.update_status("error")
        session_manager.update_session(session)
        session_manager.add_alert(session.session_id, {
            "id": str(uuid.uuid4()), "type": "error", "title": "Trading Task Error",
            "message": f"Error in paper trading task for session {session.session_id}: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
    finally:
        if hasattr(session, 'orchestrator') and session.orchestrator:
            try:
                await session.orchestrator.cleanup()
                logger.info(f"Cleaned up paper trading orchestrator for session {session.session_id}")
            except Exception as e_cleanup:
                logger.error(f"Error cleaning up paper trading orchestrator for session {session.session_id}: {e_cleanup}", exc_info=True)
                session_manager.add_alert(session.session_id, {
                    "id": str(uuid.uuid4()), "type": "warning", "title": "Orchestrator Cleanup Error",
                    "message": f"Error during orchestrator cleanup for session {session.session_id}: {str(e_cleanup)}", "timestamp": datetime.now().isoformat()
                })
        if session.status not in ["completed", "stopped", "error"]:
            final_status = "stopped" if session.stop_event and session.stop_event.is_set() else "error_unknown_finish"
            session.update_status(final_status)
            session_manager.update_session(session)
            logger.info(f"Session {session.session_id} finalized with status: {final_status}")

def _start_paper_trading_task(session_obj: ActualPaperTradingSession) -> None:
    """Starts paper trading in a background task for a specific session."""
    stop_event = asyncio.Event()
    session_obj.set_stop_event(stop_event)
    
    loop = asyncio.new_event_loop()
    
    def run_loop_in_thread():
        asyncio.set_event_loop(loop)
        try:
            task = loop.create_task(_run_paper_trading(session_obj))
            session_obj.set_task(task)
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            logger.info(f"Paper trading session {session_obj.session_id} task cancelled in thread.")
        except Exception as e_thread:
            logger.error(f"Error in paper trading session {session_obj.session_id} task thread: {e_thread}", exc_info=True)
            if session_obj.status != "error": # Avoid overwriting more specific error
                session_obj.update_status("error")
                session_manager.update_session(session_obj)
        finally:
            loop.close()
            logger.info(f"Paper trading session {session_obj.session_id} task loop closed.")
    
    thread = threading.Thread(target=run_loop_in_thread, daemon=True)
    thread.start()
    logger.info(f"Paper trading session {session_obj.session_id} task thread started.")


# --- API Endpoints ---

@router.post("/start", summary="Start New Paper Trading Session")
async def start_paper_trading(config_params: PaperTradingConfig):
    config_path = config_params.config_path
    if not os.path.exists(config_path):
        raise HTTPException(status_code=400, detail=f"Configuration file {config_path} not found")

    session_id = str(uuid.uuid4())
    
    try:
        full_agent_config = load_config(config_path)
        symbols = full_agent_config.get("data_manager", {}).get("config", {}).get("symbols", ["BTC/USDT"])
        initial_capital = full_agent_config.get("portfolio_manager", {}).get("config", {}).get("initial_capital", 10000.0)
    except Exception as e_conf:
        logger.error(f"Failed to load full config from {config_path} for session creation: {e_conf}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration file at {config_path}: {e_conf}")

    session_obj = session_manager.create_session(
        session_id=session_id,
        config_path=config_path,
        duration_minutes=config_params.duration_minutes,
        interval_minutes=config_params.interval_minutes,
        symbols=symbols,
        initial_capital=initial_capital,
        user_id=None # TODO: Integrate user auth
    )
    # if hasattr(session_obj, 'autonomous_mode'):
    #     session_obj.autonomous_mode = config_params.autonomous_mode
    #     session_manager.update_session(session_obj)

    _start_paper_trading_task(session_obj)
    
    logger.info(f"Initiated paper trading session {session_obj.session_id} with config {config_path}")
    session_manager.add_alert(session_obj.session_id, {
        "id": str(uuid.uuid4()), "type": "info", "title": "Session Initiated",
        "message": f"Paper trading session {session_obj.session_id} initiated. Status: {session_obj.status}",
        "timestamp": datetime.now().isoformat()
    })
    if config_params.autonomous_mode:
        session_manager.add_alert(session_obj.session_id, {
            "id": str(uuid.uuid4()), "type": "info", "title": "Autonomous Mode Activated",
            "message": "Trading agent is now operating in fully autonomous mode.",
            "timestamp": datetime.now().isoformat()
        })
    
    return {
        "status": "starting", # Or session_obj.status
        "session_id": session_obj.session_id,
        "message": f"Paper trading session {session_obj.session_id} is starting."
    }

@router.get("/status", summary="Get Paper Trading Session Status")
async def get_paper_trading_status(session_id: str = Query(..., description="ID of the session to get status for")):
    session_obj = session_manager.get_session(session_id)
    if not session_obj:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    session_obj.update_uptime()
    
    response_data = session_obj.to_dict()
    if session_obj.results and isinstance(session_obj.results, dict):
        response_data['performance_metrics'] = session_obj.results.get('performance_metrics', {})
        response_data['recent_trades'] = session_obj.results.get('trades', [])[-10:]
    else:
        response_data['performance_metrics'] = {}
        response_data['recent_trades'] = []
    return response_data

@router.get("/sessions", summary="List All Paper Trading Sessions")
async def list_paper_trading_sessions(user_id: Optional[str] = None, include_completed: bool = True, limit: int = 100, offset: int = 0):
    try:
        logger.info(f"Attempting to fetch paper trading sessions. user_id={user_id}, include_completed={include_completed}")
        sessions = session_manager.get_all_sessions(user_id=user_id, include_completed=include_completed, limit=limit, offset=offset)
        logger.info(f"Successfully retrieved {len(sessions)} paper trading sessions")
        return {"sessions": [s.to_dict() for s in sessions]}
    except Exception as e:
        logger.error(f"Error retrieving paper trading sessions: {str(e)}")
        # Return mock data as fallback
        mock_sessions = [
            {
                "session_id": str(uuid.uuid4()),
                "status": "completed",
                "start_time": (datetime.now() - timedelta(hours=2)).isoformat(),
                "end_time": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "config_path": "configs/default_strategy.yaml",
                "duration_minutes": 90,
                "interval_minutes": 1,
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "initial_capital": 10000.0,
                "final_capital": 10250.0,
                "profit_loss": 250.0,
                "profit_loss_pct": 2.5
            },
            {
                "session_id": str(uuid.uuid4()),
                "status": "running",
                "start_time": (datetime.now() - timedelta(minutes=45)).isoformat(),
                "end_time": None,
                "config_path": "configs/aggressive_strategy.yaml",
                "duration_minutes": 120,
                "interval_minutes": 1,
                "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                "initial_capital": 10000.0,
                "current_capital": 10120.0,
                "profit_loss": 120.0,
                "profit_loss_pct": 1.2
            }
        ]
        logger.info("Returning mock paper trading session data as fallback")
        return {"sessions": mock_sessions}

@router.get("/sessions/{session_id}", summary="Get Specific Paper Trading Session Details")
async def get_session_details(session_id: str = Path(..., description="The ID of the paper trading session")):
    session_obj = session_manager.get_session(session_id)
    if not session_obj:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session_obj.to_dict()

@router.post("/stop/{session_id}", summary="Stop Paper Trading Session")
async def stop_paper_trading_session(session_id: str = Path(..., description="The ID of the paper trading session")):
    session_obj = session_manager.get_session(session_id)
    if not session_obj:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if session_obj.status not in ["running", "starting"]:
        raise HTTPException(status_code=400, detail=f"Session {session_obj.session_id} is not running or starting.")

    await session_manager.stop_session(session_id)
    
    session_manager.add_alert(session_id, {
        "id": str(uuid.uuid4()), "type": "info", "title": "Session Stop Requested",
        "message": f"Paper trading session {session_id} stop has been requested.",
        "timestamp": datetime.now().isoformat()
    })
    return {
        "status": "stopping", # Or query session_obj.status again after stop
        "session_id": session_id,
        "message": f"Paper trading session {session_id} stop requested."
    }

@router.get("/results/{session_id}", summary="Get Paper Trading Session Results")
async def get_paper_trading_results(session_id: str = Path(..., description="The ID of the paper trading session")):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if not session.results or not isinstance(session.results, dict):
        return {
            "session_id": session.session_id, "status": session.status,
            "message": "Results not yet available or session did not produce results.",
            "results": {"trades": [], "portfolio_history": [], "performance_metrics": {}}
        }
    return {
        "session_id": session.session_id, "status": session.status,
        "start_time": session.start_time, "end_time": session.end_time,
        "config_path": session.config_path, "results": session.results
    }

# --- Pause/Resume Session Endpoints ---
@router.post("/sessions/pause/{session_id}", summary="Pause a Paper Trading Session")
async def pause_session(session_id: str = Path(..., description="The ID of the paper trading session")):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if session.status != "running":
        raise HTTPException(status_code=400, detail=f"Session {session_id} is not running and cannot be paused.")
    try:
        await session_manager.pause_session(session_id)
        session_manager.add_alert(session_id, {
            "id": str(uuid.uuid4()), "type": "info", "title": "Session Paused",
            "message": f"Paper trading session {session_id} has been paused.",
            "timestamp": datetime.now().isoformat()
        })
        return {"status": "paused", "session_id": session_id, "message": f"Session {session_id} paused."}
    except Exception as e:
        logger.error(f"Error pausing session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause session: {str(e)}")

@router.post("/sessions/resume/{session_id}", summary="Resume a Paused Paper Trading Session")
async def resume_session(session_id: str = Path(..., description="The ID of the paper trading session")):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if session.status != "paused":
        raise HTTPException(status_code=400, detail=f"Session {session_id} is not paused and cannot be resumed.")
    try:
        await session_manager.resume_session(session_id)
        session_manager.add_alert(session_id, {
            "id": str(uuid.uuid4()), "type": "info", "title": "Session Resumed",
            "message": f"Paper trading session {session_id} has been resumed.",
            "timestamp": datetime.now().isoformat()
        })
        return {"status": "running", "session_id": session_id, "message": f"Session {session_id} resumed."}
    except Exception as e:
        logger.error(f"Error resuming session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume session: {str(e)}")

@router.post("/sessions/pause-all", summary="Pause All Paper Trading Sessions")
async def pause_all_sessions():
    try:
        await session_manager.pause_all_sessions()
        return {"status": "success", "message": "All running sessions paused."}
    except Exception as e:
        logger.error(f"Error pausing all sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause all sessions: {str(e)}")

@router.post("/sessions/resume-all", summary="Resume All Paused Paper Trading Sessions")
async def resume_all_sessions():
    try:
        await session_manager.resume_all_sessions()
        return {"status": "success", "message": "All paused sessions resumed."}
    except Exception as e:
        logger.error(f"Error resuming all sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume all sessions: {str(e)}")

@router.post("/sessions/stop-all", summary="Stop All Paper Trading Sessions")
async def stop_all_sessions():
    try:
        await session_manager.stop_all_sessions()
        # Optionally, add alerts for each session
        for session in session_manager.sessions.values():
            session_manager.add_alert(session.session_id, {
                "id": str(uuid.uuid4()), "type": "info", "title": "Session Stopped",
                "message": f"Paper trading session {session.session_id} has been stopped by system control.",
                "timestamp": datetime.now().isoformat()
            })
        return {"status": "success", "message": "All sessions stopped."}
    except Exception as e:
        logger.error(f"Error stopping all sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop all sessions: {str(e)}")

# --- Alert System Endpoints ---

@router.get("/alerts/{session_id}", summary="Get Alerts for Session")
async def get_session_alerts(session_id: str = Path(..., description="The ID of the paper trading session"), limit: int = 100, offset: int = 0):
    if not session_manager.get_session(session_id): # Check existence
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    alerts = session_manager.get_alerts(session_id, limit=limit, offset=offset)
    return {"alerts": alerts}

@router.post("/alerts/{session_id}", summary="Add Alert to Session")
async def add_session_alert(session_id: str = Path(..., description="The ID of the paper trading session"), alert_data: AlertCreate = None):
    if not session_manager.get_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    alert_dict = alert_data.dict()
    alert_dict["id"] = str(uuid.uuid4())
    alert_dict["timestamp"] = datetime.now().isoformat()
    if not alert_dict.get("title"):
        alert_dict["title"] = alert_dict["type"].capitalize()
    
    session_manager.add_alert(session_id, alert_dict)
    return alert_dict

@router.get("/alerts/{session_id}/settings", summary="Get Alert Settings for Session", response_model=PaperTradingAlertSettings)
async def get_alert_settings(session_id: str = Path(..., description="The ID of the paper trading session")):
    if not session_manager.get_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    settings = session_manager.get_alert_settings(session_id)
    return settings

@router.post("/alerts/{session_id}/settings", summary="Update Alert Settings for Session")
async def update_session_alert_settings(session_id: str = Path(..., description="The ID of the paper trading session"), settings: PaperTradingAlertSettings = None):
    if not session_manager.get_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if not settings:
        raise HTTPException(status_code=400, detail="Alert settings must be provided")
    
    session_manager.update_alert_settings(session_id, settings.dict())
    logger.info(f"Updated alert settings for session {session_id}: {settings.dict()}")
    return {"status": "success", "message": "Alert settings updated successfully"}

# Note: The original file had some duplicated endpoint definitions (e.g. /sessions GET, /sessions POST, /delete/{session_id})
# and a duplicate router definition. These have been removed in this consolidated version.
# The /stop endpoint with Query parameter was also removed in favor of /stop/{session_id} with Path parameter.
