"""
Paper Trading API endpoints.

This module provides API endpoints for controlling and monitoring paper trading sessions.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..common import logger

# Create API router with prefix
router = APIRouter(
    prefix="/paper-trading",
    tags=["paper-trading"],
    responses={404: {"description": "Not found"}},
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
from ..data_acquisition.data_service import DataService
from ..agent.simple_strategy_manager import SimpleStrategyManager
from ..strategies.realtime_strategies import (
    RealtimeMACrossoverStrategy,
    RealtimeSentimentStrategy,
    RealtimeVolumeStrategy
)
from ..execution.paper_trading_handler import PaperTradingExecutionHandler
from ..portfolio.simple_portfolio_manager import SimplePortfolioManager

async def setup_realtime_components(config, mode='paper'):
    """
    Set up components optimized for real-time paper trading.
    
    Args:
        config: Trading configuration
        mode: Trading mode ('paper' or 'live')
        
    Returns:
        Tuple of (data_manager, strategy_manager, risk_manager, portfolio_manager, execution_handler)
    """
    # Initialize components
    try:
        data_sources_config = config.get('data_sources', {})
        if not data_sources_config:
            # Use default configuration if none provided
            data_sources_config = {
                "provider": "ccxt",
                "exchange": "binance",
                "symbols": ["BTC/USDT", "ETH/USDT"]
            }
        
        data_manager = DataService(data_sources_config)
        logger.info(f"Initialized DataService with config: {data_sources_config}")
    except Exception as e:
        logger.error(f"Error initializing DataService: {e}", exc_info=True)
        # Create a minimal data manager
        data_manager = DataService({"provider": "ccxt", "exchange": "binance"})
        logger.info("Using minimal DataService configuration")
    
    # Create strategy manager
    try:
        strategy_manager_config = config.get('strategy_manager', {})
        strategy_manager = SimpleStrategyManager(
            name="RealtimeStrategyManager",
            aggregation_method=strategy_manager_config.get('aggregation_method', 'weighted_average'),
            strategy_weights=strategy_manager_config.get('strategy_weights', {}),
            min_signal_interval=strategy_manager_config.get('min_signal_interval', 60),
            stale_data_threshold=strategy_manager_config.get('stale_data_threshold', 300)
        )
        logger.info(f"Initialized StrategyManager with config: {strategy_manager_config}")
    except Exception as e:
        logger.error(f"Error initializing StrategyManager: {e}", exc_info=True)
        # Create a minimal strategy manager
        strategy_manager = SimpleStrategyManager(
            name="RealtimeStrategyManager",
            aggregation_method='weighted_average',
            strategy_weights={'moving_average': 1.0}
        )
        logger.info("Using minimal StrategyManager configuration")
    
    # Add real-time optimized strategies
    strategies_config = config.get('strategies', {})
    
    # Add default strategies if none provided
    if not strategies_config:
        strategies_config = {
            'moving_average': {
                'enabled': True,
                'weight': 0.4,
                'params': {
                    'short_window': 10,
                    'long_window': 50
                }
            },
            'sentiment': {
                'enabled': True,
                'weight': 0.3,
                'params': {
                    'threshold': 0.2
                }
            },
            'volume': {
                'enabled': True,
                'weight': 0.3,
                'params': {
                    'threshold': 2.0
                }
            }
        }
    
    # Add strategies to manager
    try:
        # Moving average strategy
        if strategies_config.get('moving_average', {}).get('enabled', True):
            ma_params = strategies_config.get('moving_average', {}).get('params', {})
            ma_strategy = RealtimeMACrossoverStrategy(
                short_window=ma_params.get('short_window', 10),
                long_window=ma_params.get('long_window', 50),
                weight=strategies_config.get('moving_average', {}).get('weight', 0.4)
            )
            strategy_manager.add_strategy(ma_strategy)
            logger.info(f"Added MA Crossover strategy with params: {ma_params}")
        
        # Sentiment strategy
        if strategies_config.get('sentiment', {}).get('enabled', True):
            sentiment_params = strategies_config.get('sentiment', {}).get('params', {})
            sentiment_strategy = RealtimeSentimentStrategy(
                threshold=sentiment_params.get('threshold', 0.2),
                weight=strategies_config.get('sentiment', {}).get('weight', 0.3)
            )
            strategy_manager.add_strategy(sentiment_strategy)
            logger.info(f"Added Sentiment strategy with params: {sentiment_params}")
        
        # Volume strategy
        if strategies_config.get('volume', {}).get('enabled', True):
            volume_params = strategies_config.get('volume', {}).get('params', {})
            volume_strategy = RealtimeVolumeStrategy(
                threshold=volume_params.get('threshold', 2.0),
                weight=strategies_config.get('volume', {}).get('weight', 0.3)
            )
            strategy_manager.add_strategy(volume_strategy)
            logger.info(f"Added Volume strategy with params: {volume_params}")
    except Exception as e:
        logger.error(f"Error adding strategies: {e}", exc_info=True)
        # Add a default strategy if none were added
        if len(strategy_manager.strategies) == 0:
            ma_strategy = RealtimeMACrossoverStrategy(
                short_window=10,
                long_window=50,
                weight=1.0
            )
            strategy_manager.add_strategy(ma_strategy)
            logger.info("Added default MA Crossover strategy")
    
    # Create risk manager
    try:
        from ..risk.simple_risk_manager import SimpleRiskManager
        
        risk_config = config.get('risk_manager', {})
        risk_manager = SimpleRiskManager(
            max_position_size=risk_config.get('max_position_size', 0.1),
            max_drawdown=risk_config.get('max_drawdown', 0.05),
            stop_loss_pct=risk_config.get('stop_loss_pct', 0.02),
            take_profit_pct=risk_config.get('take_profit_pct', 0.05)
        )
        logger.info(f"Initialized RiskManager with config: {risk_config}")
    except Exception as e:
        logger.error(f"Error initializing RiskManager: {e}", exc_info=True)
        # Create a minimal risk manager
        from ..risk.simple_risk_manager import SimpleRiskManager
        risk_manager = SimpleRiskManager(
            max_position_size=0.1,
            max_drawdown=0.05,
            stop_loss_pct=0.02,
            take_profit_pct=0.05
        )
        logger.info("Using minimal RiskManager configuration")
    
    # Create portfolio manager
    try:
        portfolio_config = config.get('portfolio_manager', {})
        portfolio_manager = SimplePortfolioManager(
            initial_capital=portfolio_config.get('initial_capital', 10000.0),
            position_sizing_method=portfolio_config.get('position_sizing_method', 'equal_weight')
        )
        logger.info(f"Initialized PortfolioManager with config: {portfolio_config}")
    except Exception as e:
        logger.error(f"Error initializing PortfolioManager: {e}", exc_info=True)
        # Create a minimal portfolio manager
        portfolio_manager = SimplePortfolioManager(
            initial_capital=10000.0,
            position_sizing_method='equal_weight'
        )
        logger.info("Using minimal PortfolioManager configuration")
    
    # Create execution handler
    try:
        execution_config = config.get('execution_handler', {})
        execution_handler = PaperTradingExecutionHandler(
            slippage=execution_config.get('slippage', 0.001),
            commission=execution_config.get('commission', 0.001)
        )
        logger.info(f"Initialized ExecutionHandler with config: {execution_config}")
    except Exception as e:
        logger.error(f"Error initializing ExecutionHandler: {e}", exc_info=True)
        # Create a minimal execution handler
        execution_handler = PaperTradingExecutionHandler(
            slippage=0.001,
            commission=0.001
        )
        logger.info("Using minimal ExecutionHandler configuration")
    
    return data_manager, strategy_manager, risk_manager, portfolio_manager, execution_handler
    
    # Add MA Crossover strategy if enabled
    if strategies_config.get('ma_crossover', {}).get('enabled', False):
        ma_config = strategies_config.get('ma_crossover', {})
        ma_strategy = RealtimeMACrossoverStrategy(
            name="RealtimeMACrossover",
            short_window=ma_config.get('parameters', {}).get('short_window', 10),
            long_window=ma_config.get('parameters', {}).get('long_window', 50),
            min_data_points=ma_config.get('parameters', {}).get('min_data_points', 0),
            signal_smoothing=ma_config.get('parameters', {}).get('signal_smoothing', True),
            cache_calculations=ma_config.get('parameters', {}).get('cache_calculations', True)
        )
        strategy_manager.add_strategy(ma_strategy)
    
    # Add Sentiment strategy if enabled
    if strategies_config.get('sentiment', {}).get('enabled', False):
        sentiment_config = strategies_config.get('sentiment', {})
        sentiment_strategy = RealtimeSentimentStrategy(
            name="RealtimeSentiment",
            sentiment_threshold=sentiment_config.get('parameters', {}).get('sentiment_threshold', 0.2),
            confidence_threshold=sentiment_config.get('parameters', {}).get('confidence_threshold', 0.6),
            time_window=sentiment_config.get('parameters', {}).get('time_window', 5),
            trend_weight=sentiment_config.get('parameters', {}).get('trend_weight', 0.3)
        )
        strategy_manager.add_strategy(sentiment_strategy)
    
    # Add Volume strategy if enabled
    if strategies_config.get('volume', {}).get('enabled', False):
        volume_config = strategies_config.get('volume', {})
        volume_strategy = RealtimeVolumeStrategy(
            name="RealtimeVolume",
            volume_threshold=volume_config.get('parameters', {}).get('volume_threshold', 2.0),
            price_correlation_weight=volume_config.get('parameters', {}).get('price_correlation_weight', 0.5),
            lookback_period=volume_config.get('parameters', {}).get('lookback_period', 20)
        )
        strategy_manager.add_strategy(volume_strategy)
    
    # Create risk manager (using a simple implementation for now)
    class SimpleRiskManager:
        def __init__(self, config):
            self.name = "SimpleRiskManager"
            self.max_position_size = config.get('risk_management', {}).get('max_position_size', 0.2)
            
        def validate_signals(self, signals, portfolio):
            return signals
    
    risk_manager = SimpleRiskManager(config)
    
    # Create portfolio manager
    portfolio_manager = SimplePortfolioManager(config.get('portfolio', {}))
    
    # Create execution handler
    execution_handler = PaperTradingExecutionHandler(config)
    
    return data_manager, strategy_manager, risk_manager, portfolio_manager, execution_handler

# Create router
router = APIRouter(
    prefix="/api/paper-trading",
    tags=["paper-trading"],
    responses={404: {"description": "Not found"}},
)

# Import session manager and agent activity integration
from ..agent.session_manager import session_manager
from ..agent.agent_activity_integration import AgentActivityIntegration
from ..backtesting.performance_metrics import calculate_metrics, get_drawdown_data, calculate_trade_statistics, process_trade_history

# Global lock for thread safety
_paper_trading_lock = threading.Lock()


class PaperTradingConfig(BaseModel):
    """Paper trading configuration model."""
    config_path: str
    duration_minutes: int = 60
    interval_minutes: int = 1


# Note: We're no longer using the PaperTradingStatus model since we need to return
# additional fields that aren't defined in the model. Using a dict instead.


class PaperTradingSession(BaseModel):
    """Paper trading session model."""
    session_id: str
    status: str
    start_time: str
    config_path: str
    duration_minutes: int
    interval_minutes: int
    uptime_seconds: Optional[int] = None
    symbols: List[str] = []
    current_portfolio: Optional[Dict[str, Any]] = None


class PaperTradingAlert(BaseModel):
    """Paper trading alert model."""
    id: str
    session_id: str
    type: str
    message: str
    timestamp: int
    severity: str = "info"
    acknowledged: bool = False


class PaperTradingAlertSettings(BaseModel):
    """Paper trading alert settings model."""
    enabled: bool = True
    drawdownThreshold: float = 0.05
    gainThreshold: float = 0.05
    largeTradeThreshold: float = 0.1
    consecutiveLossesThreshold: int = 3


async def _run_paper_trading(config_path: str, duration_minutes: int, interval_minutes: int) -> None:
    """
    Run paper trading in the background.
    
    Args:
        config_path: Path to the configuration file
        duration_minutes: Duration to run paper trading in minutes
        interval_minutes: Update interval in minutes
    """
    global _paper_trading_status, _paper_trading_results, _paper_trading_orchestrator, _paper_trading_stop_event, _paper_trading_sessions
    
    try:
        # Set status to running
        _paper_trading_status = "running"
        
        # Update all running sessions to running
        for session_id, session in _paper_trading_sessions.items():
            if session["status"] == "starting":
                session["status"] = "running"
                # Add alert
                add_alert(session_id, "session_started", "Paper trading session is now running", "info")
        
        # Load configuration
        try:
            config = load_config(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}", exc_info=True)
            # Use default configuration
            config = {
                "data_sources": {
                    "provider": "ccxt",
                    "exchange": "binance",
                    "symbols": ["BTC/USDT", "ETH/USDT"]
                },
                "strategy_manager": {
                    "aggregation_method": "weighted_average",
                    "strategy_weights": {
                        "moving_average": 0.4,
                        "sentiment": 0.3,
                        "volume": 0.3
                    },
                    "min_signal_interval": 60,
                    "stale_data_threshold": 300
                },
                "risk_manager": {
                    "max_position_size": 0.1,
                    "max_drawdown": 0.05,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.05
                },
                "portfolio_manager": {
                    "initial_capital": 10000.0,
                    "position_sizing_method": "equal_weight"
                },
                "execution_handler": {
                    "mode": "paper",
                    "slippage": 0.001,
                    "commission": 0.001
                }
            }
            logger.info("Using default configuration")
        
        # Set up components
        try:
            data_manager, strategy_manager, risk_manager, portfolio_manager, execution_handler = await setup_realtime_components(
                config=config,
                mode='paper'
            )
            logger.info("Set up trading components")
        except Exception as e:
            logger.error(f"Error setting up components: {e}", exc_info=True)
            _paper_trading_status = "error"
            
            # Update all running sessions to error
            for session_id, session in _paper_trading_sessions.items():
                if session["status"] == "running":
                    session["status"] = "error"
                    # Add alert
                    add_alert(session_id, "session_error", f"Error setting up components: {str(e)}", "error")
            
            return
        
        # Create orchestrator
        orchestrator = create_trading_orchestrator(config)
        if not orchestrator:
            logger.error("Failed to create trading orchestrator")
            session.update_status("error")
            session_manager.update_session(session)
            
            # Add error alert
            session_manager.add_alert(session.session_id, {
                "type": "error",
                "title": "Orchestrator Error",
                "message": "Failed to create trading orchestrator",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        # Store orchestrator in session
        session.set_orchestrator(orchestrator)
        
        # Run paper trading with stop event
        try:
            # Convert minutes to seconds for interval
            interval_seconds = max(1, int(session.interval_minutes * 60))
            
            # Add trading started alert
            session_manager.add_alert(session.session_id, {
                "type": "info",
                "title": "Trading Started",
                "message": f"Paper trading started with {session.duration_minutes} minutes duration and {interval_seconds} seconds interval",
                "timestamp": datetime.now().isoformat()
            })
            
            # Convert minutes to timedelta for duration
            duration = timedelta(minutes=session.duration_minutes)
            interval = timedelta(seconds=interval_seconds)
            
            # Update agent statuses to active
            activity_integration = active_sessions[session_id].get('activity_integration')
            if activity_integration:
                for agent_name in activity_integration.agent_ids.keys():
                    activity_integration.update_agent_status(agent_name, 'active')
                
                for source_name in activity_integration.data_source_ids.keys():
                    activity_integration.update_data_source_status(source_name, 'active')
            
            # Run paper trading
            results = await orchestrator.run_paper_trading(
                duration=duration,
                update_interval=interval,
                stop_event=session.stop_event
            )
            
            # Store results
            active_sessions[session_id]['results'] = results
            active_sessions[session_id]['status'] = 'completed'
            
            # Update agent statuses to idle
            if activity_integration:
                for agent_name in activity_integration.agent_ids.keys():
                    activity_integration.update_agent_status(agent_name, 'idle')
            
            session_manager.update_session(session)
            
            # Add completion alert
            session_manager.add_alert(session.session_id, {
                "type": "success",
                "title": "Trading Completed",
                "message": "Paper trading completed successfully",
                "timestamp": datetime.now().isoformat()
            })
            
        except asyncio.CancelledError:
            logger.info(f"Paper trading session {session.session_id} cancelled")
            
            # Update session status
            session.update_status("stopped")
            session_manager.update_session(session)
            
            # Add cancellation alert
            session_manager.add_alert(session.session_id, {
                "type": "info",
                "title": "Trading Stopped",
                "message": "Paper trading stopped by user",
                "timestamp": datetime.now().isoformat()
            })
            
            raise
            
        except Exception as e:
            logger.error(f"Error in paper trading session {session.session_id}: {e}", exc_info=True)
            
            # Update session status
            session.update_status("error")
            session_manager.update_session(session)
            
            # Add error alert
            session_manager.add_alert(session.session_id, {
                "type": "error",
                "title": "Trading Error",
                "message": f"Paper trading error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error in paper trading session {session.session_id} task: {e}", exc_info=True)
        
        # Update session status
        session.update_status("error")
        session_manager.update_session(session)
        
        # Add error alert
        session_manager.add_alert(session.session_id, {
            "type": "error",
            "title": "Trading Error",
            "message": f"Paper trading error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
    finally:
        # Clean up orchestrator
        if session.orchestrator:
            try:
                await session.orchestrator.cleanup()
                logger.info(f"Cleaned up paper trading orchestrator for session {session.session_id}")
            except Exception as e:
                active_sessions[session_id]['status'] = 'error'
                active_sessions[session_id]['error'] = str(e)
                
                # Update agent statuses to error
                if activity_integration:
                    for agent_name in activity_integration.agent_ids.keys():
                        activity_integration.update_agent_status(agent_name, 'error')
                
                logger.error(f"Error cleaning up paper trading orchestrator for session {session.session_id}: {e}", exc_info=True)


def _start_paper_trading_task(session) -> None:
    """
    Start paper trading in a background task for a specific session.
    
    Args:
        session: The paper trading session to run
    """
    # Create stop event for this session
    stop_event = asyncio.Event()
    session.set_stop_event(stop_event)
    
    # Create and start task in a new thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Log start
    logger.info(f"Starting paper trading session {session.session_id} with config: {session.config_path}, "
               f"duration: {session.duration_minutes} minutes, interval: {session.interval_minutes} minutes")
    
    # Create task
    task = loop.create_task(_run_paper_trading(session))
    session.set_task(task)
    
    # Run loop in thread
    def run_loop_in_thread():
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            logger.info(f"Paper trading session {session.session_id} task cancelled")
        except Exception as e:
            logger.error(f"Error in paper trading session {session.session_id} task: {e}", exc_info=True)
            # Update session status to error
            session.update_status("error")
            session_manager.update_session(session)
        finally:
            loop.close()
            logger.info(f"Paper trading session {session.session_id} task loop closed")
    
    # Start thread
    thread = threading.Thread(target=run_loop_in_thread, daemon=True)
    thread.start()
    
    logger.info(f"Paper trading session {session.session_id} task started in separate thread")


@router.post("/start")
async def start_paper_trading(config: PaperTradingConfig):
    """
    Start a new paper trading session.
    
    Args:
        config: Paper trading configuration
    
    Returns:
        Status message with session ID
    """
    # Create a new session using the session manager
    session = session_manager.create_session(
        config_path=config.config_path,
        duration_minutes=config.duration_minutes,
        interval_minutes=config.interval_minutes
    )
    
    # Create agent activity integration
    activity_integration = AgentActivityIntegration(session.session_id)
    activity_integration.initialize_tracking(orchestrator)
    
    # Store session info
    active_sessions[session.session_id] = {
        'orchestrator': orchestrator,
        'status': 'starting',
        'config_path': config.config_path,
        'start_time': None,
        'results': None,
        'thread': None,
        'activity_integration': activity_integration
    }
    
    # Start paper trading task for this session
    _start_paper_trading_task(session)
    
    # Add alert
    session_manager.add_alert(session.session_id, {
        "type": "info",
        "title": "Session Started",
        "message": "Paper trading session is starting",
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "status": "starting", 
        "session_id": session.session_id,
        "message": "Paper trading session is starting"
    }


@router.post("/stop")
async def stop_paper_trading(session_id: str = Query(..., description="ID of the session to stop")):
    """
    Stop a specific paper trading session.
    
    Args:
        session_id: ID of the session to stop
    
    Returns:
        Status message
    """
    # Get session
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Check if session is running
    if session.status not in ["running", "starting"]:
        raise HTTPException(status_code=400, detail=f"Session {session_id} is not running")
    
    # Stop session
    success = session_manager.stop_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to stop session {session_id}")
    
    # Add alert
    session_manager.add_alert(session_id, {
        "type": "info",
        "title": "Session Stopping",
        "message": "Paper trading session is stopping",
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "status": "stopping",
        "session_id": session_id,
        "message": "Paper trading session is stopping"
    }


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


@router.get("/sessions")
async def get_paper_trading_sessions():
    """
    Get all paper trading sessions.
    
    Returns:
        List of paper trading sessions
    """
    # Get all sessions from the session manager
    sessions = session_manager.get_all_sessions()
    
    # Convert sessions to dictionaries for JSON serialization
    session_dicts = [session.to_dict() for session in sessions]
    
    return {"sessions": session_dicts}


@router.get("/sessions/{session_id}")
async def get_session_details(session_id: str = Path(..., description="The ID of the paper trading session")):
    """
    Get details of a specific paper trading session.
    
    Args:
        session_id: The ID of the paper trading session
    
    Returns:
        Paper trading session details
    """
    # Get session from the session manager
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Convert session to dictionary for JSON serialization
    return session.to_dict()


@router.get("/alerts/{session_id}")
async def get_session_alerts(session_id: str = Path(..., description="The ID of the paper trading session")):
    """
    Get alerts for a specific paper trading session.
    
    Args:
        session_id: The ID of the paper trading session
    
    Returns:
        List of alerts for the session
    """
    # Check if session exists
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Get alerts for session from the session manager
    alerts = session_manager.get_alerts(session_id)
    
    return {"alerts": alerts}


@router.post("/alerts/{session_id}")
async def add_session_alert(
    session_id: str = Path(..., description="The ID of the paper trading session"),
    alert: dict = None
):
    """
    Add an alert to a paper trading session.
    
    Args:
        session_id: The ID of the paper trading session
        alert: The alert to add
    
    Returns:
        The created alert
    """
    # Check if session exists
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Validate alert
    if not alert or "type" not in alert or "message" not in alert:
        raise HTTPException(status_code=400, detail="Alert must contain type and message")
    
    # Ensure timestamp is included
    if "timestamp" not in alert:
        alert["timestamp"] = datetime.now().isoformat()
        
    # Add title if not present
    if "title" not in alert:
        alert["title"] = alert["type"].capitalize()
    
    # Add alert using the session manager
    session_manager.add_alert(session_id, alert)
    
    return alert


@router.get("/alerts/{session_id}/settings")
async def get_alert_settings(session_id: str = Path(..., description="The ID of the paper trading session")):
    """
    Get alert settings for a specific paper trading session.
    
    Args:
        session_id: The ID of the paper trading session
    
    Returns:
        Alert settings for the session
    """
    # Check if session exists
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Get alert settings from the session manager
    settings = session_manager.get_alert_settings(session_id)
    if not settings:
        # Return default settings if none are set
        settings = {
            "enabled": True,
            "drawdownThreshold": 0.05,
            "gainThreshold": 0.05,
            "largeTradeThreshold": 0.1,
            "consecutiveLossesThreshold": 3
        }
    
    return {"settings": settings}


@router.post("/alerts/{session_id}/settings")
async def update_alert_settings(
    session_id: str = Path(..., description="The ID of the paper trading session"),
    settings: PaperTradingAlertSettings = None
):
    """
    Update alert settings for a specific paper trading session.
    
    Args:
        session_id: The ID of the paper trading session
        settings: The updated alert settings
    
    Returns:
        Status message
    """
    # Check if session exists
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Validate settings
    if not settings:
        raise HTTPException(status_code=400, detail="Alert settings must be provided")
    
    # Update alert settings using the session manager
    session_manager.update_alert_settings(session_id, settings.dict())
    
    # Log update
    logger.info(f"Updated alert settings for session {session_id}: {settings}")
    
    return {"status": "success", "message": "Alert settings updated successfully"}


@router.post("/stop/{session_id}")
async def stop_paper_trading_session(session_id: str = Path(..., description="The ID of the paper trading session")):
    """
    Stop a specific paper trading session.
    
    Args:
        session_id: The ID of the paper trading session
    
    Returns:
        Status message
    """
    # Get session
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Check if session is running
    if session.status not in ["running", "starting"]:
        raise HTTPException(status_code=400, detail=f"Session {session_id} is not running")
    
    # Stop session
    success = session_manager.stop_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to stop session {session_id}")
    
    # Add alert
    session_manager.add_alert(session_id, {
        "type": "info",
        "title": "Session Stopping",
        "message": "Paper trading session is stopping",
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "status": "stopping",
        "session_id": session_id,
        "message": f"Paper trading session {session_id} is being stopped"
    }
