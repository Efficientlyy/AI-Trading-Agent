from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import time
import threading
import logging
from datetime import datetime

# Import strategy classes
from ai_trading_agent.agent.strategy import BaseStrategy
from ai_trading_agent.agent.market_regime import MarketRegimeStrategy
from ai_trading_agent.agent.ml_strategy import MLStrategy
from ai_trading_agent.agent.portfolio_strategy import PortfolioStrategy
from ai_trading_agent.agent.integrated_manager import IntegratedStrategyManager

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)

# Define Pydantic models for request/response validation
class StrategyConfig(BaseModel):
    enabled: bool
    weight: float

class RiskManagementConfig(BaseModel):
    max_position_size: float
    stop_loss_pct: float
    take_profit_pct: float

class TradingAgentConfig(BaseModel):
    name: str
    description: str
    strategies: Dict[str, StrategyConfig]
    risk_management: RiskManagementConfig
    symbols: List[str]
    initial_capital: float

class PositionInfo(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

class TradingAgentStatus(BaseModel):
    status: str  # 'idle', 'running', 'stopped', 'error'
    error_message: Optional[str] = None
    uptime_seconds: int
    total_trades: int
    profitable_trades: int
    current_portfolio_value: float
    pnl_pct: float
    active_positions: List[PositionInfo]

# Global variables to track agent state
agent_config = TradingAgentConfig(
    name="Default Trading Agent",
    description="A trading agent that uses multiple strategies to make trading decisions",
    strategies={
        "market_regime": StrategyConfig(enabled=True, weight=0.33),
        "ml_strategy": StrategyConfig(enabled=True, weight=0.33),
        "portfolio_strategy": StrategyConfig(enabled=True, weight=0.34)
    },
    risk_management=RiskManagementConfig(
        max_position_size=0.1,
        stop_loss_pct=0.05,
        take_profit_pct=0.1
    ),
    symbols=["BTC/USD", "ETH/USD", "XRP/USD"],
    initial_capital=10000
)

agent_status = TradingAgentStatus(
    status="idle",
    uptime_seconds=0,
    total_trades=0,
    profitable_trades=0,
    current_portfolio_value=10000,
    pnl_pct=0,
    active_positions=[]
)

# Trading agent class
class TradingAgent:
    def __init__(self, config: TradingAgentConfig):
        self.config = config
        self.strategy_manager = IntegratedStrategyManager()
        self.start_time = None
        self.running = False
        self.thread = None
        self.error = None
        self.trades = []
        self.positions = {}
        self.portfolio_value = config.initial_capital
        
        # Initialize strategies based on config
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        # Clear existing strategies
        self.strategy_manager._strategies = {}
        
        # Add enabled strategies
        if self.config.strategies.get("market_regime", {}).get("enabled", False):
            market_regime = MarketRegimeStrategy()
            self.strategy_manager.add_strategy(
                "market_regime", 
                market_regime, 
                self.config.strategies["market_regime"].weight
            )
        
        if self.config.strategies.get("ml_strategy", {}).get("enabled", False):
            ml_strategy = MLStrategy()
            self.strategy_manager.add_strategy(
                "ml_strategy", 
                ml_strategy, 
                self.config.strategies["ml_strategy"].weight
            )
        
        if self.config.strategies.get("portfolio_strategy", {}).get("enabled", False):
            portfolio_strategy = PortfolioStrategy()
            self.strategy_manager.add_strategy(
                "portfolio_strategy", 
                portfolio_strategy, 
                self.config.strategies["portfolio_strategy"].weight
            )
    
    def start(self):
        if self.running:
            return
        
        self.start_time = time.time()
        self.running = True
        self.error = None
        
        # Start the agent in a separate thread
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
    
    def _run(self):
        try:
            logger.info(f"Starting trading agent: {self.config.name}")
            
            # Simulate trading activity for demo purposes
            # In a real implementation, this would connect to exchanges,
            # fetch market data, and execute trades based on strategy signals
            while self.running:
                # Update agent status
                global agent_status
                agent_status.status = "running"
                agent_status.uptime_seconds = int(time.time() - self.start_time)
                
                # Sleep to avoid high CPU usage
                time.sleep(5)
                
        except Exception as e:
            self.error = str(e)
            logger.error(f"Trading agent error: {self.error}")
            agent_status.status = "error"
            agent_status.error_message = self.error
        finally:
            if self.running:
                self.running = False
                agent_status.status = "stopped"
    
    def get_status(self) -> TradingAgentStatus:
        status = TradingAgentStatus(
            status="running" if self.running else "idle" if self.error is None else "error",
            error_message=self.error,
            uptime_seconds=int(time.time() - self.start_time) if self.start_time else 0,
            total_trades=len(self.trades),
            profitable_trades=sum(1 for trade in self.trades if trade.get("pnl", 0) > 0),
            current_portfolio_value=self.portfolio_value,
            pnl_pct=((self.portfolio_value / self.config.initial_capital) - 1) * 100,
            active_positions=[
                PositionInfo(
                    symbol=symbol,
                    quantity=position.get("quantity", 0),
                    entry_price=position.get("entry_price", 0),
                    current_price=position.get("current_price", 0),
                    unrealized_pnl=position.get("unrealized_pnl", 0),
                    unrealized_pnl_pct=position.get("unrealized_pnl_pct", 0)
                )
                for symbol, position in self.positions.items()
            ]
        )
        return status

# Global agent instance
trading_agent = None

@router.get("/status", response_model=TradingAgentStatus)
async def get_agent_status():
    """Get the current status of the trading agent"""
    global agent_status, trading_agent
    
    if trading_agent:
        # If agent is running, get real-time status
        return trading_agent.get_status()
    
    # Otherwise return the stored status
    return agent_status

@router.get("/config", response_model=TradingAgentConfig)
async def get_agent_config():
    """Get the current configuration of the trading agent"""
    global agent_config
    return agent_config

@router.post("/config", response_model=TradingAgentConfig)
async def update_agent_config(config: TradingAgentConfig):
    """Update the configuration of the trading agent"""
    global agent_config, trading_agent, agent_status
    
    # Check if agent is running
    if trading_agent and trading_agent.running:
        raise HTTPException(
            status_code=400,
            detail="Cannot update configuration while agent is running. Stop the agent first."
        )
    
    # Update the configuration
    agent_config = config
    
    # If agent exists, update its configuration
    if trading_agent:
        trading_agent.config = config
        trading_agent._initialize_strategies()
    
    return agent_config

@router.post("/start", response_model=TradingAgentStatus)
async def start_agent(config: Optional[TradingAgentConfig] = None):
    """Start the trading agent with the current or provided configuration"""
    global trading_agent, agent_config, agent_status
    
    # If config is provided, update the stored config
    if config:
        agent_config = config
    
    # Check if agent is already running
    if trading_agent and trading_agent.running:
        return trading_agent.get_status()
    
    # Create and start the agent
    trading_agent = TradingAgent(agent_config)
    trading_agent.start()
    
    # Update the status
    agent_status = trading_agent.get_status()
    
    return agent_status

@router.post("/stop", response_model=TradingAgentStatus)
async def stop_agent():
    """Stop the trading agent"""
    global trading_agent, agent_status
    
    # Check if agent exists and is running
    if not trading_agent or not trading_agent.running:
        if agent_status.status == "running":
            agent_status.status = "stopped"
        return agent_status
    
    # Stop the agent
    trading_agent.stop()
    
    # Update the status
    agent_status = trading_agent.get_status()
    agent_status.status = "stopped"
    
    return agent_status
