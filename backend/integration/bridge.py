"""
Trading Agent Bridge Module

This module serves as the integration layer between the backend API and the trading agent core.
It provides adapter functions that translate between API models and core domain models.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from pydantic import BaseModel

# Import API models
from backend.schemas import (
    OrderRequest, OrderResponse, TradeResponse, 
    PortfolioSummary, PositionInfo, BacktestConfig, BacktestResult
)

# Import core trading agent models
from ai_trading_agent.trading_engine.models import (
    Order, OrderSide, OrderType, OrderStatus, 
    Trade, Position, Portfolio
)
from ai_trading_agent.trading_engine.order_manager import OrderManager
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.trading_engine.execution_handler import ExecutionHandler
from ai_trading_agent.strategies.strategy_manager import StrategyManager
from ai_trading_agent.data_acquisition.data_service import DataService
from ai_trading_agent.risk_management.risk_manager import RiskManager
from ai_trading_agent.orchestrator import Orchestrator

# Setup logging
logger = logging.getLogger(__name__)

class TradingBridge:
    """
    Bridge class that connects the backend API with the trading agent core.
    Implements the adapter pattern to translate between API and domain models.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 paper_trading: bool = True):
        """
        Initialize the trading bridge with configuration.
        
        Args:
            config: Configuration dictionary
            paper_trading: If True, use paper trading mode
        """
        self.config = config
        self.paper_trading = paper_trading
        self._initialize_components()
        self._orders_cache: Dict[str, Order] = {}
        self._trades_cache: Dict[str, Trade] = {}
        self._running_backtests: Dict[str, asyncio.Task] = {}
        logger.info(f"Trading bridge initialized with paper_trading={paper_trading}")
        
    def _initialize_components(self) -> None:
        """Initialize all trading agent components."""
        # Initialize core components
        self.data_service = DataService(self.config.get("data_service", {}))
        self.order_manager = OrderManager()
        self.portfolio_manager = PortfolioManager(initial_capital=self.config.get("initial_capital", 100000))
        self.execution_handler = ExecutionHandler(paper_trading=self.paper_trading)
        self.risk_manager = RiskManager(self.config.get("risk_manager", {}))
        self.strategy_manager = StrategyManager(self.config.get("strategy_manager", {}))
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            data_manager=self.data_service,
            strategy_manager=self.strategy_manager,
            portfolio_manager=self.portfolio_manager,
            risk_manager=self.risk_manager,
            execution_handler=self.execution_handler,
            order_manager=self.order_manager
        )
        
    def convert_api_order_to_domain(self, order_request: OrderRequest) -> Order:
        """Convert API order model to domain order model."""
        try:
            # Map API order side to domain order side
            side_mapping = {
                "buy": OrderSide.BUY,
                "sell": OrderSide.SELL
            }
            
            # Map API order type to domain order type
            type_mapping = {
                "market": OrderType.MARKET,
                "limit": OrderType.LIMIT,
                "stop": OrderType.STOP,
                "stop_limit": OrderType.STOP_LIMIT
            }
            
            # Create domain order
            order = Order(
                symbol=order_request.symbol,
                side=side_mapping[order_request.side.lower()],
                order_type=type_mapping[order_request.order_type.lower()],
                quantity=order_request.quantity,
                price=order_request.price if order_request.price else None,
                stop_price=order_request.stop_price if hasattr(order_request, 'stop_price') else None,
                time_in_force=order_request.time_in_force if hasattr(order_request, 'time_in_force') else "GTC",
                status=OrderStatus.PENDING,
                timestamp=datetime.now()
            )
            
            return order
        except Exception as e:
            logger.error(f"Error converting API order to domain order: {e}")
            raise ValueError(f"Invalid order request: {e}")
            
    def convert_domain_order_to_api(self, order: Order) -> OrderResponse:
        """Convert domain order model to API order response."""
        try:
            # Map domain order side to API order side
            side_mapping = {
                OrderSide.BUY: "buy",
                OrderSide.SELL: "sell"
            }
            
            # Map domain order type to API order type
            type_mapping = {
                OrderType.MARKET: "market",
                OrderType.LIMIT: "limit",
                OrderType.STOP: "stop",
                OrderType.STOP_LIMIT: "stop_limit"
            }
            
            # Map domain order status to API order status
            status_mapping = {
                OrderStatus.PENDING: "pending",
                OrderStatus.OPEN: "open",
                OrderStatus.FILLED: "filled",
                OrderStatus.PARTIALLY_FILLED: "partially_filled",
                OrderStatus.CANCELED: "canceled",
                OrderStatus.REJECTED: "rejected",
                OrderStatus.EXPIRED: "expired"
            }
            
            # Create API order response
            order_response = OrderResponse(
                order_id=str(order.id),
                symbol=order.symbol,
                side=side_mapping[order.side],
                order_type=type_mapping[order.order_type],
                quantity=order.quantity,
                price=order.price if order.price else 0.0,
                stop_price=order.stop_price if order.stop_price else None,
                executed_quantity=order.executed_quantity,
                status=status_mapping[order.status],
                created_at=order.timestamp,
                updated_at=order.updated_timestamp if hasattr(order, 'updated_timestamp') else order.timestamp
            )
            
            return order_response
        except Exception as e:
            logger.error(f"Error converting domain order to API order: {e}")
            raise ValueError(f"Error processing order: {e}")
    
    def convert_domain_trade_to_api(self, trade: Trade) -> TradeResponse:
        """Convert domain trade model to API trade response."""
        try:
            # Map domain order side to API trade side
            side_mapping = {
                OrderSide.BUY: "buy",
                OrderSide.SELL: "sell"
            }
            
            # Create API trade response
            trade_response = TradeResponse(
                trade_id=str(trade.id),
                order_id=str(trade.order_id),
                symbol=trade.symbol,
                side=side_mapping[trade.side],
                quantity=trade.quantity,
                price=trade.price,
                timestamp=trade.timestamp,
                commission=trade.commission if hasattr(trade, 'commission') else 0.0,
                commission_asset=trade.commission_asset if hasattr(trade, 'commission_asset') else "USD"
            )
            
            return trade_response
        except Exception as e:
            logger.error(f"Error converting domain trade to API trade: {e}")
            raise ValueError(f"Error processing trade: {e}")
    
    def convert_domain_portfolio_to_api(self, portfolio: Portfolio) -> PortfolioSummary:
        """Convert domain portfolio model to API portfolio summary."""
        try:
            # Prepare positions info
            positions = []
            for symbol, position in portfolio.positions.items():
                unrealized_pnl_pct = 0
                if position.average_entry_price > 0 and position.current_price > 0:
                    if position.side == OrderSide.BUY:  # Long position
                        unrealized_pnl_pct = (position.current_price - position.average_entry_price) / position.average_entry_price
                    else:  # Short position
                        unrealized_pnl_pct = (position.average_entry_price - position.current_price) / position.average_entry_price
                
                positions.append(
                    PositionInfo(
                        symbol=symbol,
                        quantity=position.quantity,
                        side="long" if position.side == OrderSide.BUY else "short",
                        entry_price=position.average_entry_price,
                        current_price=position.current_price,
                        market_value=position.market_value,
                        unrealized_pnl=position.unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        realized_pnl=position.realized_pnl
                    )
                )
            
            # Create API portfolio summary
            portfolio_summary = PortfolioSummary(
                total_value=portfolio.total_value,
                cash=portfolio.cash,
                positions_value=portfolio.positions_value,
                positions=positions,
                realized_pnl=portfolio.realized_pnl,
                unrealized_pnl=portfolio.unrealized_pnl,
                total_pnl=portfolio.realized_pnl + portfolio.unrealized_pnl,
                timestamp=datetime.now()
            )
            
            return portfolio_summary
        except Exception as e:
            logger.error(f"Error converting domain portfolio to API portfolio summary: {e}")
            raise ValueError(f"Error processing portfolio: {e}")
    
    async def create_order(self, order_request: OrderRequest) -> OrderResponse:
        """Create a new order through the trading agent."""
        try:
            # Convert API order to domain order
            order = self.convert_api_order_to_domain(order_request)
            
            # Process order through the orchestrator
            processed_order = await self.orchestrator.process_order(order)
            
            # Cache the order for future reference
            self._orders_cache[str(processed_order.id)] = processed_order
            
            # Convert domain order to API response
            return self.convert_domain_order_to_api(processed_order)
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise ValueError(f"Failed to create order: {e}")
    
    async def get_order(self, order_id: str) -> OrderResponse:
        """Get order details by ID."""
        try:
            # Check if order is in cache
            if order_id in self._orders_cache:
                order = self._orders_cache[order_id]
            else:
                # Otherwise, retrieve from order manager
                order = await self.order_manager.get_order(order_id)
                if order:
                    self._orders_cache[order_id] = order
                else:
                    raise ValueError(f"Order not found: {order_id}")
            
            # Convert domain order to API response
            return self.convert_domain_order_to_api(order)
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            raise ValueError(f"Failed to retrieve order: {e}")
    
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an open order by ID."""
        try:
            # Process cancel request through the orchestrator
            canceled_order = await self.orchestrator.cancel_order(order_id)
            
            # Update cache
            self._orders_cache[order_id] = canceled_order
            
            # Convert domain order to API response
            return self.convert_domain_order_to_api(canceled_order)
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            raise ValueError(f"Failed to cancel order: {e}")
    
    async def get_portfolio(self) -> PortfolioSummary:
        """Get current portfolio summary."""
        try:
            # Get current portfolio from portfolio manager
            portfolio = self.portfolio_manager.get_portfolio()
            
            # Convert domain portfolio to API response
            return self.convert_domain_portfolio_to_api(portfolio)
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            raise ValueError(f"Failed to retrieve portfolio: {e}")
    
    async def start_backtest(self, config: BacktestConfig) -> str:
        """
        Start a new backtest with the provided configuration.
        Returns the backtest ID.
        """
        try:
            import uuid
            from ai_trading_agent.backtesting.backtester import Backtester
            
            # Generate a unique backtest ID
            backtest_id = str(uuid.uuid4())
            
            # Convert API config to domain config
            backtest_config = {
                "start_date": config.start_date,
                "end_date": config.end_date,
                "initial_capital": config.initial_capital,
                "symbols": config.symbols,
                "strategy_id": config.strategy_id,
                "timeframe": config.timeframe,
                "commission_rate": config.commission_rate,
                "slippage": config.slippage,
                "use_sentiment": config.use_sentiment,
                "sentiment_sources": config.sentiment_sources or ["reddit", "twitter", "news"],
                "risk_controls": config.risk_controls or {}
            }
            
            # Create and start backtest as an async task
            async def run_backtest():
                try:
                    # Initialize backtester
                    backtester = Backtester(backtest_config)
                    
                    # Run backtest
                    result = await backtester.run()
                    
                    # Store result (in a real system this would go to a database)
                    # For now we'll just log it
                    logger.info(f"Backtest {backtest_id} completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Backtest {backtest_id} failed: {e}")
                    raise
            
            # Start backtest task
            task = asyncio.create_task(run_backtest())
            self._running_backtests[backtest_id] = task
            
            return backtest_id
        except Exception as e:
            logger.error(f"Error starting backtest: {e}")
            raise ValueError(f"Failed to start backtest: {e}")
    
    async def get_backtest_status(self, backtest_id: str) -> Dict[str, Any]:
        """Get the status of a running or completed backtest."""
        try:
            if backtest_id not in self._running_backtests:
                # In a real system, we would check a database
                raise ValueError(f"Backtest {backtest_id} not found")
                
            task = self._running_backtests[backtest_id]
            
            if task.done():
                if task.exception():
                    # Backtest failed
                    return {
                        "backtest_id": backtest_id,
                        "status": "failed",
                        "progress": 1.0,
                        "error_message": str(task.exception())
                    }
                else:
                    # Backtest completed
                    return {
                        "backtest_id": backtest_id,
                        "status": "completed", 
                        "progress": 1.0
                    }
            else:
                # Backtest still running
                # In a real system, we would have a progress indicator
                return {
                    "backtest_id": backtest_id,
                    "status": "running",
                    "progress": 0.5  # Placeholder - would be actual progress in real implementation
                }
        except Exception as e:
            logger.error(f"Error getting backtest status for {backtest_id}: {e}")
            raise ValueError(f"Failed to get backtest status: {e}")
    
    async def get_backtest_result(self, backtest_id: str) -> Dict[str, Any]:
        """Get the results of a completed backtest."""
        try:
            if backtest_id not in self._running_backtests:
                # In a real system, we would check a database
                raise ValueError(f"Backtest {backtest_id} not found")
                
            task = self._running_backtests[backtest_id]
            
            if not task.done():
                raise ValueError(f"Backtest {backtest_id} is still running")
                
            if task.exception():
                raise ValueError(f"Backtest {backtest_id} failed: {task.exception()}")
                
            # Get backtest result
            result = await task
            
            # In a real system, this would be properly converted to the API model
            # For now, we'll just return the raw result
            return result
        except Exception as e:
            logger.error(f"Error getting backtest result for {backtest_id}: {e}")
            raise ValueError(f"Failed to get backtest result: {e}")


# Create a singleton instance of the bridge
_bridge_instance = None

def get_trading_bridge(config: Dict[str, Any] = None, paper_trading: bool = True) -> TradingBridge:
    """
    Get or create the trading bridge singleton instance.
    
    Args:
        config: Configuration dictionary (optional, used only when creating new instance)
        paper_trading: If True, use paper trading mode (optional, used only when creating new instance)
        
    Returns:
        TradingBridge instance
    """
    global _bridge_instance
    
    if _bridge_instance is None:
        if config is None:
            config = {}  # Default empty config
        _bridge_instance = TradingBridge(config, paper_trading)
        
    return _bridge_instance