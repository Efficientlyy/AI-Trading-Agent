"""
Shadow Trader Implementation for AI Trading Agent

This module implements the shadow trading capability, which allows the trading agent to run with
real market data but execute trades in a simulated environment. This is the first step in the
production deployment plan, enabling validation of the trading system without financial risk.
"""

import logging
import uuid
from typing import Dict, List, Optional, Union
from datetime import datetime

from ai_trading_agent.execution.models import Order, OrderStatus, OrderType, Position, Trade
from ai_trading_agent.execution.broker.base import BaseBroker
from ai_trading_agent.data.models import MarketData
from ai_trading_agent.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class ShadowTrader:
    """
    Shadow Trading implementation that uses real market data but executes trades in simulation.
    
    This component is designed for the initial deployment phase, allowing the AI Trading Agent
    to run with real market data while executing simulated trades to validate performance
    before committing real capital.
    """
    
    def __init__(self, real_broker: BaseBroker):
        """
        Initialize the shadow trader with a real broker connection.
        
        Args:
            real_broker: The real broker instance used to fetch market data
        """
        self.real_broker = real_broker
        self.paper_positions: Dict[str, Position] = {}
        self.paper_orders: Dict[str, Order] = {}
        self.paper_trades: List[Trade] = []
        
        # Starting capital for paper trading
        self.starting_capital = float(settings.SHADOW_TRADING_CAPITAL)
        self.available_cash = self.starting_capital
        self.portfolio_value = self.starting_capital
        
        # Performance tracking
        self.initial_timestamp = datetime.now()
        self.performance_snapshots: List[Dict] = []
        
        # Slippage simulation settings
        self.simulate_slippage = settings.SIMULATE_SLIPPAGE if hasattr(settings, 'SIMULATE_SLIPPAGE') else True
        self.avg_slippage_bps = float(settings.AVG_SLIPPAGE_BPS) if hasattr(settings, 'AVG_SLIPPAGE_BPS') else 5
        
        logger.info(f"Shadow trader initialized with {self.starting_capital} starting capital")

    def place_order(self, order: Order) -> str:
        """
        Place a simulated order based on real market data.
        
        Args:
            order: The order to place
            
        Returns:
            The order ID
        """
        # Generate a unique order ID
        if not order.order_id:
            order.order_id = str(uuid.uuid4())
            
        # Set initial order status
        order.status = OrderStatus.PENDING
        order.submitted_time = datetime.now()
        
        # Store the order
        self.paper_orders[order.order_id] = order
        
        logger.info(f"Shadow order placed: {order.order_id} - {order.symbol} {order.side} {order.quantity} @ {order.order_type}")
        
        # Immediately simulate execution for market orders
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order)
        
        return order.order_id
    
    def _execute_market_order(self, order: Order) -> None:
        """
        Simulate the execution of a market order using real market data.
        
        Args:
            order: The order to execute
        """
        # Get the latest market price from the real broker
        try:
            market_data = self.real_broker.get_market_data(order.symbol)
            
            # Calculate execution price with simulated slippage
            execution_price = market_data.last_price
            
            if self.simulate_slippage:
                # Apply slippage - buy orders pay more, sell orders receive less
                slippage_factor = 1.0 + (self.avg_slippage_bps / 10000.0) if order.side == 'BUY' else 1.0 - (self.avg_slippage_bps / 10000.0)
                execution_price = execution_price * slippage_factor
                
            # Update order status and execution details
            order.status = OrderStatus.FILLED
            order.filled_price = execution_price
            order.filled_time = datetime.now()
            order.filled_quantity = order.quantity
            
            # Create a trade record
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=datetime.now(),
                commission=self._calculate_commission(order.quantity, execution_price)
            )
            
            # Add to trades list
            self.paper_trades.append(trade)
            
            # Update positions and cash
            self._update_position(order, trade)
            
            logger.info(f"Shadow order executed: {order.order_id} - {order.filled_quantity} @ {order.filled_price}")
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Failed to execute shadow order {order.order_id}: {str(e)}")
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate the commission for a trade.
        
        Args:
            quantity: The quantity traded
            price: The execution price
            
        Returns:
            The commission amount
        """
        # Use the same commission structure as the real broker
        commission_rate = float(settings.COMMISSION_RATE) if hasattr(settings, 'COMMISSION_RATE') else 0.001
        min_commission = float(settings.MIN_COMMISSION) if hasattr(settings, 'MIN_COMMISSION') else 1.0
        
        commission = quantity * price * commission_rate
        return max(commission, min_commission)
    
    def _update_position(self, order: Order, trade: Trade) -> None:
        """
        Update positions and cash based on a trade.
        
        Args:
            order: The filled order
            trade: The executed trade
        """
        symbol = order.symbol
        trade_value = trade.quantity * trade.price
        trade_cost = trade_value + trade.commission
        
        # Update cash
        if order.side == 'BUY':
            self.available_cash -= trade_cost
        else:
            self.available_cash += trade_value - trade.commission
        
        # Update or create position
        if symbol in self.paper_positions:
            position = self.paper_positions[symbol]
            
            if order.side == 'BUY':
                # Calculate new average cost
                new_qty = position.quantity + trade.quantity
                new_cost = (position.quantity * position.avg_price) + trade_value
                position.avg_price = new_cost / new_qty if new_qty > 0 else 0
                position.quantity = new_qty
            else:
                position.quantity -= trade.quantity
                # If position is closed, calculate realized PnL
                if position.quantity <= 0:
                    if position.quantity < 0:
                        # Short position
                        position.avg_price = trade.price
                    else:
                        # Position closed
                        position.avg_price = 0
        else:
            # Create new position
            if order.side == 'BUY':
                self.paper_positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    avg_price=trade.price,
                    current_price=trade.price,
                    timestamp=trade.timestamp
                )
            else:
                # Short position
                self.paper_positions[symbol] = Position(
                    symbol=symbol,
                    quantity=-trade.quantity,
                    avg_price=trade.price,
                    current_price=trade.price,
                    timestamp=trade.timestamp
                )
        
        # Update portfolio value
        self._update_portfolio_value()
        
    def _update_portfolio_value(self) -> None:
        """Update the total portfolio value based on current positions and cash."""
        position_value = 0.0
        
        # Update current prices and calculate position values
        for symbol, position in self.paper_positions.items():
            if position.quantity != 0:
                try:
                    # Get latest price from real broker
                    market_data = self.real_broker.get_market_data(symbol)
                    position.current_price = market_data.last_price
                    position.timestamp = datetime.now()
                    position_value += position.quantity * position.current_price
                except Exception as e:
                    logger.warning(f"Failed to update price for {symbol}: {str(e)}")
        
        # Calculate total portfolio value
        self.portfolio_value = self.available_cash + position_value
        
        # Record performance snapshot
        self._record_performance()
    
    def _record_performance(self) -> None:
        """Record current performance metrics."""
        snapshot = {
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'available_cash': self.available_cash,
            'return_pct': (self.portfolio_value / self.starting_capital - 1) * 100
        }
        
        self.performance_snapshots.append(snapshot)
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all current shadow trading positions."""
        # Update all positions with current prices
        self._update_portfolio_value()
        return self.paper_positions
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get shadow trading orders.
        
        Args:
            status: Filter orders by status
            
        Returns:
            List of orders
        """
        if status:
            return [order for order in self.paper_orders.values() if order.status == status]
        return list(self.paper_orders.values())
    
    def get_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """
        Get shadow trading execution history.
        
        Args:
            symbol: Filter trades by symbol
            
        Returns:
            List of trades
        """
        if symbol:
            return [trade for trade in self.paper_trades if trade.symbol == symbol]
        return self.paper_trades
    
    def get_performance(self) -> Dict:
        """
        Get shadow trading performance metrics.
        
        Returns:
            Dict containing performance metrics
        """
        self._update_portfolio_value()
        
        return {
            'starting_capital': self.starting_capital,
            'current_value': self.portfolio_value,
            'available_cash': self.available_cash,
            'return_pct': (self.portfolio_value / self.starting_capital - 1) * 100,
            'position_count': len([p for p in self.paper_positions.values() if p.quantity != 0]),
            'trade_count': len(self.paper_trades),
            'start_time': self.initial_timestamp,
            'duration_days': (datetime.now() - self.initial_timestamp).days
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: The order ID to cancel
            
        Returns:
            True if the order was cancelled, False otherwise
        """
        if order_id in self.paper_orders:
            order = self.paper_orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Shadow order cancelled: {order_id}")
                return True
                
        return False
