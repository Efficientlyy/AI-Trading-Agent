"""
Live Trading Bridge Module

Provides a bridge between backtesting and live trading with safety mechanisms.
Implements a unified interface for order placement across both modes.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import time
from enum import Enum
import json

# Import API models
from backend.schemas import OrderRequest, OrderResponse, TradeResponse

# Import core trading agent models
from ai_trading_agent.trading_engine.models import (
    Order, OrderSide, OrderType, OrderStatus, 
    Trade, Position, Portfolio
)
from ai_trading_agent.trading_engine.order_manager import OrderManager
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.trading_engine.execution_handler import ExecutionHandler
from ai_trading_agent.data_acquisition.data_service import DataService
from ai_trading_agent.risk_management.risk_manager import RiskManager

# Import bridge
from backend.integration.bridge import TradingBridge, get_trading_bridge

# Setup logging
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enum."""
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"


class SafeguardType(Enum):
    """Types of trading safeguards."""
    MAX_POSITION_SIZE = "max_position_size"
    MAX_POSITION_VALUE = "max_position_value"
    MAX_ORDER_SIZE = "max_order_size"
    MAX_ORDER_VALUE = "max_order_value"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_ORDERS_PER_MINUTE = "max_orders_per_minute"
    MAX_POSITION_COUNT = "max_position_count"
    PREVENT_LIQUIDATION = "prevent_liquidation" 
    REQUIRE_STOP_LOSS = "require_stop_loss"
    BANNED_SYMBOLS = "banned_symbols"
    TRADING_HOURS = "trading_hours"
    MARKET_VOLATILITY = "market_volatility"
    NEWS_IMPACT = "news_impact"


class LiveTradingBridge:
    """
    Bridge between backtesting and live trading environments.
    
    Provides:
    1. Safety mechanisms to prevent catastrophic trading errors
    2. Unified interface for order placement
    3. Support for both paper trading and live trading modes
    4. Monitoring and alerting capabilities
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 trading_mode: TradingMode = TradingMode.PAPER_TRADING):
        """
        Initialize the live trading bridge.
        
        Args:
            config: Configuration dictionary
            trading_mode: Trading mode (backtesting, paper_trading, live_trading)
        """
        self.config = config
        self.trading_mode = trading_mode
        self.trading_bridge = get_trading_bridge(config, trading_mode != TradingMode.LIVE_TRADING)
        
        # Initialize safety limits from config
        self.safety_config = config.get("safety_limits", {})
        self._load_safety_config()
        
        # Initialize tracking variables
        self._daily_pnl = 0.0
        self._high_water_mark = 0.0
        self._order_timestamps: List[datetime] = []
        self._current_positions: Dict[str, float] = {}  # symbol -> quantity
        
        # Initialize alerting and monitoring
        self._alerts_enabled = config.get("alerts_enabled", True)
        self._alert_channels = config.get("alert_channels", ["log"])
        self._last_status_update = datetime.now()
        self._status_update_interval = config.get("status_update_interval", 300)  # 5 minutes
        
        # Register market data hooks
        self._register_market_data_hooks()
        
        logger.info(f"Live trading bridge initialized with mode: {trading_mode.value}")
    
    def _load_safety_config(self) -> None:
        """Load safety configuration with defaults."""
        # Position and order size limits
        self.max_position_size = self.safety_config.get(SafeguardType.MAX_POSITION_SIZE.value, {})
        self.max_position_value = self.safety_config.get(SafeguardType.MAX_POSITION_VALUE.value, 10000)
        self.max_order_size = self.safety_config.get(SafeguardType.MAX_ORDER_SIZE.value, {})
        self.max_order_value = self.safety_config.get(SafeguardType.MAX_ORDER_VALUE.value, 5000)
        
        # Risk limits
        self.max_daily_loss = self.safety_config.get(SafeguardType.MAX_DAILY_LOSS.value, 1000)
        self.max_drawdown = self.safety_config.get(SafeguardType.MAX_DRAWDOWN.value, 0.05)  # 5% 
        self.max_position_count = self.safety_config.get(SafeguardType.MAX_POSITION_COUNT.value, 10)
        
        # Rate limits
        self.max_orders_per_minute = self.safety_config.get(SafeguardType.MAX_ORDERS_PER_MINUTE.value, 10)
        
        # Trading restrictions
        self.require_stop_loss = self.safety_config.get(SafeguardType.REQUIRE_STOP_LOSS.value, False)
        self.banned_symbols = self.safety_config.get(SafeguardType.BANNED_SYMBOLS.value, [])
        
        # Trading hours (default: 24/7)
        self.trading_hours = self.safety_config.get(SafeguardType.TRADING_HOURS.value, {
            "enabled": False,
            "start_time": "09:30",  # Market open (ET)
            "end_time": "16:00",    # Market close (ET)
            "timezone": "America/New_York",
            "weekdays_only": True
        })
        
        # Market conditions
        self.market_volatility_limits = self.safety_config.get(SafeguardType.MARKET_VOLATILITY.value, {
            "enabled": False,
            "max_volatility": 0.03,  # 3% max volatility
            "lookback_period": "1d",
            "pause_trading": True
        })
        
        self.news_impact_settings = self.safety_config.get(SafeguardType.NEWS_IMPACT.value, {
            "enabled": False,
            "sources": ["twitter", "news_api"],
            "impact_threshold": 0.7,
            "pause_trading": False,
            "reduce_position_size": True,
            "reduction_factor": 0.5
        })
    
    def _register_market_data_hooks(self) -> None:
        """Register hooks for real-time market data."""
        # This would connect to market data streams for monitoring
        # For now, we'll just simulate it with a periodic task
        if self.trading_mode != TradingMode.BACKTESTING:
            asyncio.create_task(self._periodic_status_update())
    
    async def _periodic_status_update(self) -> None:
        """Periodically update status and check risk parameters."""
        while True:
            try:
                current_time = datetime.now()
                
                # Only update at the specified interval
                if (current_time - self._last_status_update).total_seconds() >= self._status_update_interval:
                    self._last_status_update = current_time
                    
                    # Update portfolio status
                    portfolio = self.trading_bridge.portfolio_manager.get_portfolio()
                    self._update_risk_metrics(portfolio)
                    
                    # Log status
                    if self._alerts_enabled:
                        logger.info(f"Status update - Portfolio value: ${portfolio.total_value:.2f}, " 
                                   f"Cash: ${portfolio.cash:.2f}, "
                                   f"Positions: {len(portfolio.positions)}")
                
                # Check if we need to apply any emergency measures
                await self._check_emergency_measures()
                
            except Exception as e:
                logger.error(f"Error in periodic status update: {e}")
            
            # Sleep until next check
            await asyncio.sleep(60)  # Check every minute
    
    def _update_risk_metrics(self, portfolio) -> None:
        """Update risk metrics based on current portfolio."""
        # Update high water mark if needed
        if portfolio.total_value > self._high_water_mark:
            self._high_water_mark = portfolio.total_value
        
        # Calculate drawdown
        current_drawdown = 0
        if self._high_water_mark > 0:
            current_drawdown = (self._high_water_mark - portfolio.total_value) / self._high_water_mark
        
        # Reset daily P&L if it's a new day
        current_date = datetime.now().date()
        if not hasattr(self, '_last_pnl_reset_date') or self._last_pnl_reset_date != current_date:
            self._daily_pnl = portfolio.unrealized_pnl + portfolio.realized_pnl
            self._last_pnl_reset_date = current_date
        else:
            # Update daily P&L
            self._daily_pnl = portfolio.unrealized_pnl + portfolio.realized_pnl
        
        # Update current positions dictionary
        self._current_positions = {
            symbol: position.quantity for symbol, position in portfolio.positions.items()
        }
    
    async def _check_emergency_measures(self) -> None:
        """Check if emergency measures need to be applied."""
        try:
            portfolio = self.trading_bridge.portfolio_manager.get_portfolio()
            
            # Check drawdown limit
            current_drawdown = 0
            if self._high_water_mark > 0:
                current_drawdown = (self._high_water_mark - portfolio.total_value) / self._high_water_mark
                
            if current_drawdown >= self.max_drawdown:
                await self._handle_max_drawdown_breach(current_drawdown)
            
            # Check daily loss limit
            if self._daily_pnl <= -self.max_daily_loss:
                await self._handle_max_daily_loss_breach(self._daily_pnl)
                
        except Exception as e:
            logger.error(f"Error checking emergency measures: {e}")
    
    async def _handle_max_drawdown_breach(self, current_drawdown: float) -> None:
        """Handle breach of maximum drawdown limit."""
        message = f"EMERGENCY: Maximum drawdown breach! Current drawdown: {current_drawdown:.2%}"
        logger.critical(message)
        
        if self.trading_mode == TradingMode.LIVE_TRADING:
            # Close all positions if in live trading mode
            await self._emergency_close_all_positions("Max drawdown breach")
            
            # Send alerts
            self._send_alert("critical", message)
    
    async def _handle_max_daily_loss_breach(self, daily_pnl: float) -> None:
        """Handle breach of maximum daily loss limit."""
        message = f"EMERGENCY: Maximum daily loss breach! Current daily P&L: ${daily_pnl:.2f}"
        logger.critical(message)
        
        if self.trading_mode == TradingMode.LIVE_TRADING:
            # Close all positions if in live trading mode
            await self._emergency_close_all_positions("Max daily loss breach")
            
            # Send alerts
            self._send_alert("critical", message)
    
    async def _emergency_close_all_positions(self, reason: str) -> None:
        """Emergency close all open positions."""
        try:
            portfolio = self.trading_bridge.portfolio_manager.get_portfolio()
            
            for symbol, position in portfolio.positions.items():
                if position.quantity == 0:
                    continue
                    
                # Create order to close position
                side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
                
                emergency_order = Order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=abs(position.quantity),
                    price=None,  # Market order
                    status=OrderStatus.PENDING,
                    timestamp=datetime.now(),
                    metadata={"emergency_close": True, "reason": reason}
                )
                
                # Process emergency order
                logger.warning(f"Emergency closing position for {symbol}: {position.quantity} shares")
                await self.trading_bridge.orchestrator.process_order(emergency_order)
                
            logger.critical(f"Emergency position closure completed: {reason}")
        except Exception as e:
            logger.error(f"Error during emergency position closure: {e}")
    
    def _send_alert(self, level: str, message: str) -> None:
        """Send alert through configured channels."""
        if not self._alerts_enabled:
            return
            
        # Always log the alert
        if level == "critical":
            logger.critical(message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
            
        # Send through other channels if configured
        for channel in self._alert_channels:
            if channel == "log":
                continue  # Already logged
            elif channel == "email" and "email_config" in self.config:
                self._send_email_alert(level, message)
            elif channel == "sms" and "sms_config" in self.config:
                self._send_sms_alert(level, message)
            elif channel == "webhook" and "webhook_config" in self.config:
                self._send_webhook_alert(level, message)
    
    def _send_email_alert(self, level: str, message: str) -> None:
        """Send alert via email."""
        # This would be implemented with an email library
        logger.info(f"Would send email alert ({level}): {message}")
    
    def _send_sms_alert(self, level: str, message: str) -> None:
        """Send alert via SMS."""
        # This would be implemented with an SMS service
        logger.info(f"Would send SMS alert ({level}): {message}")
    
    def _send_webhook_alert(self, level: str, message: str) -> None:
        """Send alert via webhook."""
        # This would be implemented with requests or aiohttp
        logger.info(f"Would send webhook alert ({level}): {message}")
    
    async def validate_order(self, order_request: OrderRequest) -> Tuple[bool, Optional[str]]:
        """
        Validate an order request against safety rules.
        
        Args:
            order_request: The order request to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Get current time and portfolio state
            current_time = datetime.now()
            portfolio = self.trading_bridge.portfolio_manager.get_portfolio()
            
            # 1. Check if symbol is banned
            if order_request.symbol in self.banned_symbols:
                return False, f"Trading of {order_request.symbol} is not allowed"
            
            # 2. Check trading hours if enabled
            if self.trading_hours.get("enabled", False):
                if not self._is_within_trading_hours(current_time):
                    return False, "Order rejected: Outside of allowed trading hours"
            
            # 3. Check order frequency (rate limiting)
            if not self._check_order_frequency(current_time):
                return False, f"Order rejected: Exceeded maximum order frequency of {self.max_orders_per_minute} orders per minute"
            
            # 4. Check position count limit
            current_position_count = len([p for p in portfolio.positions.values() if p.quantity != 0])
            if current_position_count >= self.max_position_count and order_request.side.lower() == "buy":
                # If this is opening a new position
                if order_request.symbol not in portfolio.positions or portfolio.positions[order_request.symbol].quantity == 0:
                    return False, f"Order rejected: Maximum position count ({self.max_position_count}) reached"
            
            # 5. Check order size limit for this symbol
            symbol_max_order = self.max_order_size.get(order_request.symbol, float('inf'))
            general_max_order = self.max_order_size.get("default", float('inf'))
            effective_max_order = min(symbol_max_order, general_max_order)
            
            if order_request.quantity > effective_max_order:
                return False, f"Order rejected: Quantity {order_request.quantity} exceeds maximum allowed {effective_max_order}"
            
            # 6. Check order value limit
            # Get current market price for the symbol
            current_price = await self._get_current_price(order_request.symbol)
            order_value = order_request.quantity * current_price
            
            if order_value > self.max_order_value:
                return False, f"Order rejected: Order value ${order_value:.2f} exceeds maximum allowed ${self.max_order_value:.2f}"
            
            # 7. Check position size limit if this would increase a position
            if order_request.side.lower() == "buy":
                current_position_size = portfolio.positions.get(order_request.symbol, Position(symbol=order_request.symbol, quantity=0)).quantity
                new_position_size = current_position_size + order_request.quantity
                
                symbol_max_position = self.max_position_size.get(order_request.symbol, float('inf'))
                general_max_position = self.max_position_size.get("default", float('inf'))
                effective_max_position = min(symbol_max_position, general_max_position)
                
                if new_position_size > effective_max_position:
                    return False, f"Order rejected: Resulting position size {new_position_size} exceeds maximum allowed {effective_max_position}"
            
            # 8. Check position value limit if this would increase a position
            if order_request.side.lower() == "buy":
                current_position = portfolio.positions.get(order_request.symbol, Position(symbol=order_request.symbol, quantity=0))
                new_position_value = (current_position.quantity + order_request.quantity) * current_price
                
                if new_position_value > self.max_position_value:
                    return False, f"Order rejected: Resulting position value ${new_position_value:.2f} exceeds maximum allowed ${self.max_position_value:.2f}"
            
            # 9. Check stop loss requirement
            if self.require_stop_loss and order_request.side.lower() == "buy" and not self._has_stop_loss(order_request):
                return False, "Order rejected: Stop loss is required for all buy orders"
            
            # 10. Check market volatility if enabled
            if self.market_volatility_limits.get("enabled", False):
                volatility = await self._get_market_volatility(order_request.symbol)
                if volatility > self.market_volatility_limits.get("max_volatility", 0.03):
                    return False, f"Order rejected: Market volatility ({volatility:.2%}) exceeds maximum allowed ({self.market_volatility_limits.get('max_volatility', 0.03):.2%})"
            
            # Order passed all validation rules
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False, f"Order validation error: {str(e)}"
    
    def _is_within_trading_hours(self, current_time: datetime) -> bool:
        """Check if the current time is within allowed trading hours."""
        if not self.trading_hours.get("enabled", False):
            return True
        
        # Check weekday if weekdays_only is enabled
        if self.trading_hours.get("weekdays_only", True) and current_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        
        # Parse trading hours
        try:
            import pytz
            from datetime import datetime, time
            
            # Get timezone
            timezone_str = self.trading_hours.get("timezone", "America/New_York")
            timezone = pytz.timezone(timezone_str)
            
            # Convert current time to target timezone
            current_time_tz = current_time.astimezone(timezone)
            
            # Parse start and end times
            start_time_str = self.trading_hours.get("start_time", "09:30")
            end_time_str = self.trading_hours.get("end_time", "16:00")
            
            start_hour, start_minute = map(int, start_time_str.split(":"))
            end_hour, end_minute = map(int, end_time_str.split(":"))
            
            start_time = time(hour=start_hour, minute=start_minute)
            end_time = time(hour=end_hour, minute=end_minute)
            
            # Check if current time is within range
            current_time_only = current_time_tz.time()
            return start_time <= current_time_only <= end_time
            
        except Exception as e:
            logger.error(f"Error checking trading hours: {e}")
            return True  # Default to allowing trading if there's an error
    
    def _check_order_frequency(self, current_time: datetime) -> bool:
        """Check if order frequency is within limits."""
        # Clean up old timestamps
        one_minute_ago = current_time - timedelta(minutes=1)
        self._order_timestamps = [ts for ts in self._order_timestamps if ts > one_minute_ago]
        
        # Check if we've exceeded the limit
        return len(self._order_timestamps) < self.max_orders_per_minute
    
    def _has_stop_loss(self, order_request: OrderRequest) -> bool:
        """Check if the order has a stop loss."""
        # In a real implementation, this would check if a stop loss order is submitted alongside the main order
        # For now, we'll just check if stop_price is set (for stop or stop-limit orders)
        return hasattr(order_request, 'stop_price') and order_request.stop_price is not None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        try:
            # In production, this would call the actual data service
            return await self.trading_bridge.data_service.get_latest_price(symbol)
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            # Return a default price for testing - in production this would raise an error
            return 100.0
    
    async def _get_market_volatility(self, symbol: str) -> float:
        """Calculate market volatility for a symbol."""
        try:
            # In production, this would calculate actual volatility
            # Here we'll return a simulated value
            import random
            return random.uniform(0.01, 0.04)  # Random volatility between 1% and 4%
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.02  # Default to 2% volatility
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Place an order with safety checks.
        
        Args:
            order_request: The order request
            
        Returns:
            Order response
        """
        # Track order timestamp for rate limiting
        self._order_timestamps.append(datetime.now())
        
        # Validate order
        is_valid, error_message = await self.validate_order(order_request)
        
        if not is_valid:
            logger.warning(f"Order validation failed: {error_message}")
            raise ValueError(error_message)
        
        # Process the order through the trading bridge
        try:
            return await self.trading_bridge.create_order(order_request)
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an open order."""
        try:
            return await self.trading_bridge.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            raise


# Create a singleton instance
_live_trading_bridge_instance = None

def get_live_trading_bridge(config: Dict[str, Any] = None, trading_mode: TradingMode = TradingMode.PAPER_TRADING) -> LiveTradingBridge:
    """
    Get or create the live trading bridge singleton instance.
    
    Args:
        config: Configuration dictionary (optional, used only when creating new instance)
        trading_mode: Trading mode (optional, used only when creating new instance)
        
    Returns:
        LiveTradingBridge instance
    """
    global _live_trading_bridge_instance
    
    if _live_trading_bridge_instance is None:
        if config is None:
            config = {}  # Default empty config
        _live_trading_bridge_instance = LiveTradingBridge(config, trading_mode)
        
    return _live_trading_bridge_instance