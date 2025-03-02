"""Portfolio manager for the AI Crypto Trading System.

This module defines the PortfolioManager class, which manages the portfolio of positions,
implements position sizing strategies, and provides risk management.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.models.events import (
    CandleDataEvent, ErrorEvent, SignalEvent, SystemStatusEvent
)
from src.models.position import Position, PositionSide, PositionStatus


class PortfolioManager(Component):
    """Manager for the portfolio of positions.
    
    This component manages position creation, tracking, and closing based on
    signals from strategies and real-time market data.
    """
    
    def __init__(self):
        """Initialize the portfolio manager."""
        super().__init__("portfolio")
        self.logger = get_logger("portfolio", "manager")
        self.positions: Dict[str, Position] = {}
        self.active_symbols: Set[str] = set()
        self.current_prices: Dict[str, float] = {}
        
        # Configuration values
        self.max_positions = 0
        self.max_risk_per_trade = 0.0
        self.default_position_size = 0.0
        self.default_leverage = 1.0
        self.position_sizing_method = "fixed"  # fixed, risk-based, kelly
        
    async def _initialize(self) -> None:
        """Initialize the portfolio manager."""
        self.logger.info("Initializing portfolio manager")
        
        # Load configuration
        self.max_positions = self.get_config("max_positions", 5)
        self.max_risk_per_trade = self.get_config("max_risk_per_trade", 0.02)  # 2% max risk per trade
        self.default_position_size = self.get_config("default_position_size", 0.1)  # 10% of available capital
        self.default_leverage = self.get_config("default_leverage", 1.0)
        self.position_sizing_method = self.get_config("position_sizing_method", "fixed")
        
        self.logger.info("Portfolio configuration loaded", 
                        max_positions=self.max_positions,
                        max_risk_per_trade=self.max_risk_per_trade,
                        position_sizing_method=self.position_sizing_method)
    
    async def _start(self) -> None:
        """Start the portfolio manager."""
        self.logger.info("Starting portfolio manager")
        
        # Register event handlers
        event_bus.subscribe("SignalEvent", self._handle_signal_event)
        event_bus.subscribe("CandleDataEvent", self._handle_candle_event)
        
        # Start position monitoring task
        self.create_task(self._monitor_positions())
        
        self.logger.info("Portfolio manager started")
    
    async def _stop(self) -> None:
        """Stop the portfolio manager."""
        self.logger.info("Stopping portfolio manager")
        
        # Unregister event handlers
        event_bus.unsubscribe("SignalEvent", self._handle_signal_event)
        event_bus.unsubscribe("CandleDataEvent", self._handle_candle_event)
        
        # Close all open positions (in a real system, we might not want to do this automatically)
        if self.get_config("close_positions_on_shutdown", False):
            await self._close_all_positions("System shutdown")
        
        self.logger.info("Portfolio manager stopped")
    
    async def _handle_signal_event(self, event: SignalEvent) -> None:
        """Handle a signal event from a strategy.
        
        Args:
            event: The signal event
        """
        self.logger.debug("Received signal event", 
                         strategy=event.source,
                         symbol=event.symbol,
                         signal_type=event.signal_type,
                         confidence=event.confidence)
        
        # Check if we should act on this signal
        if event.confidence < self.get_config("min_signal_confidence", 0.7):
            self.logger.debug("Signal confidence below threshold", confidence=event.confidence)
            return
        
        # Check if we have the current price for this symbol
        symbol_key = f"{event.exchange}:{event.symbol}"
        if symbol_key not in self.current_prices:
            self.logger.warning("No current price for symbol", symbol=event.symbol)
            return
        
        current_price = self.current_prices[symbol_key]
        
        # Handle the signal based on its type
        if event.signal_type == "ENTRY_LONG":
            await self._open_position(
                event.exchange,
                event.symbol,
                PositionSide.LONG,
                current_price,
                event.stop_loss,
                event.take_profit,
                event.metadata.get("tags", []) + [f"strategy:{event.source}"]
            )
        elif event.signal_type == "ENTRY_SHORT":
            await self._open_position(
                event.exchange,
                event.symbol,
                PositionSide.SHORT,
                current_price,
                event.stop_loss,
                event.take_profit,
                event.metadata.get("tags", []) + [f"strategy:{event.source}"]
            )
        elif event.signal_type == "EXIT":
            await self._close_positions_for_symbol(
                event.exchange, 
                event.symbol, 
                f"Signal from {event.source}"
            )
    
    async def _handle_candle_event(self, event: CandleDataEvent) -> None:
        """Handle a candle event to update current prices.
        
        Args:
            event: The candle event
        """
        # Update the current price for this symbol
        symbol_key = f"{event.exchange}:{event.symbol}"
        self.current_prices[symbol_key] = event.close
        self.active_symbols.add(symbol_key)
    
    async def _monitor_positions(self) -> None:
        """Monitor open positions for stop loss and take profit triggers."""
        while True:
            try:
                for position_id, position in list(self.positions.items()):
                    if position.status != PositionStatus.OPEN:
                        continue
                    
                    symbol_key = f"{position.exchange}:{position.symbol}"
                    if symbol_key not in self.current_prices:
                        continue
                    
                    current_price = self.current_prices[symbol_key]
                    
                    # Check stop loss
                    if position.is_stop_loss_triggered(current_price):
                        self.logger.info("Stop loss triggered", 
                                        position_id=position.id,
                                        symbol=position.symbol,
                                        stop_loss=position.stop_loss,
                                        current_price=current_price)
                        
                        await self._close_position(
                            position_id, 
                            current_price, 
                            "Stop loss triggered"
                        )
                        continue
                    
                    # Check take profit
                    if position.is_take_profit_triggered(current_price):
                        self.logger.info("Take profit triggered", 
                                        position_id=position.id,
                                        symbol=position.symbol,
                                        take_profit=position.take_profit,
                                        current_price=current_price)
                        
                        await self._close_position(
                            position_id, 
                            current_price, 
                            "Take profit triggered"
                        )
                        continue
                
                # Sleep for a short interval
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in position monitoring", error=str(e))
                await asyncio.sleep(1)  # Sleep longer if there was an error
    
    async def _open_position(
        self, 
        exchange: str,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Position]:
        """Open a new position.
        
        Args:
            exchange: The exchange to use
            symbol: The trading pair symbol
            side: Long or short
            entry_price: The entry price
            stop_loss: The stop loss price (optional)
            take_profit: The take profit price (optional)
            tags: Tags for the position (optional)
            
        Returns:
            The new position, or None if it couldn't be created
        """
        # Check if we already have too many positions
        open_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        if len(open_positions) >= self.max_positions:
            self.logger.warning("Maximum number of positions reached", 
                              max_positions=self.max_positions)
            return None
        
        # Calculate position size
        position_size, leverage = self._calculate_position_size(
            exchange, symbol, side, entry_price, stop_loss
        )
        
        # Create the position
        position = Position(
            exchange=exchange,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            amount=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            tags=tags or []
        )
        
        # Mark the position as open
        position.open()
        
        # Store the position
        self.positions[position.id] = position
        
        self.logger.info("Opened new position", 
                       position_id=position.id,
                       symbol=symbol,
                       side=side.value,
                       entry_price=entry_price,
                       amount=position_size,
                       stop_loss=stop_loss,
                       take_profit=take_profit)
        
        # In a real system, we would place the order on the exchange here
        
        return position
    
    async def _close_position(
        self, 
        position_id: str,
        exit_price: float,
        reason: str
    ) -> bool:
        """Close an open position.
        
        Args:
            position_id: The ID of the position to close
            exit_price: The exit price
            reason: The reason for closing the position
            
        Returns:
            bool: True if the position was closed, False otherwise
        """
        if position_id not in self.positions:
            self.logger.warning("Position not found", position_id=position_id)
            return False
        
        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN:
            self.logger.warning("Position is not open", 
                              position_id=position_id, 
                              status=position.status.value)
            return False
        
        # Close the position
        position.close(exit_price)
        
        self.logger.info("Closed position", 
                       position_id=position.id,
                       symbol=position.symbol,
                       side=position.side.value,
                       entry_price=position.entry_price,
                       exit_price=exit_price,
                       realized_pnl=position.realized_pnl,
                       reason=reason)
        
        # In a real system, we would place the order on the exchange here
        
        return True
    
    async def _close_positions_for_symbol(
        self, 
        exchange: str,
        symbol: str,
        reason: str
    ) -> int:
        """Close all open positions for a symbol.
        
        Args:
            exchange: The exchange
            symbol: The trading pair symbol
            reason: The reason for closing the positions
            
        Returns:
            int: The number of positions closed
        """
        symbol_key = f"{exchange}:{symbol}"
        if symbol_key not in self.current_prices:
            self.logger.warning("No current price for symbol", symbol=symbol)
            return 0
        
        current_price = self.current_prices[symbol_key]
        
        # Find all open positions for this symbol
        positions_to_close = [
            p for p in self.positions.values() 
            if p.status == PositionStatus.OPEN 
            and p.exchange == exchange 
            and p.symbol == symbol
        ]
        
        # Close each position
        closed_count = 0
        for position in positions_to_close:
            if await self._close_position(position.id, current_price, reason):
                closed_count += 1
        
        return closed_count
    
    async def _close_all_positions(self, reason: str) -> int:
        """Close all open positions.
        
        Args:
            reason: The reason for closing the positions
            
        Returns:
            int: The number of positions closed
        """
        # Find all open positions
        positions_to_close = [
            p for p in self.positions.values() 
            if p.status == PositionStatus.OPEN
        ]
        
        # Close each position
        closed_count = 0
        for position in positions_to_close:
            symbol_key = f"{position.exchange}:{position.symbol}"
            
            # If we don't have the current price, use the entry price (not ideal but safer than guessing)
            current_price = self.current_prices.get(symbol_key, position.entry_price)
            
            if await self._close_position(position.id, current_price, reason):
                closed_count += 1
        
        return closed_count
    
    def _calculate_position_size(
        self, 
        exchange: str,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        stop_loss: Optional[float]
    ) -> Tuple[float, float]:
        """Calculate the position size and leverage based on risk parameters.
        
        Args:
            exchange: The exchange
            symbol: The trading pair symbol
            side: Long or short
            entry_price: The entry price
            stop_loss: The stop loss price (optional)
            
        Returns:
            Tuple containing (position_size, leverage)
        """
        # This is a simplified implementation
        # In a real system, we would calculate this based on account balance,
        # risk preferences, volatility, and other factors
        
        leverage = self.default_leverage
        
        if self.position_sizing_method == "fixed":
            # Use a fixed position size
            position_size = self.default_position_size
            
        elif self.position_sizing_method == "risk-based" and stop_loss is not None:
            # Calculate position size based on risk
            risk_amount = self.get_config("risk_capital", 1000.0) * self.max_risk_per_trade
            
            if side == PositionSide.LONG:
                price_risk = entry_price - stop_loss
            else:
                price_risk = stop_loss - entry_price
                
            if price_risk <= 0:
                # Invalid stop loss
                position_size = self.default_position_size
            else:
                position_size = risk_amount / price_risk
        else:
            # Default to fixed position size
            position_size = self.default_position_size
        
        return position_size, leverage
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions.
        
        Returns:
            List of open positions
        """
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get a position by its ID.
        
        Args:
            position_id: The position ID
            
        Returns:
            The position, or None if not found
        """
        return self.positions.get(position_id)
    
    def calculate_portfolio_stats(self) -> Dict:
        """Calculate statistics for the portfolio.
        
        Returns:
            Dictionary with portfolio statistics
        """
        open_positions = self.get_open_positions()
        closed_positions = [p for p in self.positions.values() if p.status == PositionStatus.CLOSED]
        
        # Calculate total PnL
        total_realized_pnl = sum(p.realized_pnl or 0.0 for p in closed_positions)
        
        # Calculate total unrealized PnL
        total_unrealized_pnl = 0.0
        for position in open_positions:
            symbol_key = f"{position.exchange}:{position.symbol}"
            if symbol_key in self.current_prices:
                total_unrealized_pnl += position.calculate_unrealized_pnl(self.current_prices[symbol_key])
        
        # Calculate win rate
        winning_trades = sum(1 for p in closed_positions if (p.realized_pnl or 0.0) > 0)
        win_rate = winning_trades / len(closed_positions) if closed_positions else 0.0
        
        # Calculate other metrics
        avg_win = 0.0
        avg_loss = 0.0
        
        # Filter out None values before summing
        wins = [p.realized_pnl for p in closed_positions if (p.realized_pnl or 0.0) > 0]
        filtered_wins = [w for w in wins if w is not None]
        
        losses = [p.realized_pnl for p in closed_positions if (p.realized_pnl or 0.0) < 0]
        filtered_losses = [l for l in losses if l is not None]
        
        if filtered_wins:
            avg_win = sum(filtered_wins) / len(filtered_wins)
        
        if filtered_losses:
            avg_loss = sum(filtered_losses) / len(filtered_losses)
        
        profit_factor = abs(sum(filtered_wins)) / abs(sum(filtered_losses)) if filtered_losses and sum(filtered_losses) != 0 else 0.0
        
        return {
            "open_positions": len(open_positions),
            "closed_positions": len(closed_positions),
            "total_realized_pnl": total_realized_pnl,
            "total_unrealized_pnl": total_unrealized_pnl,
            "win_rate": win_rate * 100.0,  # as percentage
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "timestamp": datetime.utcnow().isoformat()
        } 