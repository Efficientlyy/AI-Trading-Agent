"""
Multi-Broker Execution Manager

Provides a unified execution layer that can manage multiple brokers, intelligently
route orders, and handle failover between execution venues.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from datetime import datetime
import uuid

from ..common.enums import OrderType, OrderSide, OrderStatus, TimeInForce
from ..common.models import Order, Trade, Position, Balance, Portfolio
from ..common.enhanced_circuit_breaker import EnhancedCircuitBreaker
from ..common.error_handling import TradingAgentError, ErrorCode
from .broker_interface import BrokerInterface

logger = logging.getLogger(__name__)


class ExecutionRouterMode(Enum):
    """Execution router modes for the multi-broker manager."""
    SINGLE = "single"          # Use only one broker
    FAILOVER = "failover"      # Failover to backup broker on failure
    COST_OPTIMIZED = "cost"    # Route based on best execution cost
    LIQUIDITY = "liquidity"    # Route based on available liquidity
    PARALLEL = "parallel"      # Parallel execution across brokers


class ExecutionManager:
    """
    Multi-broker execution manager that provides unified execution across
    multiple brokers with intelligent routing, failover, and position tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the execution manager.
        
        Args:
            config: Configuration dictionary containing broker settings and routing
                   preferences.
        """
        self.name = "ExecutionManager"
        
        # Router configuration
        self.router_mode = ExecutionRouterMode(config.get('router_mode', 'failover'))
        self.primary_broker = config.get('primary_broker', 'paper')
        
        # Broker configurations
        self.broker_configs = config.get('brokers', {})
        
        # Initialize broker instances
        self.brokers: Dict[str, BrokerInterface] = {}
        
        # Broker health status
        self.broker_health: Dict[str, bool] = {}
        
        # Position consolidation settings
        self.consolidate_positions = config.get('consolidate_positions', True)
        
        # Route optimization preferences
        self.cost_weights = {
            'fee': config.get('cost_weights', {}).get('fee', 0.7),
            'slippage': config.get('cost_weights', {}).get('slippage', 0.3)
        }
        
        # Global portfolio view
        self.global_portfolio = Portfolio(
            balances={},
            positions={},
            order_history=[],
            trade_history=[],
            realized_pnl=0.0,
            unrealized_pnl=0.0
        )
        
        # Order to broker mapping for tracking
        self.order_broker_map: Dict[str, str] = {}
        
        # Order execution metrics for optimization
        self.execution_metrics: Dict[str, Dict[str, float]] = {}  # broker -> metric -> value
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Order tracking across brokers
        self.all_orders: Dict[str, Order] = {}
        
        logger.info(f"Initialized {self.name} with router mode: {self.router_mode.value}")
    
    async def initialize(self) -> bool:
        """
        Initialize connections to all configured brokers.
        
        Returns:
            True if at least one broker was successfully initialized
        """
        initialized_count = 0
        
        for broker_id, broker_config in self.broker_configs.items():
            try:
                broker_type = broker_config.get('type', 'paper')
                
                # Import the right broker implementation
                if broker_type == 'paper':
                    from .paper_trading_manager import PaperTradingManager
                    broker = PaperTradingManager(broker_config)
                elif broker_type == 'binance':
                    from .binance_broker import BinanceBroker
                    broker = BinanceBroker(broker_config)
                else:
                    logger.error(f"Unknown broker type: {broker_type}")
                    continue
                
                # Initialize the broker
                success = await broker.initialize()
                if success:
                    self.brokers[broker_id] = broker
                    self.broker_health[broker_id] = True
                    initialized_count += 1
                    logger.info(f"Successfully initialized broker: {broker_id}")
                else:
                    logger.error(f"Failed to initialize broker: {broker_id}")
                    self.broker_health[broker_id] = False
                    
            except Exception as e:
                logger.error(f"Error initializing broker {broker_id}: {str(e)}")
                self.broker_health[broker_id] = False
        
        # Check if we have a working broker
        if initialized_count == 0:
            logger.error("Failed to initialize any brokers")
            return False
            
        # Start health monitoring task
        asyncio.create_task(self._health_monitor())
        
        # Refresh global portfolio view
        await self.refresh_global_portfolio()
        
        logger.info(f"Execution manager initialized with {initialized_count} brokers")
        return True
    
    async def _health_monitor(self) -> None:
        """
        Continuously monitor broker health and update status.
        """
        while True:
            try:
                for broker_id, broker in self.brokers.items():
                    try:
                        # Check if broker is responsive
                        # Try to get a simple balance or ping
                        balances = await broker.get_balances()
                        if balances is not None:
                            self.broker_health[broker_id] = True
                        else:
                            logger.warning(f"Broker {broker_id} health check failed")
                            self.broker_health[broker_id] = False
                            
                    except Exception as e:
                        logger.error(f"Error checking broker {broker_id} health: {str(e)}")
                        self.broker_health[broker_id] = False
                
                # Log overall health status
                healthy_brokers = sum(1 for status in self.broker_health.values() if status)
                logger.debug(f"Broker health status: {healthy_brokers}/{len(self.broker_health)} brokers healthy")
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {str(e)}")
                
            # Sleep between checks
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def place_order(self, order: Order) -> Tuple[bool, str, Optional[str]]:
        """
        Place an order through the execution manager, which will route to
        the appropriate broker based on the routing mode.
        
        Args:
            order: Order to place
            
        Returns:
            Tuple of (success, message, order_id)
        """
        async with self.lock:
            # Generate order ID if not provided
            if not order.order_id:
                order.order_id = str(uuid.uuid4())
            
            # Determine which broker to use based on router mode
            broker_id = await self._select_broker_for_order(order)
            
            if not broker_id:
                return False, "No suitable broker available", None
                
            broker = self.brokers[broker_id]
            
            try:
                # Place the order with the selected broker
                success, message, broker_order_id = await broker.place_order(order)
                
                if success and broker_order_id:
                    # Update internal tracking
                    self.order_broker_map[order.order_id] = broker_id
                    
                    # Store order with updated ID from broker
                    order_copy = Order(
                        order_id=broker_order_id,
                        symbol=order.symbol,
                        type=order.type,
                        side=order.side,
                        quantity=order.quantity,
                        price=order.price,
                        status=OrderStatus.OPEN,
                        time_in_force=order.time_in_force,
                        created_at=datetime.now().isoformat()
                    )
                    self.all_orders[broker_order_id] = order_copy
                    
                    logger.info(f"Order {broker_order_id} placed via broker {broker_id}")
                    return True, f"Order placed via {broker_id}", broker_order_id
                else:
                    if self.router_mode == ExecutionRouterMode.FAILOVER:
                        # Try the next available broker
                        logger.warning(f"Failover: Primary order placement failed, trying alternative broker")
                        return await self._failover_place_order(order, exclude_broker=broker_id)
                    
                    logger.error(f"Failed to place order via {broker_id}: {message}")
                    return False, message, None
                    
            except Exception as e:
                logger.error(f"Error placing order via {broker_id}: {str(e)}")
                if self.router_mode == ExecutionRouterMode.FAILOVER:
                    # Try the next available broker
                    logger.warning(f"Failover: Primary order placement failed with exception, trying alternative broker")
                    return await self._failover_place_order(order, exclude_broker=broker_id)
                
                return False, f"Error placing order: {str(e)}", None
    
    async def _failover_place_order(self, order: Order, exclude_broker: str) -> Tuple[bool, str, Optional[str]]:
        """
        Attempt to place an order with an alternative broker after primary fails.
        
        Args:
            order: Order to place
            exclude_broker: Broker ID to exclude (failed broker)
            
        Returns:
            Same as place_order
        """
        # Find alternative healthy broker
        for broker_id, healthy in self.broker_health.items():
            if broker_id != exclude_broker and healthy:
                broker = self.brokers[broker_id]
                try:
                    # Place the order with the alternative broker
                    success, message, broker_order_id = await broker.place_order(order)
                    
                    if success and broker_order_id:
                        # Update internal tracking
                        self.order_broker_map[order.order_id] = broker_id
                        
                        # Store order
                        order_copy = Order(
                            order_id=broker_order_id,
                            symbol=order.symbol,
                            type=order.type,
                            side=order.side,
                            quantity=order.quantity,
                            price=order.price,
                            status=OrderStatus.OPEN,
                            time_in_force=order.time_in_force,
                            created_at=datetime.now().isoformat()
                        )
                        self.all_orders[broker_order_id] = order_copy
                        
                        logger.info(f"Failover success: Order {broker_order_id} placed via broker {broker_id}")
                        return True, f"Order placed via {broker_id} (failover)", broker_order_id
                except Exception as e:
                    logger.error(f"Failover error with broker {broker_id}: {str(e)}")
                    continue
        
        # If we get here, all failover attempts failed
        logger.error("All failover attempts failed")
        return False, "All broker placement attempts failed", None
    
    async def _select_broker_for_order(self, order: Order) -> Optional[str]:
        """
        Select the appropriate broker for an order based on the routing mode.
        
        Args:
            order: Order to place
            
        Returns:
            Selected broker ID or None if no suitable broker found
        """
        # Check for healthy brokers
        healthy_brokers = [broker_id for broker_id, status in self.broker_health.items() if status]
        if not healthy_brokers:
            logger.error("No healthy brokers available")
            return None
        
        # Different routing strategies
        if self.router_mode == ExecutionRouterMode.SINGLE:
            # Use primary broker if healthy, otherwise any healthy broker
            if self.primary_broker in healthy_brokers:
                return self.primary_broker
            return healthy_brokers[0]
            
        elif self.router_mode == ExecutionRouterMode.FAILOVER:
            # Start with primary broker
            if self.primary_broker in healthy_brokers:
                return self.primary_broker
            # Failover to first healthy broker
            return healthy_brokers[0]
            
        elif self.router_mode == ExecutionRouterMode.COST_OPTIMIZED:
            # Route based on cost optimization
            return await self._select_cost_optimized_broker(order, healthy_brokers)
            
        elif self.router_mode == ExecutionRouterMode.LIQUIDITY:
            # Route based on liquidity
            return await self._select_liquidity_optimized_broker(order, healthy_brokers)
            
        elif self.router_mode == ExecutionRouterMode.PARALLEL:
            # Parallel execution not implemented for order placement (only for market data)
            # Default to primary/first healthy
            if self.primary_broker in healthy_brokers:
                return self.primary_broker
            return healthy_brokers[0]
            
        # Default case
        if self.primary_broker in healthy_brokers:
            return self.primary_broker
        return healthy_brokers[0] if healthy_brokers else None
    
    async def _select_cost_optimized_broker(self, order: Order, healthy_brokers: List[str]) -> str:
        """
        Select the broker with the lowest expected execution cost.
        
        Args:
            order: Order to place
            healthy_brokers: List of healthy broker IDs
            
        Returns:
            Broker ID with lowest expected cost
        """
        best_broker = None
        lowest_cost = float('inf')
        
        for broker_id in healthy_brokers:
            broker = self.brokers[broker_id]
            
            # Get fee rate
            fee_rate = getattr(broker, 'fee_rate', 0.001)  # Default to 0.1%
            
            # Get historical slippage for this broker/symbol
            slippage = self.execution_metrics.get(broker_id, {}).get(f"slippage_{order.symbol}", 0.001)
            
            # Calculate weighted cost
            cost = (fee_rate * self.cost_weights['fee'] + 
                   slippage * self.cost_weights['slippage'])
            
            if cost < lowest_cost:
                lowest_cost = cost
                best_broker = broker_id
        
        return best_broker or healthy_brokers[0]
    
    async def _select_liquidity_optimized_broker(self, order: Order, healthy_brokers: List[str]) -> str:
        """
        Select the broker with the best liquidity for the order.
        
        Args:
            order: Order to place
            healthy_brokers: List of healthy broker IDs
            
        Returns:
            Broker ID with best liquidity
        """
        # This would ideally check order book depth, but simplified version here
        # Just uses execution metrics for now
        best_broker = None
        best_score = -float('inf')
        
        for broker_id in healthy_brokers:
            # Get success rate for this broker/symbol
            success_rate = self.execution_metrics.get(broker_id, {}).get(f"success_rate_{order.symbol}", 0.9)
            
            # Get average fill time
            fill_time = self.execution_metrics.get(broker_id, {}).get(f"fill_time_{order.symbol}", 5.0)
            
            # Higher success rate and lower fill time is better
            score = success_rate - (fill_time / 60.0)  # Normalize fill time to ~1 minute max
            
            if score > best_score:
                best_score = score
                best_broker = broker_id
        
        return best_broker or healthy_brokers[0]
    
    async def cancel_order(self, order_id: str, symbol: str) -> Tuple[bool, str]:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Symbol for the order
            
        Returns:
            Tuple of (success, message)
        """
        async with self.lock:
            # Find which broker has this order
            broker_id = self.order_broker_map.get(order_id)
            
            if not broker_id:
                # Try to find the order in all brokers if not tracked
                broker_id = await self._find_broker_for_order(order_id, symbol)
            
            if not broker_id:
                return False, f"Order {order_id} not found in any broker"
                
            if broker_id not in self.brokers:
                return False, f"Broker {broker_id} not available"
                
            broker = self.brokers[broker_id]
            
            try:
                # Cancel the order
                success, message = await broker.cancel_order(order_id, symbol)
                
                if success:
                    logger.info(f"Order {order_id} cancelled via broker {broker_id}")
                    
                    # Update order status in tracking
                    if order_id in self.all_orders:
                        self.all_orders[order_id].status = OrderStatus.CANCELLED
                    
                    return True, message
                else:
                    logger.error(f"Failed to cancel order {order_id} via broker {broker_id}: {message}")
                    return False, message
                    
            except Exception as e:
                logger.error(f"Error cancelling order {order_id} via broker {broker_id}: {str(e)}")
                return False, f"Error cancelling order: {str(e)}"
    
    async def _find_broker_for_order(self, order_id: str, symbol: str) -> Optional[str]:
        """
        Find which broker has a specific order.
        
        Args:
            order_id: Order ID to find
            symbol: Symbol for the order
            
        Returns:
            Broker ID or None if not found
        """
        for broker_id, broker in self.brokers.items():
            try:
                order = await broker.get_order(order_id, symbol)
                if order:
                    # Cache the mapping for future reference
                    self.order_broker_map[order_id] = broker_id
                    return broker_id
            except Exception as e:
                logger.debug(f"Error checking for order {order_id} in broker {broker_id}: {str(e)}")
                continue
        
        return None
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Get order details by ID.
        
        Args:
            order_id: Order ID to retrieve
            symbol: Symbol for the order
            
        Returns:
            Order object if found, None otherwise
        """
        # Check our cached orders first
        if order_id in self.all_orders:
            return self.all_orders[order_id]
        
        # Find which broker has this order
        broker_id = self.order_broker_map.get(order_id)
        
        if not broker_id:
            # Try to find the order in all brokers if not tracked
            broker_id = await self._find_broker_for_order(order_id, symbol)
        
        if not broker_id:
            return None
            
        if broker_id not in self.brokers:
            return None
            
        broker = self.brokers[broker_id]
        
        try:
            # Get the order
            order = await broker.get_order(order_id, symbol)
            
            if order:
                # Cache the order
                self.all_orders[order_id] = order
            
            return order
                
        except Exception as e:
            logger.error(f"Error retrieving order {order_id} from broker {broker_id}: {str(e)}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders across all brokers.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of open orders
        """
        all_open_orders = []
        
        for broker_id, broker in self.brokers.items():
            try:
                # Only query healthy brokers
                if not self.broker_health.get(broker_id, False):
                    continue
                    
                # Get open orders from this broker
                orders = await broker.get_open_orders(symbol)
                
                if orders:
                    # Cache orders and broker mapping
                    for order in orders:
                        self.all_orders[order.order_id] = order
                        self.order_broker_map[order.order_id] = broker_id
                    
                    all_open_orders.extend(orders)
                    
            except Exception as e:
                logger.error(f"Error retrieving open orders from broker {broker_id}: {str(e)}")
                continue
        
        return all_open_orders
    
    async def refresh_global_portfolio(self) -> None:
        """
        Refresh the global portfolio view by consolidating data from all brokers.
        """
        async with self.lock:
            # Reset global portfolio
            self.global_portfolio = Portfolio(
                balances={},
                positions={},
                order_history=[],
                trade_history=[],
                realized_pnl=0.0,
                unrealized_pnl=0.0
            )
            
            for broker_id, broker in self.brokers.items():
                try:
                    # Only query healthy brokers
                    if not self.broker_health.get(broker_id, False):
                        continue
                        
                    # Get portfolio from this broker
                    portfolio = await broker.get_portfolio()
                    
                    if not portfolio:
                        continue
                    
                    # Consolidate balances
                    for currency, balance in portfolio.balances.items():
                        if currency in self.global_portfolio.balances:
                            # Add to existing balance
                            existing = self.global_portfolio.balances[currency]
                            existing.free += balance.free
                            existing.locked += balance.locked
                        else:
                            # New currency
                            self.global_portfolio.balances[currency] = Balance(
                                currency=currency,
                                free=balance.free,
                                locked=balance.locked
                            )
                    
                    # Consolidate positions
                    for symbol, position in portfolio.positions.items():
                        if symbol in self.global_portfolio.positions:
                            # Add to existing position
                            existing = self.global_portfolio.positions[symbol]
                            total_quantity = existing.quantity + position.quantity
                            
                            # Calculate weighted average entry price
                            if total_quantity > 0:
                                existing.entry_price = (
                                    (existing.entry_price * existing.quantity + 
                                     position.entry_price * position.quantity) / total_quantity
                                )
                            
                            existing.quantity = total_quantity
                            existing.current_price = position.current_price  # Use latest price
                            existing.unrealized_pnl += position.unrealized_pnl
                        else:
                            # New position
                            self.global_portfolio.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=position.quantity,
                                entry_price=position.entry_price,
                                current_price=position.current_price,
                                unrealized_pnl=position.unrealized_pnl
                            )
                    
                    # Track PnL
                    self.global_portfolio.realized_pnl += portfolio.realized_pnl
                    self.global_portfolio.unrealized_pnl += portfolio.unrealized_pnl
                    
                except Exception as e:
                    logger.error(f"Error retrieving portfolio from broker {broker_id}: {str(e)}")
                    continue
            
            logger.info(f"Global portfolio refreshed: {len(self.global_portfolio.balances)} currencies, "
                      f"{len(self.global_portfolio.positions)} positions")
    
    async def get_portfolio(self) -> Portfolio:
        """
        Get the global portfolio consolidating all brokers.
        
        Returns:
            Portfolio object
        """
        await self.refresh_global_portfolio()
        return self.global_portfolio
    
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol from any available broker.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if unavailable
        """
        # Try primary broker first
        if self.primary_broker in self.brokers:
            broker = self.brokers[self.primary_broker]
            try:
                price = await broker.get_market_price(symbol)
                if price:
                    return price
            except Exception:
                pass
        
        # Try all other brokers
        for broker_id, broker in self.brokers.items():
            if broker_id == self.primary_broker:
                continue
                
            try:
                price = await broker.get_market_price(symbol)
                if price:
                    return price
            except Exception:
                continue
        
        return None
    
    async def close(self) -> None:
        """
        Close all broker connections and clean up resources.
        """
        for broker_id, broker in self.brokers.items():
            try:
                await broker.close()
                logger.info(f"Closed broker {broker_id}")
            except Exception as e:
                logger.error(f"Error closing broker {broker_id}: {str(e)}")
        
        # Clear internal state
        self.brokers = {}
        self.broker_health = {}
        self.order_broker_map = {}
        self.all_orders = {}
        
        logger.info("Execution manager closed")


# Example usage:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'router_mode': 'failover',
        'primary_broker': 'paper',
        'brokers': {
            'paper': {
                'type': 'paper',
                'initial_balance': {'USDT': 10000.0},
                'fee_rate': 0.001,
                'slippage': 0.0005
            },
            'binance_test': {
                'type': 'binance',
                'api_key': 'your_test_api_key',
                'api_secret': 'your_test_api_secret',
                'testnet': True
            }
        }
    }
    
    async def test_execution_manager():
        # Create execution manager
        manager = ExecutionManager(config)
        
        # Initialize
        success = await manager.initialize()
        print(f"Initialization: {'successful' if success else 'failed'}")
        
        if success:
            # Get portfolio
            portfolio = await manager.get_portfolio()
            print(f"Global portfolio balances: {portfolio.balances}")
            
            # Place a test order
            test_order = Order(
                order_id="",  # Will be generated
                symbol="BTC/USDT",
                type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=0.01,
                price=None,
                status=OrderStatus.NEW,
                time_in_force=TimeInForce.GTC
            )
            
            success, message, order_id = await manager.place_order(test_order)
            print(f"Order placement: {success}, {message}, Order ID: {order_id}")
            
            # Close
            await manager.close()
    
    # Run the test
    asyncio.run(test_execution_manager())
