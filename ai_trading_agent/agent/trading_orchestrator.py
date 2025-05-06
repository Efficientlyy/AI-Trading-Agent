"""
Trading orchestrator module for the AI Trading Agent.

This module coordinates the interaction between different components of the trading system.
"""

from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import threading
import asyncio
from copy import deepcopy

from ..common import logger
from ..common.event_emitter import global_event_emitter
from ..common.error_handling import (
    TradingAgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    retry,
    circuit_breaker,
    error_handler,
    log_error
)
from ..common.circuit_breaker_executor import CircuitBreakerExecutor


class TradingOrchestrator:
    """
    Coordinates the flow of data and actions between all components of the trading system.
    
    This class is responsible for:
    1. Managing the trading loop (backtest or live)
    2. Passing data between components
    3. Executing the trading logic
    4. Collecting and reporting results
    """
    
    def __init__(self, data_manager, strategy_manager, risk_manager, 
                portfolio_manager, execution_handler):
        """
        Initialize the trading orchestrator.
        
        Args:
            data_manager: Data manager instance
            strategy_manager: Strategy manager instance
            risk_manager: Risk manager instance
            portfolio_manager: Portfolio manager instance
            execution_handler: Execution handler instance
        """
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.execution_handler = execution_handler
        
        self._running = False
        self._stop_requested = False
        self._trading_thread = None
        self._start_time = None
        
        self.results = {
            'trades': [],
            'portfolio_history': [],
            'performance_metrics': {}
        }
        
        # Initialize circuit breaker executors for different components
        self.data_executor = CircuitBreakerExecutor(
            name="data_provider",
            failure_threshold=3,
            reset_timeout=60.0,
            retry_attempts=3,
            retry_delay=1.0,
            retry_backoff_factor=2.0,
            on_circuit_open=self._on_data_circuit_open,
            on_circuit_close=self._on_data_circuit_close
        )
        
        self.strategy_executor = CircuitBreakerExecutor(
            name="strategy",
            failure_threshold=3,
            reset_timeout=30.0,
            retry_attempts=2,
            retry_delay=0.5,
            retry_backoff_factor=2.0,
            on_circuit_open=self._on_strategy_circuit_open,
            on_circuit_close=self._on_strategy_circuit_close
        )
        
        self.execution_executor = CircuitBreakerExecutor(
            name="execution",
            failure_threshold=3,
            reset_timeout=60.0,
            retry_attempts=3,
            retry_delay=1.0,
            retry_backoff_factor=2.0,
            on_circuit_open=self._on_execution_circuit_open,
            on_circuit_close=self._on_execution_circuit_close
        )
        
        # Error handling state
        self.error_state = {
            "data_provider": {
                "errors": [],
                "last_error_time": 0,
                "recovery_attempts": 0
            },
            "strategy": {
                "errors": [],
                "last_error_time": 0,
                "recovery_attempts": 0
            },
            "execution": {
                "errors": [],
                "last_error_time": 0,
                "recovery_attempts": 0
            }
        }
        
        logger.info("TradingOrchestrator initialized with error handling capabilities")
        
    def get_uptime_seconds(self) -> int:
        """
        Get the number of seconds the orchestrator has been running.
        
        Returns:
            Uptime in seconds
        """
        if not hasattr(self, '_start_time') or self._start_time is None:
            return 0
        
        return int(time.time() - self._start_time)
        
    def get_symbols(self) -> List[str]:
        """
        Get the list of symbols being traded.
        
        Returns:
            List of symbols
        """
        if self.data_manager:
            try:
                return self.data_manager.get_symbols()
            except Exception as e:
                logger.error(f"Error getting symbols: {e}", exc_info=True)
                return []
        return []
        
    def get_recent_trades(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent trades.
        
        Args:
            count: Number of recent trades to return
            
        Returns:
            List of recent trades
        """
        if not self.results or 'trades' not in self.results:
            return []
            
        trades = self.results.get('trades', [])
        return trades[-count:] if trades else []
        
    def get_current_portfolio(self) -> Optional[Dict[str, Any]]:
        """
        Get the current portfolio.
        
        Returns:
            Current portfolio or None if not available
        """
        if not self.results or 'portfolio_history' not in self.results:
            return None
            
        portfolio_history = self.results.get('portfolio_history', [])
        return portfolio_history[-1] if portfolio_history else None
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get the current performance metrics.
        
        Returns:
            Performance metrics
        """
        if not self.results or 'performance_metrics' not in self.results:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
            
        return self.results.get('performance_metrics', {})
        
    async def cleanup(self) -> None:
        """
        Clean up resources used by the orchestrator.
        
        This method should be called when the orchestrator is no longer needed.
        """
        self._running = False
        self._stop_requested = True
        
        # Clean up data manager
        if self.data_manager:
            try:
                if hasattr(self.data_manager, 'cleanup'):
                    await self.data_manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up data manager: {e}", exc_info=True)
                
        # Clean up execution handler
        if self.execution_handler:
            try:
                if hasattr(self.execution_handler, 'cleanup'):
                    await self.execution_handler.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up execution handler: {e}", exc_info=True)
                
        logger.info("TradingOrchestrator cleaned up")
    
    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Run a backtest from start_date to end_date.
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Reset results
        self.results = {
            'trades': [],
            'portfolio_history': [],
            'performance_metrics': {}
        }
        
        # Set running flag
        self._running = True
        self._stop_requested = False
        
        # Get all dates in the backtest period
        dates = self.data_manager.get_dates_in_range(start_date, end_date)
        
        if not dates:
            logger.warning("No dates available in the specified range")
            self._running = False
            return self.results
        
        # Initialize portfolio
        initial_portfolio = {
            'total_value': self.portfolio_manager.initial_capital,
            'cash': self.portfolio_manager.initial_capital,
            'positions': {},
            'position_values': {}
        }
        
        current_portfolio = deepcopy(initial_portfolio)
        
        # Run backtest for each date
        for current_date in dates:
            if self._stop_requested:
                logger.info("Backtest stopped by user request")
                break
            
            logger.info(f"Processing date: {current_date}")
            
            # Get current market data
            current_data = self.data_manager.get_current_data(current_date)
            
            if not current_data:
                logger.warning(f"No data available for {current_date}")
                continue
            
            # Get historical data
            historical_data = self.data_manager.get_historical_data(
                end_date=current_date,
                lookback_periods=100  # Use last 100 periods
            )
            
            # Generate signals
            signals = self.strategy_manager.process_data_and_generate_signals(
                current_data=current_data,
                historical_data=historical_data,
                current_portfolio=current_portfolio
            )
            
            # Apply risk management
            risk_adjusted_signals = self.risk_manager.apply_risk_constraints(
                signals=signals,
                current_portfolio=current_portfolio,
                market_data=current_data
            )
            
            # Calculate position sizes
            target_positions = self.portfolio_manager.calculate_position_sizes(
                signals=risk_adjusted_signals,
                current_portfolio=current_portfolio,
                market_data=current_data
            )
            
            # Get current positions
            current_positions = current_portfolio.get('positions', {})
            
            # Calculate position adjustments
            position_adjustments = self.portfolio_manager.rebalance_portfolio(
                current_positions=current_positions,
                target_positions=target_positions
            )
            
            # Generate orders from position adjustments
            orders = []
            for symbol, adjustment in position_adjustments.items():
                if adjustment != 0:
                    # Create order
                    from ai_trading_agent.agent.execution_handler_new import Order
                    order = Order(
                        symbol=symbol,
                        quantity=adjustment
                    )
                    orders.append(order)
            
            # Execute orders
            executed_orders = self.execution_handler.submit_orders(
                orders=orders,
                market_data=current_data
            )
            
            # Update portfolio
            for order in executed_orders:
                if order.status == 'FILLED':
                    # Update positions
                    symbol = order.symbol
                    quantity = order.executed_quantity
                    price = order.executed_price
                    commission = order.commission
                    
                    # Update current positions
                    if symbol not in current_positions:
                        current_positions[symbol] = 0
                    
                    current_positions[symbol] += quantity
                    
                    # If position is now zero, remove it
                    if current_positions[symbol] == 0:
                        del current_positions[symbol]
                    
                    # Update cash
                    current_portfolio['cash'] -= (quantity * price + commission)
                    
                    # Record trade
                    trade = {
                        'timestamp': current_date,
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'commission': commission,
                        'order_id': order.order_id,
                        'pnl': 0.0  # Will be updated later
                    }
                    
                    self.results['trades'].append(trade)
            
            # Update portfolio value
            prices = {symbol: data.get('close', 0.0) if isinstance(data, dict) else data['close'] 
                     for symbol, data in current_data.items()}
            
            portfolio_value = self.portfolio_manager.update_portfolio_value(
                positions=current_positions,
                prices=prices,
                timestamp=current_date
            )
            
            # Update current portfolio
            current_portfolio = self.portfolio_manager.get_current_portfolio()
            
            # Add to portfolio history
            self.results['portfolio_history'].append(current_portfolio)
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Set running flag
        self._running = False
        
        logger.info("Backtest completed")
        
        return self.results
    
    async def run_paper_trading(self, duration: timedelta, update_interval: timedelta, stop_event=None) -> Dict[str, Any]:
        """
        Run paper trading for a specified duration.
        
        Args:
            duration: Duration to run paper trading
            update_interval: Interval between updates
            stop_event: Optional asyncio.Event to signal stopping
        
        Returns:
            Dictionary with paper trading results
        """
        import asyncio
        
        logger.info(f"Starting paper trading for {duration} with {update_interval} updates")
        
        # Reset results
        self.results = {
            'trades': [],
            'portfolio_history': [],
            'performance_metrics': {}
        }
        
        # Set running flag and start time
        self._running = True
        self._stop_requested = False
        self._start_time = time.time()
        
        # Emit status update event
        global_event_emitter.emit('paper_trading_status', {
            'status': 'running',
            'uptime_seconds': 0,
            'symbols': [],
            'current_portfolio': None,
            'recent_trades': [],
            'performance_metrics': None
        })
        
        # Calculate end time
        start_time = datetime.now()
        end_time = start_time + duration
        
        # Initialize portfolio
        initial_portfolio = {
            'total_value': self.portfolio_manager.initial_capital,
            'cash': self.portfolio_manager.initial_capital,
            'positions': {},
            'position_values': {},
            'timestamp': start_time.isoformat()
        }
        
        current_portfolio = deepcopy(initial_portfolio)
        self.results['portfolio_history'].append(current_portfolio)
        
        # Get available symbols using circuit breaker
        try:
            symbols = self.data_executor.execute(
                self.data_manager.get_symbols,
                retry_on_exceptions=Exception
            )
            if not symbols:
                logger.warning("No symbols available for paper trading")
                symbols = ["BTC/USDT", "ETH/USDT"]  # Default symbols if none available
                
                # Emit error event
                global_event_emitter.emit('paper_trading_error', {
                    'error_category': 'data_provider',
                    'error_code': ErrorCode.DATA_PROVIDER_SYMBOL_ERROR.value,
                    'message': 'No symbols available for paper trading. Using default symbols.',
                    'severity': ErrorSeverity.WARNING.value,
                    'timestamp': time.time(),
                    'details': {
                        'default_symbols': symbols
                    }
                })
        except TradingAgentError as e:
            logger.error(f"Error getting symbols: {e}", exc_info=True)
            symbols = ["BTC/USDT", "ETH/USDT"]  # Default symbols if error
            
            # Emit error event
            global_event_emitter.emit('paper_trading_error', {
                'error_category': e.error_category.value,
                'error_code': e.error_code.value,
                'message': f"Error getting symbols: {e.message}",
                'severity': e.severity.value,
                'timestamp': time.time(),
                'details': e.details,
                'troubleshooting': e.troubleshooting
            })
            
        logger.info(f"Paper trading with symbols: {symbols}")
        
        # Main paper trading loop
        current_time = start_time
        try:
            while current_time < end_time and not self._stop_requested:
                if stop_event and stop_event.is_set():
                    logger.info("Stop event received, ending paper trading")
                    break
                
                logger.info(f"Processing time: {current_time}")
                
                # Get current market data for all symbols using circuit breaker
                current_data = {}
                for symbol in symbols:
                    try:
                        # Define an async wrapper for the circuit breaker executor
                        async def get_data_with_circuit_breaker():
                            return await self.data_manager.get_latest_data(symbol)
                            
                        # Fetch latest market data with circuit breaker
                        latest_data = await self.data_executor.execute_async(
                            get_data_with_circuit_breaker,
                            retry_on_exceptions=Exception
                        )
                        
                        if latest_data:
                            current_data[symbol] = latest_data
                    except TradingAgentError as e:
                        logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
                        
                        # Emit error event
                        global_event_emitter.emit('paper_trading_error', {
                            'error_category': e.error_category.value,
                            'error_code': e.error_code.value,
                            'message': f"Error fetching data for {symbol}: {e.message}",
                            'severity': e.severity.value,
                            'timestamp': time.time(),
                            'details': e.details,
                            'troubleshooting': e.troubleshooting
                        })
                
                if not current_data:
                    logger.warning("No data available")
                    await asyncio.sleep(update_interval.total_seconds())
                    current_time = datetime.now()
                    continue
                
                # Emit market data update event
                global_event_emitter.emit('market_data_update', {
                    'timestamp': current_time.isoformat(),
                    'data': current_data
                })
                
                # Get historical data using circuit breaker
                historical_data = {}
                for symbol in symbols:
                    try:
                        # Define an async wrapper for the circuit breaker executor
                        async def get_history_with_circuit_breaker():
                            return await self.data_manager.get_historical_data(
                                symbol=symbol,
                                lookback_periods=100  # Use last 100 periods
                            )
                            
                        # Fetch historical data with circuit breaker
                        symbol_history = await self.data_executor.execute_async(
                            get_history_with_circuit_breaker,
                            retry_on_exceptions=Exception
                        )
                        
                        if symbol_history is not None and len(symbol_history) > 0:
                            historical_data[symbol] = symbol_history
                    except TradingAgentError as e:
                        logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
                        
                        # Emit error event
                        global_event_emitter.emit('paper_trading_error', {
                            'error_category': e.error_category.value,
                            'error_code': e.error_code.value,
                            'message': f"Error fetching historical data for {symbol}: {e.message}",
                            'severity': e.severity.value,
                            'timestamp': time.time(),
                            'details': e.details,
                            'troubleshooting': e.troubleshooting
                        })
                
                # Generate signals using circuit breaker
                try:
                    # Use strategy circuit breaker for signal generation
                    signals = self.strategy_executor.execute(
                        self.strategy_manager.process_data_and_generate_signals,
                        current_data=current_data,
                        historical_data=historical_data,
                        current_portfolio=current_portfolio,
                        retry_on_exceptions=Exception
                    )
                    
                    # Apply risk management with strategy circuit breaker
                    risk_adjusted_signals = self.strategy_executor.execute(
                        self.risk_manager.apply_risk_constraints,
                        signals=signals,
                        current_portfolio=current_portfolio,
                        market_data=current_data,
                        retry_on_exceptions=Exception
                    )
                except TradingAgentError as e:
                    logger.error(f"Error in strategy processing: {e}", exc_info=True)
                    signals = {}
                    risk_adjusted_signals = {}
                    
                    # Emit error event
                    global_event_emitter.emit('paper_trading_error', {
                        'error_category': e.error_category.value,
                        'error_code': e.error_code.value,
                        'message': f"Strategy error: {e.message}",
                        'severity': e.severity.value,
                        'timestamp': time.time(),
                        'details': e.details,
                        'troubleshooting': e.troubleshooting
                    })
                    
                    # Emit signal generation event
                    global_event_emitter.emit('signal_update', {
                        'timestamp': current_time.isoformat(),
                        'raw_signals': signals,
                        'risk_adjusted_signals': risk_adjusted_signals
                    })
                    
                    # Calculate position sizes
                    target_positions = self.portfolio_manager.calculate_position_sizes(
                        signals=risk_adjusted_signals,
                        current_portfolio=current_portfolio,
                        market_data=current_data
                    )
                    
                    # Get current positions
                    current_positions = current_portfolio.get('positions', {})
                    
                    # Calculate position adjustments
                    position_adjustments = self.portfolio_manager.rebalance_portfolio(
                        current_positions=current_positions,
                        target_positions=target_positions
                    )
                    
                    # Generate orders from position adjustments
                    orders = []
                    for symbol, adjustment in position_adjustments.items():
                        if adjustment != 0:
                            # Create order
                            try:
                                from ai_trading_agent.agent.execution_handler_new import Order
                                order = Order(
                                    symbol=symbol,
                                    quantity=adjustment
                                )
                                orders.append(order)
                            except ImportError:
                                # Fallback if Order class not available
                                order = {
                                    'symbol': symbol,
                                    'quantity': adjustment,
                                    'type': 'market'
                                }
                                orders.append(order)
                    
                    # Execute orders using circuit breaker
                    try:
                        # Define an async wrapper for the circuit breaker executor
                        async def execute_orders_with_circuit_breaker():
                            return await self.execution_handler.submit_orders(
                                orders=orders,
                                market_data=current_data
                            )
                            
                        # Execute orders with circuit breaker
                        executed_orders = await self.execution_executor.execute_async(
                            execute_orders_with_circuit_breaker,
                            retry_on_exceptions=Exception
                        )
                    except TradingAgentError as e:
                        logger.error(f"Error executing orders: {e}", exc_info=True)
                        executed_orders = []
                        
                        # Emit error event
                        global_event_emitter.emit('paper_trading_error', {
                            'error_category': e.error_category.value,
                            'error_code': e.error_code.value,
                            'message': f"Order execution error: {e.message}",
                            'severity': e.severity.value,
                            'timestamp': time.time(),
                            'details': e.details,
                            'troubleshooting': e.troubleshooting
                        })
                    
                    # Emit order execution event
                    global_event_emitter.emit('order_execution', {
                        'timestamp': current_time.isoformat(),
                        'orders': [order.__dict__ if hasattr(order, '__dict__') else order for order in executed_orders]
                    })
                    
                    # Update portfolio
                    for order in executed_orders:
                        try:
                            # Handle both Order objects and dictionaries
                            if hasattr(order, 'status') and order.status == 'FILLED':
                                symbol = order.symbol
                                quantity = order.executed_quantity
                                price = order.executed_price
                                commission = order.commission
                                order_id = order.order_id
                            elif isinstance(order, dict) and order.get('status') == 'FILLED':
                                symbol = order['symbol']
                                quantity = order['executed_quantity']
                                price = order['executed_price']
                                commission = order.get('commission', 0.0)
                                order_id = order.get('order_id', 'unknown')
                            else:
                                continue
                                
                            # Update current positions
                            if symbol not in current_positions:
                                current_positions[symbol] = 0
                            
                            current_positions[symbol] += quantity
                            
                            # If position is now zero, remove it
                            if current_positions[symbol] == 0:
                                del current_positions[symbol]
                            
                            # Update cash
                            current_portfolio['cash'] -= (quantity * price + commission)
                            
                            # Record trade
                            trade = {
                                'timestamp': current_time.isoformat(),
                                'symbol': symbol,
                                'quantity': quantity,
                                'price': price,
                                'commission': commission,
                                'order_id': order_id,
                                'pnl': 0.0  # Will be updated later
                            }
                            
                            self.results['trades'].append(trade)
                        except Exception as e:
                            logger.error(f"Error processing executed order: {e}", exc_info=True)
                    
                    # Update portfolio value using circuit breaker
                    try:
                        prices = {}
                        for symbol, data in current_data.items():
                            if isinstance(data, dict) and 'close' in data:
                                prices[symbol] = data['close']
                            elif hasattr(data, 'close'):
                                prices[symbol] = data.close
                            elif isinstance(data, (list, tuple)) and len(data) > 0:
                                last_item = data[-1]
                                if isinstance(last_item, dict) and 'close' in last_item:
                                    prices[symbol] = last_item['close']
                        
                        # Update portfolio value with circuit breaker
                        portfolio_value = self.strategy_executor.execute(
                            self.portfolio_manager.update_portfolio_value,
                            positions=current_positions,
                            prices=prices,
                            timestamp=current_time,
                            retry_on_exceptions=Exception
                        )
                        
                        # Update current portfolio
                        current_portfolio = self.portfolio_manager.get_current_portfolio()
                        current_portfolio['timestamp'] = current_time.isoformat()
                    except TradingAgentError as e:
                        logger.error(f"Error updating portfolio value: {e}", exc_info=True)
                        
                        # Emit error event
                        global_event_emitter.emit('paper_trading_error', {
                            'error_category': e.error_category.value,
                            'error_code': e.error_code.value,
                            'message': f"Portfolio error: {e.message}",
                            'severity': e.severity.value,
                            'timestamp': time.time(),
                            'details': e.details,
                            'troubleshooting': e.troubleshooting
                        })
                        
                        # Add to portfolio history
                        self.results['portfolio_history'].append(deepcopy(current_portfolio))
                        
                        # Emit portfolio update event
                        global_event_emitter.emit('portfolio_update', {
                            'timestamp': current_time.isoformat(),
                            'portfolio': deepcopy(current_portfolio)
                        })
                        
                        # Emit status update event with all current information
                        global_event_emitter.emit('paper_trading_status', {
                            'status': 'running',
                            'uptime_seconds': self.get_uptime_seconds(),
                            'symbols': self.get_symbols(),
                            'current_portfolio': deepcopy(current_portfolio),
                            'recent_trades': self.get_recent_trades(10),
                            'performance_metrics': self.results.get('performance_metrics')
                        })
                    except Exception as e:
                        logger.error(f"Error updating portfolio value: {e}", exc_info=True)
                
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}", exc_info=True)
                
                    # Sleep until next update
                await asyncio.sleep(update_interval.total_seconds())
                current_time = datetime.now()
            
            # Calculate performance metrics
            try:
                self._calculate_performance_metrics()
                
                # Emit performance metrics event
                global_event_emitter.emit('performance_metrics', {
                    'timestamp': datetime.now().isoformat(),
                    'metrics': deepcopy(self.results['performance_metrics'])
                })
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
                self.results['performance_metrics'] = {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
                
                # Emit error performance metrics event
                global_event_emitter.emit('performance_metrics', {
                    'timestamp': datetime.now().isoformat(),
                    'metrics': deepcopy(self.results['performance_metrics']),
                    'error': str(e)
                })
        
        except asyncio.CancelledError:
            logger.info("Paper trading task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in paper trading: {e}", exc_info=True)
        finally:
            # Set running flag to false
            self._running = False
            
            # Emit completion status update event
            status = "completed"
            if self._stop_requested:
                status = "stopped"
            
            global_event_emitter.emit('paper_trading_status', {
                'status': status,
                'uptime_seconds': self.get_uptime_seconds(),
                'symbols': self.get_symbols(),
                'current_portfolio': self.get_current_portfolio(),
                'recent_trades': self.get_recent_trades(10),
                'performance_metrics': self.results.get('performance_metrics')
            })
            
            # Ensure we have performance metrics
            if not self.results['performance_metrics']:
                try:
                    self._calculate_performance_metrics()
                except Exception as e:
                    logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
                    self.results['performance_metrics'] = {
                        'total_return': 0.0,
                        'annualized_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'win_rate': 0.0
                    }
            
            logger.info("Paper trading completed")
        
        return self.results
    
    def run_live_trading(self, update_interval: timedelta) -> None:
        """
        Run live trading in a separate thread.
        
        Args:
            update_interval: Interval between updates
        """
        logger.info(f"Starting live trading with {update_interval} updates")
        
        # Reset results
        self.results = {
            'trades': [],
            'portfolio_history': [],
            'performance_metrics': {}
        }
        
        # Set running flag
        self._running = True
        self._stop_requested = False
        
        # Start trading thread
        self._trading_thread = threading.Thread(
            target=self._live_trading_loop,
            args=(update_interval,)
        )
        
        self._trading_thread.daemon = True
        self._trading_thread.start()
        
        logger.info("Live trading thread started")
    
    def _live_trading_loop(self, update_interval: timedelta) -> None:
        """
        Live trading loop to be run in a separate thread.
        
        Args:
            update_interval: Interval between updates
        """
        # Initialize portfolio
        initial_portfolio = {
            'total_value': self.portfolio_manager.initial_capital,
            'cash': self.portfolio_manager.initial_capital,
            'positions': {},
            'position_values': {}
        }
        
        current_portfolio = deepcopy(initial_portfolio)
        
        # Run live trading loop
        while not self._stop_requested:
            current_time = datetime.now()
            logger.info(f"Processing time: {current_time}")
            
            try:
                # Get current market data
                current_data = self.data_manager.get_latest_data()
                
                if not current_data:
                    logger.warning("No data available")
                    time.sleep(update_interval.total_seconds())
                    continue
                
                # Get historical data
                historical_data = self.data_manager.get_historical_data(
                    lookback_periods=100  # Use last 100 periods
                )
                
                # Generate signals
                signals = self.strategy_manager.process_data_and_generate_signals(
                    current_data=current_data,
                    historical_data=historical_data,
                    current_portfolio=current_portfolio
                )
                
                # Apply risk management
                risk_adjusted_signals = self.risk_manager.apply_risk_constraints(
                    signals=signals,
                    current_portfolio=current_portfolio,
                    market_data=current_data
                )
                
                # Calculate position sizes
                target_positions = self.portfolio_manager.calculate_position_sizes(
                    signals=risk_adjusted_signals,
                    current_portfolio=current_portfolio,
                    market_data=current_data
                )
                
                # Get current positions
                current_positions = current_portfolio.get('positions', {})
                
                # Calculate position adjustments
                position_adjustments = self.portfolio_manager.rebalance_portfolio(
                    current_positions=current_positions,
                    target_positions=target_positions
                )
                
                # Generate orders from position adjustments
                orders = []
                for symbol, adjustment in position_adjustments.items():
                    if adjustment != 0:
                        # Create order
                        from ai_trading_agent.agent.execution_handler_new import Order
                        order = Order(
                            symbol=symbol,
                            quantity=adjustment
                        )
                        orders.append(order)
                
                # Execute orders
                executed_orders = self.execution_handler.submit_orders(
                    orders=orders,
                    market_data=current_data
                )
                
                # Update portfolio
                for order in executed_orders:
                    if order.status == 'FILLED':
                        # Update positions
                        symbol = order.symbol
                        quantity = order.executed_quantity
                        price = order.executed_price
                        commission = order.commission
                        
                        # Update current positions
                        if symbol not in current_positions:
                            current_positions[symbol] = 0
                        
                        current_positions[symbol] += quantity
                        
                        # If position is now zero, remove it
                        if current_positions[symbol] == 0:
                            del current_positions[symbol]
                        
                        # Update cash
                        current_portfolio['cash'] -= (quantity * price + commission)
                        
                        # Record trade
                        trade = {
                            'timestamp': current_time,
                            'symbol': symbol,
                            'quantity': quantity,
                            'price': price,
                            'commission': commission,
                            'order_id': order.order_id,
                            'pnl': 0.0  # Will be updated later
                        }
                        
                        self.results['trades'].append(trade)
                
                # Update portfolio value
                prices = {symbol: data.get('close', 0.0) if isinstance(data, dict) else data['close'] 
                         for symbol, data in current_data.items()}
                
                portfolio_value = self.portfolio_manager.update_portfolio_value(
                    positions=current_positions,
                    prices=prices,
                    timestamp=current_time
                )
                
                # Update current portfolio
                current_portfolio = self.portfolio_manager.get_current_portfolio()
                
                # Add to portfolio history
                self.results['portfolio_history'].append(current_portfolio)
                
                # Calculate performance metrics periodically
                if len(self.results['portfolio_history']) % 10 == 0:
                    self._calculate_performance_metrics()
            
            except Exception as e:
                logger.error(f"Error in live trading loop: {e}")
            
            # Sleep until next update
            time.sleep(update_interval.total_seconds())
        
        # Set running flag
        self._running = False
        
        logger.info("Live trading stopped")
    
    def stop_live_trading(self) -> None:
        """Stop live trading."""
        if self._running:
            logger.info("Stopping live trading")
            self._stop_requested = True
            
            if self._trading_thread:
                self._trading_thread.join(timeout=30)
                
                if self._trading_thread.is_alive():
                    logger.warning("Trading thread did not stop gracefully")
                else:
                    logger.info("Trading thread stopped")
            
            self._running = False
        else:
            logger.info("Live trading not running")
    
    # Circuit breaker callback methods
    def _on_data_circuit_open(self) -> None:
        """Callback when the data provider circuit breaker opens."""
        logger.warning("Data provider circuit breaker opened - too many failures")
        self.error_state["data_provider"]["last_error_time"] = time.time()
        
        # Emit event for frontend notification
        global_event_emitter.emit('paper_trading_error', {
            'error_category': 'data_provider',
            'error_code': ErrorCode.DATA_PROVIDER_CONNECTION_ERROR.value,
            'message': 'Data provider connection issues detected. Automatic recovery in progress.',
            'severity': ErrorSeverity.ERROR.value,
            'timestamp': time.time(),
            'details': {
                'circuit_state': 'OPEN',
                'recovery_in_progress': True
            }
        })
    
    def _on_data_circuit_close(self) -> None:
        """Callback when the data provider circuit breaker closes."""
        logger.info("Data provider circuit breaker closed - connection restored")
        self.error_state["data_provider"]["recovery_attempts"] = 0
        
        # Emit event for frontend notification
        global_event_emitter.emit('paper_trading_error', {
            'error_category': 'data_provider',
            'error_code': ErrorCode.DATA_PROVIDER_CONNECTION_ERROR.value,
            'message': 'Data provider connection restored.',
            'severity': ErrorSeverity.INFO.value,
            'timestamp': time.time(),
            'details': {
                'circuit_state': 'CLOSED',
                'recovery_successful': True
            }
        })
    
    def _on_strategy_circuit_open(self) -> None:
        """Callback when the strategy circuit breaker opens."""
        logger.warning("Strategy circuit breaker opened - too many calculation failures")
        self.error_state["strategy"]["last_error_time"] = time.time()
        
        # Emit event for frontend notification
        global_event_emitter.emit('paper_trading_error', {
            'error_category': 'strategy',
            'error_code': ErrorCode.STRATEGY_CALCULATION_ERROR.value,
            'message': 'Strategy calculation issues detected. Automatic recovery in progress.',
            'severity': ErrorSeverity.ERROR.value,
            'timestamp': time.time(),
            'details': {
                'circuit_state': 'OPEN',
                'recovery_in_progress': True
            }
        })
    
    def _on_strategy_circuit_close(self) -> None:
        """Callback when the strategy circuit breaker closes."""
        logger.info("Strategy circuit breaker closed - calculations restored")
        self.error_state["strategy"]["recovery_attempts"] = 0
        
        # Emit event for frontend notification
        global_event_emitter.emit('paper_trading_error', {
            'error_category': 'strategy',
            'error_code': ErrorCode.STRATEGY_CALCULATION_ERROR.value,
            'message': 'Strategy calculations restored.',
            'severity': ErrorSeverity.INFO.value,
            'timestamp': time.time(),
            'details': {
                'circuit_state': 'CLOSED',
                'recovery_successful': True
            }
        })
    
    def _on_execution_circuit_open(self) -> None:
        """Callback when the execution circuit breaker opens."""
        logger.warning("Execution circuit breaker opened - too many execution failures")
        self.error_state["execution"]["last_error_time"] = time.time()
        
        # Emit event for frontend notification
        global_event_emitter.emit('paper_trading_error', {
            'error_category': 'execution',
            'error_code': ErrorCode.EXECUTION_ORDER_ERROR.value,
            'message': 'Order execution issues detected. Automatic recovery in progress.',
            'severity': ErrorSeverity.ERROR.value,
            'timestamp': time.time(),
            'details': {
                'circuit_state': 'OPEN',
                'recovery_in_progress': True
            }
        })
    
    def _on_execution_circuit_close(self) -> None:
        """Callback when the execution circuit breaker closes."""
        logger.info("Execution circuit breaker closed - execution restored")
        self.error_state["execution"]["recovery_attempts"] = 0
        
        # Emit event for frontend notification
        global_event_emitter.emit('paper_trading_error', {
            'error_category': 'execution',
            'error_code': ErrorCode.EXECUTION_ORDER_ERROR.value,
            'message': 'Order execution restored.',
            'severity': ErrorSeverity.INFO.value,
            'timestamp': time.time(),
            'details': {
                'circuit_state': 'CLOSED',
                'recovery_successful': True
            }
        })
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics from backtest results."""
        if not self.results['portfolio_history']:
            logger.warning("No portfolio history available for performance calculation")
            return
        
        # Convert portfolio history to DataFrame
        if isinstance(self.results['portfolio_history'][0], dict):
            portfolio_df = pd.DataFrame(self.results['portfolio_history'])
        else:
            portfolio_df = self.results['portfolio_history']
        
        # Calculate metrics
        try:
            # Extract portfolio values
            if 'total_value' in portfolio_df.columns:
                portfolio_values = portfolio_df['total_value'].values
                
                # Calculate returns
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                
                # Calculate metrics
                total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
                
                # Annualized return
                n_periods = len(portfolio_values)
                annualized_return = ((1 + total_return / 100) ** (252 / n_periods) - 1) * 100
                
                # Volatility
                daily_volatility = np.std(returns) * 100
                annualized_volatility = daily_volatility * np.sqrt(252)
                
                # Sharpe ratio
                risk_free_rate = 0.0  # Assume zero risk-free rate
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
                
                # Maximum drawdown
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (peak - portfolio_values) / peak
                max_drawdown = np.max(drawdown) * 100
                
                # Win rate
                if self.results['trades']:
                    trades_df = pd.DataFrame(self.results['trades'])
                    win_count = len(trades_df[trades_df['pnl'] > 0])
                    total_trades = len(trades_df)
                    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
                else:
                    win_rate = 0
                
                # Store metrics
                self.results['performance_metrics'] = {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'daily_volatility': daily_volatility,
                    'annualized_volatility': annualized_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate
                }
                
                logger.info(f"Performance metrics calculated: {self.results['performance_metrics']}")
            else:
                logger.warning("Portfolio history does not contain 'total_value' column")
                self.results['performance_metrics'] = {'error': 'No total_value in portfolio history'}
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            self.results['performance_metrics'] = {'error': str(e)}
