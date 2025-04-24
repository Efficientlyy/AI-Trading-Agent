#!/usr/bin/env python
"""
Main script for AI Trading Agent.

This script integrates the Reddit API collector to gather sentiment data,
processes it with the NLP pipeline, and runs a backtest using the
processed sentiment data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import necessary components
from ai_trading_agent.common.logging_config import setup_logging
from ai_trading_agent.sentiment_analysis.data_collection import RedditSentimentCollector
from ai_trading_agent.nlp_processing.sentiment_processor import SentimentProcessor
from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
from ai_trading_agent.agent.data_manager import SimpleDataManager, InMemoryDataManager
from ai_trading_agent.agent.risk_manager import SimpleRiskManager
from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
from ai_trading_agent.backtesting.backtester import Backtester
from ai_trading_agent.backtesting.performance_metrics import calculate_metrics
from ai_trading_agent.visualization.backtest_viz import (
    visualize_backtest_results,
    visualize_sentiment_vs_price,
    visualize_all_sentiment
)
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager

# Try to import RustBacktester if available
try:
    from ai_trading_agent.backtesting.rust_backtester import RustBacktester
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

class CustomPortfolioManagerWrapper:
    """A wrapper to adapt our PortfolioManager to the interface expected by the BacktestOrchestrator."""
    
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        
    def update_market_prices(self, market_data, timestamp=None):
        """Adapts the market_data to the format expected by the PortfolioManager."""
        prices = {}
        
        # Extract prices from market_data (expected to be a dict of Series with 'close' prices)
        for symbol, series in market_data.items():
            if hasattr(series, 'close'):
                prices[symbol] = series.close
            elif isinstance(series, dict) and 'close' in series:
                prices[symbol] = series['close']
                
        # If timestamp is None, use current time
        if timestamp is None:
            timestamp = pd.Timestamp.now()
            
        # Call the original method with both required parameters
        self.portfolio_manager.update_market_prices(prices, timestamp)
        
    def generate_orders(self, signals, market_data, risk_constraints=None):
        """Generate orders based on signals with advanced position sizing and risk management.
        
        This enhanced implementation uses signal strength, portfolio value, 
        volatility-adjusted position sizing and advanced entry/exit logic.
        """
        orders = []
        
        # Get current portfolio state
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        current_positions = portfolio_state.get('positions', {})
        portfolio_value = portfolio_state.get('total_value', 0)
        
        if portfolio_value <= 0:
            return []
        
        # Get risk parameters (either from risk constraints or defaults)
        max_position_size = 0.15  # Default max position size
        if risk_constraints and 'max_position_size' in risk_constraints:
            max_position_size = risk_constraints['max_position_size']
            
        # Track allocated capital to ensure we don't exceed portfolio constraints
        allocated_capital = sum(
            position.get('current_value', 0) 
            for position in current_positions.values()
        )
        available_capital = portfolio_value - allocated_capital
        
        # Sort signals by strength (absolute value) to prioritize stronger signals
        sorted_signals = sorted(
            signals.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
            
        from ai_trading_agent.trading_engine.models import Order
        from ai_trading_agent.trading_engine.enums import OrderSide, OrderType
        
        for symbol, signal in sorted_signals:
            # Skip if signal is too weak (noise)
            if abs(signal) < 0.12:  # Adjusted threshold to filter out noise
                continue
                
            # Get current price for the symbol
            current_price = None
            if symbol in market_data:
                data = market_data[symbol]
                if hasattr(data, 'close'):
                    current_price = data.close
                elif isinstance(data, dict) and 'close' in data:
                    current_price = data['close']
                    
            if current_price is None:
                continue
            
            # Get volatility if available for volatility-adjusted position sizing
            volatility = 0.02  # Default volatility assumption
            try:
                # Try to get historical prices for volatility calculation
                if hasattr(self.portfolio_manager, 'data_manager'):
                    hist_data = self.portfolio_manager.data_manager.get_historical_data(
                        symbol, lookback=20
                    )
                    if hist_data is not None and 'close' in hist_data:
                        returns = hist_data['close'].pct_change().dropna()
                        volatility = max(0.01, returns.std())  # Min volatility of 1%
            except Exception:
                pass  # Use default if calculation fails
                
            # Current position information
            current_qty = current_positions.get(symbol, {}).get('quantity', 0)
            current_value = current_positions.get(symbol, {}).get('current_value', 0)
            
            # Calculate position size based on signal strength, volatility and available capital
            # Stronger signals and lower volatility lead to larger position sizes
            signal_strength_factor = min(1.0, abs(signal) * 3)  # Scale signal strength (0.12-0.33 => 0.36-1.0)
            volatility_factor = min(1.0, 0.05 / volatility)  # Inverse volatility scaling
            
            # Base position size as percentage of portfolio
            position_size_pct = max_position_size * signal_strength_factor * volatility_factor
            
            # Ensure we respect portfolio constraints
            target_position_value = portfolio_value * position_size_pct
            adjusted_position_value = min(target_position_value, available_capital)
            
            # Calculate quantity based on adjusted position value
            quantity = adjusted_position_value / current_price
            
            # Minimum meaningful order size (avoid dust orders)
            min_order_value = portfolio_value * 0.01  # 1% of portfolio minimum
            
            if signal > 0:  # Buy signal
                side = OrderSide.BUY
                # More sophisticated logic for buy decisions
                if current_qty <= 0:  # No position or short position
                    # Create new buy order
                    if adjusted_position_value >= min_order_value:
                        # For short positions, first close the short
                        if current_qty < 0:
                            close_order = Order(
                                symbol=symbol,
                                side=OrderSide.BUY,
                                quantity=abs(current_qty),
                                order_type=OrderType.MARKET,
                                notes="Close short position"
                            )
                            orders.append(close_order)
                            
                        # Create new long position
                        new_order = Order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET,
                            notes=f"New long ({signal_strength_factor:.2f} signal)"
                        )
                        orders.append(new_order)
                else:
                    # Already have a long position
                    # Increase position if signal strength is significantly higher
                    if signal > 0.25 and adjusted_position_value >= min_order_value:
                        # Calculate additional quantity to buy
                        additional_value = adjusted_position_value - current_value
                        if additional_value > min_order_value:
                            additional_qty = additional_value / current_price
                            add_order = Order(
                                symbol=symbol,
                                side=OrderSide.BUY,
                                quantity=additional_qty,
                                order_type=OrderType.MARKET,
                                notes=f"Increase long ({signal_strength_factor:.2f} signal)"
                            )
                            orders.append(add_order)
            else:  # Sell signal
                side = OrderSide.SELL
                # More sophisticated logic for sell decisions
                if current_qty >= 0:  # No position or long position
                    if current_qty > 0:
                        # Close existing long position
                        close_order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=current_qty,
                            order_type=OrderType.MARKET,
                            notes=f"Close long ({abs(signal):.2f} signal)"
                        )
                        orders.append(close_order)
                    
                    # If signal is strong enough, open a short position
                    if abs(signal) > 0.25 and adjusted_position_value >= min_order_value:
                        short_order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=quantity,
                            order_type=OrderType.MARKET,
                            notes=f"New short ({abs(signal):.2f} signal)"
                        )
                        orders.append(short_order)
                else:
                    # Already have a short position
                    # Increase short if signal strength is significantly higher
                    if abs(signal) > 0.25 and adjusted_position_value >= min_order_value:
                        # Calculate additional quantity to short
                        additional_value = adjusted_position_value - abs(current_value)
                        if additional_value > min_order_value:
                            additional_qty = additional_value / current_price
                            add_short_order = Order(
                                symbol=symbol,
                                side=OrderSide.SELL,
                                quantity=additional_qty,
                                order_type=OrderType.MARKET,
                                notes=f"Increase short ({abs(signal):.2f} signal)"
                            )
                            orders.append(add_short_order)
                            
        return orders
        
    def __getattr__(self, name):
        """Delegate all other method calls to the wrapped portfolio_manager."""
        return getattr(self.portfolio_manager, name)

class CustomRiskManagerWrapper:
    """A wrapper to adapt our RiskManager to the interface expected by the BacktestOrchestrator."""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        
    def get_risk_constraints(self, current_positions, current_value):
        """Adapts parameters to match the SimpleRiskManager interface.
        
        The orchestrator passes current_positions and current_value,
        but our SimpleRiskManager expects portfolio_state and market_data.
        """
        # Create a portfolio state dictionary from the current positions and value
        portfolio_state = {
            'positions': current_positions,
            'total_value': current_value
        }
        
        # For market_data, we need to have access to it
        # Since we don't have direct access to market_data here, 
        # we'll need a different approach
        
        try:
            # Try calling with just portfolio_state (some implementations might use defaults)
            return self.risk_manager.get_risk_constraints(portfolio_state=portfolio_state, market_data={})
        except Exception as e:
            # If that fails, log the error and return default constraints
            import logging
            logging.warning(f"Error in get_risk_constraints adapter: {e}")
            
            # Return sensible defaults
            return {
                'max_position_size': self.risk_manager.config.get('max_position_size', 0.2),
                'max_leverage': 1.0,
                'max_concentration': 0.25,
                'max_drawdown': 0.25
            }
    
    def generate_stop_loss_signals(self, portfolio_state, market_data):
        """Adapts parameters to match the SimpleRiskManager interface."""
        # Extract positions from portfolio state
        positions = portfolio_state.get('positions', {})
        # Pass the expected parameters
        return self.risk_manager.generate_stop_loss_signals(positions, market_data)
    
    def assess_risk(self, order, positions, portfolio_value):
        """Adapts parameters to match the SimpleRiskManager interface."""
        # SimpleRiskManager likely has a different signature for assess_risk
        return self.risk_manager.assess_risk(order)
        
    def __getattr__(self, name):
        """Delegate all other method calls to the wrapped risk_manager."""
        return getattr(self.risk_manager, name)

class CustomBacktestOrchestratorWrapper:
    """A wrapper to adapt the BacktestOrchestrator to handle the performance metrics calculation properly."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        # Copy all relevant attributes from the wrapped orchestrator
        self.data_manager = orchestrator.data_manager
        self.strategy_manager = orchestrator.strategy_manager
        self.portfolio_manager = orchestrator.portfolio_manager
        self.risk_manager = orchestrator.risk_manager
        self.execution_handler = orchestrator.execution_handler
        self.config = orchestrator.config
        
    def run(self):
        """Complete override of the orchestrator's run method to properly handle performance metrics."""
        import logging
        from ai_trading_agent.backtesting.performance_metrics import calculate_metrics
        
        logger = logging.getLogger(__name__)
        logger.info("Starting backtest with custom orchestrator wrapper")
        
        # Extract configuration
        start_date = self.config.get('start_date')
        end_date = self.config.get('end_date')
        symbols = self.config.get('symbols', [])
        
        if not symbols:
            logger.error("No symbols provided for backtest")
            return None
        
        # Initialize results dictionary
        results = {
            'portfolio_history': [],
            'trades': [],
            'signals': [],
            'risk_events': []
        }
        
        # Get initial portfolio state
        initial_portfolio_state = self.portfolio_manager.get_portfolio_state()
        initial_capital = initial_portfolio_state.get('total_value', 100000.0)
        results['portfolio_history'].append({
            'timestamp': start_date,
            'value': initial_capital
        })
        
        # Generate date range for the backtest
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Run the backtest
        for date in dates:
            try:
                # Get market data for current date
                market_data = {}
                for symbol in symbols:
                    try:
                        data = self.data_manager.get_price_data(symbol, date)
                        if data is not None:
                            market_data[symbol] = data
                    except Exception as e:
                        logger.warning(f"Error getting price data for {symbol} at {date}: {e}")
                
                if not market_data:
                    logger.warning(f"No market data available for {date}")
                    continue
                
                # Update portfolio with latest prices
                self.portfolio_manager.update_market_prices(market_data, date)
                
                # Get current portfolio state
                portfolio_state = self.portfolio_manager.get_portfolio_state()
                current_positions = portfolio_state.get('positions', {})
                current_portfolio_value = portfolio_state.get('total_value', 0)
                
                # Generate signals
                signals = {}
                try:
                    signals = self.strategy_manager.generate_signals(date)
                except Exception as e:
                    logger.warning(f"Failed to generate signals: {e}")
                
                # Get risk constraints
                risk_constraints = {}
                try:
                    risk_constraints = self.risk_manager.get_risk_constraints(
                        current_positions=current_positions,
                        current_value=current_portfolio_value
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate risk constraints: {e}")
                    risk_constraints = {
                        'max_position_size': 0.2,
                        'max_leverage': 1.0
                    }
                
                # Generate stop loss signals
                stop_loss_signals = {}
                try:
                    stop_loss_signals = self.risk_manager.generate_stop_loss_signals(
                        portfolio_state, market_data
                    )
                    # Merge stop loss signals with strategy signals (stop loss takes precedence)
                    signals.update(stop_loss_signals)
                except Exception as e:
                    logger.warning(f"Failed to generate stop loss signals: {e}")
                
                # Generate orders based on signals
                proposed_orders = []
                try:
                    proposed_orders = self.portfolio_manager.generate_orders(
                        signals, market_data, risk_constraints
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate proposed orders: {e}")
                
                # Execute orders
                executed_orders = []
                for order in proposed_orders:
                    try:
                        executed_order = self.execution_handler.execute_order(order, date)
                        if executed_order:
                            executed_orders.append(executed_order)
                    except Exception as e:
                        logger.warning(f"Failed to execute order {order}: {e}")
                
                # Record results
                results['portfolio_history'].append({
                    'timestamp': date,
                    'value': self.portfolio_manager.get_portfolio_state().get('total_value', 0)
                })
                
                results['signals'].extend([
                    {'timestamp': date, 'symbol': symbol, 'signal': signal}
                    for symbol, signal in signals.items()
                ])
                
                results['trades'].extend(executed_orders)
                
            except Exception as e:
                logger.error(f"Error in backtest at {date}: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
        # Calculate performance metrics
        try:
            # Format portfolio history as expected by calculate_metrics
            formatted_history = [
                {'timestamp': entry['timestamp'], 'total_value': entry['value']}
                for entry in results['portfolio_history']
            ]
            
            metrics = calculate_metrics(
                portfolio_history=formatted_history,
                trade_history=results['trades'],
                initial_capital=initial_capital,
                risk_free_rate=0.0
            )
            
            # Add metrics to results
            results['performance_metrics'] = metrics.__dict__ if hasattr(metrics, '__dict__') else {}
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("Backtest completed successfully")
        return results
        
    def __getattr__(self, name):
        """Delegate all other method calls to the wrapped orchestrator."""
        return getattr(self.orchestrator, name)

class CustomDataManagerWrapper:
    """A wrapper to adapt our data manager to the interface expected by the BacktestOrchestrator."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        
    def get_price_data(self, symbol, date):
        """Gets price data for a specific symbol and date.
        
        This method adapts the InMemoryDataManager interface to provide price data 
        in the format expected by the BacktestOrchestrator.
        """
        if not hasattr(self.data_manager, 'price_data'):
            return None
            
        if symbol not in self.data_manager.price_data:
            return None
            
        # Get the price DataFrame for the symbol
        price_df = self.data_manager.price_data.get(symbol)
        
        if price_df is None or price_df.empty:
            return None
            
        # Convert date to a comparable format if it's not already
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
            
        # Get the closest date (exact match or closest prior date)
        closest_date = None
        for idx in price_df.index:
            # If we find an exact match, use it
            if idx == date:
                closest_date = idx
                break
                
            # If index date is before our target date and closer than current closest
            if idx < date and (closest_date is None or idx > closest_date):
                closest_date = idx
                
        # If we found a date, return the data for that date
        if closest_date is not None:
            return price_df.loc[closest_date]
            
        return None
        
    def __getattr__(self, name):
        """Delegate all other method calls to the wrapped data_manager."""
        return getattr(self.data_manager, name)

class CustomStrategyManagerWrapper:
    """A wrapper to adapt our strategy manager to the interface expected by the BacktestOrchestrator."""
    
    def __init__(self, strategy_manager, portfolio_manager, data_manager=None):
        self.strategy_manager = strategy_manager
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager
        
    def generate_signals(self, date):
        """Adapts the generate_signals method to include portfolio_state parameter and market_data."""
        # Get the current portfolio state from the portfolio manager
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        
        # If we have a data manager, fetch the market data for this date
        market_data = {}
        if hasattr(self.strategy_manager, 'data_manager'):
            data_manager = self.strategy_manager.data_manager
            
            # Get symbols from the data manager if available
            if hasattr(data_manager, 'symbols'):
                symbols = data_manager.symbols
            elif hasattr(data_manager, 'price_data'):
                symbols = list(data_manager.price_data.keys())
            elif hasattr(data_manager, 'config') and 'symbols' in data_manager.config:
                symbols = data_manager.config['symbols']
            else:
                symbols = []
                
            # Fetch market data for each symbol
            for symbol in symbols:
                try:
                    # Try to get price data for this symbol and date
                    price_data = data_manager.get_price_data(symbol, date)
                    if price_data is not None:
                        market_data[symbol] = price_data
                except Exception as e:
                    import logging
                    logging.warning(f"Error getting price data for {symbol} at {date}: {e}")
        
        # Call the original method with both required parameters
        try:
            # Try the most complete form with date, portfolio_state, and market_data
            if market_data:
                return self.strategy_manager.generate_signals(market_data, portfolio_state)
            else:
                # If no market data, try the simplified form with just date and portfolio_state
                return self.strategy_manager.generate_signals(date, portfolio_state)
        except TypeError:
            # If that fails, try different combinations
            try:
                # Try with just market_data
                if market_data:
                    return self.strategy_manager.generate_signals(market_data)
                else:
                    # Try with just date
                    return self.strategy_manager.generate_signals(date)
            except Exception as e:
                import logging
                logging.warning(f"Error generating signals: {e}")
                return {}  # Return empty signals as fallback
        
    def __getattr__(self, name):
        """Delegate all other method calls to the wrapped strategy_manager."""
        return getattr(self.strategy_manager, name)

def load_market_data(symbols, start_date, end_date):
    """
    Load historical price data for the specified symbols.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        Dictionary mapping symbols to DataFrames with OHLCV data
    """
    logger.info(f"Loading market data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # For this example, we'll generate synthetic data
    # In a production environment, replace this with real data from a provider
    data = {}
    
    for symbol in symbols:
        # Generate synthetic price data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Start with a random walk, but with symbol-specific trends
        seed = sum(ord(c) for c in symbol)  # Create a seed from symbol name
        np.random.seed(seed)
        
        # Different symbols will have different trends
        if symbol in ['BTC', 'ETH', 'DOGE']:
            trend = 0.001  # Crypto slight uptrend
            volatility = 0.02  # Higher volatility
        elif symbol in ['AAPL', 'MSFT', 'GOOGL']:
            trend = 0.0005  # Tech stocks slight uptrend
            volatility = 0.01  # Moderate volatility
        else:
            trend = 0.0002  # Other stocks smaller uptrend
            volatility = 0.008  # Lower volatility
            
        # Generate price series with trend and volatility
        returns = np.random.normal(trend, volatility, n)
        close = 100 * (1 + returns).cumprod()
        
        # Ensure price doesn't go negative
        close = np.maximum(close, 1)
        
        # Generate other OHLCV data
        high = close * (1 + np.random.uniform(0, volatility, n))
        low = close * (1 - np.random.uniform(0, volatility, n))
        open_price = low + np.random.uniform(0, 1, n) * (high - low)
        volume = np.random.uniform(100000, 1000000, n) * (1 + returns * 5)  # Volume increases with returns
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        data[symbol] = df
    
    logger.info(f"Market data loaded successfully with {len(data[symbols[0]])} bars per symbol")
    return data

def collect_reddit_sentiment(symbols, start_date, end_date):
    """
    Collect sentiment data from Reddit for the specified symbols.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date for sentiment data 
        end_date: End date for sentiment data
        
    Returns:
        Dictionary mapping symbols to DataFrames with sentiment scores
    """
    logger.info(f"Collecting Reddit sentiment data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Reddit API credentials - use environment variables in production
    config = {
        # Use empty values to trigger environment variable fallbacks
        'client_id': '',
        'client_secret': '',
        'user_agent': 'AI Trading Agent Backtester',
        'subreddits': ['wallstreetbets', 'investing', 'stocks', 'cryptocurrency'],
        'keywords': ['buy', 'sell', 'bullish', 'bearish', 'moon', 'crash'],
        'comment_limit': 5,  # Limit comments per post for faster collection
        'post_limit': 20,    # Limit posts per symbol for faster collection
    }
    
    # Initialize Reddit collector
    collector = RedditSentimentCollector(config)
    
    sentiment_data = {}
    
    try:
        # Collect sentiment data for each symbol
        for symbol in symbols:
            symbol_data = collector.collect([symbol], start_date, end_date)
            
            # Process the data: aggregate by date and calculate average sentiment
            if not symbol_data.empty:
                daily_sentiment = symbol_data.groupby(symbol_data['timestamp'].dt.date)['sentiment_score'].mean()
                daily_volume = symbol_data.groupby(symbol_data['timestamp'].dt.date)['volume'].sum()
                
                # Convert to DataFrame with date as index
                dates = [pd.Timestamp(d) for d in daily_sentiment.index]
                df = pd.DataFrame({
                    'sentiment': daily_sentiment.values,
                    'volume': daily_volume.values
                }, index=dates)
                
                # Reindex to ensure all dates are present
                full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                df = df.reindex(full_date_range)
                
                # Forward fill missing values, then backfill any remaining NaNs
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                # If still has NaNs (e.g., no data at all), fill with zeros
                df = df.fillna(0)
                
                sentiment_data[symbol] = df
                logger.info(f"Collected {len(symbol_data)} sentiment data points for {symbol}")
            else:
                logger.warning(f"No sentiment data collected for {symbol}, using synthetic data")
                
                # Generate synthetic sentiment data when real data collection fails
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                n = len(dates)
                
                # Generate sentiment scores between -1 and 1
                np.random.seed(sum(ord(c) for c in symbol))  # Consistent seed for reproducibility
                sentiment = np.random.normal(0, 0.3, n)
                sentiment = np.clip(sentiment, -1, 1)
                volume = np.random.uniform(10, 100, n)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'sentiment': sentiment,
                    'volume': volume
                }, index=dates)
                
                sentiment_data[symbol] = df
                
    except Exception as e:
        logger.error(f"Error collecting Reddit sentiment data: {e}")
        logger.info("Falling back to synthetic sentiment data")
        
        # Generate synthetic sentiment data as fallback
        for symbol in symbols:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            n = len(dates)
            
            # Generate sentiment scores between -1 and 1
            np.random.seed(sum(ord(c) for c in symbol))  # Consistent seed for reproducibility
            sentiment = np.random.normal(0, 0.3, n)
            sentiment = np.clip(sentiment, -1, 1)
            volume = np.random.uniform(10, 100, n)
            
            # Create DataFrame
            df = pd.DataFrame({
                'sentiment': sentiment,
                'volume': volume
            }, index=dates)
            
            sentiment_data[symbol] = df
    
    logger.info(f"Sentiment data processing completed")
    return sentiment_data

def run_backtest(symbols, start_date, end_date, initial_capital=100000.0, use_real_sentiment=True):
    """
    Run a backtest for the specified symbols and date range.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date for the backtest
        end_date: End date for the backtest
        initial_capital: Initial capital for the portfolio
        use_real_sentiment: Whether to use real Reddit sentiment data or synthetic
        
    Returns:
        Dictionary containing backtest results
    """
    logger.info(f"Starting backtest with {len(symbols)} symbols and {initial_capital} initial capital")
    
    # Load market data
    price_data = load_market_data(symbols, start_date, end_date)
    
    # Load sentiment data
    if use_real_sentiment:
        sentiment_data = collect_reddit_sentiment(symbols, start_date, end_date)
    else:
        # Use synthetic sentiment data
        logger.info("Using synthetic sentiment data")
        sentiment_data = {}
        for symbol in symbols:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            n = len(dates)
            
            # Generate sentiment scores between -1 and 1
            np.random.seed(sum(ord(c) for c in symbol))  # Consistent seed for reproducibility
            sentiment = np.random.normal(0, 0.3, n)
            sentiment = np.clip(sentiment, -1, 1)
            volume = np.random.uniform(10, 100, n)
            
            # Create DataFrame
            df = pd.DataFrame({
                'sentiment': sentiment,
                'volume': volume
            }, index=dates)
            
            sentiment_data[symbol] = df
    
    # Create data manager with direct access to price and sentiment data
    data_manager_config = {
        'price_data': price_data,
        'sentiment_data': sentiment_data,
        'symbols': symbols
    }
    data_manager = InMemoryDataManager(data_manager_config)
    
    # Ensure the data manager has the price_data attribute
    if not hasattr(data_manager, 'price_data'):
        setattr(data_manager, 'price_data', price_data)
    
    # Wrap data manager with custom adapter
    data_manager = CustomDataManagerWrapper(data_manager)
    
    # Create portfolio manager (we need to create this before the strategy manager)
    portfolio_manager = PortfolioManager(
        initial_capital=initial_capital,
        risk_per_trade=0.02,
        max_position_size=0.1  # Maximum single position size as percentage of portfolio
    )
    
    # Wrap portfolio manager with custom adapter
    portfolio_manager = CustomPortfolioManagerWrapper(portfolio_manager)
    
    # Create strategy manager
    strategy_config = {
        'buy_threshold': 0.15,  # Lower threshold to generate more buy signals (was 0.2)
        'sell_threshold': -0.15, # Higher threshold to generate more sell signals (was -0.2)
        'lookback_period': 3,    # Use a shorter lookback period to be more responsive
        'smoothing_factor': 0.7, # Add smoothing factor for sentiment data
        'use_volume_weighting': True, # Weight sentiment by volume for better signal quality
        'signal_scaling': True   # Scale signal strength based on sentiment intensity
    }
    strategy = SentimentStrategy(name="RedditSentimentStrategy", config=strategy_config)

    strategy_manager_config = {
        'name': 'SentimentStrategyManager',
        'signal_processing': {
            'smoothing': True,     # Enable signal smoothing
            'noise_threshold': 0.1, # Filter out noise below this threshold
            'amplify_factor': 1.5  # Amplify stronger signals
        }
    }
    strategy_manager = SimpleStrategyManager(strategy_manager_config, data_manager)
    strategy_manager.add_strategy(strategy)
    
    # Wrap strategy manager with custom adapter (passing the portfolio manager and data manager)
    strategy_manager = CustomStrategyManagerWrapper(strategy_manager, portfolio_manager, data_manager)
    
    # Create risk manager
    risk_manager_config = {
        'max_position_size': 0.2,      # Maximum single position size
        'max_portfolio_risk_pct': 0.05, # Maximum portfolio risk
        'stop_loss_pct': 0.05          # Stop loss percentage
    }
    risk_manager = SimpleRiskManager(config=risk_manager_config)
    
    # Wrap risk manager with custom adapter
    risk_manager = CustomRiskManagerWrapper(risk_manager)
    
    # Create execution handler with portfolio manager
    execution_handler_config = {
        'commission_rate': 0.001,  # 0.1% commission
        'slippage_pct': 0.001      # 0.1% slippage
    }
    execution_handler = SimulatedExecutionHandler(
        portfolio_manager=portfolio_manager,
        config=execution_handler_config
    )
    
    # Create orchestrator config
    orchestrator_config = {
        'start_date': start_date,
        'end_date': end_date,
        'symbols': symbols
    }
    
    # Create orchestrator
    orchestrator = BacktestOrchestrator(
        data_manager=data_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        config=orchestrator_config
    )
    
    # Wrap orchestrator with custom adapter
    orchestrator = CustomBacktestOrchestratorWrapper(orchestrator)
    
    # Run backtest
    try:
        # Use the orchestrator directly instead of a separate backtester
        logger.info("Running backtest using Orchestrator")
        results = orchestrator.run()
        
        if results is None:
            logger.error("Backtest failed - orchestrator returned None")
            raise RuntimeError("Backtest orchestrator returned None")
            
        logger.info(f"Backtest completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise

def analyze_results(results, save_to_file=True):
    """
    Analyze backtest results and print performance metrics.
    
    Args:
        results: Dictionary containing backtest results
        save_to_file: Whether to save results to a file
        
    Returns:
        Performance metrics object
    """
    logger.info("Analyzing backtest results")
    
    # Extract results
    portfolio_history = results.get('portfolio_history', [])
    trades = results.get('trades', [])
    
    if not portfolio_history:
        logger.warning("No portfolio history found in results. Cannot analyze.")
        return None

    # Format portfolio history for metrics calculation if needed
    formatted_history = []
    for entry in portfolio_history:
        # Check if we have the expected structure
        if isinstance(entry, dict):
            # Make sure we have all the necessary keys
            if 'timestamp' in entry:
                formatted_entry = {
                    'timestamp': entry['timestamp'],
                    'total_value': entry.get('value', 0)  # Fallback to 0 if 'value' missing
                }
                formatted_history.append(formatted_entry)
    
    if not formatted_history:
        logger.warning("Could not format portfolio history for analysis")
        return None
        
    # Calculate performance metrics
    initial_capital = formatted_history[0]['total_value'] if formatted_history else 0
    try:
        metrics = calculate_metrics(
            portfolio_history=formatted_history,
            trade_history=trades,
            initial_capital=initial_capital,
            risk_free_rate=0.0
        )
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None
    
    # Print performance metrics
    logger.info(f"Total Return: {getattr(metrics, 'total_return', 'N/A')}")
    logger.info(f"Annualized Return: {getattr(metrics, 'annualized_return', 'N/A')}")
    logger.info(f"Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 'N/A')}")
    logger.info(f"Max Drawdown: {getattr(metrics, 'max_drawdown', 'N/A')}")
    logger.info(f"Win Rate: {getattr(metrics, 'win_rate', 'N/A')}")
    logger.info(f"Profit Factor: {getattr(metrics, 'profit_factor', 'N/A')}")
    
    if save_to_file:
        # Save results to file
        output = {
            'total_return': getattr(metrics, 'total_return', 0),
            'annualized_return': getattr(metrics, 'annualized_return', 0),
            'sharpe_ratio': getattr(metrics, 'sharpe_ratio', 0),
            'max_drawdown': getattr(metrics, 'max_drawdown', 0),
            'win_rate': getattr(metrics, 'win_rate', 0),
            'profit_factor': getattr(metrics, 'profit_factor', 0),
            'portfolio_history': [
                {
                    'timestamp': str(entry['timestamp']),
                    'total_value': entry['total_value']
                } for entry in formatted_history
            ],
            'trade_summary': {
                'total_trades': len(trades),
                'winning_trades': sum(1 for trade in trades if trade.get('pnl', 0) > 0),
                'losing_trades': sum(1 for trade in trades if trade.get('pnl', 0) < 0)
            }
        }
        
        with open('backtest_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
            
        logger.info(f"Results saved to backtest_results.json")
    
    return metrics

def main():
    """Main function to run the backtest with Reddit sentiment data."""
    try:
        # Define backtest parameters
        symbols = ['BTC', 'ETH', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        # Increase backtest period from 30 days to 90 days to ensure enough historical data
        start_date = datetime.now() - timedelta(days=90)  # Use last 90 days instead of 30
        end_date = datetime.now()
        initial_capital = 100000.0
        
        # Run backtest with real sentiment data
        logger.info(f"Running backtest with Reddit sentiment data from {start_date} to {end_date}")
        price_data = load_market_data(symbols, start_date, end_date)
        sentiment_data = collect_reddit_sentiment(symbols, start_date, end_date)
        
        # Save raw sentiment data for inspection
        sentiment_df_for_file = {}
        for symbol, df in sentiment_data.items():
            sentiment_df_for_file[symbol] = df.reset_index().to_dict('records')
            
        with open('sentiment_data.json', 'w') as f:
            json.dump(sentiment_df_for_file, f, indent=2, default=str)
        logger.info("Raw sentiment data saved to sentiment_data.json")
        
        # Run the backtest
        results = run_backtest(symbols, start_date, end_date, initial_capital, use_real_sentiment=True)
        
        if not results:
            logger.error("No backtest results to analyze")
            return 1
            
        # Check if performance metrics were successfully calculated by our wrapper
        if 'performance_metrics' in results:
            logger.info("Performance metrics calculated successfully by orchestrator wrapper")
            
            # Print some key metrics
            metrics = results['performance_metrics']
            logger.info(f"Total Return: {metrics.get('total_return', 'N/A')}")
            logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
            logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A')}")
            
            # Format portfolio history for visualization (ensure we have 'total_value' instead of just 'value')
            formatted_history = []
            for entry in results.get('portfolio_history', []):
                formatted_entry = {
                    'timestamp': entry['timestamp'],
                    'total_value': entry.get('total_value', entry.get('value', 0))
                }
                formatted_history.append(formatted_entry)
            
            # Update results with formatted history
            formatted_results = {
                'portfolio_history': formatted_history,
                'trades': results.get('trades', []),
                'signals': results.get('signals', []),
                'risk_events': results.get('risk_events', []),
                'performance_metrics': results.get('performance_metrics', {})
            }
            
            # Format results for visualization
            with open('backtest_results.json', 'w') as f:
                json.dump(formatted_results, f, indent=2, default=str)
            
            # Visualize backtest results
            visualize_backtest_results('backtest_results.json', 'backtest_results.png')
            
            # Visualize sentiment vs price for each symbol
            for symbol in symbols:
                output_file = f'{symbol}_sentiment_vs_price.png'
                visualize_sentiment_vs_price(sentiment_data, price_data, symbol, output_file)
                
            # Visualize all sentiment together
            visualize_all_sentiment(sentiment_data, 'all_sentiment.png')
            
            logger.info("Backtest completed successfully")
            logger.info("Results visualized and saved to PNG files")
        else:
            logger.warning("Performance metrics not found in results, running separate analysis")
            analyze_results(results)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())