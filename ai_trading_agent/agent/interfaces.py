"""
Base interfaces for the modular agent architecture.

This module defines the abstract base classes that form the foundation of the
agent architecture, ensuring proper separation of concerns and modularity.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime

from ai_trading_agent.trading_engine.models import Order, Portfolio, Fill


class DataManager(ABC):
    """
    Responsible for acquiring and preprocessing market data from various sources.
    Provides clean, normalized data to strategies.
    """
    
    @abstractmethod
    def get_market_data(self, symbols: List[str], start_date: datetime, 
                       end_date: datetime, timeframe: str = "1d") -> pd.DataFrame:
        """
        Retrieve market data for the specified symbols and time range.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Data timeframe (e.g., "1m", "1h", "1d")
            
        Returns:
            DataFrame with market data (OHLCV)
        """
        pass
    
    @abstractmethod
    def get_sentiment_data(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """
        Retrieve sentiment data for the specified symbols and time range.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with sentiment data
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.
        
        Args:
            data: Raw data to preprocess
            
        Returns:
            Preprocessed data
        """
        pass


class Strategy(ABC):
    """
    Base class for all trading strategies.
    Receives data and generates trading signals.
    """
    
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame, 
                        additional_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Generate trading signals based on market data and optional additional data.
        
        Args:
            market_data: Market data (OHLCV)
            additional_data: Optional additional data (e.g., sentiment)
            
        Returns:
            DataFrame with signals
        """
        pass


class StrategyManager(ABC):
    """
    Manages multiple trading strategies and combines their signals.
    """
    
    @abstractmethod
    def add_strategy(self, strategy: Strategy, weight: float = 1.0) -> None:
        """
        Add a strategy to the manager with an optional weight.
        
        Args:
            strategy: Strategy to add
            weight: Weight to assign to the strategy
        """
        pass
    
    @abstractmethod
    def remove_strategy(self, strategy_id: str) -> None:
        """
        Remove a strategy from the manager.
        
        Args:
            strategy_id: ID of the strategy to remove
        """
        pass
    
    @abstractmethod
    def generate_combined_signals(self, market_data: pd.DataFrame, 
                                 additional_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Generate combined signals from all managed strategies.
        
        Args:
            market_data: Market data (OHLCV)
            additional_data: Optional additional data (e.g., sentiment)
            
        Returns:
            DataFrame with combined signals
        """
        pass


class RiskManager(ABC):
    """
    Enforces risk limits and constraints.
    Provides position sizing recommendations based on risk metrics.
    """
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, signal: float, portfolio: Portfolio, 
                               market_data: pd.DataFrame) -> float:
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            symbol: Ticker symbol
            signal: Trading signal (-1 to 1)
            portfolio: Current portfolio state
            market_data: Market data for risk calculations
            
        Returns:
            Recommended position size
        """
        pass
    
    @abstractmethod
    def check_risk_limits(self, portfolio: Portfolio, 
                         proposed_orders: List[Order]) -> List[Order]:
        """
        Check if proposed orders comply with risk limits and adjust if necessary.
        
        Args:
            portfolio: Current portfolio state
            proposed_orders: List of proposed orders
            
        Returns:
            Adjusted list of orders that comply with risk limits
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_risk(self, portfolio: Portfolio, 
                                market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various risk metrics for the current portfolio.
        
        Args:
            portfolio: Current portfolio state
            market_data: Market data for risk calculations
            
        Returns:
            Dictionary of risk metrics
        """
        pass


class PortfolioManager(ABC):
    """
    Takes trading signals and translates them into concrete orders.
    Manages the portfolio state and tracks performance.
    """
    
    @abstractmethod
    def generate_orders(self, signals: pd.DataFrame, portfolio: Portfolio, 
                       risk_manager: RiskManager, market_data: pd.DataFrame) -> List[Order]:
        """
        Generate orders based on signals, current portfolio, and risk constraints.
        
        Args:
            signals: Trading signals
            portfolio: Current portfolio state
            risk_manager: Risk manager for position sizing
            market_data: Current market data
            
        Returns:
            List of orders
        """
        pass
    
    @abstractmethod
    def update_portfolio(self, portfolio: Portfolio, fills: List[Fill], 
                        current_prices: Dict[str, float]) -> Portfolio:
        """
        Update the portfolio based on fills and current market prices.
        
        Args:
            portfolio: Current portfolio state
            fills: List of fills to apply
            current_prices: Dictionary of current prices by symbol
            
        Returns:
            Updated portfolio
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self, portfolio_history: List[Portfolio]) -> Dict[str, float]:
        """
        Calculate performance metrics based on portfolio history.
        
        Args:
            portfolio_history: Historical portfolio states
            
        Returns:
            Dictionary of performance metrics
        """
        pass


class ExecutionHandler(ABC):
    """
    Takes order requests and handles execution details.
    Returns fill information.
    """
    
    @abstractmethod
    def execute_orders(self, orders: List[Order], market_data: pd.DataFrame) -> List[Fill]:
        """
        Execute orders against the market (real or simulated).
        
        Args:
            orders: Orders to execute
            market_data: Current market data
            
        Returns:
            List of fills
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if successful, False otherwise
        """
        pass


class Orchestrator(ABC):
    """
    Coordinates the flow between all components.
    Manages the trading lifecycle.
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        pass
    
    @abstractmethod
    def step(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Execute a single step of the trading process at the given timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary with step results
        """
        pass
    
    @abstractmethod
    def run(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Run the trading process from start_date to end_date.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with run results
        """
        pass
