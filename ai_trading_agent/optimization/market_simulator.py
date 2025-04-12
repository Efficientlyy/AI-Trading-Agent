"""
Market Condition Simulator for AI Trading Agent.

This module provides tools for simulating various market conditions to test
trading strategies under different scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Tuple, Optional
import datetime
from enum import Enum
import matplotlib.pyplot as plt
import os

class MarketCondition(Enum):
    """Types of market conditions to simulate."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CRASH = "crash"
    RECOVERY = "recovery"
    CUSTOM = "custom"

class MarketSimulator:
    """
    Simulator for generating synthetic market data under various conditions.
    
    Features:
    - Generate synthetic price data for multiple assets
    - Simulate various market conditions (bull, bear, volatile, etc.)
    - Control correlation between assets
    - Add realistic noise and anomalies
    - Generate volume and other market data
    """
    
    def __init__(
        self,
        start_date: Union[str, datetime.datetime] = "2020-01-01",
        end_date: Union[str, datetime.datetime] = "2020-12-31",
        symbols: List[str] = None,
        base_prices: Dict[str, float] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the market simulator.
        
        Args:
            start_date: Start date for the simulation
            end_date: End date for the simulation
            symbols: List of symbols to simulate
            base_prices: Dictionary of base prices for each symbol
            random_seed: Random seed for reproducibility
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.symbols = symbols or ["BTC/USD", "ETH/USD", "SOL/USD"]
        self.base_prices = base_prices or {
            "BTC/USD": 50000,
            "ETH/USD": 3000,
            "SOL/USD": 150
        }
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate date range
        self.dates = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        self.n_days = len(self.dates)
    
    def generate_bull_market(
        self,
        trend_strength: float = 0.1,
        volatility: float = 0.02,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a bull market scenario with upward trend.
        
        Args:
            trend_strength: Strength of the upward trend (daily percentage)
            volatility: Daily volatility
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        return self._generate_market_data(
            trend=trend_strength,
            volatility=volatility,
            correlation_matrix=correlation_matrix
        )
    
    def generate_bear_market(
        self,
        trend_strength: float = -0.08,
        volatility: float = 0.025,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a bear market scenario with downward trend.
        
        Args:
            trend_strength: Strength of the downward trend (daily percentage)
            volatility: Daily volatility
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        return self._generate_market_data(
            trend=trend_strength,
            volatility=volatility,
            correlation_matrix=correlation_matrix
        )
    
    def generate_sideways_market(
        self,
        volatility: float = 0.01,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a sideways market with no clear trend.
        
        Args:
            volatility: Daily volatility
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        return self._generate_market_data(
            trend=0.0,
            volatility=volatility,
            correlation_matrix=correlation_matrix
        )
    
    def generate_volatile_market(
        self,
        trend: float = 0.0,
        volatility: float = 0.04,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a volatile market with high price fluctuations.
        
        Args:
            trend: Overall trend (daily percentage)
            volatility: Daily volatility
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        return self._generate_market_data(
            trend=trend,
            volatility=volatility,
            correlation_matrix=correlation_matrix
        )
    
    def generate_crash(
        self,
        crash_day: Optional[int] = None,
        crash_severity: float = -0.3,
        recovery_rate: float = 0.02,
        pre_crash_trend: float = 0.05,
        volatility: float = 0.03,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a market crash scenario.
        
        Args:
            crash_day: Day of the crash (None = random)
            crash_severity: Severity of the crash (percentage drop)
            recovery_rate: Daily recovery rate after crash
            pre_crash_trend: Trend before the crash
            volatility: Daily volatility
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        if crash_day is None:
            # Random crash day in the first 2/3 of the period
            crash_day = np.random.randint(int(self.n_days * 0.1), int(self.n_days * 0.7))
        
        # Generate correlated returns
        returns = self._generate_correlated_returns(
            n_days=self.n_days,
            n_assets=len(self.symbols),
            correlation_matrix=correlation_matrix
        )
        
        # Apply trends and crash
        market_data = {}
        for i, symbol in enumerate(self.symbols):
            base_price = self.base_prices.get(symbol, 100)
            
            # Create price series with pre-crash trend
            prices = np.zeros(self.n_days)
            prices[0] = base_price
            
            # Pre-crash phase
            for day in range(1, crash_day):
                daily_return = pre_crash_trend + volatility * returns[day, i]
                prices[day] = prices[day-1] * (1 + daily_return)
            
            # Crash day
            prices[crash_day] = prices[crash_day-1] * (1 + crash_severity)
            
            # Post-crash recovery
            for day in range(crash_day + 1, self.n_days):
                # Gradually decreasing volatility after crash
                days_since_crash = day - crash_day
                current_volatility = volatility * (1 + 0.5 * np.exp(-days_since_crash / 10))
                
                daily_return = recovery_rate + current_volatility * returns[day, i]
                prices[day] = prices[day-1] * (1 + daily_return)
            
            # Create OHLCV data
            market_data[symbol] = self._create_ohlcv_from_prices(prices)
        
        return market_data
    
    def generate_custom_scenario(
        self,
        phases: List[Dict[str, Any]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a custom market scenario with multiple phases.
        
        Args:
            phases: List of phase configurations
                Format: [
                    {
                        "duration": 60,  # days
                        "trend": 0.05,
                        "volatility": 0.02
                    },
                    ...
                ]
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        # Generate correlated returns
        returns = self._generate_correlated_returns(
            n_days=self.n_days,
            n_assets=len(self.symbols),
            correlation_matrix=correlation_matrix
        )
        
        # Apply phases
        market_data = {}
        for i, symbol in enumerate(self.symbols):
            base_price = self.base_prices.get(symbol, 100)
            
            # Create price series
            prices = np.zeros(self.n_days)
            prices[0] = base_price
            
            day = 1
            for phase in phases:
                duration = min(phase.get("duration", 30), self.n_days - day)
                trend = phase.get("trend", 0.0)
                volatility = phase.get("volatility", 0.02)
                
                for d in range(duration):
                    if day >= self.n_days:
                        break
                    
                    daily_return = trend + volatility * returns[day, i]
                    prices[day] = prices[day-1] * (1 + daily_return)
                    day += 1
            
            # Fill any remaining days with sideways market
            while day < self.n_days:
                daily_return = 0.0 + 0.01 * returns[day, i]
                prices[day] = prices[day-1] * (1 + daily_return)
                day += 1
            
            # Create OHLCV data
            market_data[symbol] = self._create_ohlcv_from_prices(prices)
        
        return market_data
    
    def generate_market_condition(
        self,
        condition: MarketCondition,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate market data for a specific condition.
        
        Args:
            condition: Market condition to simulate
            **kwargs: Additional parameters for the condition
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        if condition == MarketCondition.BULL_MARKET:
            return self.generate_bull_market(**kwargs)
        elif condition == MarketCondition.BEAR_MARKET:
            return self.generate_bear_market(**kwargs)
        elif condition == MarketCondition.SIDEWAYS:
            return self.generate_sideways_market(**kwargs)
        elif condition == MarketCondition.VOLATILE:
            return self.generate_volatile_market(**kwargs)
        elif condition == MarketCondition.CRASH:
            return self.generate_crash(**kwargs)
        elif condition == MarketCondition.CUSTOM:
            return self.generate_custom_scenario(**kwargs)
        else:
            raise ValueError(f"Unsupported market condition: {condition}")
    
    def _generate_market_data(
        self,
        trend: float,
        volatility: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate market data with specified trend and volatility.
        
        Args:
            trend: Daily trend (percentage)
            volatility: Daily volatility
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        # Generate correlated returns
        returns = self._generate_correlated_returns(
            n_days=self.n_days,
            n_assets=len(self.symbols),
            correlation_matrix=correlation_matrix
        )
        
        # Apply trend and volatility
        market_data = {}
        for i, symbol in enumerate(self.symbols):
            base_price = self.base_prices.get(symbol, 100)
            
            # Create price series
            prices = np.zeros(self.n_days)
            prices[0] = base_price
            
            for day in range(1, self.n_days):
                daily_return = trend + volatility * returns[day, i]
                prices[day] = prices[day-1] * (1 + daily_return)
            
            # Create OHLCV data
            market_data[symbol] = self._create_ohlcv_from_prices(prices)
        
        return market_data
    
    def _generate_correlated_returns(
        self,
        n_days: int,
        n_assets: int,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate correlated random returns.
        
        Args:
            n_days: Number of days
            n_assets: Number of assets
            correlation_matrix: Correlation matrix between assets
            
        Returns:
            Array of correlated returns [n_days, n_assets]
        """
        # Default correlation matrix (moderate positive correlation)
        if correlation_matrix is None:
            correlation_matrix = np.ones((n_assets, n_assets)) * 0.5
            np.fill_diagonal(correlation_matrix, 1.0)
        
        # Ensure correlation matrix is valid
        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"Correlation matrix shape {correlation_matrix.shape} "
                           f"does not match number of assets {n_assets}")
        
        # Cholesky decomposition for correlated random numbers
        L = np.linalg.cholesky(correlation_matrix)
        
        # Generate uncorrelated random returns
        uncorrelated_returns = np.random.normal(0, 1, size=(n_days, n_assets))
        
        # Apply correlation
        correlated_returns = np.zeros((n_days, n_assets))
        for day in range(n_days):
            correlated_returns[day] = np.dot(L, uncorrelated_returns[day])
        
        return correlated_returns
    
    def _create_ohlcv_from_prices(self, prices: np.ndarray) -> pd.DataFrame:
        """
        Create OHLCV data from price series.
        
        Args:
            prices: Array of close prices
            
        Returns:
            DataFrame with OHLCV data
        """
        n_days = len(prices)
        
        # Generate realistic OHLC data
        high = np.zeros(n_days)
        low = np.zeros(n_days)
        open_prices = np.zeros(n_days)
        
        # First day
        open_prices[0] = prices[0] * (1 - 0.005 + 0.01 * np.random.random())
        high[0] = max(open_prices[0], prices[0]) * (1 + 0.005 * np.random.random())
        low[0] = min(open_prices[0], prices[0]) * (1 - 0.005 * np.random.random())
        
        # Remaining days
        for i in range(1, n_days):
            # Open price is influenced by previous close
            open_prices[i] = prices[i-1] * (1 - 0.01 + 0.02 * np.random.random())
            
            # High and low based on open and close
            high[i] = max(open_prices[i], prices[i]) * (1 + 0.01 * np.random.random())
            low[i] = min(open_prices[i], prices[i]) * (1 - 0.01 * np.random.random())
        
        # Generate volume (higher on volatile days)
        volume = np.zeros(n_days)
        for i in range(n_days):
            if i > 0:
                price_change = abs(prices[i] / prices[i-1] - 1)
                # Higher volume on days with larger price changes
                volume_factor = 1.0 + 5.0 * price_change
            else:
                volume_factor = 1.0
            
            # Base volume with some randomness
            base_volume = 1000 * prices[i]
            volume[i] = base_volume * volume_factor * (0.5 + np.random.random())
        
        # Create DataFrame
        df = pd.DataFrame({
            "open": open_prices,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume
        }, index=self.dates)
        
        return df
    
    def save_market_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        output_dir: str,
        condition_name: str = "custom"
    ):
        """
        Save market data to CSV files.
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data
            output_dir: Output directory
            condition_name: Name of the market condition
        """
        # Create output directory
        condition_dir = os.path.join(output_dir, condition_name)
        os.makedirs(condition_dir, exist_ok=True)
        
        # Save each symbol's data
        for symbol, df in market_data.items():
            # Replace / with _ in symbol name for filename
            safe_symbol = symbol.replace("/", "_")
            filename = os.path.join(condition_dir, f"{safe_symbol}.csv")
            df.to_csv(filename)
        
        print(f"Market data saved to {condition_dir}")
    
    def plot_market_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        output_dir: Optional[str] = None,
        condition_name: str = "custom"
    ):
        """
        Plot market data.
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data
            output_dir: Output directory (None = display only)
            condition_name: Name of the market condition
        """
        # Plot price chart for each symbol
        for symbol, df in market_data.items():
            plt.figure(figsize=(12, 8))
            
            # Plot OHLC
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df["close"], label="Close")
            plt.title(f"{symbol} - {condition_name}")
            plt.ylabel("Price")
            plt.grid(True)
            plt.legend()
            
            # Plot volume
            plt.subplot(2, 1, 2)
            plt.bar(df.index, df["volume"], width=1.0, alpha=0.5)
            plt.ylabel("Volume")
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save or display
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                safe_symbol = symbol.replace("/", "_")
                filename = os.path.join(output_dir, f"{safe_symbol}_{condition_name}.png")
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()
        
        # Plot comparison of all symbols
        plt.figure(figsize=(12, 8))
        
        for symbol, df in market_data.items():
            # Normalize to starting price
            normalized_price = df["close"] / df["close"].iloc[0]
            plt.plot(df.index, normalized_price, label=symbol)
        
        plt.title(f"Normalized Price Comparison - {condition_name}")
        plt.ylabel("Normalized Price")
        plt.grid(True)
        plt.legend()
        
        # Save or display
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"comparison_{condition_name}.png")
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

def generate_market_scenarios(
    output_dir: str = "market_scenarios",
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
    symbols: List[str] = None,
    base_prices: Dict[str, float] = None,
    random_seed: int = 42
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate a set of standard market scenarios for strategy testing.
    
    Args:
        output_dir: Output directory for saving data
        start_date: Start date for the simulation
        end_date: End date for the simulation
        symbols: List of symbols to simulate
        base_prices: Dictionary of base prices for each symbol
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary of market data for each scenario
    """
    # Create simulator
    simulator = MarketSimulator(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        base_prices=base_prices,
        random_seed=random_seed
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate scenarios
    scenarios = {}
    
    # Bull market
    bull_market = simulator.generate_bull_market(trend_strength=0.08, volatility=0.02)
    scenarios["bull_market"] = bull_market
    simulator.save_market_data(bull_market, output_dir, "bull_market")
    simulator.plot_market_data(bull_market, output_dir, "bull_market")
    
    # Bear market
    bear_market = simulator.generate_bear_market(trend_strength=-0.06, volatility=0.025)
    scenarios["bear_market"] = bear_market
    simulator.save_market_data(bear_market, output_dir, "bear_market")
    simulator.plot_market_data(bear_market, output_dir, "bear_market")
    
    # Sideways market
    sideways = simulator.generate_sideways_market(volatility=0.015)
    scenarios["sideways"] = sideways
    simulator.save_market_data(sideways, output_dir, "sideways")
    simulator.plot_market_data(sideways, output_dir, "sideways")
    
    # Volatile market
    volatile = simulator.generate_volatile_market(trend=0.02, volatility=0.04)
    scenarios["volatile"] = volatile
    simulator.save_market_data(volatile, output_dir, "volatile")
    simulator.plot_market_data(volatile, output_dir, "volatile")
    
    # Market crash
    crash = simulator.generate_crash(
        crash_day=int(simulator.n_days * 0.4),
        crash_severity=-0.25,
        recovery_rate=0.015,
        pre_crash_trend=0.04,
        volatility=0.03
    )
    scenarios["crash"] = crash
    simulator.save_market_data(crash, output_dir, "crash")
    simulator.plot_market_data(crash, output_dir, "crash")
    
    # Custom multi-phase scenario
    custom = simulator.generate_custom_scenario(phases=[
        {"duration": 60, "trend": 0.05, "volatility": 0.02},  # Initial bull
        {"duration": 30, "trend": -0.02, "volatility": 0.03},  # Correction
        {"duration": 90, "trend": 0.03, "volatility": 0.02},  # Recovery
        {"duration": 45, "trend": -0.04, "volatility": 0.035},  # Decline
        {"duration": 60, "trend": 0.01, "volatility": 0.015},  # Consolidation
        {"duration": 80, "trend": 0.06, "volatility": 0.025},  # Final rally
    ])
    scenarios["multi_phase"] = custom
    simulator.save_market_data(custom, output_dir, "multi_phase")
    simulator.plot_market_data(custom, output_dir, "multi_phase")
    
    return scenarios
