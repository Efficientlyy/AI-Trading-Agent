#!/usr/bin/env python3
"""
Dynamic Risk Limits Demo

This script demonstrates the functionality of the Dynamic Risk Limits module,
showcasing adaptive position sizing based on volatility, drawdown protection,
and circuit breakers for rapid market movements.
"""

import sys
import os
import logging
import asyncio
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Dynamic Risk Limits module
from src.risk.dynamic_risk_limits import DynamicRiskLimits, RiskLimitStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dynamic_risk_limits_demo')

# Create data directory
data_dir = Path('examples/data/risk')
data_dir.mkdir(parents=True, exist_ok=True)

def generate_price_data(symbol: str, days: int = 60, volatility: float = 0.02, 
                        trend: float = 0.0001, start_price: float = 100.0) -> pd.DataFrame:
    """
    Generate synthetic price data for a symbol with specified volatility and trend
    
    Args:
        symbol: Trading symbol
        days: Number of days of data to generate
        volatility: Daily volatility
        trend: Daily trend (positive for uptrend, negative for downtrend)
        start_price: Starting price
        
    Returns:
        DataFrame with timestamp and OHLCV data
    """
    # Generate timestamps (hourly data)
    now = datetime.now()
    hours = days * 24
    timestamps = [now - timedelta(hours=i) for i in range(hours)]
    timestamps.reverse()  # Oldest first
    
    # Generate returns with random noise and trend
    returns = np.random.normal(trend, volatility, hours)
    
    # Calculate prices from returns
    prices = [start_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    prices = prices[1:]  # Remove the seed price
    
    # Create OHLCV data
    data = []
    for i, timestamp in enumerate(timestamps):
        price = prices[i]
        high = price * (1 + random.uniform(0, volatility/2))
        low = price * (1 - random.uniform(0, volatility/2))
        open_price = price * (1 + random.uniform(-volatility/4, volatility/4))
        volume = random.uniform(10000, 100000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'symbol': symbol
        })
    
    return pd.DataFrame(data)

def simulate_portfolio_performance(initial_value: float = 100000.0, 
                                 days: int = 60, 
                                 data: Optional[Dict[str, pd.DataFrame]] = None) -> List[Tuple[datetime, float]]:
    """
    Simulate portfolio performance over time
    
    Args:
        initial_value: Initial portfolio value
        days: Number of days to simulate
        data: Dictionary of price data by symbol
        
    Returns:
        List of (timestamp, value) tuples
    """
    # Generate timestamps (daily data)
    now = datetime.now()
    timestamps = [now - timedelta(days=i) for i in range(days)]
    timestamps.reverse()  # Oldest first
    
    # Calculate portfolio value over time
    values = []
    current_value = initial_value
    
    # If no data provided, return constant value
    if data is None:
        return [(timestamp, initial_value) for timestamp in timestamps]
    
    # Randomly assign weights to each symbol
    symbols = list(data.keys())
    weights = [random.uniform(0.5, 1.5) for _ in symbols]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Track contribution of each symbol to total value
    symbol_values = {symbol: initial_value * weight for symbol, weight in zip(symbols, weights)}
    
    for i, timestamp in enumerate(timestamps):
        # Update value for each symbol based on price movement
        for j, symbol in enumerate(symbols):
            if i > 0:
                price_yesterday = data[symbol].iloc[i-1]['close']
                price_today = data[symbol].iloc[i]['close']
                price_change = price_today / price_yesterday - 1
                symbol_values[symbol] *= (1 + price_change)
        
        # Calculate total portfolio value
        portfolio_value = sum(symbol_values.values())
        values.append((timestamp, portfolio_value))
    
    return values

def create_market_scenarios(base_data: dict) -> dict:
    """
    Create market scenarios by modifying base data
    
    Args:
        base_data: Dictionary of base price data by symbol
        
    Returns:
        Dictionary of scenarios with modified price data
    """
    scenarios = {
        "base": base_data,
        "high_volatility": {},
        "sharp_decline": {},
        "slow_drawdown": {}
    }
    
    # High volatility scenario
    for symbol, data in base_data.items():
        # Copy the data
        high_vol_data = data.copy()
        
        # Increase volatility by applying random shocks
        for i in range(len(high_vol_data)):
            if i > 0 and random.random() < 0.2:  # 20% chance of shock
                shock_factor = random.uniform(-0.05, 0.05)  # 5% shock
                high_vol_data.at[i, 'close'] *= (1 + shock_factor)
                high_vol_data.at[i, 'high'] = max(high_vol_data.at[i, 'high'], high_vol_data.at[i, 'close'])
                high_vol_data.at[i, 'low'] = min(high_vol_data.at[i, 'low'], high_vol_data.at[i, 'close'])
        
        scenarios["high_volatility"][symbol] = high_vol_data
    
    # Sharp decline scenario
    for symbol, data in base_data.items():
        # Copy the data
        sharp_decline_data = data.copy()
        
        # Apply a sharp decline in the middle of the dataset
        midpoint = len(sharp_decline_data) // 2
        decline_factor = 0.15  # 15% decline
        
        for i in range(midpoint, min(midpoint + 10, len(sharp_decline_data))):
            decline_step = decline_factor / 10 * (i - midpoint + 1)
            sharp_decline_data.at[i, 'close'] *= (1 - decline_step)
            sharp_decline_data.at[i, 'high'] *= (1 - decline_step * 0.8)
            sharp_decline_data.at[i, 'low'] *= (1 - decline_step * 1.2)
            sharp_decline_data.at[i, 'open'] *= (1 - decline_step * 0.9)
        
        scenarios["sharp_decline"][symbol] = sharp_decline_data
    
    # Slow drawdown scenario
    for symbol, data in base_data.items():
        # Copy the data
        slow_drawdown_data = data.copy()
        
        # Apply a gradual decline over the entire dataset
        for i in range(1, len(slow_drawdown_data)):
            decline_step = 0.001 * i  # 0.1% per day, cumulative
            slow_drawdown_data.at[i, 'close'] *= (1 - decline_step)
            slow_drawdown_data.at[i, 'high'] *= (1 - decline_step * 0.9)
            slow_drawdown_data.at[i, 'low'] *= (1 - decline_step * 1.1)
            slow_drawdown_data.at[i, 'open'] *= (1 - decline_step * 0.95)
        
        scenarios["slow_drawdown"][symbol] = slow_drawdown_data
    
    return scenarios

def plot_risk_limits_effects(scenario_results: dict, output_path: str) -> None:
    """
    Create visualizations of risk limits effects
    
    Args:
        scenario_results: Dictionary of results by scenario
        output_path: Path to save the visualizations
    """
    # Set up the plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Position Size Multipliers
    plt.subplot(3, 1, 1)
    for scenario, results in scenario_results.items():
        for symbol, data in results['position_size_multipliers'].items():
            dates = [item[0] for item in data]
            values = [item[1] for item in data]
            plt.plot(dates, values, label=f"{scenario} - {symbol}")
    
    plt.title("Position Size Multipliers Over Time")
    plt.ylabel("Multiplier")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Portfolio Values
    plt.subplot(3, 1, 2)
    for scenario, results in scenario_results.items():
        dates = [item[0] for item in results['portfolio_values']]
        values = [item[1] for item in results['portfolio_values']]
        plt.plot(dates, values, label=f"{scenario}")
    
    plt.title("Portfolio Values Over Time")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Drawdowns
    plt.subplot(3, 1, 3)
    for scenario, results in scenario_results.items():
        dates = [item[0] for item in results['drawdowns']]
        values = [item[1] * 100 for item in results['drawdowns']]  # Convert to percentage
        plt.plot(dates, values, label=f"{scenario}")
    
    plt.title("Portfolio Drawdowns Over Time")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=5, color='r', linestyle='--', label="5% Threshold")
    plt.axhline(y=10, color='r', linestyle='-', label="10% Threshold")
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_circuit_breaker_events(scenario_results: dict, output_path: str) -> None:
    """
    Analyze and visualize circuit breaker events
    
    Args:
        scenario_results: Dictionary of results by scenario
        output_path: Path to save the visualization
    """
    # Set up the plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Collect circuit breaker events
    events_by_scenario = {}
    
    for scenario, results in scenario_results.items():
        events = results['circuit_breaker_events']
        events_by_scenario[scenario] = events
    
    # Plot price movement for each symbol with circuit breaker events
    symbols = set()
    for events in events_by_scenario.values():
        for event in events:
            symbols.add(event['symbol'])
    
    for i, symbol in enumerate(symbols):
        plt.subplot(len(symbols), 1, i+1)
        
        # Plot price for each scenario
        for scenario, results in scenario_results.items():
            if symbol in results['price_data']:
                price_data = results['price_data'][symbol]
                dates = price_data['timestamp']
                prices = price_data['close']
                plt.plot(dates, prices, label=f"{scenario}")
        
        # Mark circuit breaker events
        for scenario, events in events_by_scenario.items():
            for event in events:
                if event['symbol'] == symbol:
                    plt.axvline(x=event['timestamp'], color='r', linestyle='--', alpha=0.5)
                    plt.text(event['timestamp'], plt.ylim()[1]*0.9, 
                             f"Circuit Breaker: {scenario}", 
                             rotation=90, verticalalignment='top')
        
        plt.title(f"Price Movement with Circuit Breaker Events - {symbol}")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

async def run_scenario(scenario_name: str, price_data: dict, risk_manager: DynamicRiskLimits, 
                     symbols: list, initial_portfolio_value: float = 100000.0) -> dict:
    """
    Run a scenario with the risk manager
    
    Args:
        scenario_name: Name of the scenario
        price_data: Dictionary of price data by symbol
        risk_manager: DynamicRiskLimits instance
        symbols: List of symbols to monitor
        initial_portfolio_value: Initial portfolio value
        
    Returns:
        Dictionary with scenario results
    """
    logger.info(f"Starting scenario: {scenario_name}")
    
    # Set up tracking
    position_size_multipliers = {symbol: [] for symbol in symbols}
    portfolio_values = []
    drawdowns = []
    circuit_breaker_events = []
    limit_status_history = {symbol: [] for symbol in symbols}
    
    # Update market data
    for symbol, data in price_data.items():
        risk_manager.update_market_data(symbol, data)
    
    # Simulate portfolio performance
    portfolio_value_history = simulate_portfolio_performance(
        initial_value=initial_portfolio_value,
        days=len(price_data[symbols[0]]) // 24,  # Convert hourly to daily
        data=price_data
    )
    
    # Process each timestamp
    for timestamp, value in portfolio_value_history:
        # Update portfolio value
        risk_manager.update_portfolio_value(value)
        portfolio_values.append((timestamp, value))
        
        # Calculate and track drawdown
        drawdown = risk_manager.calculate_current_drawdown()
        drawdowns.append((timestamp, drawdown))
        
        # Check drawdown limits
        is_breached, current_drawdown = risk_manager.check_drawdown_limits()
        if is_breached:
            logger.warning(f"[{scenario_name}] Drawdown limit breached: {current_drawdown:.2%}")
        
        # Process each symbol
        daily_data = {symbol: data[data['timestamp'].dt.date == timestamp.date()] 
                      for symbol, data in price_data.items()}
        
        for symbol in symbols:
            # Skip if no data for this day
            if symbol not in daily_data or daily_data[symbol].empty:
                continue
                
            # Get last price for the day
            current_price = daily_data[symbol].iloc[-1]['close']
            
            # Check circuit breaker
            if risk_manager.check_circuit_breaker(symbol, current_price):
                logger.warning(f"[{scenario_name}] Circuit breaker triggered for {symbol} at price {current_price}")
                circuit_breaker_events.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': current_price
                })
            
            # Get position size multiplier
            multiplier = risk_manager.get_position_size_multiplier(symbol)
            position_size_multipliers[symbol].append((timestamp, multiplier))
            
            # Get limit statuses for this symbol
            for limit in risk_manager.risk_limits.get(symbol, []):
                limit_status_history[symbol].append({
                    'timestamp': timestamp,
                    'limit_type': limit.limit_type.name,
                    'status': limit.status.name,
                    'current_value': limit.current_value,
                    'threshold': limit.threshold
                })
    
    # Get final risk summary
    risk_summary = risk_manager.get_risk_summary()
    
    # Prepare results
    results = {
        'scenario': scenario_name,
        'position_size_multipliers': position_size_multipliers,
        'portfolio_values': portfolio_values,
        'drawdowns': drawdowns,
        'circuit_breaker_events': circuit_breaker_events,
        'limit_status_history': limit_status_history,
        'final_risk_summary': risk_summary,
        'price_data': price_data  # Include price data for analysis
    }
    
    logger.info(f"Completed scenario: {scenario_name}")
    return results

async def main():
    """Main function for the Dynamic Risk Limits demo"""
    logger.info("Starting Dynamic Risk Limits demo")
    
    # Generate synthetic data for multiple symbols
    symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD']
    
    # Generate base price data
    base_data = {}
    for symbol in symbols:
        params = {
            'BTC-USD': {'volatility': 0.03, 'trend': 0.0005, 'start_price': 30000},
            'ETH-USD': {'volatility': 0.04, 'trend': 0.0003, 'start_price': 2000},
            'ADA-USD': {'volatility': 0.05, 'trend': 0.0002, 'start_price': 0.5},
            'XRP-USD': {'volatility': 0.045, 'trend': 0.0001, 'start_price': 0.6}
        }
        
        data = generate_price_data(
            symbol=symbol,
            days=60,
            volatility=params[symbol]['volatility'],
            trend=params[symbol]['trend'],
            start_price=params[symbol]['start_price']
        )
        
        base_data[symbol] = data
    
    # Create market scenarios
    scenarios = create_market_scenarios(base_data)
    
    # Initialize risk manager
    initial_portfolio_value = 100000.0
    
    # Run each scenario
    scenario_results = {}
    
    for scenario_name, price_data in scenarios.items():
        # Create a fresh risk manager for each scenario
        risk_manager = DynamicRiskLimits()
        
        # Add risk limits for each symbol
        for symbol in symbols:
            # Add volatility-based limit
            volatility_threshold = 0.05  # 5% daily volatility threshold
            risk_manager.add_volatility_limit(
                symbol=symbol,
                threshold=volatility_threshold,
                lookback_period=20,
                volatility_scale=1.0
            )
            
            # Add drawdown limit
            max_drawdown = 0.1  # 10% max drawdown
            risk_manager.add_drawdown_limit(
                symbol=symbol,
                max_drawdown=max_drawdown,
                recovery_threshold=0.5
            )
            
            # Add circuit breaker
            price_move_threshold = 0.08  # 8% price move threshold
            risk_manager.add_circuit_breaker(
                symbol=symbol,
                price_move_threshold=price_move_threshold,
                time_window_minutes=60 * 24,  # 1 day
                cooldown_minutes=60 * 24 * 2  # 2 days
            )
        
        # Run the scenario
        results = await run_scenario(
            scenario_name=scenario_name,
            price_data=price_data,
            risk_manager=risk_manager,
            symbols=symbols,
            initial_portfolio_value=initial_portfolio_value
        )
        
        scenario_results[scenario_name] = results
    
    # Save results to file
    results_file = data_dir / 'dynamic_risk_limits_results.json'
    
    # Convert datetime objects to strings for JSON serialization
    for scenario, results in scenario_results.items():
        for key in ['position_size_multipliers', 'portfolio_values', 'drawdowns']:
            for symbol, data_list in results[key].items() if key == 'position_size_multipliers' else [('', results[key])]:
                for i, (timestamp, value) in enumerate(data_list):
                    if isinstance(timestamp, datetime):
                        if key == 'position_size_multipliers':
                            results[key][symbol][i] = (timestamp.isoformat(), value)
                        else:
                            results[key][i] = (timestamp.isoformat(), value)
        
        for event in results['circuit_breaker_events']:
            if isinstance(event['timestamp'], datetime):
                event['timestamp'] = event['timestamp'].isoformat()
        
        for symbol, history in results['limit_status_history'].items():
            for item in history:
                if isinstance(item['timestamp'], datetime):
                    item['timestamp'] = item['timestamp'].isoformat()
                    
        # Convert DataFrame to dict for JSON serialization
        for symbol, df in results['price_data'].items():
            df_dict = df.to_dict(orient='records')
            for record in df_dict:
                if isinstance(record['timestamp'], datetime):
                    record['timestamp'] = record['timestamp'].isoformat()
            results['price_data'][symbol] = df_dict
    
    # Save to file
    with open(results_file, 'w') as f:
        json.dump(scenario_results, f, indent=2)
    
    logger.info(f"Saved results to {results_file}")
    
    # Create visualizations
    plot_risk_limits_effects(
        scenario_results=scenario_results,
        output_path=str(data_dir / 'risk_limits_effects.png')
    )
    
    analyze_circuit_breaker_events(
        scenario_results=scenario_results,
        output_path=str(data_dir / 'circuit_breaker_events.png')
    )
    
    logger.info("Created visualizations")
    
    # Print summary
    print("\nDynamic Risk Limits Demo Summary:")
    print("=" * 50)
    
    for scenario, results in scenario_results.items():
        print(f"\nScenario: {scenario}")
        print("-" * 40)
        
        # Calculate average position size multiplier
        avg_multipliers = {}
        for symbol, data in results['position_size_multipliers'].items():
            values = [item[1] for item in data]
            avg_multipliers[symbol] = sum(values) / len(values) if values else 0
        
        print("Average Position Size Multipliers:")
        for symbol, avg in avg_multipliers.items():
            print(f"  {symbol}: {avg:.2f}")
        
        # Drawdown statistics
        drawdown_values = [item[1] for item in results['drawdowns']]
        max_drawdown = max(drawdown_values) if drawdown_values else 0
        
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        
        # Circuit breaker events
        print(f"Circuit Breaker Events: {len(results['circuit_breaker_events'])}")
        
        # Final portfolio value
        final_value = results['portfolio_values'][-1][1] if results['portfolio_values'] else 0
        value_change = final_value - initial_portfolio_value
        value_change_pct = value_change / initial_portfolio_value
        
        print(f"Final Portfolio Value: ${final_value:.2f} ({value_change_pct:+.2%})")
    
    print("\nDemo completed successfully!")
    logger.info("Dynamic Risk Limits demo completed")

if __name__ == "__main__":
    asyncio.run(main()) 