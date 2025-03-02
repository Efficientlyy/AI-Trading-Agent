#!/usr/bin/env python3
"""
Portfolio Management Example.

This script demonstrates how to use the PortfolioManager to manage trading positions
based on signals from strategies. It shows:
1. How to initialize a portfolio manager
2. How to process trading signals
3. How to update positions with price changes
4. How to calculate and display portfolio performance

This example simulates a trading session with randomly generated signals and price movements.
"""

import asyncio
import random
import sys
from datetime import datetime, timedelta
from decimal import Decimal
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.signals import Signal, SignalType
from src.portfolio.portfolio_manager import (
    PortfolioManager, Position, PositionType, PositionStatus, RiskParameters
)


async def generate_random_signal(
    strategy_id: str, 
    symbols: list,
    last_prices: dict,
    signal_type: SignalType = SignalType.ENTRY,
    confidence_range: tuple = (0.6, 0.95)
) -> Signal:
    """Generate a random trading signal for demonstration purposes.
    
    Args:
        strategy_id: ID of the strategy generating the signal
        symbols: List of available trading symbols
        last_prices: Dictionary of last known prices for each symbol
        signal_type: Type of signal to generate
        confidence_range: Range of possible confidence values
        
    Returns:
        A randomly generated Signal object
    """
    symbol = random.choice(symbols)
    direction = random.choice(["long", "short"])
    
    # Generate a price near the last known price
    base_price = last_prices.get(symbol, 1000.0)
    price_change_pct = random.uniform(-0.005, 0.005)  # ±0.5%
    price = base_price * (1 + price_change_pct)
    
    # Generate a confidence score
    confidence = random.uniform(*confidence_range)
    
    # Generate metadata with stop loss and take profit
    if direction == "long":
        stop_loss = price * 0.99  # 1% below entry
        take_profit = price * 1.02  # 2% above entry
    else:
        stop_loss = price * 1.01  # 1% above entry
        take_profit = price * 0.98  # 2% below entry
    
    metadata = {
        "stop_loss_price": stop_loss,
        "take_profit_price": take_profit,
        "suggested_size": random.uniform(0.01, 0.2)  # Suggest a position size
    }
    
    reason = f"Random {direction} signal for demonstration"
    
    return Signal(
        symbol=symbol,
        signal_type=signal_type,
        direction=direction,
        price=price,
        confidence=confidence,
        strategy_id=strategy_id,
        reason=reason,
        metadata=metadata
    )


async def simulate_price_updates(
    portfolio_manager: PortfolioManager,
    symbols: list,
    last_prices: dict,
    steps: int = 10,
    max_change_pct: float = 0.01
) -> None:
    """Simulate price movements and update the portfolio positions.
    
    Args:
        portfolio_manager: The portfolio manager to update
        symbols: List of trading symbols to simulate
        last_prices: Dictionary of last known prices for each symbol
        steps: Number of price update steps to simulate
        max_change_pct: Maximum percentage price change per step
    """
    print("\n== Simulating price movements ==")
    
    for step in range(steps):
        await asyncio.sleep(0.5)  # Simulate time passing
        
        print(f"\nPrice update step {step + 1}/{steps}")
        
        for symbol in symbols:
            # Generate a random price movement
            price_change_pct = random.uniform(-max_change_pct, max_change_pct)
            old_price = last_prices[symbol]
            new_price = old_price * (1 + price_change_pct)
            last_prices[symbol] = new_price
            
            print(f"  {symbol}: {old_price:.2f} -> {new_price:.2f} ({price_change_pct * 100:+.2f}%)")
            
            # Update the portfolio with the new price
            await portfolio_manager.update_prices(symbol, Decimal(str(new_price)))
        
        # Print the current portfolio state after each update
        portfolio_state = portfolio_manager.get_portfolio_state()
        print(f"\nPortfolio after step {step + 1}:")
        print(f"  Balance: {portfolio_state['current_balance']:.2f}")
        print(f"  Unrealized P&L: {portfolio_state['unrealized_pnl']:+.2f}")
        print(f"  Total value: {portfolio_state['total_value']:.2f}")
        print(f"  Return: {portfolio_state['total_return_pct']:+.2f}%")
        print(f"  Open positions: {portfolio_state['open_positions_count']}")


async def display_position_details(portfolio_manager: PortfolioManager) -> None:
    """Display detailed information about all positions.
    
    Args:
        portfolio_manager: The portfolio manager containing the positions
    """
    print("\n== Position Details ==")
    
    # Get all positions (open and closed)
    positions = portfolio_manager.positions.values()
    
    if not positions:
        print("No positions to display.")
        return
    
    print(f"Total positions: {len(positions)}")
    
    # Display open positions first
    open_positions = portfolio_manager.get_open_positions()
    if open_positions:
        print("\nOpen Positions:")
        for position in open_positions:
            print(f"  ID: {position.position_id[:8]}... | {position.symbol} | "
                  f"{position.position_type.value} | Size: {position.size} | "
                  f"Entry: {position.entry_price} | Current: {position.last_price or 'N/A'} | "
                  f"Unrealized P&L: {position.calculate_unrealized_pnl(position.last_price) if position.last_price else 'N/A'}")
    
    # Display closed positions
    closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
    if closed_positions:
        print("\nClosed Positions:")
        for position in closed_positions:
            print(f"  ID: {position.position_id[:8]}... | {position.symbol} | "
                  f"{position.position_type.value} | Size: {position.size} | "
                  f"Entry: {position.entry_price} | Exit: {position.exit_price} | "
                  f"P&L: {position.realized_pnl:+.4f} ({position.calculate_pnl_pct(position.exit_price):+.2f}%)")


async def main() -> None:
    """Run the portfolio management example."""
    print("=== Portfolio Management Example ===")
    
    # Set up initial parameters
    initial_balance = Decimal("10000.00")  # Starting with $10,000
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    
    # Initial prices for our symbols
    last_prices = {
        "BTC/USDT": 50000.0,
        "ETH/USDT": 3000.0,
        "SOL/USDT": 100.0,
        "XRP/USDT": 0.5
    }
    
    # Create strategies that will generate signals
    strategies = {
        "trend_following": "Trend Following Strategy",
        "mean_reversion": "Mean Reversion Strategy",
        "breakout": "Breakout Strategy"
    }
    
    # Custom risk parameters
    risk_params = RiskParameters(
        max_position_size=Decimal("0.1"),  # Max 10% of portfolio per position
        max_risk_per_trade_pct=Decimal("0.01"),  # Risk 1% per trade
        max_risk_per_day_pct=Decimal("0.03"),  # Risk 3% per day
        max_open_positions=5,  # Maximum 5 open positions
        max_open_positions_per_symbol=1,  # Maximum 1 position per symbol
        max_drawdown_pct=Decimal("0.2")  # Stop trading at 20% drawdown
    )
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(
        initial_balance=initial_balance,
        risk_parameters=risk_params,
        name="example_portfolio"
    )
    
    # Simulate receiving signals and opening positions
    print("\n== Simulating Trading Signals ==")
    
    # Generate and process 10 random signals
    signals_to_generate = 10
    for i in range(signals_to_generate):
        await asyncio.sleep(0.2)  # Simulate time passing between signals
        
        # Choose random strategy
        strategy_id = random.choice(list(strategies.keys()))
        
        # Generate a random signal
        signal = await generate_random_signal(
            strategy_id=strategy_id,
            symbols=symbols,
            last_prices=last_prices
        )
        
        print(f"\nReceived signal {i+1}/{signals_to_generate}:")
        print(f"  {signal}")
        
        # Process the signal
        position = await portfolio_manager.process_signal(signal)
        
        if position:
            print(f"  Opened position: {position.position_id[:8]}... | "
                  f"{position.symbol} {position.position_type.value} | "
                  f"Size: {position.size} | Price: {position.entry_price}")
        else:
            print("  Signal did not result in a position")
    
    # Simulate price movements and portfolio updates
    await simulate_price_updates(
        portfolio_manager=portfolio_manager,
        symbols=symbols,
        last_prices=last_prices,
        steps=15,
        max_change_pct=0.02  # Maximum 2% price change per step
    )
    
    # Display detailed position information
    await display_position_details(portfolio_manager)
    
    # Final portfolio state
    print("\n== Final Portfolio State ==")
    portfolio_state = portfolio_manager.get_portfolio_state()
    for key, value in portfolio_state.items():
        print(f"{key}: {value}")
    
    # Check risk limits
    print("\n== Risk Limit Status ==")
    risk_status = portfolio_manager.check_risk_limits()
    for key, value in risk_status.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main()) 