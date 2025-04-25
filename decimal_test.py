"""
Simple test script to verify the Decimal refactoring.
"""
from decimal import Decimal
import sys

# Add the project root to the Python path
sys.path.append(".")

from ai_trading_agent.trading_engine.models import Position, Portfolio, PositionSide, OrderSide
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager

def test_position_decimal():
    """Test that Position class correctly handles Decimal values."""
    print("\n=== Testing Position with Decimal ===")
    
    # Create a position with Decimal values
    position = Position(
        symbol="BTC/USD",
        side=PositionSide.LONG,
        quantity=Decimal("1.0"),
        entry_price=Decimal("50000.0")
    )
    
    print(f"Position created: {position}")
    print(f"Quantity type: {type(position.quantity)}")
    print(f"Entry price type: {type(position.entry_price)}")
    
    # Test updating market price
    position.update_market_price(Decimal("55000.0"))
    print(f"Updated market price to 55000.0")
    print(f"Unrealized PnL: {position.unrealized_pnl}")
    print(f"Unrealized PnL type: {type(position.unrealized_pnl)}")
    
    # Test updating position
    position.update_position(
        trade_qty=Decimal("0.5"),
        trade_price=Decimal("60000.0"),
        trade_side=OrderSide.BUY,
        current_market_price=Decimal("60000.0")
    )
    print(f"Updated position with 0.5 BTC at 60000.0")
    print(f"New quantity: {position.quantity}")
    print(f"New entry price: {position.entry_price}")
    
    return position

def test_portfolio_decimal():
    """Test that Portfolio class correctly handles Decimal values."""
    print("\n=== Testing Portfolio with Decimal ===")
    
    # Create a portfolio with Decimal initial capital
    portfolio = Portfolio(initial_capital=Decimal("10000.0"))
    
    print(f"Portfolio created with initial capital: {portfolio.starting_balance}")
    print(f"Starting balance type: {type(portfolio.starting_balance)}")
    print(f"Current balance type: {type(portfolio.current_balance)}")
    
    # Add a position through a trade
    from ai_trading_agent.trading_engine.models import Trade
    
    trade = Trade(
        symbol="ETH/USD",
        order_id="order123",
        side=OrderSide.BUY,
        quantity=Decimal("2.0"),
        price=Decimal("3000.0")
    )
    
    # Update portfolio with the trade
    current_prices = {"ETH/USD": Decimal("3100.0")}
    portfolio.update_from_trade(trade, current_prices)
    
    print(f"Updated portfolio with ETH trade")
    print(f"Current balance: {portfolio.current_balance}")
    print(f"Position quantity: {portfolio.positions['ETH/USD'].quantity}")
    print(f"Position entry price: {portfolio.positions['ETH/USD'].entry_price}")
    print(f"Position unrealized PnL: {portfolio.positions['ETH/USD'].unrealized_pnl}")
    
    # Test total equity
    print(f"Total equity: {portfolio.total_equity}")
    print(f"Total equity type: {type(portfolio.total_equity)}")
    
    return portfolio

def test_portfolio_manager_decimal():
    """Test that PortfolioManager class correctly handles Decimal values."""
    print("\n=== Testing PortfolioManager with Decimal ===")
    
    # Create a portfolio manager with Decimal values
    manager = PortfolioManager(
        initial_capital=Decimal("10000.0"),
        risk_per_trade=Decimal("0.02"),
        max_position_size=Decimal("0.2")
    )
    
    print(f"Portfolio manager created")
    print(f"Risk per trade: {manager.risk_per_trade}")
    print(f"Risk per trade type: {type(manager.risk_per_trade)}")
    print(f"Max position size: {manager.max_position_size}")
    print(f"Max position size type: {type(manager.max_position_size)}")
    
    # Test calculate_position_size
    symbol = "BTC/USD"
    price = Decimal("50000.0")
    stop_loss = Decimal("45000.0")
    
    position_size = manager.calculate_position_size(
        symbol=symbol,
        price=price,
        stop_loss=stop_loss
    )
    
    print(f"Calculated position size for {symbol} at {price} with stop loss {stop_loss}: {position_size}")
    print(f"Position size type: {type(position_size)}")
    
    # Test with float inputs (should be converted to Decimal)
    position_size_from_float = manager.calculate_position_size(
        symbol=symbol,
        price=float(50000.0),
        stop_loss=float(45000.0)
    )
    
    print(f"Calculated position size with float inputs: {position_size_from_float}")
    print(f"Position size type: {type(position_size_from_float)}")
    
    return manager

if __name__ == "__main__":
    print("Running Decimal refactoring tests...")
    
    position = test_position_decimal()
    portfolio = test_portfolio_decimal()
    manager = test_portfolio_manager_decimal()
    
    print("\nAll tests completed successfully!")
