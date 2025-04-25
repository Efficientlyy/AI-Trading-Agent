"""
Unit tests specifically for Decimal precision in the trading engine.

These tests verify that all monetary calculations in the trading engine
use Decimal for proper precision and avoid floating-point errors.
"""
import pytest
from decimal import Decimal
import pandas as pd

from ai_trading_agent.trading_engine.models import (
    Position, Portfolio, Trade, Order,
    OrderSide, OrderType, PositionSide
)
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager


class TestDecimalPrecision:
    """Test cases for Decimal precision in the trading engine."""
    
    @pytest.fixture
    def sample_position(self):
        """Create a sample position with Decimal values."""
        return Position(
            symbol="BTC/USD",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0")
        )
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio with Decimal initial capital."""
        return Portfolio(initial_capital=Decimal("10000.0"))
    
    @pytest.fixture
    def sample_portfolio_manager(self):
        """Create a sample portfolio manager with Decimal values."""
        manager = PortfolioManager(
            initial_capital=Decimal("10000.0"),
            risk_per_trade=Decimal("0.02"),
            max_position_size=Decimal("0.2")
        )
        # Ensure portfolio.total_value is set correctly
        manager.portfolio.total_value = Decimal("10000.0")
        return manager
    
    def test_position_decimal_attributes(self, sample_position):
        """Test that Position attributes are Decimal."""
        assert isinstance(sample_position.quantity, Decimal)
        assert isinstance(sample_position.entry_price, Decimal)
        assert isinstance(sample_position.unrealized_pnl, Decimal)
        assert isinstance(sample_position.realized_pnl, Decimal)
    
    def test_position_update_market_price(self, sample_position):
        """Test that Position.update_market_price handles Decimal correctly."""
        # Update with Decimal
        sample_position.update_market_price(Decimal("55000.0"))
        assert isinstance(sample_position.unrealized_pnl, Decimal)
        assert sample_position.unrealized_pnl == Decimal("5000.0")
        
        # Update with float (should be converted to Decimal)
        sample_position.update_market_price(60000.0)
        assert isinstance(sample_position.unrealized_pnl, Decimal)
        assert sample_position.unrealized_pnl == Decimal("10000.0")
    
    def test_position_update_position(self, sample_position):
        """Test that Position.update_position handles Decimal correctly."""
        # Update with Decimal
        sample_position.update_position(
            Decimal("0.5"),
            Decimal("60000.0"),
            OrderSide.BUY,
            Decimal("60000.0")
        )
        assert isinstance(sample_position.quantity, Decimal)
        assert isinstance(sample_position.entry_price, Decimal)
        assert sample_position.quantity == Decimal("1.5")
        
        # Calculate expected new entry price: ((1.0 * 50000) + (0.5 * 60000)) / 1.5
        # Don't check exact equality due to potential precision differences
        # Instead, check that the values are close enough
        assert abs(sample_position.entry_price - Decimal("53333.33333333")) < Decimal("0.00001")
        
        # Update with float (should be converted to Decimal)
        sample_position.update_position(
            0.5,
            65000.0,
            OrderSide.BUY,
            65000.0
        )
        assert isinstance(sample_position.quantity, Decimal)
        assert isinstance(sample_position.entry_price, Decimal)
        assert sample_position.quantity == Decimal("2.0")
    
    def test_portfolio_decimal_attributes(self, sample_portfolio):
        """Test that Portfolio attributes are Decimal."""
        assert isinstance(sample_portfolio.starting_balance, Decimal)
        assert isinstance(sample_portfolio.current_balance, Decimal)
        assert isinstance(sample_portfolio.total_realized_portfolio_pnl, Decimal)
        assert isinstance(sample_portfolio.total_value, Decimal)
    
    def test_portfolio_update_from_trade(self, sample_portfolio):
        """Test that Portfolio.update_from_trade handles Decimal correctly."""
        # Create a trade with Decimal values
        trade = Trade(
            symbol="ETH/USD",
            order_id="order123",
            side=OrderSide.BUY,
            quantity=Decimal("2.0"),
            price=Decimal("3000.0")
        )
        
        # Update portfolio with the trade
        current_prices = {"ETH/USD": Decimal("3100.0")}
        sample_portfolio.update_from_trade(trade, current_prices)
        
        # Check that the position was created with Decimal values
        assert "ETH/USD" in sample_portfolio.positions
        position = sample_portfolio.positions["ETH/USD"]
        assert isinstance(position.quantity, Decimal)
        assert isinstance(position.entry_price, Decimal)
        assert isinstance(position.unrealized_pnl, Decimal)
        
        # Check that the cash balance was updated correctly
        expected_balance = Decimal("10000.0") - (Decimal("2.0") * Decimal("3000.0"))
        assert sample_portfolio.current_balance == expected_balance
        
        # Create a trade with float values (should be converted to Decimal)
        trade2 = Trade(
            symbol="BTC/USD",
            order_id="order456",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        
        # Update portfolio with the trade
        current_prices["BTC/USD"] = 51000.0
        sample_portfolio.update_from_trade(trade2, current_prices)
        
        # Check that the position was created with Decimal values
        assert "BTC/USD" in sample_portfolio.positions
        position = sample_portfolio.positions["BTC/USD"]
        assert isinstance(position.quantity, Decimal)
        assert isinstance(position.entry_price, Decimal)
        assert isinstance(position.unrealized_pnl, Decimal)
        
        # Check that the cash balance was updated correctly
        expected_balance -= Decimal("0.1") * Decimal("50000.0")
        assert sample_portfolio.current_balance == expected_balance
    
    def test_portfolio_total_equity(self, sample_portfolio):
        """Test that Portfolio.total_equity handles Decimal correctly."""
        # Add positions to the portfolio
        sample_portfolio.positions["BTC/USD"] = Position(
            symbol="BTC/USD",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.0")
        )
        sample_portfolio.positions["ETH/USD"] = Position(
            symbol="ETH/USD",
            side=PositionSide.LONG,
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.0")
        )
        
        # Update unrealized PnL
        current_prices = {
            "BTC/USD": Decimal("55000.0"),
            "ETH/USD": Decimal("3100.0")
        }
        sample_portfolio.update_all_unrealized_pnl(current_prices)
        
        # Calculate expected total equity
        btc_pnl = (Decimal("55000.0") - Decimal("50000.0")) * Decimal("0.1")
        eth_pnl = (Decimal("3100.0") - Decimal("3000.0")) * Decimal("2.0")
        expected_equity = sample_portfolio.current_balance + btc_pnl + eth_pnl
        
        # Check that total_equity is a Decimal and has the correct value
        assert isinstance(sample_portfolio.total_equity, Decimal)
        assert sample_portfolio.total_equity == expected_equity
    
    def test_portfolio_manager_decimal_attributes(self, sample_portfolio_manager):
        """Test that PortfolioManager attributes are Decimal."""
        assert isinstance(sample_portfolio_manager.risk_per_trade, Decimal)
        assert isinstance(sample_portfolio_manager.max_position_size, Decimal)
        assert isinstance(sample_portfolio_manager.portfolio.starting_balance, Decimal)
        assert isinstance(sample_portfolio_manager.portfolio.current_balance, Decimal)
    
    def test_portfolio_manager_calculate_position_size(self, sample_portfolio_manager):
        """Test that PortfolioManager.calculate_position_size handles Decimal correctly."""
        # Ensure portfolio value is set correctly
        assert sample_portfolio_manager.portfolio.total_value == Decimal("10000.0")
        
        # Calculate position size with Decimal inputs
        position_size = sample_portfolio_manager.calculate_position_size(
            symbol="BTC/USD",
            price=Decimal("50000.0"),
            stop_loss=Decimal("45000.0")
        )
        
        # Check that position_size is a Decimal
        assert isinstance(position_size, Decimal)
        
        # Calculate expected position size
        # risk_amount = portfolio_value * risk_per_trade = 10000 * 0.02 = 200
        # risk_per_unit = price - stop_loss = 50000 - 45000 = 5000
        # position_size = risk_amount / risk_per_unit = 200 / 5000 = 0.04
        expected_position_size = Decimal("0.04")
        assert position_size == expected_position_size
        
        # Calculate position size with float inputs (should be converted to Decimal)
        position_size = sample_portfolio_manager.calculate_position_size(
            symbol="BTC/USD",
            price=50000.0,
            stop_loss=45000.0
        )
        
        # Check that position_size is a Decimal and has the correct value
        assert isinstance(position_size, Decimal)
        assert position_size == expected_position_size
    
    def test_portfolio_manager_update_market_prices(self, sample_portfolio_manager):
        """Test that PortfolioManager.update_market_prices handles Decimal correctly."""
        # Add a position to the portfolio
        sample_portfolio_manager.portfolio.positions["BTC/USD"] = Position(
            symbol="BTC/USD",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.0")
        )
        
        # Update market prices with Decimal values
        prices = {"BTC/USD": Decimal("55000.0")}
        timestamp = pd.Timestamp('2023-01-01')
        sample_portfolio_manager.update_market_prices(prices, timestamp)
        
        # Check that the position's unrealized PnL was updated correctly
        position = sample_portfolio_manager.portfolio.positions["BTC/USD"]
        expected_pnl = (Decimal("55000.0") - Decimal("50000.0")) * Decimal("0.1")
        assert position.unrealized_pnl == expected_pnl
        
        # Update market prices with float values (should be converted to Decimal)
        prices = {"BTC/USD": 60000.0}
        timestamp = pd.Timestamp('2023-01-02')
        sample_portfolio_manager.update_market_prices(prices, timestamp)
        
        # Check that the position's unrealized PnL was updated correctly
        position = sample_portfolio_manager.portfolio.positions["BTC/USD"]
        expected_pnl = (Decimal("60000.0") - Decimal("50000.0")) * Decimal("0.1")
        assert position.unrealized_pnl == expected_pnl
