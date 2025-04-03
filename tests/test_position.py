"""Tests for the Position model."""

import datetime
import unittest
from unittest import mock
from uuid import UUID

import pytest

from src.models.position import Position, PositionSide, PositionStatus


class TestPosition(unittest.TestCase):
    """Test cases for the Position model."""
    
    def test_position_initialization(self):
        """Test that a position can be initialized correctly."""
        position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1,
            stop_loss=48000.0,
            take_profit=55000.0,
            leverage=1.0
        )
        
        assert position.exchange == "binance"
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        assert position.entry_price == 50000.0
        assert position.amount == 0.1
        assert position.stop_loss == 48000.0
        assert position.take_profit == 55000.0
        assert position.leverage == 1.0
        assert position.status == PositionStatus.PENDING
        assert position.realized_pnl is None
        assert position.exit_price is None
        
    def test_position_open(self):
        """Test that a position can be opened."""
        now = datetime.datetime.now(datetime.timezone.utc)
        position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1
        )
        
        # Mock datetime for both the test and the position module
        with mock.patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.timezone.utc = datetime.timezone.utc
            # Also patch the used datetime in the Position module
            with mock.patch('src.models.position.datetime') as mock_position_datetime:
                mock_position_datetime.now.return_value = now
                mock_position_datetime.timezone.utc = datetime.timezone.utc
                position.open()
        
        assert position.status == PositionStatus.OPEN
        assert position.opened_at == now
        
    def test_position_close(self):
        """Test that a position can be closed."""
        now = datetime.datetime.now(datetime.timezone.utc)
        position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1
        )
        position.open()
        
        # Mock datetime for both the test and the position module
        with mock.patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.timezone.utc = datetime.timezone.utc
            with mock.patch('src.models.position.datetime') as mock_position_datetime:
                mock_position_datetime.now.return_value = now
                mock_position_datetime.timezone.utc = datetime.timezone.utc
                position.close(52000.0)
        
        assert position.status == PositionStatus.CLOSED
        assert position.closed_at == now
        assert position.exit_price == 52000.0
        assert position.realized_pnl == 200.0  # (52000 - 50000) * 0.1
        
    def test_unrealized_pnl_long(self):
        """Test unrealized PnL calculation for a long position."""
        position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1
        )
        position.open()
        
        # Price increase (profit)
        pnl = position.calculate_unrealized_pnl(52000.0)
        assert pnl == 200.0  # (52000 - 50000) * 0.1
        
        # Price decrease (loss)
        pnl = position.calculate_unrealized_pnl(48000.0)
        assert pnl == -200.0  # (48000 - 50000) * 0.1
        
    def test_unrealized_pnl_short(self):
        """Test unrealized PnL calculation for a short position."""
        position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=50000.0,
            amount=0.1
        )
        position.open()
        
        # Price decrease (profit)
        pnl = position.calculate_unrealized_pnl(48000.0)
        assert pnl == 200.0  # (50000 - 48000) * 0.1
        
        # Price increase (loss)
        pnl = position.calculate_unrealized_pnl(52000.0)
        assert pnl == -200.0  # (50000 - 52000) * 0.1
    
    def test_leverage_effect(self):
        """Test that leverage correctly affects the PnL calculations."""
        position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1,
            leverage=5.0  # 5x leverage
        )
        position.open()
        
        # Price increase (profit) with leverage
        pnl = position.calculate_unrealized_pnl(51000.0)
        assert pnl == 500.0  # (51000 - 50000) * 0.1 * 5
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1
        )
        position.open()
        
        # 10% price increase
        roi = position.calculate_roi(55000.0)
        assert roi == 10.0  # ((55000 - 50000) * 0.1) / (50000 * 0.1) * 100
    
    def test_stop_loss_triggered(self):
        """Test stop loss detection."""
        # Long position
        long_position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1,
            stop_loss=48000.0
        )
        long_position.open()
        
        assert not long_position.is_stop_loss_triggered(49000.0)
        assert long_position.is_stop_loss_triggered(48000.0)
        assert long_position.is_stop_loss_triggered(47000.0)
        
        # Short position
        short_position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=50000.0,
            amount=0.1,
            stop_loss=52000.0
        )
        short_position.open()
        
        assert not short_position.is_stop_loss_triggered(51000.0)
        assert short_position.is_stop_loss_triggered(52000.0)
        assert short_position.is_stop_loss_triggered(53000.0)
    
    def test_take_profit_triggered(self):
        """Test take profit detection."""
        # Long position
        long_position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1,
            take_profit=55000.0
        )
        long_position.open()
        
        assert not long_position.is_take_profit_triggered(54000.0)
        assert long_position.is_take_profit_triggered(55000.0)
        assert long_position.is_take_profit_triggered(56000.0)
        
        # Short position
        short_position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=50000.0,
            amount=0.1,
            take_profit=45000.0
        )
        short_position.open()
        
        assert not short_position.is_take_profit_triggered(46000.0)
        assert short_position.is_take_profit_triggered(45000.0)
        assert short_position.is_take_profit_triggered(44000.0)
    
    def test_fee_deduction(self):
        """Test that fees are correctly deducted from realized PnL."""
        position = Position(
            exchange="binance",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            amount=0.1,
            fee_paid=25.0  # $25 in fees
        )
        position.open()
        position.close(52000.0)
        
        # PnL should be (52000 - 50000) * 0.1 - 25 = 175
        assert position.realized_pnl == 175.0 