"""
Backtest model for storing backtest results.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Text, Float, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..config import Base


class Backtest(Base):
    """Backtest model for storing backtest results."""
    
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=False)  # Strategy parameters used for backtest
    start_date = Column(DateTime(timezone=True), nullable=False, index=True)
    end_date = Column(DateTime(timezone=True), nullable=False, index=True)
    initial_capital = Column(Float, nullable=False, default=10000.0)
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    results = Column(JSON, nullable=True)  # Backtest results including performance metrics
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="backtests")
    strategy = relationship("Strategy", back_populates="backtests")
    trades = relationship("Trade", back_populates="backtest", cascade="all, delete-orphan")
    portfolio_snapshots = relationship("PortfolioSnapshot", back_populates="backtest", cascade="all, delete-orphan")
    
    # Create indexes
    __table_args__ = (
        Index('idx_backtest_user_strategy', 'user_id', 'strategy_id'),
        Index('idx_backtest_date_range', 'start_date', 'end_date'),
    )
    
    def __repr__(self):
        return f"<Backtest {self.name}>"


class Trade(Base):
    """Trade model for storing individual trades from backtests."""
    
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    order_type = Column(String, nullable=False)  # market, limit, stop, stop_limit
    side = Column(String, nullable=False, index=True)  # buy, sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    commission = Column(Float, nullable=False, default=0.0)
    slippage = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    backtest = relationship("Backtest", back_populates="trades")
    
    # Create indexes
    __table_args__ = (
        Index('idx_trade_backtest_symbol', 'backtest_id', 'symbol'),
        Index('idx_trade_backtest_timestamp', 'backtest_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Trade {self.symbol} {self.side} {self.quantity}>"


class PortfolioSnapshot(Base):
    """Portfolio snapshot model for storing portfolio state at different points in time."""
    
    __tablename__ = "portfolio_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions = Column(JSON, nullable=False)  # Dictionary of symbol -> quantity, avg_price
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    backtest = relationship("Backtest", back_populates="portfolio_snapshots")
    
    # Create indexes
    __table_args__ = (
        Index('idx_snapshot_backtest_time', 'backtest_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<PortfolioSnapshot {self.timestamp} {self.equity}>"
