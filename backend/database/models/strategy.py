"""
Strategy model for storing trading strategies.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..config import Base


class Strategy(Base):
    """Strategy model for storing trading strategies."""
    
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    strategy_type = Column(String, nullable=False)  # e.g., MovingAverageCrossover, RSI, SentimentStrategy
    config = Column(JSON, nullable=False)  # Strategy-specific configuration
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="strategies")
    backtests = relationship("Backtest", back_populates="strategy", cascade="all, delete-orphan")
    optimizations = relationship("Optimization", back_populates="strategy", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Strategy {self.name}>"


class Optimization(Base):
    """Optimization model for storing strategy optimization results."""
    
    __tablename__ = "optimizations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    optimization_type = Column(String, nullable=False)  # e.g., GeneticAlgorithm, GridSearch, BayesianOptimization
    parameters = Column(JSON, nullable=False)  # Parameter space definition
    results = Column(JSON, nullable=True)  # Optimization results
    best_parameters = Column(JSON, nullable=True)  # Best parameters found
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    strategy = relationship("Strategy", back_populates="optimizations")
    
    def __repr__(self):
        return f"<Optimization {self.name}>"
