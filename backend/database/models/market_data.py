"""
Market data models for storing historical price and sentiment data.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, JSON, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..config import Base


class Asset(Base):
    """Asset model for storing information about tradable assets."""
    
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    asset_type = Column(String, nullable=False, index=True)  # crypto, stock, forex, etc.
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    ohlcv_data = relationship("OHLCV", back_populates="asset", cascade="all, delete-orphan")
    sentiment_data = relationship("SentimentData", back_populates="asset", cascade="all, delete-orphan")
    
    # Create indexes
    __table_args__ = (
        Index('idx_asset_symbol_type', 'symbol', 'asset_type'),
        Index('idx_asset_active_type', 'is_active', 'asset_type'),
    )
    
    def __repr__(self):
        return f"<Asset {self.symbol}>"


class OHLCV(Base):
    """OHLCV model for storing price data."""
    
    __tablename__ = "ohlcv"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    timeframe = Column(String, nullable=False, index=True)  # 1m, 5m, 15m, 1h, 4h, 1d, etc.
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    asset = relationship("Asset", back_populates="ohlcv_data")
    
    # Create indexes
    __table_args__ = (
        Index('idx_ohlcv_asset_timeframe_timestamp', 'asset_id', 'timeframe', 'timestamp'),
        Index('idx_ohlcv_timestamp_range', 'asset_id', 'timeframe', 'timestamp'),
        UniqueConstraint('asset_id', 'timestamp', 'timeframe', name='unique_ohlcv_data'),
        Index('idx_ohlcv_asset_time', 'asset_id', 'timestamp', 'timeframe'),
    )
    
    def __repr__(self):
        return f"<OHLCV {self.asset_id} {self.timeframe} {self.timestamp}>"


class SentimentData(Base):
    """Sentiment data model for storing sentiment analysis results."""
    
    __tablename__ = "sentiment_data"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    source = Column(String, nullable=False, index=True)  # twitter, news, reddit, etc.
    sentiment_score = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)  # Number of mentions, can be null
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    asset = relationship("Asset", back_populates="sentiment_data")
    
    # Create indexes
    __table_args__ = (
        Index('idx_sentiment_asset_source_timestamp', 'asset_id', 'source', 'timestamp'),
        Index('idx_sentiment_timestamp_range', 'asset_id', 'timestamp'),
        Index('idx_sentiment_asset_time', 'asset_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SentimentData {self.asset_id} {self.source} {self.timestamp}>"
