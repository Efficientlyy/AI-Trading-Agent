"""
Market data repositories for managing assets, OHLCV, and sentiment data.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from ..models.market_data import Asset, OHLCV, SentimentData
from .base import BaseRepository
from ..errors import with_error_handling, RecordNotFoundError
from ..cache import cached, invalidate_cache


class AssetRepository(BaseRepository[Asset, Dict[str, Any], Dict[str, Any]]):
    """Repository for managing assets."""
    
    def __init__(self):
        super().__init__(Asset)
    
    @with_error_handling
    @cached(ttl=300)  # Cache for 5 minutes
    def get_by_symbol(self, db: Session, symbol: str) -> Optional[Asset]:
        """
        Get an asset by symbol.
        
        Args:
            db: Database session
            symbol: Asset symbol
            
        Returns:
            Asset if found, None otherwise
        """
        return db.query(self.model).filter(self.model.symbol == symbol).first()
    
    @with_error_handling
    @cached(ttl=300)  # Cache for 5 minutes
    def get_active_assets(self, db: Session, asset_type: Optional[str] = None) -> List[Asset]:
        """
        Get all active assets, optionally filtered by type.
        
        Args:
            db: Database session
            asset_type: Optional asset type filter
            
        Returns:
            List of active assets
        """
        query = db.query(self.model).filter(self.model.is_active == True)
        
        if asset_type:
            query = query.filter(self.model.asset_type == asset_type)
            
        return query.all()
    
    @with_error_handling
    def update_asset(self, db: Session, asset_id: int, **kwargs) -> Asset:
        """
        Update an asset.
        
        Args:
            db: Database session
            asset_id: Asset ID
            **kwargs: Fields to update
            
        Returns:
            Updated asset
            
        Raises:
            RecordNotFoundError: If asset not found
        """
        asset = self.get(db, asset_id)
        if not asset:
            raise RecordNotFoundError(f"Asset with ID {asset_id} not found")
        
        # Update the asset
        result = self.update(db, asset, kwargs)
        
        # Invalidate cache
        invalidate_cache(self.get_by_symbol, db, asset.symbol)
        invalidate_cache(self.get_active_assets, db)
        
        return result


class OHLCVRepository(BaseRepository[OHLCV, Dict[str, Any], Dict[str, Any]]):
    """Repository for managing OHLCV data."""
    
    def __init__(self):
        super().__init__(OHLCV)
    
    @with_error_handling
    @cached(ttl=60)  # Cache for 1 minute
    def get_ohlcv_data(
        self,
        db: Session,
        asset_id: int,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[OHLCV]:
        """
        Get OHLCV data for an asset within a date range.
        
        Args:
            db: Database session
            asset_id: Asset ID
            timeframe: Timeframe (e.g., "1d", "1h")
            start_date: Start date
            end_date: End date
            
        Returns:
            List of OHLCV data
        """
        return db.query(self.model).filter(
            self.model.asset_id == asset_id,
            self.model.timeframe == timeframe,
            self.model.timestamp >= start_date,
            self.model.timestamp <= end_date
        ).order_by(self.model.timestamp).all()
    
    @with_error_handling
    @cached(ttl=60)  # Cache for 1 minute
    def get_latest_ohlcv(
        self,
        db: Session,
        asset_id: int,
        timeframe: str,
        limit: int = 1
    ) -> List[OHLCV]:
        """
        Get the latest OHLCV data for an asset.
        
        Args:
            db: Database session
            asset_id: Asset ID
            timeframe: Timeframe (e.g., "1d", "1h")
            limit: Maximum number of records to return
            
        Returns:
            List of latest OHLCV data
        """
        return db.query(self.model).filter(
            self.model.asset_id == asset_id,
            self.model.timeframe == timeframe
        ).order_by(desc(self.model.timestamp)).limit(limit).all()
    
    @with_error_handling
    def bulk_insert_ohlcv(self, db: Session, ohlcv_data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert OHLCV data.
        
        Args:
            db: Database session
            ohlcv_data: List of OHLCV data dictionaries
            
        Returns:
            Number of records inserted
        """
        # Extract unique asset IDs and timeframes for cache invalidation
        asset_ids = set()
        timeframes = set()
        
        # Create OHLCV objects
        ohlcv_objects = []
        for data in ohlcv_data:
            asset_ids.add(data["asset_id"])
            timeframes.add(data["timeframe"])
            ohlcv_objects.append(self.model(**data))
        
        # Bulk insert
        db.bulk_save_objects(ohlcv_objects)
        db.commit()
        
        # Invalidate cache for affected assets and timeframes
        from ..cache import clear_cache  # Import here to avoid circular imports
        clear_cache()  # For simplicity, clear the entire cache
        
        return len(ohlcv_objects)


class SentimentRepository(BaseRepository[SentimentData, Dict[str, Any], Dict[str, Any]]):
    """Repository for managing sentiment data."""
    
    def __init__(self):
        super().__init__(SentimentData)
    
    @with_error_handling
    @cached(ttl=120)  # Cache for 2 minutes
    def get_sentiment_data(
        self,
        db: Session,
        asset_id: int,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[SentimentData]:
        """
        Get sentiment data for an asset.
        
        Args:
            db: Database session
            asset_id: Asset ID
            source: Optional source filter
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            List of sentiment data
        """
        query = db.query(self.model).filter(self.model.asset_id == asset_id)
        
        if source:
            query = query.filter(self.model.source == source)
            
        if start_date:
            query = query.filter(self.model.timestamp >= start_date)
            
        if end_date:
            query = query.filter(self.model.timestamp <= end_date)
            
        return query.order_by(self.model.timestamp).all()
    
    @with_error_handling
    @cached(ttl=120)  # Cache for 2 minutes
    def get_average_sentiment(
        self,
        db: Session,
        asset_id: int,
        source: Optional[str] = None,
        days: int = 7
    ) -> float:
        """
        Get average sentiment for an asset over a period.
        
        Args:
            db: Database session
            asset_id: Asset ID
            source: Optional source filter
            days: Number of days to look back
            
        Returns:
            Average sentiment score
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        query = db.query(func.avg(self.model.sentiment_score)).filter(
            self.model.asset_id == asset_id,
            self.model.timestamp >= start_date,
            self.model.timestamp <= end_date
        )
        
        if source:
            query = query.filter(self.model.source == source)
            
        result = query.scalar()
        return float(result) if result is not None else 0.0
    
    @with_error_handling
    def bulk_insert_sentiment(self, db: Session, sentiment_data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert sentiment data.
        
        Args:
            db: Database session
            sentiment_data: List of sentiment data dictionaries
            
        Returns:
            Number of records inserted
        """
        # Extract unique asset IDs for cache invalidation
        asset_ids = set()
        
        # Create sentiment objects
        sentiment_objects = []
        for data in sentiment_data:
            asset_ids.add(data["asset_id"])
            sentiment_objects.append(self.model(**data))
        
        # Bulk insert
        db.bulk_save_objects(sentiment_objects)
        db.commit()
        
        # Invalidate cache for affected assets
        from ..cache import clear_cache  # Import here to avoid circular imports
        clear_cache()  # For simplicity, clear the entire cache
        
        return len(sentiment_objects)
