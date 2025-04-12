"""
Backtest repository for database operations.
"""

from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime

from .base import BaseRepository
from ..models import Backtest, Trade, PortfolioSnapshot


class BacktestRepository(BaseRepository[Backtest, Dict[str, Any], Dict[str, Any]]):
    """Backtest repository for database operations."""
    
    def __init__(self):
        """Initialize the repository with the Backtest model."""
        super().__init__(Backtest)
    
    def get_user_backtests(
        self,
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Backtest]:
        """
        Get backtests for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of items to skip
            limit: Maximum number of items to return
            
        Returns:
            List of backtests
        """
        return db.query(Backtest).filter(
            Backtest.user_id == user_id
        ).order_by(
            desc(Backtest.created_at)
        ).offset(skip).limit(limit).all()
    
    def get_strategy_backtests(
        self,
        db: Session,
        strategy_id: int,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Backtest]:
        """
        Get backtests for a specific strategy.
        
        Args:
            db: Database session
            strategy_id: Strategy ID
            user_id: User ID
            skip: Number of items to skip
            limit: Maximum number of items to return
            
        Returns:
            List of backtests
        """
        return db.query(Backtest).filter(
            Backtest.strategy_id == strategy_id,
            Backtest.user_id == user_id
        ).order_by(
            desc(Backtest.created_at)
        ).offset(skip).limit(limit).all()
    
    def create_backtest(
        self,
        db: Session,
        user_id: int,
        strategy_id: int,
        name: str,
        parameters: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        description: Optional[str] = None
    ) -> Backtest:
        """
        Create a new backtest.
        
        Args:
            db: Database session
            user_id: User ID
            strategy_id: Strategy ID
            name: Backtest name
            parameters: Strategy parameters used for backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for backtest
            description: Backtest description
            
        Returns:
            Created backtest
        """
        # Create backtest
        backtest_data = {
            "user_id": user_id,
            "strategy_id": strategy_id,
            "name": name,
            "parameters": parameters,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "description": description,
            "status": "pending"
        }
        
        return self.create(db, backtest_data)
    
    def update_backtest_status(
        self,
        db: Session,
        backtest_id: int,
        user_id: int,
        status: str,
        results: Optional[Dict[str, Any]] = None
    ) -> Optional[Backtest]:
        """
        Update a backtest's status.
        
        Args:
            db: Database session
            backtest_id: Backtest ID
            user_id: User ID
            status: New status
            results: Backtest results
            
        Returns:
            Updated backtest if found, None otherwise
        """
        backtest = db.query(Backtest).filter(
            Backtest.id == backtest_id,
            Backtest.user_id == user_id
        ).first()
        
        if not backtest:
            return None
        
        # Update status
        backtest.status = status
        
        # Update results if provided
        if results is not None:
            backtest.results = results
        
        db.add(backtest)
        db.commit()
        db.refresh(backtest)
        
        return backtest
    
    def add_trades(
        self,
        db: Session,
        backtest_id: int,
        trades: List[Dict[str, Any]]
    ) -> List[Trade]:
        """
        Add trades to a backtest.
        
        Args:
            db: Database session
            backtest_id: Backtest ID
            trades: List of trade data
            
        Returns:
            List of created trades
        """
        created_trades = []
        
        for trade_data in trades:
            trade_data["backtest_id"] = backtest_id
            trade = Trade(**trade_data)
            db.add(trade)
            created_trades.append(trade)
        
        db.commit()
        
        for trade in created_trades:
            db.refresh(trade)
        
        return created_trades
    
    def add_portfolio_snapshots(
        self,
        db: Session,
        backtest_id: int,
        snapshots: List[Dict[str, Any]]
    ) -> List[PortfolioSnapshot]:
        """
        Add portfolio snapshots to a backtest.
        
        Args:
            db: Database session
            backtest_id: Backtest ID
            snapshots: List of snapshot data
            
        Returns:
            List of created snapshots
        """
        created_snapshots = []
        
        for snapshot_data in snapshots:
            snapshot_data["backtest_id"] = backtest_id
            snapshot = PortfolioSnapshot(**snapshot_data)
            db.add(snapshot)
            created_snapshots.append(snapshot)
        
        db.commit()
        
        for snapshot in created_snapshots:
            db.refresh(snapshot)
        
        return created_snapshots
    
    def get_backtest_trades(
        self,
        db: Session,
        backtest_id: int,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Trade]:
        """
        Get trades for a specific backtest.
        
        Args:
            db: Database session
            backtest_id: Backtest ID
            user_id: User ID
            skip: Number of items to skip
            limit: Maximum number of items to return
            
        Returns:
            List of trades
        """
        # First check if backtest belongs to user
        backtest = db.query(Backtest).filter(
            Backtest.id == backtest_id,
            Backtest.user_id == user_id
        ).first()
        
        if not backtest:
            return []
        
        return db.query(Trade).filter(
            Trade.backtest_id == backtest_id
        ).order_by(
            Trade.timestamp
        ).offset(skip).limit(limit).all()
    
    def get_backtest_portfolio_snapshots(
        self,
        db: Session,
        backtest_id: int,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[PortfolioSnapshot]:
        """
        Get portfolio snapshots for a specific backtest.
        
        Args:
            db: Database session
            backtest_id: Backtest ID
            user_id: User ID
            skip: Number of items to skip
            limit: Maximum number of items to return
            
        Returns:
            List of portfolio snapshots
        """
        # First check if backtest belongs to user
        backtest = db.query(Backtest).filter(
            Backtest.id == backtest_id,
            Backtest.user_id == user_id
        ).first()
        
        if not backtest:
            return []
        
        return db.query(PortfolioSnapshot).filter(
            PortfolioSnapshot.backtest_id == backtest_id
        ).order_by(
            PortfolioSnapshot.timestamp
        ).offset(skip).limit(limit).all()
