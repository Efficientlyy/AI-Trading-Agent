"""
Strategy repository for database operations.
"""

from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .base import BaseRepository
from ..models import Strategy, Optimization


class StrategyRepository(BaseRepository[Strategy, Dict[str, Any], Dict[str, Any]]):
    """Strategy repository for database operations."""
    
    def __init__(self):
        """Initialize the repository with the Strategy model."""
        super().__init__(Strategy)
    
    def get_user_strategies(self, db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Strategy]:
        """
        Get strategies for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of items to skip
            limit: Maximum number of items to return
            
        Returns:
            List of strategies
        """
        return db.query(Strategy).filter(
            Strategy.user_id == user_id
        ).order_by(
            desc(Strategy.updated_at)
        ).offset(skip).limit(limit).all()
    
    def get_public_strategies(self, db: Session, skip: int = 0, limit: int = 100) -> List[Strategy]:
        """
        Get public strategies.
        
        Args:
            db: Database session
            skip: Number of items to skip
            limit: Maximum number of items to return
            
        Returns:
            List of public strategies
        """
        return db.query(Strategy).filter(
            Strategy.is_public == True
        ).order_by(
            desc(Strategy.updated_at)
        ).offset(skip).limit(limit).all()
    
    def get_strategy_by_name(self, db: Session, user_id: int, name: str) -> Optional[Strategy]:
        """
        Get a strategy by name for a specific user.
        
        Args:
            db: Database session
            user_id: User ID
            name: Strategy name
            
        Returns:
            Strategy if found, None otherwise
        """
        return db.query(Strategy).filter(
            Strategy.user_id == user_id,
            Strategy.name == name
        ).first()
    
    def create_strategy(
        self,
        db: Session,
        user_id: int,
        name: str,
        strategy_type: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        is_public: bool = False
    ) -> Strategy:
        """
        Create a new strategy.
        
        Args:
            db: Database session
            user_id: User ID
            name: Strategy name
            strategy_type: Strategy type
            config: Strategy configuration
            description: Strategy description
            is_public: Whether the strategy is public
            
        Returns:
            Created strategy
        """
        # Check if strategy already exists
        existing = self.get_strategy_by_name(db, user_id, name)
        if existing:
            raise ValueError(f"Strategy with name '{name}' already exists for this user")
        
        # Create strategy
        strategy_data = {
            "user_id": user_id,
            "name": name,
            "strategy_type": strategy_type,
            "config": config,
            "description": description,
            "is_public": is_public
        }
        
        return self.create(db, strategy_data)
    
    def update_strategy_config(
        self,
        db: Session,
        strategy_id: int,
        user_id: int,
        config: Dict[str, Any]
    ) -> Optional[Strategy]:
        """
        Update a strategy's configuration.
        
        Args:
            db: Database session
            strategy_id: Strategy ID
            user_id: User ID
            config: New configuration
            
        Returns:
            Updated strategy if found, None otherwise
        """
        strategy = db.query(Strategy).filter(
            Strategy.id == strategy_id,
            Strategy.user_id == user_id
        ).first()
        
        if not strategy:
            return None
        
        strategy.config = config
        db.add(strategy)
        db.commit()
        db.refresh(strategy)
        
        return strategy


class OptimizationRepository(BaseRepository[Optimization, Dict[str, Any], Dict[str, Any]]):
    """Optimization repository for database operations."""
    
    def __init__(self):
        """Initialize the repository with the Optimization model."""
        super().__init__(Optimization)
    
    def get_strategy_optimizations(
        self,
        db: Session,
        strategy_id: int,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Optimization]:
        """
        Get optimizations for a specific strategy.
        
        Args:
            db: Database session
            strategy_id: Strategy ID
            user_id: User ID
            skip: Number of items to skip
            limit: Maximum number of items to return
            
        Returns:
            List of optimizations
        """
        return db.query(Optimization).filter(
            Optimization.strategy_id == strategy_id,
            Optimization.user_id == user_id
        ).order_by(
            desc(Optimization.created_at)
        ).offset(skip).limit(limit).all()
    
    def create_optimization(
        self,
        db: Session,
        user_id: int,
        strategy_id: int,
        name: str,
        optimization_type: str,
        parameters: Dict[str, Any],
        description: Optional[str] = None
    ) -> Optimization:
        """
        Create a new optimization.
        
        Args:
            db: Database session
            user_id: User ID
            strategy_id: Strategy ID
            name: Optimization name
            optimization_type: Optimization type
            parameters: Parameter space definition
            description: Optimization description
            
        Returns:
            Created optimization
        """
        # Create optimization
        optimization_data = {
            "user_id": user_id,
            "strategy_id": strategy_id,
            "name": name,
            "optimization_type": optimization_type,
            "parameters": parameters,
            "description": description,
            "status": "pending"
        }
        
        return self.create(db, optimization_data)
    
    def update_optimization_status(
        self,
        db: Session,
        optimization_id: int,
        user_id: int,
        status: str,
        results: Optional[Dict[str, Any]] = None,
        best_parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Optimization]:
        """
        Update an optimization's status.
        
        Args:
            db: Database session
            optimization_id: Optimization ID
            user_id: User ID
            status: New status
            results: Optimization results
            best_parameters: Best parameters found
            
        Returns:
            Updated optimization if found, None otherwise
        """
        optimization = db.query(Optimization).filter(
            Optimization.id == optimization_id,
            Optimization.user_id == user_id
        ).first()
        
        if not optimization:
            return None
        
        # Update status
        optimization.status = status
        
        # Update results if provided
        if results is not None:
            optimization.results = results
        
        # Update best parameters if provided
        if best_parameters is not None:
            optimization.best_parameters = best_parameters
        
        # Update timestamps
        if status == "running" and not optimization.start_time:
            from datetime import datetime
            optimization.start_time = datetime.utcnow()
        
        if status in ["completed", "failed"] and not optimization.end_time:
            from datetime import datetime
            optimization.end_time = datetime.utcnow()
        
        db.add(optimization)
        db.commit()
        db.refresh(optimization)
        
        return optimization
