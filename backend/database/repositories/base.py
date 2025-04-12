"""
Base repository for database operations.
"""

from typing import Generic, TypeVar, Type, List, Optional, Any, Dict, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, NoResultFound, MultipleResultsFound
from pydantic import BaseModel
import logging

from ..config import Base
from ..errors import with_error_handling, RecordNotFoundError, DatabaseError, DuplicateRecordError

# Define type variables
T = TypeVar('T', bound=Base)
CreateSchemaType = TypeVar('CreateSchemaType', bound=BaseModel)
UpdateSchemaType = TypeVar('UpdateSchemaType', bound=BaseModel)

# Logger
logger = logging.getLogger(__name__)


class BaseRepository(Generic[T, CreateSchemaType, UpdateSchemaType]):
    """
    Base repository for database operations.
    
    This class provides common database operations for all models.
    """
    
    def __init__(self, model: Type[T]):
        """
        Initialize the repository with a model.
        
        Args:
            model: SQLAlchemy model class
        """
        self.model = model
        logger.debug(f"Initialized {self.__class__.__name__} with model {model.__name__}")
    
    @with_error_handling
    def get(self, db: Session, id: int) -> Optional[T]:
        """
        Get an item by ID.
        
        Args:
            db: Database session
            id: Item ID
            
        Returns:
            Item if found, None otherwise
        """
        logger.debug(f"Getting {self.model.__name__} with ID {id}")
        return db.query(self.model).filter(self.model.id == id).first()
    
    @with_error_handling
    def get_by_id(self, db: Session, id: int) -> Optional[T]:
        """
        Get an item by ID (alias for get).
        
        Args:
            db: Database session
            id: Item ID
            
        Returns:
            Item if found, None otherwise
        """
        return self.get(db, id)
    
    @with_error_handling
    def get_by(self, db: Session, **kwargs) -> Optional[T]:
        """
        Get an item by arbitrary filters.
        
        Args:
            db: Database session
            **kwargs: Filter conditions
            
        Returns:
            Item if found, None otherwise
        """
        logger.debug(f"Getting {self.model.__name__} with filters: {kwargs}")
        return db.query(self.model).filter_by(**kwargs).first()
    
    @with_error_handling
    def get_by_or_404(self, db: Session, **kwargs) -> T:
        """
        Get an item by arbitrary filters or raise RecordNotFoundError.
        
        Args:
            db: Database session
            **kwargs: Filter conditions
            
        Returns:
            Item if found
            
        Raises:
            RecordNotFoundError: If item not found
        """
        item = self.get_by(db, **kwargs)
        if not item:
            error_msg = f"{self.model.__name__} not found with filters: {kwargs}"
            logger.warning(error_msg)
            raise RecordNotFoundError(error_msg)
        return item
    
    @with_error_handling
    def list(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        **kwargs
    ) -> List[T]:
        """
        Get a list of items with optional filtering.
        
        Args:
            db: Database session
            skip: Number of items to skip
            limit: Maximum number of items to return
            **kwargs: Filter conditions
            
        Returns:
            List of items
        """
        logger.debug(f"Listing {self.model.__name__} with filters: {kwargs}, skip: {skip}, limit: {limit}")
        query = db.query(self.model)
        
        if kwargs:
            query = query.filter_by(**kwargs)
            
        return query.offset(skip).limit(limit).all()
    
    @with_error_handling
    def create(self, db: Session, obj_in: Union[CreateSchemaType, Dict[str, Any]]) -> T:
        """
        Create a new item.
        
        Args:
            db: Database session
            obj_in: Input data
            
        Returns:
            Created item
        """
        try:
            # Convert to dict if it's a Pydantic model
            obj_data = obj_in.dict() if hasattr(obj_in, "dict") else obj_in
            
            logger.debug(f"Creating {self.model.__name__} with data: {obj_data}")
            
            # Create model instance
            db_obj = self.model(**obj_data)
            
            # Add to session
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            
            logger.info(f"Created {self.model.__name__} with ID: {db_obj.id}")
            return db_obj
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise
    
    @with_error_handling
    def update(
        self,
        db: Session,
        db_obj: T,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> T:
        """
        Update an existing item.
        
        Args:
            db: Database session
            db_obj: Existing database object
            obj_in: New data
            
        Returns:
            Updated item
        """
        try:
            # Convert to dict if it's a Pydantic model
            update_data = obj_in.dict(exclude_unset=True) if hasattr(obj_in, "dict") else obj_in
            
            logger.debug(f"Updating {self.model.__name__} with ID {db_obj.id} with data: {update_data}")
            
            # Update attributes
            for field, value in update_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            # Commit changes
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            
            logger.info(f"Updated {self.model.__name__} with ID: {db_obj.id}")
            return db_obj
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating {self.model.__name__}: {e}")
            raise
    
    @with_error_handling
    def delete(self, db: Session, id: int) -> bool:
        """
        Delete an item by ID.
        
        Args:
            db: Database session
            id: Item ID
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            logger.debug(f"Deleting {self.model.__name__} with ID: {id}")
            
            obj = db.query(self.model).get(id)
            if not obj:
                logger.warning(f"{self.model.__name__} with ID {id} not found for deletion")
                return False
                
            db.delete(obj)
            db.commit()
            
            logger.info(f"Deleted {self.model.__name__} with ID: {id}")
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error deleting {self.model.__name__}: {e}")
            raise
    
    @with_error_handling
    def count(self, db: Session, **kwargs) -> int:
        """
        Count items with optional filtering.
        
        Args:
            db: Database session
            **kwargs: Filter conditions
            
        Returns:
            Number of items
        """
        logger.debug(f"Counting {self.model.__name__} with filters: {kwargs}")
        
        query = db.query(self.model)
        
        if kwargs:
            query = query.filter_by(**kwargs)
            
        return query.count()
    
    @with_error_handling
    def exists(self, db: Session, **kwargs) -> bool:
        """
        Check if an item exists with the given filters.
        
        Args:
            db: Database session
            **kwargs: Filter conditions
            
        Returns:
            True if exists, False otherwise
        """
        logger.debug(f"Checking if {self.model.__name__} exists with filters: {kwargs}")
        
        return db.query(
            db.query(self.model).filter_by(**kwargs).exists()
        ).scalar()
