"""
Database module for the AI Trading Agent.
"""

from .config import Base, engine, SessionLocal, get_db
from .init_db import init_db, create_admin_user
from .repositories import (
    UserRepository,
    StrategyRepository,
    OptimizationRepository,
    BacktestRepository,
    AssetRepository,
    OHLCVRepository,
    SentimentRepository,
)

# Export all components
__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "create_admin_user",
    "UserRepository",
    "StrategyRepository",
    "OptimizationRepository",
    "BacktestRepository",
    "AssetRepository",
    "OHLCVRepository",
    "SentimentRepository",
]
