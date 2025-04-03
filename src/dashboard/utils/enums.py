"""
Dashboard Enums Module

This module contains enums used by the modern dashboard implementation.
Following the Single Responsibility Principle, these enums are separated
from the main dashboard module.
"""

from enum import Enum

class ExchangeProvider(Enum):
    """Exchange provider enum"""
    BINANCE = "binance"
    KRAKEN = "kraken"
    COINBASE = "coinbase"
    BITVAVO = "bitvavo"  # Added Bitvavo exchange

class SystemState:
    """System operational state enum"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class TradingState:
    """Trading activity state enum"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    PAUSED = "paused"

class SystemMode:
    """System operating mode enum"""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"

class UserRole:
    """User permission role enum"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    
class DataSource:
    """Data sourcing option enum"""
    MOCK = "mock"
    REAL = "real"
