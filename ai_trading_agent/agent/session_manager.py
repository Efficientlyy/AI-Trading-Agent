"""
Session Manager module for the AI Trading Agent.

This module manages multiple paper trading sessions, providing isolation and persistence.
"""

from typing import Dict, Any, Optional, List, Union
import asyncio
import threading
import time
import uuid
from datetime import datetime
import json
import os
import sqlite3
from pathlib import Path

from ..common import logger
from ..common.error_handling import (
    TradingAgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity
)

class PaperTradingSession:
    """
    Represents a single paper trading session with its own isolated state.
    """
    
    def __init__(
        self,
        session_id: str,
        config_path: str,
        duration_minutes: int,
        interval_minutes: int,
        symbols: List[str] = None,
        initial_capital: float = 10000.0,
        user_id: str = None
    ):
        """
        Initialize a paper trading session.
        
        Args:
            session_id: Unique identifier for the session
            config_path: Path to the configuration file
            duration_minutes: Duration to run paper trading in minutes
            interval_minutes: Update interval in minutes
            symbols: List of symbols to trade
            initial_capital: Initial capital for the session
            user_id: Optional user ID for the session owner
        """
        self.session_id = session_id
        self.config_path = config_path
        self.duration_minutes = duration_minutes
        self.interval_minutes = interval_minutes
        self.symbols = symbols or []
        self.initial_capital = initial_capital
        self.user_id = user_id
        
        self.status = "starting"
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.uptime_seconds = 0
        self.current_portfolio = None
        self.orchestrator = None
        self.stop_event = None
        self.task = None
        self.results = {
            'trades': [],
            'portfolio_history': [],
            'performance_metrics': {}
        }
        
        self.last_updated = time.time()
        
        logger.info(f"Created paper trading session {session_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary for API responses.
        
        Returns:
            Session as a dictionary
        """
        return {
            "session_id": self.session_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config_path": self.config_path,
            "duration_minutes": self.duration_minutes,
            "interval_minutes": self.interval_minutes,
            "uptime_seconds": self.uptime_seconds,
            "symbols": self.symbols,
            "current_portfolio": self.current_portfolio,
            "user_id": self.user_id
        }
    
    def to_db_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary for database storage.
        
        Returns:
            Session as a dictionary for database storage
        """
        return {
            "session_id": self.session_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config_path": self.config_path,
            "duration_minutes": self.duration_minutes,
            "interval_minutes": self.interval_minutes,
            "uptime_seconds": self.uptime_seconds,
            "symbols": json.dumps(self.symbols),
            "current_portfolio": json.dumps(self.current_portfolio) if self.current_portfolio else None,
            "results": json.dumps(self.results),
            "user_id": self.user_id,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'PaperTradingSession':
        """
        Create a session from a database dictionary.
        
        Args:
            data: Dictionary from database
            
        Returns:
            PaperTradingSession instance
        """
        session = cls(
            session_id=data["session_id"],
            config_path=data["config_path"],
            duration_minutes=data["duration_minutes"],
            interval_minutes=data["interval_minutes"],
            symbols=json.loads(data["symbols"]) if data["symbols"] else [],
            user_id=data.get("user_id")
        )
        
        session.status = data["status"]
        session.start_time = data["start_time"]
        session.end_time = data["end_time"]
        session.uptime_seconds = data["uptime_seconds"]
        session.current_portfolio = json.loads(data["current_portfolio"]) if data["current_portfolio"] else None
        session.results = json.loads(data["results"]) if data["results"] else {
            'trades': [],
            'portfolio_history': [],
            'performance_metrics': {}
        }
        session.last_updated = data.get("last_updated", time.time())
        
        return session
    
    def update_uptime(self) -> None:
        """Update the session uptime."""
        if self.status in ["running", "starting"]:
            start_time = datetime.fromisoformat(self.start_time)
            self.uptime_seconds = int((datetime.now() - start_time).total_seconds())
    
    def set_stop_event(self, stop_event: asyncio.Event) -> None:
        """
        Set the stop event for the session.
        
        Args:
            stop_event: Asyncio event to signal stopping
        """
        self.stop_event = stop_event
    
    def set_task(self, task: asyncio.Task) -> None:
        """
        Set the asyncio task for the session.
        
        Args:
            task: Asyncio task running the session
        """
        self.task = task
    
    def set_orchestrator(self, orchestrator: Any) -> None:
        """
        Set the trading orchestrator for the session.
        
        Args:
            orchestrator: Trading orchestrator instance
        """
        self.orchestrator = orchestrator
    
    def update_status(self, status: str) -> None:
        """
        Update the session status.
        
        Args:
            status: New status
        """
        self.status = status
        self.last_updated = time.time()
        
        if status in ["completed", "stopped", "error"]:
            self.end_time = datetime.now().isoformat()
        
        logger.info(f"Session {self.session_id} status updated to {status}")
    
    def update_results(self, results: Dict[str, Any]) -> None:
        """
        Update the session results.
        
        Args:
            results: New results
        """
        self.results = results
        self.last_updated = time.time()
        
        # Update current portfolio from results
        if results and 'portfolio_history' in results and results['portfolio_history']:
            self.current_portfolio = results['portfolio_history'][-1]
    
    async def stop(self) -> None:
        """Stop the session."""
        if self.status not in ["running", "starting"]:
            logger.warning(f"Cannot stop session {self.session_id} with status {self.status}")
            return
        
        self.update_status("stopping")
        
        if self.stop_event:
            self.stop_event.set()
            
        # Wait for task to complete
        if self.task and not self.task.done():
            try:
                # Wait with timeout
                await asyncio.wait_for(self.task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for session {self.session_id} to stop")
            except Exception as e:
                logger.error(f"Error stopping session {self.session_id}: {e}")
        
        self.update_status("stopped")


class SessionDatabaseManager:
    """
    Manages the database for session persistence.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the session database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        if db_path is None:
            # Use default path in the project directory
            project_dir = Path(__file__).parent.parent.parent
            db_dir = project_dir / "data"
            os.makedirs(db_dir, exist_ok=True)
            db_path = str(db_dir / "sessions.db")
        
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                config_path TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                interval_minutes INTEGER NOT NULL,
                uptime_seconds INTEGER NOT NULL,
                symbols TEXT,
                current_portfolio TEXT,
                results TEXT,
                user_id TEXT,
                last_updated REAL NOT NULL
            )
            ''')
            
            # Create alerts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                read INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            # Create alert settings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_settings (
                session_id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                drawdown_threshold REAL NOT NULL DEFAULT 0.05,
                gain_threshold REAL NOT NULL DEFAULT 0.05,
                large_trade_threshold REAL NOT NULL DEFAULT 0.1,
                consecutive_losses_threshold INTEGER NOT NULL DEFAULT 3,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            conn.commit()
            conn.close()
    
    def save_session(self, session: PaperTradingSession) -> None:
        """
        Save a session to the database.
        
        Args:
            session: The session to save
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            session_dict = session.to_db_dict()
            
            # Check if session exists
            cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session.session_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing session
                placeholders = ", ".join([f"{k} = ?" for k in session_dict.keys()])
                query = f"UPDATE sessions SET {placeholders} WHERE session_id = ?"
                values = list(session_dict.values()) + [session.session_id]
                cursor.execute(query, values)
            else:
                # Insert new session
                placeholders = ", ".join(["?"] * len(session_dict))
                columns = ", ".join(session_dict.keys())
                query = f"INSERT INTO sessions ({columns}) VALUES ({placeholders})"
                cursor.execute(query, list(session_dict.values()))
            
            conn.commit()
            conn.close()
    
    def load_session(self, session_id: str) -> Optional[PaperTradingSession]:
        """
        Load a session from the database.
        
        Args:
            session_id: The ID of the session to load
            
        Returns:
            The loaded session or None if not found
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return PaperTradingSession.from_db_dict(dict(row))
            
            return None
    
    def load_all_sessions(self, user_id: str = None, limit: int = 100, offset: int = 0) -> List[PaperTradingSession]:
        """
        Load all sessions from the database.
        
        Args:
            user_id: Optional user ID to filter sessions
            limit: Maximum number of sessions to return
            offset: Offset for pagination
            
        Returns:
            List of sessions
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute(
                    "SELECT * FROM sessions WHERE user_id = ? ORDER BY last_updated DESC LIMIT ? OFFSET ?",
                    (user_id, limit, offset)
                )
            else:
                cursor.execute(
                    "SELECT * FROM sessions ORDER BY last_updated DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )
            
            rows = cursor.fetchall()
            
            conn.close()
            
            return [PaperTradingSession.from_db_dict(dict(row)) for row in rows]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database.
        
        Args:
            session_id: The ID of the session to delete
            
        Returns:
            True if the session was deleted, False otherwise
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete alerts first (foreign key constraint)
            cursor.execute("DELETE FROM alerts WHERE session_id = ?", (session_id,))
            
            # Delete alert settings
            cursor.execute("DELETE FROM alert_settings WHERE session_id = ?", (session_id,))
            
            # Delete session
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            return deleted
    
    def save_alert(self, alert: Dict[str, Any]) -> None:
        """
        Save an alert to the database.
        
        Args:
            alert: The alert to save
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if alert exists
            cursor.execute("SELECT id FROM alerts WHERE id = ?", (alert["id"],))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing alert
                cursor.execute(
                    "UPDATE alerts SET timestamp = ?, type = ?, title = ?, message = ?, read = ? WHERE id = ?",
                    (
                        alert["timestamp"],
                        alert["type"],
                        alert["title"],
                        alert["message"],
                        1 if alert.get("read", False) else 0,
                        alert["id"]
                    )
                )
            else:
                # Insert new alert
                cursor.execute(
                    "INSERT INTO alerts (id, session_id, timestamp, type, title, message, read) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        alert["id"],
                        alert["session_id"],
                        alert["timestamp"],
                        alert["type"],
                        alert["title"],
                        alert["message"],
                        1 if alert.get("read", False) else 0
                    )
                )
            
            conn.commit()
            conn.close()
    
    def load_alerts(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Load alerts for a session from the database.
        
        Args:
            session_id: The ID of the session
            limit: Maximum number of alerts to return
            offset: Offset for pagination
            
        Returns:
            List of alerts
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM alerts WHERE session_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (session_id, limit, offset)
            )
            
            rows = cursor.fetchall()
            
            conn.close()
            
            return [dict(row) for row in rows]
    
    def mark_alert_as_read(self, alert_id: str) -> bool:
        """
        Mark an alert as read.
        
        Args:
            alert_id: The ID of the alert
            
        Returns:
            True if the alert was marked as read, False otherwise
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("UPDATE alerts SET read = 1 WHERE id = ?", (alert_id,))
            
            updated = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            return updated
    
    def save_alert_settings(self, session_id: str, settings: Dict[str, Any]) -> None:
        """
        Save alert settings for a session.
        
        Args:
            session_id: The ID of the session
            settings: The alert settings
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if settings exist
            cursor.execute("SELECT session_id FROM alert_settings WHERE session_id = ?", (session_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing settings
                cursor.execute(
                    """
                    UPDATE alert_settings SET 
                    enabled = ?, 
                    drawdown_threshold = ?, 
                    gain_threshold = ?, 
                    large_trade_threshold = ?, 
                    consecutive_losses_threshold = ? 
                    WHERE session_id = ?
                    """,
                    (
                        1 if settings.get("enabled", True) else 0,
                        settings.get("drawdown_threshold", 0.05),
                        settings.get("gain_threshold", 0.05),
                        settings.get("large_trade_threshold", 0.1),
                        settings.get("consecutive_losses_threshold", 3),
                        session_id
                    )
                )
            else:
                # Insert new settings
                cursor.execute(
                    """
                    INSERT INTO alert_settings (
                    session_id, enabled, drawdown_threshold, gain_threshold, 
                    large_trade_threshold, consecutive_losses_threshold
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        1 if settings.get("enabled", True) else 0,
                        settings.get("drawdown_threshold", 0.05),
                        settings.get("gain_threshold", 0.05),
                        settings.get("large_trade_threshold", 0.1),
                        settings.get("consecutive_losses_threshold", 3)
                    )
                )
            
            conn.commit()
            conn.close()
    
    def load_alert_settings(self, session_id: str) -> Dict[str, Any]:
        """
        Load alert settings for a session.
        
        Args:
            session_id: The ID of the session
            
        Returns:
            The alert settings
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM alert_settings WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                settings = dict(row)
                # Convert SQLite integer to boolean
                settings["enabled"] = settings["enabled"] == 1
                return settings
            
            # Return default settings
            return {
                "enabled": True,
                "drawdown_threshold": 0.05,
                "gain_threshold": 0.05,
                "large_trade_threshold": 0.1,
                "consecutive_losses_threshold": 3
            }


class SessionManager:
    """
    Manages multiple paper trading sessions.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the session manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.sessions: Dict[str, PaperTradingSession] = {}
        self.db_manager = SessionDatabaseManager(db_path)
        self.lock = threading.Lock()
        
        # Load active sessions from database
        self._load_active_sessions()
    
    def _load_active_sessions(self) -> None:
        """Load active sessions from the database."""
        sessions = self.db_manager.load_all_sessions()
        
        for session in sessions:
            if session.status in ["running", "starting", "stopping"]:
                # Only load active sessions
                self.sessions[session.session_id] = session
    
    def create_session(
        self,
        config_path: str,
        duration_minutes: int,
        interval_minutes: int,
        symbols: List[str] = None,
        initial_capital: float = 10000.0,
        user_id: str = None,
        session_id: str = None
    ) -> PaperTradingSession:
        """
        Create a new paper trading session.
        
        Args:
            config_path: Path to the configuration file
            duration_minutes: Duration to run paper trading in minutes
            interval_minutes: Update interval in minutes
            symbols: List of symbols to trade
            initial_capital: Initial capital for the session
            user_id: Optional user ID for the session owner
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            The created session
        """
        with self.lock:
            # Generate session ID if not provided
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            # Create session
            session = PaperTradingSession(
                session_id=session_id,
                config_path=config_path,
                duration_minutes=duration_minutes,
                interval_minutes=interval_minutes,
                symbols=symbols,
                initial_capital=initial_capital,
                user_id=user_id
            )
            
            # Store session
            self.sessions[session_id] = session
            
            # Save to database
            self.db_manager.save_session(session)
            
            # Initialize alert settings
            self.db_manager.save_alert_settings(session_id, {})
            
            return session
    
    def get_session(self, session_id: str) -> Optional[PaperTradingSession]:
        """
        Get a session by ID.
        
        Args:
            session_id: The ID of the session
            
        Returns:
            The session or None if not found
        """
        # Check in-memory sessions first
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try to load from database
        session = self.db_manager.load_session(session_id)
        
        if session and session.status in ["running", "starting", "stopping"]:
            # Add to in-memory sessions if active
            self.sessions[session_id] = session
        
        return session
    
    def get_all_sessions(self, user_id: str = None, include_completed: bool = True, limit: int = 100, offset: int = 0) -> List[PaperTradingSession]:
        """
        Get all sessions.
        
        Args:
            user_id: Optional user ID to filter sessions
            include_completed: Whether to include completed sessions
            limit: Maximum number of sessions to return
            offset: Offset for pagination
            
        Returns:
            List of sessions
        """
        # Get sessions from database
        db_sessions = self.db_manager.load_all_sessions(user_id, limit, offset)
        
        if not include_completed:
            # Filter out completed sessions
            db_sessions = [s for s in db_sessions if s.status not in ["completed", "stopped", "error"]]
        
        # Update in-memory sessions with active sessions from database
        for session in db_sessions:
            if session.status in ["running", "starting", "stopping"]:
                self.sessions[session.session_id] = session
        
        return db_sessions
    
    def update_session(self, session: PaperTradingSession) -> None:
        """
        Update a session.
        
        Args:
            session: The session to update
        """
        with self.lock:
            # Update in-memory session
            self.sessions[session.session_id] = session
            
            # Save to database
            self.db_manager.save_session(session)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The ID of the session to delete
            
        Returns:
            True if the session was deleted, False otherwise
        """
        with self.lock:
            # Remove from in-memory sessions
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            # Delete from database
            return self.db_manager.delete_session(session_id)
    
    async def stop_session(self, session_id: str) -> bool:
        """
        Stop a session.
        
        Args:
            session_id: The ID of the session to stop
            
        Returns:
            True if the session was stopped, False otherwise
        """
        session = self.get_session(session_id)
        
        if not session:
            logger.warning(f"Session {session_id} not found")
            return False
        
        if session.status not in ["running", "starting"]:
            logger.warning(f"Cannot stop session {session_id} with status {session.status}")
            return False
        
        # Stop the session
        await session.stop()
        
        # Update session in database
        self.db_manager.save_session(session)
        
        return True
    
    async def stop_all_sessions(self) -> None:
        """Stop all active sessions."""
        with self.lock:
            active_sessions = [
                session for session in self.sessions.values()
                if session.status in ["running", "starting"]
            ]
        
        # Stop each session
        for session in active_sessions:
            await session.stop()
            
            # Update session in database
            self.db_manager.save_session(session)
    
    def add_alert(self, session_id: str, alert: Dict[str, Any]) -> None:
        """
        Add an alert for a session.
        
        Args:
            session_id: The ID of the session
            alert: The alert to add
        """
        # Ensure alert has an ID
        if "id" not in alert:
            alert["id"] = str(uuid.uuid4())
        
        # Ensure alert has a session ID
        alert["session_id"] = session_id
        
        # Save to database
        self.db_manager.save_alert(alert)
    
    def get_alerts(self, session_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get alerts for a session.
        
        Args:
            session_id: The ID of the session
            limit: Maximum number of alerts to return
            offset: Offset for pagination
            
        Returns:
            List of alerts
        """
        return self.db_manager.load_alerts(session_id, limit, offset)
    
    def mark_alert_as_read(self, alert_id: str) -> bool:
        """
        Mark an alert as read.
        
        Args:
            alert_id: The ID of the alert
            
        Returns:
            True if the alert was marked as read, False otherwise
        """
        return self.db_manager.mark_alert_as_read(alert_id)
    
    def update_alert_settings(self, session_id: str, settings: Dict[str, Any]) -> None:
        """
        Update alert settings for a session.
        
        Args:
            session_id: The ID of the session
            settings: The alert settings
        """
        self.db_manager.save_alert_settings(session_id, settings)
    
    def get_alert_settings(self, session_id: str) -> Dict[str, Any]:
        """
        Get alert settings for a session.
        
        Args:
            session_id: The ID of the session
            
        Returns:
            The alert settings
        """
        return self.db_manager.load_alert_settings(session_id)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Nothing to clean up for now
        pass


# Create singleton instance
session_manager = SessionManager()
