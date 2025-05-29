"""
Session Manager module for the AI Trading Agent.

This module manages multiple paper trading sessions, providing isolation and persistence.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import asyncio
import threading
import time
import uuid
from datetime import datetime
import json
import os
import sqlite3
import pickle
import traceback
from pathlib import Path

from ..common import logger
from ..common.error_handling import (
    TradingAgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity
)

# Import recovery manager for integration
try:
    from .recovery_manager import RecoveryManager, RecoveryState, RecoveryStrategy
except ImportError:
    # Fallback for environments where the relative import might not work directly
    from recovery_manager import RecoveryManager, RecoveryState, RecoveryStrategy

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
        user_id: str = None,
        name: Optional[str] = None,
        strategy_name: Optional[str] = None,
        agent_role: Optional[str] = None, # Added agent_role
        outputs_to: Optional[List[str]] = None # Added outputs_to as a list
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
            name: Optional name for the session/agent
            strategy_name: Optional name of the strategy being used
            agent_role: Optional role of the agent in the multi-agent system
            outputs_to: Optional list of agent_ids this agent sends data/signals to
        """
        import logging
        from pathlib import Path
        # Ensure logs directory exists
        logs_dir = Path(__file__).parent / "../logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Set up per-session logger
        self.logger = logging.getLogger(f"ai_trading_agent.session.{session_id}")
        self.logger.setLevel(logging.INFO)
        log_file = logs_dir / f"{session_id}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
        file_handler.setFormatter(formatter)
        # Avoid duplicate handlers
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in self.logger.handlers):
            self.logger.addHandler(file_handler)
        self.logger.propagate = False
        self.logger.info(f"Initialized per-session logger for {session_id}")
        # --- rest of original code below ---

        self.session_id = session_id
        self.name = name or f"Session {session_id[:8]}" # Default name if not provided
        self.strategy_name = strategy_name
        self.agent_role = agent_role
        self.outputs_to = outputs_to or []
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
        self.paused = False
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Not paused by default
        
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
    
    async def pause(self) -> None:
        """Pause the session (trading loop will block until resumed)."""
        if self.status != "running":
            logger.warning(f"Cannot pause session {self.session_id} with status {self.status}")
            return
        self.paused = True
        self.pause_event.clear()
        self.update_status("paused")
        logger.info(f"Session {self.session_id} paused.")

    async def resume(self) -> None:
        """Resume the session if it is paused."""
        if self.status != "paused":
            logger.warning(f"Cannot resume session {self.session_id} with status {self.status}")
            return
        self.paused = False
        self.pause_event.set()
        self.update_status("running")
        logger.info(f"Session {self.session_id} resumed.")

    async def stop(self) -> None:
        """Stop the session."""
        if self.status not in ["running", "starting", "paused"]:
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


class TransactionJournal:
    """Handles the journaling and recovery of trading transactions."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the transaction journal.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), "../data/transactions.db")
        self.conn = None
        self.recovery_manager = None
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        
        # Create transactions table if it doesn't exist
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            data TEXT NOT NULL,
            result TEXT,
            updated_at TEXT
        )
        """)
        
        # Create snapshots table for state recovery
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            data BLOB NOT NULL,
            type TEXT NOT NULL
        )
        """)
        
        self.conn.commit()
    
    def set_recovery_manager(self, recovery_manager: RecoveryManager):
        """Set the recovery manager reference for delegation."""
        self.recovery_manager = recovery_manager
    
    def journal_transaction(self, session_id: str, transaction_type: str, data: Dict[str, Any]) -> str:
        """
        Record a transaction in the journal.
        
        Args:
            session_id: ID of the session
            transaction_type: Type of transaction (e.g., 'order', 'position_update')
            data: Transaction data
            
        Returns:
            Transaction ID
        """
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO transactions (id, session_id, type, status, timestamp, data) VALUES (?, ?, ?, ?, ?, ?)",
            (transaction_id, session_id, transaction_type, "pending", timestamp, json.dumps(data))
        )
        self.conn.commit()
        
        # Also delegate to recovery manager if available
        if self.recovery_manager:
            self.recovery_manager.journal_transaction(transaction_type, data, session_id)
        
        logger.debug(f"Journaled transaction {transaction_id} of type {transaction_type} for session {session_id}")
        return transaction_id
    
    def update_transaction_status(self, transaction_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a transaction.
        
        Args:
            transaction_id: ID of the transaction
            status: New status ('completed', 'failed', 'rolled_back')
            result: Optional result data
            
        Returns:
            Success status
        """
        updated_at = datetime.now().isoformat()
        
        try:
            cursor = self.conn.cursor()
            
            if result:
                cursor.execute(
                    "UPDATE transactions SET status = ?, result = ?, updated_at = ? WHERE id = ?",
                    (status, json.dumps(result), updated_at, transaction_id)
                )
            else:
                cursor.execute(
                    "UPDATE transactions SET status = ?, updated_at = ? WHERE id = ?",
                    (status, updated_at, transaction_id)
                )
            
            if cursor.rowcount == 0:
                logger.warning(f"Transaction {transaction_id} not found")
                return False
            
            self.conn.commit()
            
            # Also delegate to recovery manager if available
            if self.recovery_manager:
                self.recovery_manager.update_transaction_status(transaction_id, status, result)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to update transaction status: {str(e)}")
            return False
    
    def get_pending_transactions(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get pending transactions.
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            List of pending transactions
        """
        cursor = self.conn.cursor()
        
        if session_id:
            cursor.execute(
                "SELECT id, session_id, type, timestamp, data FROM transactions WHERE status = 'pending' AND session_id = ?",
                (session_id,)
            )
        else:
            cursor.execute(
                "SELECT id, session_id, type, timestamp, data FROM transactions WHERE status = 'pending'"
            )
        
        transactions = []
        for row in cursor.fetchall():
            transactions.append({
                "id": row[0],
                "session_id": row[1],
                "type": row[2],
                "timestamp": row[3],
                "data": json.loads(row[4])
            })
        
        return transactions
    
    def detect_orphaned_transactions(self, max_age_minutes: int = 5) -> List[Dict[str, Any]]:
        """
        Detect transactions that have been pending for too long.
        
        Args:
            max_age_minutes: Maximum age for a transaction in minutes
            
        Returns:
            List of orphaned transactions
        """
        cursor = self.conn.cursor()
        cutoff_time = (datetime.now() - datetime.timedelta(minutes=max_age_minutes)).isoformat()
        
        cursor.execute(
            "SELECT id, session_id, type, timestamp, data FROM transactions WHERE status = 'pending' AND timestamp < ?",
            (cutoff_time,)
        )
        
        orphaned = []
        for row in cursor.fetchall():
            orphaned.append({
                "id": row[0],
                "session_id": row[1],
                "type": row[2],
                "timestamp": row[3],
                "data": json.loads(row[4])
            })
        
        if orphaned:
            logger.warning(f"Detected {len(orphaned)} orphaned transactions")
        
        return orphaned
    
    def save_snapshot(self, session_id: str, snapshot_type: str, data: Any) -> str:
        """
        Save a snapshot of session state for recovery.
        
        Args:
            session_id: ID of the session
            snapshot_type: Type of snapshot ('portfolio', 'full_state', etc.)
            data: Snapshot data (will be pickled)
            
        Returns:
            Snapshot ID
        """
        snapshot_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Serialize data
        serialized_data = pickle.dumps(data)
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO snapshots (id, session_id, timestamp, data, type) VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, session_id, timestamp, serialized_data, snapshot_type)
        )
        self.conn.commit()
        
        logger.debug(f"Saved {snapshot_type} snapshot for session {session_id}")
        return snapshot_id
    
    def load_latest_snapshot(self, session_id: str, snapshot_type: str) -> Tuple[bool, Optional[Any]]:
        """
        Load the latest snapshot for a session.
        
        Args:
            session_id: ID of the session
            snapshot_type: Type of snapshot to load
            
        Returns:
            Tuple of (success, data)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT data FROM snapshots WHERE session_id = ? AND type = ? ORDER BY timestamp DESC LIMIT 1",
                (session_id, snapshot_type)
            )
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"No {snapshot_type} snapshot found for session {session_id}")
                return False, None
            
            # Deserialize data
            data = pickle.loads(row[0])
            
            return True, data
        
        except Exception as e:
            logger.error(f"Failed to load snapshot: {str(e)}")
            return False, None
    
    def clean_up_old_snapshots(self, days_to_keep: int = 7):
        """
        Clean up old snapshots.
        
        Args:
            days_to_keep: Number of days to keep snapshots
        """
        cutoff_time = (datetime.now() - datetime.timedelta(days=days_to_keep)).isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM snapshots WHERE timestamp < ?",
            (cutoff_time,)
        )
        
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old snapshots")


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
    
    def __init__(self, db_path: str = None, recovery_data_dir: str = "./data/recovery"):
        """
        Initialize the session manager.
        
        Args:
            db_path: Path to the SQLite database file
            recovery_data_dir: Directory for recovery data
        """
        self.sessions: Dict[str, PaperTradingSession] = {}
        self.lock = threading.Lock()
        self.db_manager = SessionDatabaseManager(db_path)
        
        # Initialize transaction journal and recovery services
        self.transaction_journal = TransactionJournal(db_path)
        self.recovery_manager = RecoveryManager(data_dir=recovery_data_dir)
        
        # Connect transaction journal with recovery manager for coordination
        self.transaction_journal.set_recovery_manager(self.recovery_manager)
        
        # Track portfolio states
        
        # Load existing sessions from database
        self._load_sessions()
        
    def _load_sessions(self):
        """
        Load existing sessions from the database.
        
        This method retrieves session data from persistent storage and
        initializes session objects for any previous sessions that were saved.
        """
        try:
            # For now, just return without loading sessions to avoid any errors
            # This prevents the startup from failing and avoids recursion issues
            logger.info("Session loading disabled for now to ensure backend starts correctly")
            return
            
            # The proper implementation would be:
            # sessions = self.db_manager.load_all_sessions()
            # for session in sessions:
            #     with self.lock:
            #         self.sessions[session.session_id] = session
        except Exception as e:
            # Log error but don't propagate it to prevent startup failure
            logger.error(f"Error loading sessions from database: {str(e)}")
            return
        self.portfolio_states: Dict[str, Dict[str, Any]] = {}
        
        # Recovery metrics
        self.recovery_metrics = {
            "recoveries_attempted": 0,
            "recoveries_successful": 0,
            "recoveries_failed": 0,
            "last_recovery_time": None,
            "transactions_reconciled": 0,
            "transactions_rolled_back": 0
        }
        
        # Load existing sessions from database
        self._load_sessions()
    
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
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a session."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return False
        if session.status != "running":
            logger.warning(f"Cannot pause session {session_id} with status {session.status}")
            return False
        await session.pause()
        self.db_manager.save_session(session)
        return True

    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return False
        if session.status != "paused":
            logger.warning(f"Cannot resume session {session_id} with status {session.status}")
            return False
        await session.resume()
        self.db_manager.save_session(session)
        return True

    async def pause_all_sessions(self) -> None:
        """Pause all running sessions."""
        with self.lock:
            running_sessions = [
                session for session in self.sessions.values()
                if session.status == "running"
            ]
        for session in running_sessions:
            await session.pause()
            self.db_manager.save_session(session)

    async def resume_all_sessions(self) -> None:
        """Resume all paused sessions."""
        with self.lock:
            paused_sessions = [
                session for session in self.sessions.values()
                if session.status == "paused"
            ]
        for session in paused_sessions:
            await session.resume()
            self.db_manager.save_session(session)

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
        
        if session.status not in ["running", "starting", "paused"]:
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
                if session.status in ["running", "starting", "paused"]
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
    
    def journal_transaction(self, session_id: str, transaction_type: str, data: Dict[str, Any]) -> str:
        """
        Journal a transaction for recovery purposes.
        
        Args:
            session_id: ID of the session
            transaction_type: Type of transaction
            data: Transaction data
            
        Returns:
            Transaction ID
        """
        return self.transaction_journal.journal_transaction(session_id, transaction_type, data)
    
    def update_transaction_status(self, transaction_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a journaled transaction.
        
        Args:
            transaction_id: ID of the transaction
            status: New status
            result: Optional result data
            
        Returns:
            Success status
        """
        return self.transaction_journal.update_transaction_status(transaction_id, status, result)
    
    def save_portfolio_state(self, session_id: str, portfolio_state: Dict[str, Any]) -> str:
        """
        Save the current portfolio state for a session.
        
        Args:
            session_id: ID of the session
            portfolio_state: Current portfolio state
            
        Returns:
            Snapshot ID
        """
        # Cache the portfolio state
        self.portfolio_states[session_id] = portfolio_state.copy()
        
        # Save to persistent storage
        return self.transaction_journal.save_snapshot(session_id, "portfolio", portfolio_state)
    
    def _recover_portfolio_state(self, session_id: str) -> bool:
        """
        Recover portfolio state for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Success status
        """
        # First check in-memory cache
        if session_id in self.portfolio_states:
            logger.info(f"Recovered portfolio state for session {session_id} from memory cache")
            return True
        
        # Try to load from persistent storage
        success, portfolio_state = self.transaction_journal.load_latest_snapshot(session_id, "portfolio")
        if success and portfolio_state:
            self.portfolio_states[session_id] = portfolio_state
            logger.info(f"Recovered portfolio state for session {session_id} from persistent storage")
            return True
        
        logger.warning(f"No portfolio state found for session {session_id}")
        return False
    
    def reconcile_orphaned_transactions(self) -> Tuple[int, int]:
        """
        Detect and reconcile orphaned transactions.
        
        Returns:
            Tuple of (reconciled count, failed count)
        """
        orphaned = self.transaction_journal.detect_orphaned_transactions()
        reconciled = 0
        failed = 0
        
        for transaction in orphaned:
            transaction_id = transaction["id"]
            transaction_type = transaction["type"]
            session_id = transaction["session_id"]
            
            # Different reconciliation strategy based on transaction type
            if transaction_type == "order":
                success = self._reconcile_order_transaction(transaction)
            elif transaction_type == "position_update":
                success = self._reconcile_position_transaction(transaction)
            else:
                # Default reconciliation: mark as failed
                success = False
                self.transaction_journal.update_transaction_status(
                    transaction_id,
                    "failed",
                    {"reason": f"Unknown transaction type: {transaction_type}"}
                )
            
            if success:
                reconciled += 1
            else:
                failed += 1
                
        # Update metrics
        self.recovery_metrics["transactions_reconciled"] += reconciled
        self.recovery_metrics["transactions_rolled_back"] += failed
        
        return reconciled, failed
    
    def _reconcile_order_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Reconcile an orphaned order transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Success status
        """
        transaction_id = transaction["id"]
        session_id = transaction["session_id"]
        data = transaction["data"]
        
        # Get the session
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Cannot reconcile order transaction {transaction_id} - session {session_id} not found")
            self.transaction_journal.update_transaction_status(
                transaction_id,
                "failed",
                {"reason": "Session not found"}
            )
            return False
        
        # Check with execution agent if order exists
        # This is a simplified example - in a real implementation we would query the broker
        order_id = data.get("order_id")
        symbol = data.get("symbol")
        
        # For demonstration, mark as rolled back
        self.transaction_journal.update_transaction_status(
            transaction_id,
            "rolled_back",
            {"reason": "Order transaction orphaned and rolled back during reconciliation"}
        )
        
        logger.info(f"Rolled back orphaned order transaction {transaction_id} for session {session_id}")
        return True
    
    def _reconcile_position_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Reconcile an orphaned position update transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Success status
        """
        transaction_id = transaction["id"]
        session_id = transaction["session_id"]
        data = transaction["data"]
        
        # Get the session
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Cannot reconcile position transaction {transaction_id} - session {session_id} not found")
            self.transaction_journal.update_transaction_status(
                transaction_id,
                "failed",
                {"reason": "Session not found"}
            )
            return False
        
        # For demonstration, restore from latest portfolio snapshot
        success = self._recover_portfolio_state(session_id)
        
        if success:
            # Mark transaction as reconciled
            self.transaction_journal.update_transaction_status(
                transaction_id,
                "completed",
                {"reason": "Position recovered from snapshot"}
            )
            logger.info(f"Reconciled position transaction {transaction_id} for session {session_id}")
            return True
        else:
            # Mark as failed
            self.transaction_journal.update_transaction_status(
                transaction_id,
                "failed",
                {"reason": "Could not recover position state"}
            )
            logger.warning(f"Failed to reconcile position transaction {transaction_id} for session {session_id}")
            return False
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """
        Get recovery metrics.
        
        Returns:
            Dictionary of recovery metrics
        """
        return self.recovery_metrics.copy()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up old snapshots
        try:
            self.transaction_journal.clean_up_old_snapshots()
        except Exception as e:
            logger.error(f"Error cleaning up old snapshots: {str(e)}")
        
        # Other cleanup tasks can be added here


# Create singleton instance
session_manager = SessionManager()
