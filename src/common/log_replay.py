"""Log replay system for debugging and testing.

This module allows replaying log events for debugging or testing purposes.
It can be used to recreate specific scenarios or test how the system would
respond to particular sequences of events.
"""

import gzip
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union

import structlog

from src.common.config import config
from src.common.logging import LOG_DIR


class LogReplay:
    """Replay logs for testing and debugging purposes."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        handlers: Optional[Dict[str, Callable]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the log replay system.
        
        Args:
            log_file: Optional specific log file to replay
            log_dir: Optional directory containing log files
            handlers: Dict of event type to handler function
            filters: Dict of field name to filter value
        """
        self.log_file = log_file
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.handlers = handlers or {}
        self.filters = filters or {}
        self.logger = structlog.get_logger("log_replay")
        
    def add_handler(self, event_type: str, handler: Callable) -> None:
        """
        Add a handler for a specific event type.
        
        Args:
            event_type: Event type to handle
            handler: Function that takes a log entry and processes it
        """
        self.handlers[event_type] = handler
        
    def add_filter(self, field: str, value: Any) -> None:
        """
        Add a filter for log entries.
        
        Args:
            field: Field name to filter on
            value: Value or pattern to match
        """
        self.filters[field] = value
        
    def _should_process(self, entry: Dict[str, Any]) -> bool:
        """
        Check if an entry matches the filters.
        
        Args:
            entry: Log entry to check
            
        Returns:
            True if the entry should be processed
        """
        for field, filter_value in self.filters.items():
            # Skip if field not in entry
            if field not in entry:
                return False
                
            value = entry[field]
            
            # Handle regex pattern matching
            if isinstance(filter_value, Pattern) and isinstance(value, str):
                if not filter_value.search(value):
                    return False
            # Handle exact match
            elif value != filter_value:
                return False
                
        return True
        
    def _process_entry(self, entry: Dict[str, Any]) -> None:
        """
        Process a log entry.
        
        Args:
            entry: Log entry to process
        """
        # Skip if it doesn't match filters
        if not self._should_process(entry):
            return
            
        # Get event type
        event_type = entry.get('event_type', entry.get('level', 'info'))
        
        # Call specific handler if exists
        if event_type in self.handlers:
            try:
                self.handlers[event_type](entry)
            except Exception as e:
                self.logger.error(f"Error in handler for {event_type}", error=str(e))
        # Call default handler if exists
        elif 'default' in self.handlers:
            try:
                self.handlers['default'](entry)
            except Exception as e:
                self.logger.error("Error in default handler", error=str(e))
                
    def _parse_log_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a log file.
        
        Args:
            file_path: Path to log file
            
        Returns:
            List of parsed log entries
        """
        entries = []
        
        try:
            # Handle compressed log files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entries.append(entry)
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
            else:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entries.append(entry)
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
        except Exception as e:
            self.logger.error(f"Error parsing log file: {file_path}", error=str(e))
            
        return entries
        
    def _get_timestamp(self, entry: Dict[str, Any]) -> Optional[datetime]:
        """
        Extract timestamp from log entry.
        
        Args:
            entry: Log entry
            
        Returns:
            Datetime object or None if not found
        """
        # Try to get timestamp from entry
        ts_str = entry.get('timestamp')
        if not ts_str:
            return None
            
        try:
            # Handle different timestamp formats
            if 'T' in ts_str:
                # ISO format
                return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                # Try common formats
                for fmt in [
                    '%Y-%m-%d %H:%M:%S.%f',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y/%m/%d %H:%M:%S.%f',
                    '%Y/%m/%d %H:%M:%S'
                ]:
                    try:
                        return datetime.strptime(ts_str, fmt)
                    except ValueError:
                        continue
        except Exception:
            pass
            
        return None
        
    def replay_from_file(
        self,
        file_path: str,
        speed_factor: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_entries: Optional[int] = None
    ) -> int:
        """
        Replay logs from a specific file.
        
        Args:
            file_path: Path to log file
            speed_factor: Speed multiplier (1.0 = realtime)
            start_time: Optional start time to filter entries
            end_time: Optional end time to filter entries
            max_entries: Maximum number of entries to process
            
        Returns:
            Number of entries processed
        """
        self.logger.info(f"Replaying logs from file: {file_path}")
        
        # Parse log file
        entries = self._parse_log_file(file_path)
        self.logger.info(f"Found {len(entries)} log entries")
        
        # Filter by timestamp if requested
        if start_time or end_time:
            filtered_entries = []
            for entry in entries:
                ts = self._get_timestamp(entry)
                if ts is None:
                    continue
                    
                if start_time and ts < start_time:
                    continue
                    
                if end_time and ts > end_time:
                    continue
                    
                filtered_entries.append(entry)
                
            entries = filtered_entries
            self.logger.info(f"Filtered to {len(entries)} log entries by time range")
            
        # Sort by timestamp
        entries.sort(key=lambda e: self._get_timestamp(e) or datetime.min)
        
        # Apply max entries limit
        if max_entries is not None and len(entries) > max_entries:
            entries = entries[:max_entries]
            self.logger.info(f"Limited to {len(entries)} log entries")
            
        # Process entries
        count = 0
        last_ts = None
        
        for entry in entries:
            # Get timestamp for delay calculation
            ts = self._get_timestamp(entry)
            
            # Calculate delay
            if last_ts and ts and speed_factor > 0:
                delay = (ts - last_ts).total_seconds() / speed_factor
                if delay > 0:
                    time.sleep(delay)
                    
            # Process the entry
            self._process_entry(entry)
            count += 1
            last_ts = ts
            
        self.logger.info(f"Replay complete: processed {count} entries")
        return count
        
    def replay_all(
        self,
        pattern: str = "*.log*",
        speed_factor: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_entries: Optional[int] = None
    ) -> int:
        """
        Replay logs from all matching files in log directory.
        
        Args:
            pattern: Glob pattern for log files
            speed_factor: Speed multiplier (1.0 = realtime)
            start_time: Optional start time to filter entries
            end_time: Optional end time to filter entries
            max_entries: Maximum number of entries to process
            
        Returns:
            Number of entries processed
        """
        # Get list of log files
        log_files = list(self.log_dir.glob(pattern))
        self.logger.info(f"Found {len(log_files)} log files matching pattern: {pattern}")
        
        # Sort by modification time
        log_files.sort(key=lambda p: p.stat().st_mtime)
        
        # Process each file
        total_count = 0
        entries_left = max_entries
        
        for file_path in log_files:
            if entries_left is not None and entries_left <= 0:
                break
                
            count = self.replay_from_file(
                str(file_path),
                speed_factor=speed_factor,
                start_time=start_time,
                end_time=end_time,
                max_entries=entries_left
            )
            
            total_count += count
            
            if entries_left is not None:
                entries_left -= count
                
        self.logger.info(f"Replay complete: processed {total_count} entries from {len(log_files)} files")
        return total_count
        
    def replay_by_request_id(self, request_id: str, speed_factor: float = 1.0) -> int:
        """
        Replay all log entries for a specific request ID.
        
        Args:
            request_id: Request ID to filter by
            speed_factor: Speed multiplier (1.0 = realtime)
            
        Returns:
            Number of entries processed
        """
        self.logger.info(f"Replaying logs for request ID: {request_id}")
        
        # Add filter for request ID
        self.add_filter('request_id', request_id)
        
        # Replay from all log files
        return self.replay_all(speed_factor=speed_factor)
        
    def replay_by_component(self, component: str, speed_factor: float = 1.0) -> int:
        """
        Replay all log entries for a specific component.
        
        Args:
            component: Component name to filter by
            speed_factor: Speed multiplier (1.0 = realtime)
            
        Returns:
            Number of entries processed
        """
        self.logger.info(f"Replaying logs for component: {component}")
        
        # Add filter for component
        self.add_filter('component', component)
        
        # Replay from all log files
        return self.replay_all(speed_factor=speed_factor)
        
    def replay_by_pattern(self, field: str, pattern: str, speed_factor: float = 1.0) -> int:
        """
        Replay all log entries matching a regex pattern in a specific field.
        
        Args:
            field: Field name to match against
            pattern: Regex pattern to match
            speed_factor: Speed multiplier (1.0 = realtime)
            
        Returns:
            Number of entries processed
        """
        self.logger.info(f"Replaying logs matching pattern '{pattern}' in field '{field}'")
        
        # Compile regex pattern
        regex = re.compile(pattern)
        
        # Add filter
        self.add_filter(field, regex)
        
        # Replay from all log files
        return self.replay_all(speed_factor=speed_factor)


def create_replay_session(
    handlers: Optional[Dict[str, Callable]] = None,
    log_file: Optional[str] = None
) -> LogReplay:
    """
    Create a new log replay session.
    
    Args:
        handlers: Dict of event type to handler function
        log_file: Optional specific log file to replay
        
    Returns:
        Configured LogReplay instance
    """
    return LogReplay(log_file=log_file, handlers=handlers)
