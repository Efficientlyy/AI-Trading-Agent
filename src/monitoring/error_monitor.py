"""
Error Monitor

This module provides monitoring for API errors.
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorMonitor:
    """
    Monitor API errors and provide alerts when thresholds are reached.
    """
    
    def __init__(self, error_threshold: int = 3, window_seconds: int = 300, check_interval: int = 60):
        """
        Initialize error monitor.
        
        Args:
            error_threshold: Number of errors to trigger alert
            window_seconds: Time window for counting errors (seconds)
            check_interval: Interval for checking errors (seconds)
        """
        self.error_threshold = error_threshold
        self.window_seconds = window_seconds
        self.check_interval = check_interval
        self.errors = defaultdict(list)
        self.alert_callbacks = []
        self.circuit_breakers = {}
        self.running = False
        self.thread = None
        
    def register_error(self, source: str, error_type: str, message: str, details: Dict[str, Any] = None):
        """
        Register an error.
        
        Args:
            source: Error source
            error_type: Error type
            message: Error message
            details: Additional error details
        """
        now = datetime.now()
        
        # Create error record
        error = {
            'timestamp': now,
            'type': error_type,
            'message': message,
            'details': details or {}
        }
        
        # Add error to list
        self.errors[source].append(error)
        
        # Log error
        logger.warning(f"Error in {source}: {error_type} - {message}")
        
        # Remove old errors
        self._prune_old_errors()
        
        # Check if threshold is exceeded
        source_errors = self.errors[source]
        if len(source_errors) >= self.error_threshold:
            self._trigger_alert(source, source_errors)
            self._check_circuit_breaker(source, source_errors)
            
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback for alerts.
        
        Args:
            callback: Callback function that takes alert data as argument
        """
        self.alert_callbacks.append(callback)
        logger.info(f"Registered alert callback: {callback.__name__}")
        
    def start(self):
        """Start monitoring."""
        if self.running:
            logger.warning("Error monitor already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        logger.info("Error monitor started")
        
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
            
        logger.info("Error monitor stopped")
        
    def _monitor_loop(self):
        """Monitor loop."""
        while self.running:
            try:
                self._prune_old_errors()
                self._check_circuit_breakers()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in error monitor: {e}")
                
    def _prune_old_errors(self):
        """Remove old errors."""
        now = datetime.now().timestamp()
        cutoff_time = now - self.window_seconds
        
        for source in self.errors:
            self.errors[source] = [
                e for e in self.errors[source]
                if e['timestamp'].timestamp() > cutoff_time
            ]
            
    def _trigger_alert(self, source: str, errors: List[Dict[str, Any]]):
        """
        Trigger error alert.
        
        Args:
            source: Error source
            errors: List of errors
        """
        # Count errors by type
        error_counts = defaultdict(int)
        for error in errors:
            error_counts[error['type']] += 1
            
        # Create alert data
        alert_data = {
            'type': 'error',
            'source': source,
            'error_count': len(errors),
            'error_types': dict(error_counts),
            'errors': [
                {
                    'type': e['type'],
                    'message': e['message'],
                    'timestamp': e['timestamp'].isoformat()
                }
                for e in errors
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.warning(f"Error alert: {source} with {len(errors)} errors")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
    def _check_circuit_breaker(self, source: str, errors: List[Dict[str, Any]]):
        """
        Check if circuit breaker should be triggered.
        
        Args:
            source: Error source
            errors: List of errors
        """
        # Count errors by type
        error_counts = defaultdict(int)
        for error in errors:
            error_counts[error['type']] += 1
            
        # Check if any error type exceeds threshold
        for error_type, count in error_counts.items():
            if count >= self.error_threshold:
                self._trip_circuit_breaker(source, error_type, count)
                
    def _trip_circuit_breaker(self, source: str, error_type: str, count: int):
        """
        Trip circuit breaker.
        
        Args:
            source: Error source
            error_type: Error type
            count: Error count
        """
        key = f"{source}:{error_type}"
        
        # Check if circuit breaker already tripped
        if key in self.circuit_breakers:
            return
            
        # Calculate reset time
        reset_time = datetime.now().timestamp() + (count * 60)  # 1 minute per error
        
        # Trip circuit breaker
        self.circuit_breakers[key] = {
            'source': source,
            'error_type': error_type,
            'tripped_at': datetime.now(),
            'reset_time': reset_time,
            'error_count': count
        }
        
        logger.warning(f"Circuit breaker tripped for {source}:{error_type}, reset in {count} minutes")
        
        # Trigger alert
        alert_data = {
            'type': 'circuit_breaker',
            'source': source,
            'error_type': error_type,
            'error_count': count,
            'reset_time': reset_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
    def _check_circuit_breakers(self):
        """Check circuit breakers for reset."""
        now = datetime.now().timestamp()
        
        # Find circuit breakers to reset
        to_reset = []
        for key, breaker in self.circuit_breakers.items():
            if breaker['reset_time'] <= now:
                to_reset.append(key)
                
        # Reset circuit breakers
        for key in to_reset:
            breaker = self.circuit_breakers[key]
            logger.info(f"Circuit breaker reset for {breaker['source']}:{breaker['error_type']}")
            del self.circuit_breakers[key]
            
    def is_circuit_open(self, source: str, error_type: str = None) -> bool:
        """
        Check if circuit breaker is open.
        
        Args:
            source: Error source
            error_type: Error type (optional)
            
        Returns:
            bool: True if circuit breaker is open
        """
        if error_type:
            # Check specific circuit breaker
            key = f"{source}:{error_type}"
            return key in self.circuit_breakers
        else:
            # Check any circuit breaker for source
            for key in self.circuit_breakers:
                if key.startswith(f"{source}:"):
                    return True
            return False
            
    def get_circuit_breakers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all circuit breakers.
        
        Returns:
            Dict with circuit breaker status
        """
        result = {}
        
        for key, breaker in self.circuit_breakers.items():
            result[key] = {
                'source': breaker['source'],
                'error_type': breaker['error_type'],
                'tripped_at': breaker['tripped_at'].isoformat(),
                'reset_time': breaker['reset_time'],
                'error_count': breaker['error_count']
            }
            
        return result
        
    def get_error_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get error statistics.
        
        Returns:
            Dict with error statistics
        """
        stats = {}
        
        for source, errors in self.errors.items():
            # Count errors by type
            error_counts = defaultdict(int)
            for error in errors:
                error_counts[error['type']] += 1
                
            # Get latest error
            latest_error = max(errors, key=lambda e: e['timestamp']) if errors else None
            
            # Add to stats
            stats[source] = {
                'total_errors': len(errors),
                'error_types': dict(error_counts),
                'latest_error': {
                    'type': latest_error['type'],
                    'message': latest_error['message'],
                    'timestamp': latest_error['timestamp'].isoformat()
                } if latest_error else None,
                'circuit_breaker_status': self.is_circuit_open(source)
            }
            
        return stats