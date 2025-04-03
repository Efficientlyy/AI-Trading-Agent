"""
Connection Pool

This module provides a connection pool for HTTP requests.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConnectionPool:
    """
    Connection pool for HTTP requests.
    
    This class provides a pool of HTTP connections for making requests,
    with features like connection reuse, automatic retries, and timeouts.
    """
    
    def __init__(self, pool_size: int = 10, max_retries: int = 3, 
                 backoff_factor: float = 0.5, timeout: int = 30):
        """
        Initialize connection pool.
        
        Args:
            pool_size: Maximum number of connections in the pool
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
            timeout: Request timeout in seconds
        """
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.sessions = {}
        self.lock = threading.RLock()
        self.stats = {
            'requests': 0,
            'retries': 0,
            'errors': 0,
            'timeouts': 0
        }
        
        logger.info(f"Initialized connection pool with size {pool_size}")
        
    def get_session(self, name: str = 'default') -> requests.Session:
        """
        Get a session from the pool.
        
        Args:
            name: Session name
            
        Returns:
            requests.Session: Session object
        """
        with self.lock:
            # Check if session exists
            if name in self.sessions:
                return self.sessions[name]
                
            # Create new session
            session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.max_retries,
                backoff_factor=self.backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
            )
            
            # Configure adapter
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=self.pool_size,
                pool_maxsize=self.pool_size
            )
            
            # Mount adapter
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Store session
            self.sessions[name] = session
            
            logger.debug(f"Created new session: {name}")
            return session
            
    def request(self, method: str, url: str, session_name: str = 'default', 
                **kwargs) -> requests.Response:
        """
        Make a request using a session from the pool.
        
        Args:
            method: HTTP method
            url: URL to request
            session_name: Session name
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response: Response object
        """
        # Get session
        session = self.get_session(session_name)
        
        # Set timeout
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
            
        # Make request
        try:
            self.stats['requests'] += 1
            start_time = time.time()
            response = session.request(method, url, **kwargs)
            elapsed = time.time() - start_time
            
            # Check for retries
            retries = getattr(response.raw, 'retries', None)
            if retries and retries.history:
                self.stats['retries'] += len(retries.history)
                
            logger.debug(f"Request {method} {url} completed in {elapsed:.2f}s")
            return response
        except requests.exceptions.Timeout:
            self.stats['timeouts'] += 1
            logger.warning(f"Request {method} {url} timed out after {self.timeout}s")
            raise
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Request {method} {url} failed: {e}")
            raise
            
    def close(self, session_name: str = None):
        """
        Close session(s).
        
        Args:
            session_name: Session name (None for all sessions)
        """
        with self.lock:
            if session_name:
                # Close specific session
                if session_name in self.sessions:
                    self.sessions[session_name].close()
                    del self.sessions[session_name]
                    logger.debug(f"Closed session: {session_name}")
            else:
                # Close all sessions
                for name, session in self.sessions.items():
                    session.close()
                    logger.debug(f"Closed session: {name}")
                self.sessions = {}
                
    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dict with statistics
        """
        with self.lock:
            stats = {
                'pool_size': self.pool_size,
                'active_sessions': len(self.sessions),
                'requests': self.stats['requests'],
                'retries': self.stats['retries'],
                'errors': self.stats['errors'],
                'timeouts': self.stats['timeouts']
            }
            
            return stats


class RequestBatcher:
    """
    Batch multiple requests into a single request.
    
    This class allows batching multiple API requests into a single request,
    reducing the number of HTTP connections and improving performance.
    """
    
    def __init__(self, batch_size: int = 10, batch_interval: float = 0.1):
        """
        Initialize request batcher.
        
        Args:
            batch_size: Maximum number of requests in a batch
            batch_interval: Maximum time to wait for a batch (seconds)
        """
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.batches = {}
        self.lock = threading.RLock()
        self.stats = {
            'batches_created': 0,
            'batches_sent': 0,
            'requests_batched': 0,
            'requests_sent': 0
        }
        
        logger.info(f"Initialized request batcher with size {batch_size}")
        
    def add_request(self, batch_key: str, request_data: Dict[str, Any], 
                    callback: Callable[[Dict[str, Any]], None] = None) -> str:
        """
        Add a request to a batch.
        
        Args:
            batch_key: Batch key
            request_data: Request data
            callback: Callback function for response
            
        Returns:
            str: Request ID
        """
        with self.lock:
            # Generate request ID
            request_id = f"{batch_key}_{int(time.time() * 1000)}_{self.stats['requests_batched']}"
            
            # Check if batch exists
            if batch_key not in self.batches:
                # Create new batch
                self.batches[batch_key] = {
                    'requests': [],
                    'callbacks': {},
                    'created_at': time.time(),
                    'last_update': time.time()
                }
                self.stats['batches_created'] += 1
                
                # Start batch timer
                threading.Timer(self.batch_interval, self._process_batch, args=[batch_key]).start()
                
            # Add request to batch
            self.batches[batch_key]['requests'].append({
                'id': request_id,
                'data': request_data
            })
            
            # Store callback
            if callback:
                self.batches[batch_key]['callbacks'][request_id] = callback
                
            # Update timestamp
            self.batches[batch_key]['last_update'] = time.time()
            
            self.stats['requests_batched'] += 1
            
            # Check if batch is full
            if len(self.batches[batch_key]['requests']) >= self.batch_size:
                # Process batch
                threading.Thread(target=self._process_batch, args=[batch_key]).start()
                
            return request_id
            
    def _process_batch(self, batch_key: str):
        """
        Process a batch.
        
        Args:
            batch_key: Batch key
        """
        with self.lock:
            # Check if batch exists
            if batch_key not in self.batches:
                return
                
            # Get batch
            batch = self.batches[batch_key]
            
            # Check if batch is empty
            if not batch['requests']:
                del self.batches[batch_key]
                return
                
            # Extract batch data
            requests = batch['requests']
            callbacks = batch['callbacks']
            
            # Remove batch
            del self.batches[batch_key]
            
        # Process batch (outside lock)
        try:
            self.stats['batches_sent'] += 1
            self.stats['requests_sent'] += len(requests)
            
            # Call batch processor
            self._process_batch_requests(batch_key, requests, callbacks)
        except Exception as e:
            logger.error(f"Error processing batch {batch_key}: {e}")
            
    def _process_batch_requests(self, batch_key: str, requests: List[Dict[str, Any]], 
                               callbacks: Dict[str, Callable[[Dict[str, Any]], None]]):
        """
        Process batch requests.
        
        This method should be overridden by subclasses to implement
        the actual batch processing logic.
        
        Args:
            batch_key: Batch key
            requests: List of requests
            callbacks: Dict of callbacks
        """
        logger.warning("Request batching not implemented for base class")
        
        # Call callbacks with empty responses
        for request in requests:
            request_id = request['id']
            if request_id in callbacks:
                try:
                    callbacks[request_id]({
                        'id': request_id,
                        'error': 'Batch processing not implemented'
                    })
                except Exception as e:
                    logger.error(f"Error in callback for {request_id}: {e}")
                    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batcher statistics.
        
        Returns:
            Dict with statistics
        """
        with self.lock:
            stats = {
                'batch_size': self.batch_size,
                'batch_interval': self.batch_interval,
                'active_batches': len(self.batches),
                'batches_created': self.stats['batches_created'],
                'batches_sent': self.stats['batches_sent'],
                'requests_batched': self.stats['requests_batched'],
                'requests_sent': self.stats['requests_sent']
            }
            
            return stats