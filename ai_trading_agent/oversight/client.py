"""
LLM Oversight Client.

This module provides a client for interacting with the LLM Oversight service
from other components of the AI Trading Agent system.
"""

import logging
import time
import json
import requests
from typing import Any, Dict, Optional, Union
from enum import Enum
from urllib.parse import urljoin

# Configure logging
logger = logging.getLogger(__name__)

class OversightAction(Enum):
    """Possible oversight actions that can be taken on trading decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    LOG = "log"

class OversightClient:
    """
    Client for interacting with the LLM Oversight service.
    
    This client allows other components of the AI Trading Agent system to
    leverage the LLM oversight capabilities for decision validation,
    market analysis, and other functions.
    """
    
    def __init__(
        self, 
        base_url: str = "http://llm-oversight-service", 
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the LLM Oversight client.
        
        Args:
            base_url: Base URL of the oversight service
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
        logger.info(f"LLM Oversight client initialized with base URL: {base_url}")
    
    def _make_request(
        self, 
        endpoint: str, 
        method: str = "POST", 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the oversight service with retry logic.
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method to use
            data: JSON data to send in the request body
            params: URL parameters to include
            
        Returns:
            Response from the oversight service
            
        Raises:
            ConnectionError: If connection to the oversight service fails
            TimeoutError: If the request times out
            Exception: For other request failures
        """
        url = urljoin(self.base_url, endpoint)
        attempts = 0
        
        while attempts < self.retry_attempts:
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=self.timeout)
                else:
                    response = self.session.post(url, json=data, timeout=self.timeout)
                
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.ConnectionError as e:
                attempts += 1
                if attempts >= self.retry_attempts:
                    logger.error(f"Failed to connect to oversight service after {self.retry_attempts} attempts: {str(e)}")
                    raise ConnectionError(f"Failed to connect to oversight service: {str(e)}")
                else:
                    logger.warning(f"Connection failed, retrying in {self.retry_delay}s (attempt {attempts}/{self.retry_attempts})")
                    time.sleep(self.retry_delay)
                    # Increase delay for subsequent retries
                    self.retry_delay *= 1.5
            
            except requests.exceptions.Timeout as e:
                attempts += 1
                if attempts >= self.retry_attempts:
                    logger.error(f"Request to oversight service timed out after {self.retry_attempts} attempts")
                    raise TimeoutError(f"Request to oversight service timed out: {str(e)}")
                else:
                    logger.warning(f"Request timed out, retrying in {self.retry_delay}s (attempt {attempts}/{self.retry_attempts})")
                    time.sleep(self.retry_delay)
                    # Increase delay for subsequent retries
                    self.retry_delay *= 1.5
            
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error from oversight service: {str(e)}")
                # Get the error details if available
                try:
                    error_detail = response.json().get("detail", str(e))
                    raise Exception(f"Oversight service error: {error_detail}")
                except Exception:
                    raise Exception(f"Oversight service error: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error communicating with oversight service: {str(e)}")
                raise Exception(f"Oversight service request failed: {str(e)}")
    
    def check_health(self) -> bool:
        """
        Check if the oversight service is healthy.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            response = self._make_request("health", method="GET")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Oversight service health check failed: {str(e)}")
            return False
    
    def validate_trading_decision(
        self, 
        decision: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a trading decision using LLM oversight.
        
        Args:
            decision: Trading decision to validate
            context: Context for decision validation
            
        Returns:
            Validation results including approval/rejection
        """
        data = {
            "decision": decision,
            "context": context
        }
        
        logger.debug(f"Validating trading decision: {json.dumps(decision)}")
        response = self._make_request("validate/decision", data=data)
        
        result = response.get("result", {})
        logger.info(f"Decision validation result: {result.get('action', 'unknown')}")
        
        return result
    
    def get_decision_action(
        self, 
        decision: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> OversightAction:
        """
        Get the recommended action for a trading decision.
        
        Args:
            decision: Trading decision to validate
            context: Context for decision validation
            
        Returns:
            Recommended oversight action (APPROVE, REJECT, MODIFY, LOG)
        """
        try:
            result = self.validate_trading_decision(decision, context)
            action_str = result.get("action", "").lower()
            
            if action_str == "approve":
                return OversightAction.APPROVE
            elif action_str == "reject":
                return OversightAction.REJECT
            elif action_str == "modify":
                return OversightAction.MODIFY
            else:
                # Default to logging if action is unclear
                logger.warning(f"Unclear oversight action: {action_str}, defaulting to LOG")
                return OversightAction.LOG
                
        except Exception as e:
            logger.error(f"Failed to get decision action: {str(e)}")
            # If service fails, default to logging the decision without blocking
            return OversightAction.LOG
    
    def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions using LLM oversight.
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Analysis results including regime identification
        """
        data = {"data": market_data}
        response = self._make_request("analyze/market", data=data)
        return response.get("result", {})
    
    def detect_anomalies(
        self, 
        data: Dict[str, Any], 
        thresholds: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect anomalies in data using LLM oversight.
        
        Args:
            data: Data to analyze for anomalies
            thresholds: Threshold parameters
            
        Returns:
            Anomaly detection results
        """
        request_data = {
            "data": data,
            "thresholds": thresholds
        }
        response = self._make_request("detect/anomalies", data=request_data)
        return response.get("result", {})
    
    def explain_market_event(
        self, 
        event: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain a market event using LLM oversight.
        
        Args:
            event: Market event to explain
            context: Additional context
            
        Returns:
            Explanation of the market event
        """
        data = {
            "event": event,
            "context": context
        }
        response = self._make_request("explain/event", data=data)
        return response.get("result", {})
    
    def suggest_strategy_adjustments(
        self, 
        current_strategy: Dict[str, Any], 
        performance_metrics: Dict[str, Any], 
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest strategy adjustments using LLM oversight.
        
        Args:
            current_strategy: Current trading strategy
            performance_metrics: Performance metrics
            market_conditions: Market conditions
            
        Returns:
            Suggested strategy adjustments
        """
        data = {
            "current_strategy": current_strategy,
            "performance_metrics": performance_metrics,
            "market_conditions": market_conditions
        }
        response = self._make_request("suggest/strategy-adjustments", data=data)
        return response.get("result", {})
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current oversight configuration.
        
        Returns:
            Current oversight configuration
        """
        return self._make_request("config", method="GET")
