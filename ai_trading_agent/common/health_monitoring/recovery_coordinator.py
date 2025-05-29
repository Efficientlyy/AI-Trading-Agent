"""
Recovery Coordinator for Health Monitoring System.

This module implements the recovery coordination functionality,
managing recovery actions and their execution in response to health issues.
"""

import threading
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Set

# Import from core definitions to avoid circular dependencies
from .core_definitions import AlertSeverity
from .health_status import AlertData

# Set up logger
logger = logging.getLogger(__name__)


class RecoveryAction:
    """
    Represents a recovery action that can be taken in response to health issues.
    """
    
    def __init__(
        self,
        action_id: str,
        description: str,
        action_func: Callable[..., bool],
        component_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        severity_threshold: AlertSeverity = AlertSeverity.ERROR
    ):
        """
        Initialize a recovery action.
        
        Args:
            action_id: Unique identifier for the action
            description: Human-readable description of the action
            action_func: Function to call when action is executed
            component_id: Optional component ID this action applies to
            parameters: Optional parameters to pass to action function
            severity_threshold: Only execute for alerts of this severity or higher
        """
        self.action_id = action_id
        self.description = description
        self.action_func = action_func
        self.component_id = component_id
        self.parameters = parameters or {}
        self.severity_threshold = severity_threshold
        self.last_executed = None
        self.success_count = 0
        self.failure_count = 0
    
    def execute(self) -> bool:
        """
        Execute the recovery action.
        
        Returns:
            True if action was successful, False otherwise
        """
        self.last_executed = time.time()
        
        try:
            result = self.action_func(**self.parameters)
            
            if result:
                self.success_count += 1
            else:
                self.failure_count += 1
                
            return result
        except Exception as e:
            logger.error(f"Error executing recovery action {self.action_id}: {str(e)}")
            self.failure_count += 1
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the recovery action to a dictionary representation."""
        return {
            "action_id": self.action_id,
            "description": self.description,
            "component_id": self.component_id,
            "parameters": {k: str(v) for k, v in self.parameters.items()},
            "severity_threshold": self.severity_threshold.value,
            "last_executed": self.last_executed,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }


class RecoveryCoordinator:
    """
    Coordinates recovery actions for system components.
    
    Manages the selection and execution of recovery actions in response to health issues,
    provides logging and tracking of recovery attempts, and prevents action conflicts.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the recovery coordinator.
        
        Args:
            max_history: Maximum number of recovery records to keep
        """
        self.actions = {}  # Dict[str, RecoveryAction]
        self.component_actions = {}  # Dict[str, List[str]]
        self.recovery_history = []  # List of recovery attempt records
        self.executing_actions = set()  # Set of action IDs currently executing
        self.max_history = max_history
        self._lock = threading.RLock()
    
    def register_action(self, action: RecoveryAction) -> None:
        """
        Register a recovery action.
        
        Args:
            action: The recovery action to register
        """
        with self._lock:
            if action.action_id in self.actions:
                logger.warning(f"Recovery action {action.action_id} already registered")
                return
                
            self.actions[action.action_id] = action
            
            # Index by component if specified
            if action.component_id:
                if action.component_id not in self.component_actions:
                    self.component_actions[action.component_id] = []
                    
                self.component_actions[action.component_id].append(action.action_id)
                
            logger.info(f"Registered recovery action {action.action_id}: {action.description}")
    
    def unregister_action(self, action_id: str) -> bool:
        """
        Unregister a recovery action.
        
        Args:
            action_id: ID of the action to unregister
            
        Returns:
            True if action was unregistered, False if not found
        """
        with self._lock:
            if action_id not in self.actions:
                logger.warning(f"Recovery action {action_id} not found for unregistration")
                return False
                
            action = self.actions[action_id]
            del self.actions[action_id]
            
            # Remove from component index if applicable
            if action.component_id and action.component_id in self.component_actions:
                if action_id in self.component_actions[action.component_id]:
                    self.component_actions[action.component_id].remove(action_id)
                    
            logger.info(f"Unregistered recovery action {action_id}")
            return True
    
    def handle_alert(self, alert: AlertData) -> bool:
        """
        Handle an alert by finding and executing appropriate recovery actions.
        
        Args:
            alert: The alert to handle
            
        Returns:
            True if recovery action was taken, False otherwise
        """
        with self._lock:
            component_id = alert.component_id
            actions_to_execute = []
            
            # Find applicable actions
            if component_id in self.component_actions:
                for action_id in self.component_actions[component_id]:
                    action = self.actions[action_id]
                    
                    # Check if severity threshold is met
                    severity_values = {s: i for i, s in enumerate(AlertSeverity)}
                    if severity_values[alert.severity] >= severity_values[action.severity_threshold]:
                        actions_to_execute.append(action)
            
            # Execute actions
            if not actions_to_execute:
                logger.info(f"No recovery actions available for alert {alert.alert_id}")
                return False
                
            # Execute first applicable action
            action = actions_to_execute[0]
            
            # Check if already executing
            if action.action_id in self.executing_actions:
                logger.info(f"Recovery action {action.action_id} already executing")
                return False
                
            # Mark as executing
            self.executing_actions.add(action.action_id)
            
            try:
                logger.info(f"Executing recovery action {action.action_id} for alert {alert.alert_id}")
                success = action.execute()
                
                # Record recovery attempt
                recovery_record = {
                    "timestamp": time.time(),
                    "alert_id": alert.alert_id,
                    "action_id": action.action_id,
                    "component_id": component_id,
                    "success": success
                }
                
                self.recovery_history.append(recovery_record)
                
                # Keep recovery history size reasonable
                if len(self.recovery_history) > self.max_history:
                    self.recovery_history = self.recovery_history[-self.max_history:]
                    
                return success
            finally:
                # Remove from executing set
                self.executing_actions.remove(action.action_id)
    
    def get_actions_for_component(self, component_id: str) -> List[RecoveryAction]:
        """
        Get all recovery actions for a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            List of recovery actions
        """
        with self._lock:
            if component_id not in self.component_actions:
                return []
                
            return [self.actions[action_id] for action_id in self.component_actions[component_id]]
    
    def get_recovery_history(
        self,
        component_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recovery history with optional filtering.
        
        Args:
            component_id: Optional component ID to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of recovery attempt records
        """
        with self._lock:
            if component_id is None:
                filtered_history = self.recovery_history
            else:
                filtered_history = [
                    record for record in self.recovery_history
                    if record["component_id"] == component_id
                ]
                
            # Sort by timestamp (newest first)
            sorted_history = sorted(
                filtered_history,
                key=lambda r: r["timestamp"],
                reverse=True
            )
            
            return sorted_history[:limit]
    
    def get_all_actions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered recovery actions.
        
        Returns:
            Dictionary of recovery actions by ID
        """
        with self._lock:
            return {
                action_id: action.to_dict()
                for action_id, action in self.actions.items()
            }
