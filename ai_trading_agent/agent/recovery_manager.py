"""
Recovery Manager for AI Trading Agent.

This module implements state preservation, autonomous recovery strategies,
and coordination of recovery operations across the AI Trading Agent system.
"""
import logging
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
import threading
import uuid
import pickle
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecoveryState(Enum):
    """States for the recovery process."""
    NORMAL = "normal"               # System is operating normally
    RECOVERY_NEEDED = "recovery"    # Recovery is needed
    RECOVERING = "recovering"       # Recovery is in progress
    DEGRADED = "degraded"           # System is operating in degraded mode
    FAILED = "failed"               # Recovery failed


class RecoveryStrategy(Enum):
    """Types of recovery strategies available."""
    RESTART = "restart"             # Simple restart of agent
    STATE_RESTORE = "state_restore" # Restore agent from saved state
    ROLLBACK = "rollback"           # Rollback to a previous consistent state
    CHECKPOINT = "checkpoint"       # Restore from checkpoint
    GRACEFUL_DEGRADATION = "graceful_degradation" # Continue in degraded mode


class RecoveryManager:
    """
    Manages the recovery process for AI Trading Agent components.
    
    Responsibilities:
    - State preservation for agents and transactions
    - Coordinating recovery operations
    - Managing recovery strategies based on failure type
    - Maintaining system resilience during recovery
    """
    
    def __init__(self, data_dir: str = "./data/recovery"):
        """
        Initialize the recovery manager.
        
        Args:
            data_dir: Directory for storing recovery data
        """
        self.data_dir = data_dir
        self.checkpoint_dir = os.path.join(data_dir, "checkpoints")
        self.state_dir = os.path.join(data_dir, "state")
        self.journal_dir = os.path.join(data_dir, "journal")
        
        # Create required directories
        for directory in [self.data_dir, self.checkpoint_dir, self.state_dir, self.journal_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Recovery state and properties
        self.recovery_state = RecoveryState.NORMAL
        self.failed_components: Set[str] = set()
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        
        # Agent state cache for fast recovery
        self.agent_state_cache: Dict[str, Dict[str, Any]] = {}
        
        # Transaction journal
        self.transaction_journal: List[Dict[str, Any]] = []
        self.journal_lock = threading.Lock()
        
        # Recovery metrics
        self.recovery_metrics = {
            "recoveries_attempted": 0,
            "recoveries_successful": 0,
            "recoveries_failed": 0,
            "last_recovery_time": None,
            "average_recovery_time_ms": 0,
            "total_recovery_time_ms": 0,
        }
        
        logger.info(f"Recovery Manager initialized with data directory: {data_dir}")
    
    def checkpoint_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """
        Create a checkpoint of agent state for recovery.
        
        Args:
            agent_id: ID of the agent
            state: Agent state dictionary
            
        Returns:
            Success status
        """
        try:
            # Cache state in memory
            self.agent_state_cache[agent_id] = copy.deepcopy(state)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{agent_id}_{timestamp}.checkpoint"
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            # Save state to file
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            # Create a pointer to the latest checkpoint
            latest_filepath = os.path.join(self.state_dir, f"{agent_id}_latest.state")
            with open(latest_filepath, 'w') as f:
                f.write(filepath)
            
            logger.debug(f"Checkpointed agent state for {agent_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to checkpoint agent state for {agent_id}: {str(e)}")
            return False
    
    def restore_agent_state(self, agent_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Restore agent state from the latest checkpoint.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Tuple of (success status, restored state)
        """
        # First try to restore from memory cache for speed
        if agent_id in self.agent_state_cache:
            logger.info(f"Restored agent state for {agent_id} from memory cache")
            return True, copy.deepcopy(self.agent_state_cache[agent_id])
        
        try:
            # Get latest checkpoint file path
            latest_filepath = os.path.join(self.state_dir, f"{agent_id}_latest.state")
            if not os.path.exists(latest_filepath):
                logger.warning(f"No state file found for agent {agent_id}")
                return False, None
            
            # Read the checkpoint path
            with open(latest_filepath, 'r') as f:
                checkpoint_path = f.read().strip()
            
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint file {checkpoint_path} not found")
                return False, None
            
            # Load state from checkpoint
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            
            # Cache state in memory
            self.agent_state_cache[agent_id] = copy.deepcopy(state)
            
            logger.info(f"Restored agent state for {agent_id} from {checkpoint_path}")
            return True, state
        
        except Exception as e:
            logger.error(f"Failed to restore agent state for {agent_id}: {str(e)}")
            return False, None
    
    def journal_transaction(self, 
                          transaction_type: str, 
                          transaction_data: Dict[str, Any],
                          agent_id: str) -> str:
        """
        Record a transaction in the journal for potential recovery.
        
        Args:
            transaction_type: Type of transaction
            transaction_data: Transaction data
            agent_id: ID of the agent creating the transaction
            
        Returns:
            Transaction ID
        """
        # Generate transaction ID if not provided
        transaction_id = transaction_data.get('transaction_id', str(uuid.uuid4()))
        transaction_data['transaction_id'] = transaction_id
        
        # Create journal entry
        journal_entry = {
            'transaction_id': transaction_id,
            'transaction_type': transaction_type,
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'data': transaction_data
        }
        
        # Add to in-memory journal with thread safety
        with self.journal_lock:
            self.transaction_journal.append(journal_entry)
        
        # Persist to disk
        self._persist_journal_entry(journal_entry)
        
        logger.debug(f"Journaled transaction {transaction_id} of type {transaction_type}")
        return transaction_id
    
    def _persist_journal_entry(self, journal_entry: Dict[str, Any]) -> bool:
        """
        Persist a journal entry to disk.
        
        Args:
            journal_entry: Journal entry to persist
            
        Returns:
            Success status
        """
        try:
            transaction_id = journal_entry['transaction_id']
            filepath = os.path.join(self.journal_dir, f"{transaction_id}.journal")
            
            with open(filepath, 'w') as f:
                json.dump(journal_entry, f, indent=2)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to persist journal entry: {str(e)}")
            return False
    
    def update_transaction_status(self, 
                               transaction_id: str, 
                               status: str,
                               result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a transaction in the journal.
        
        Args:
            transaction_id: ID of the transaction
            status: New status ('completed', 'failed', 'rolled_back')
            result: Optional result data
            
        Returns:
            Success status
        """
        try:
            # Update in-memory journal with thread safety
            with self.journal_lock:
                for entry in self.transaction_journal:
                    if entry['transaction_id'] == transaction_id:
                        entry['status'] = status
                        entry['updated_at'] = datetime.now().isoformat()
                        if result:
                            entry['result'] = result
                        break
            
            # Update on disk
            filepath = os.path.join(self.journal_dir, f"{transaction_id}.journal")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    journal_entry = json.load(f)
                
                journal_entry['status'] = status
                journal_entry['updated_at'] = datetime.now().isoformat()
                if result:
                    journal_entry['result'] = result
                
                with open(filepath, 'w') as f:
                    json.dump(journal_entry, f, indent=2)
            
            logger.debug(f"Updated transaction {transaction_id} status to {status}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update transaction status: {str(e)}")
            return False
    
    def get_pending_transactions(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all pending transactions, optionally filtered by agent.
        
        Args:
            agent_id: Optional agent ID to filter by
            
        Returns:
            List of pending transactions
        """
        with self.journal_lock:
            if agent_id:
                return [entry for entry in self.transaction_journal 
                        if entry['status'] == 'pending' and entry['agent_id'] == agent_id]
            else:
                return [entry for entry in self.transaction_journal 
                        if entry['status'] == 'pending']
    
    def detect_orphaned_transactions(self, 
                                  max_age_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Detect transactions that have been pending for too long.
        
        Args:
            max_age_seconds: Maximum age for a transaction to be considered orphaned
            
        Returns:
            List of orphaned transactions
        """
        orphaned = []
        current_time = datetime.now()
        
        with self.journal_lock:
            for entry in self.transaction_journal:
                if entry['status'] == 'pending':
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    age_seconds = (current_time - timestamp).total_seconds()
                    
                    if age_seconds > max_age_seconds:
                        orphaned.append(entry)
        
        if orphaned:
            logger.warning(f"Detected {len(orphaned)} orphaned transactions")
        
        return orphaned
    
    def reconcile_orphaned_transactions(self) -> Tuple[int, int]:
        """
        Attempt to reconcile orphaned transactions.
        
        Returns:
            Tuple of (reconciled count, failed count)
        """
        orphaned = self.detect_orphaned_transactions()
        reconciled = 0
        failed = 0
        
        for entry in orphaned:
            # Strategy depends on transaction type
            transaction_type = entry['transaction_type']
            transaction_id = entry['transaction_id']
            
            if transaction_type == 'order':
                # For orders, check with the execution system
                # This is a placeholder - actual implementation would query the trading system
                success = self._reconcile_order_transaction(entry)
            elif transaction_type == 'position_update':
                # For position updates, verify the current position
                success = self._reconcile_position_transaction(entry)
            else:
                # Unknown transaction type
                logger.warning(f"Unknown transaction type {transaction_type} for reconciliation")
                success = False
            
            if success:
                reconciled += 1
            else:
                failed += 1
        
        logger.info(f"Reconciled {reconciled} transactions, {failed} failed")
        return reconciled, failed
    
    def _reconcile_order_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Reconcile an orphaned order transaction.
        
        Args:
            transaction: Transaction entry
            
        Returns:
            Success status
        """
        # Placeholder - actual implementation would:
        # 1. Query the broker/exchange for order status
        # 2. Update local state based on actual order status
        # 3. Mark transaction as reconciled
        
        # For now, just mark as failed
        self.update_transaction_status(
            transaction['transaction_id'], 
            'failed',
            {'reason': 'Orphaned and could not verify'}
        )
        
        return True
    
    def _reconcile_position_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Reconcile an orphaned position transaction.
        
        Args:
            transaction: Transaction entry
            
        Returns:
            Success status
        """
        # Placeholder - actual implementation would:
        # 1. Query the current position state
        # 2. Compare with intended update
        # 3. Apply or reject the update
        
        # For now, just mark as failed
        self.update_transaction_status(
            transaction['transaction_id'], 
            'failed',
            {'reason': 'Orphaned and could not verify'}
        )
        
        return True
    
    def handle_agent_failure(self, 
                          agent_id: str, 
                          failure_info: Dict[str, Any]) -> Tuple[bool, RecoveryStrategy]:
        """
        Handle an agent failure by determining and applying the appropriate recovery strategy.
        
        Args:
            agent_id: ID of the failed agent
            failure_info: Information about the failure
            
        Returns:
            Tuple of (success status, strategy used)
        """
        logger.info(f"Handling failure of agent {agent_id}: {failure_info.get('reason', 'Unknown')}")
        
        # Mark as failed and update recovery state
        self.failed_components.add(agent_id)
        if self.recovery_state == RecoveryState.NORMAL:
            self.recovery_state = RecoveryState.RECOVERY_NEEDED
        
        # Track recovery attempts
        self.recovery_attempts[agent_id] = self.recovery_attempts.get(agent_id, 0) + 1
        
        # Check if we've exceeded max attempts
        if self.recovery_attempts[agent_id] > self.max_recovery_attempts:
            logger.warning(f"Agent {agent_id} has exceeded maximum recovery attempts")
            return False, RecoveryStrategy.GRACEFUL_DEGRADATION
        
        # Determine recovery strategy based on failure type
        failure_type = failure_info.get('type', 'unknown')
        severity = failure_info.get('severity', 'medium')
        
        # Choose strategy based on failure characteristics
        if failure_type == 'crash':
            strategy = RecoveryStrategy.RESTART
        elif failure_type == 'state_corruption':
            strategy = RecoveryStrategy.STATE_RESTORE
        elif failure_type == 'deadlock':
            strategy = RecoveryStrategy.RESTART
        elif failure_type == 'performance_degradation':
            strategy = RecoveryStrategy.RESTART if severity == 'high' else RecoveryStrategy.GRACEFUL_DEGRADATION
        else:
            # Default strategy
            strategy = RecoveryStrategy.STATE_RESTORE
        
        logger.info(f"Selected recovery strategy for {agent_id}: {strategy.value}")
        return True, strategy
    
    def apply_recovery_strategy(self, 
                             agent_id: str, 
                             strategy: RecoveryStrategy,
                             agent_instance: Any) -> bool:
        """
        Apply the selected recovery strategy to an agent.
        
        Args:
            agent_id: ID of the agent
            strategy: Recovery strategy to apply
            agent_instance: Agent instance to recover
            
        Returns:
            Success status
        """
        logger.info(f"Applying recovery strategy {strategy.value} to agent {agent_id}")
        
        # Update recovery state
        self.recovery_state = RecoveryState.RECOVERING
        
        # Track recovery metrics
        start_time = time.time()
        self.recovery_metrics["recoveries_attempted"] += 1
        
        success = False
        
        try:
            if strategy == RecoveryStrategy.RESTART:
                success = self._apply_restart_strategy(agent_id, agent_instance)
            elif strategy == RecoveryStrategy.STATE_RESTORE:
                success = self._apply_state_restore_strategy(agent_id, agent_instance)
            elif strategy == RecoveryStrategy.ROLLBACK:
                success = self._apply_rollback_strategy(agent_id, agent_instance)
            elif strategy == RecoveryStrategy.CHECKPOINT:
                success = self._apply_checkpoint_strategy(agent_id, agent_instance)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success = self._apply_degradation_strategy(agent_id, agent_instance)
        
        except Exception as e:
            logger.error(f"Error applying recovery strategy {strategy.value} to {agent_id}: {str(e)}")
            success = False
        
        # Calculate recovery time
        end_time = time.time()
        recovery_time_ms = (end_time - start_time) * 1000
        
        # Update recovery metrics
        if success:
            self.recovery_metrics["recoveries_successful"] += 1
            self.failed_components.remove(agent_id)
            
            # Update average recovery time
            total_recoveries = self.recovery_metrics["recoveries_successful"]
            current_avg = self.recovery_metrics["average_recovery_time_ms"]
            
            self.recovery_metrics["total_recovery_time_ms"] += recovery_time_ms
            self.recovery_metrics["average_recovery_time_ms"] = self.recovery_metrics["total_recovery_time_ms"] / total_recoveries
        else:
            self.recovery_metrics["recoveries_failed"] += 1
        
        self.recovery_metrics["last_recovery_time"] = datetime.now().isoformat()
        
        # Update system state
        if not self.failed_components:
            self.recovery_state = RecoveryState.NORMAL
        else:
            self.recovery_state = RecoveryState.DEGRADED
        
        return success
    
    def _apply_restart_strategy(self, agent_id: str, agent_instance: Any) -> bool:
        """
        Apply the restart recovery strategy.
        
        Args:
            agent_id: ID of the agent
            agent_instance: Agent instance to recover
            
        Returns:
            Success status
        """
        try:
            # Stop the agent if running
            if hasattr(agent_instance, 'stop'):
                agent_instance.stop()
            
            # Give it time to cleanup
            time.sleep(1)
            
            # Restart the agent
            if hasattr(agent_instance, 'start'):
                agent_instance.start()
                logger.info(f"Successfully restarted agent {agent_id}")
                return True
            else:
                logger.error(f"Agent {agent_id} doesn't have a start method")
                return False
        
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_id}: {str(e)}")
            return False
    
    def _apply_state_restore_strategy(self, agent_id: str, agent_instance: Any) -> bool:
        """
        Apply the state restore recovery strategy.
        
        Args:
            agent_id: ID of the agent
            agent_instance: Agent instance to recover
            
        Returns:
            Success status
        """
        try:
            # Stop the agent if running
            if hasattr(agent_instance, 'stop'):
                agent_instance.stop()
            
            # Restore state
            success, state = self.restore_agent_state(agent_id)
            
            if not success or not state:
                logger.warning(f"Could not restore state for agent {agent_id}")
                # Fallback to restart
                return self._apply_restart_strategy(agent_id, agent_instance)
            
            # Apply restored state
            if hasattr(agent_instance, 'restore_state'):
                agent_instance.restore_state(state)
            else:
                # Manual state restoration
                for key, value in state.items():
                    if hasattr(agent_instance, key):
                        setattr(agent_instance, key, value)
            
            # Restart the agent
            if hasattr(agent_instance, 'start'):
                agent_instance.start()
                logger.info(f"Successfully restored state for agent {agent_id}")
                return True
            else:
                logger.error(f"Agent {agent_id} doesn't have a start method")
                return False
        
        except Exception as e:
            logger.error(f"Failed to restore state for agent {agent_id}: {str(e)}")
            return False
    
    def _apply_rollback_strategy(self, agent_id: str, agent_instance: Any) -> bool:
        """
        Apply the transaction rollback recovery strategy.
        
        Args:
            agent_id: ID of the agent
            agent_instance: Agent instance to recover
            
        Returns:
            Success status
        """
        try:
            # Get pending transactions for this agent
            pending_transactions = self.get_pending_transactions(agent_id)
            
            # Roll back each transaction
            for transaction in pending_transactions:
                transaction_id = transaction['transaction_id']
                logger.info(f"Rolling back transaction {transaction_id}")
                
                # Update transaction status
                self.update_transaction_status(
                    transaction_id,
                    'rolled_back',
                    {'reason': 'Agent failure recovery'}
                )
            
            # Restart the agent with clean state
            return self._apply_restart_strategy(agent_id, agent_instance)
        
        except Exception as e:
            logger.error(f"Failed to rollback transactions for agent {agent_id}: {str(e)}")
            return False
    
    def _apply_checkpoint_strategy(self, agent_id: str, agent_instance: Any) -> bool:
        """
        Apply the checkpoint restore recovery strategy.
        
        Args:
            agent_id: ID of the agent
            agent_instance: Agent instance to recover
            
        Returns:
            Success status
        """
        # For now, this is identical to state restore
        return self._apply_state_restore_strategy(agent_id, agent_instance)
    
    def _apply_degradation_strategy(self, agent_id: str, agent_instance: Any) -> bool:
        """
        Apply the graceful degradation recovery strategy.
        
        Args:
            agent_id: ID of the agent
            agent_instance: Agent instance to recover
            
        Returns:
            Success status
        """
        try:
            # Stop the agent if running
            if hasattr(agent_instance, 'stop'):
                agent_instance.stop()
            
            # Put agent in degraded mode if supported
            if hasattr(agent_instance, 'set_degraded_mode'):
                agent_instance.set_degraded_mode(True)
            
            # Restart in degraded mode
            if hasattr(agent_instance, 'start'):
                agent_instance.start()
                logger.info(f"Agent {agent_id} set to degraded mode")
                return True
            else:
                logger.error(f"Agent {agent_id} doesn't have a start method")
                return False
        
        except Exception as e:
            logger.error(f"Failed to set degraded mode for agent {agent_id}: {str(e)}")
            return False
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """
        Get the current recovery status.
        
        Returns:
            Dictionary with recovery status information
        """
        return {
            "state": self.recovery_state.value,
            "failed_components": list(self.failed_components),
            "recovery_attempts": self.recovery_attempts,
            "metrics": self.recovery_metrics
        }
