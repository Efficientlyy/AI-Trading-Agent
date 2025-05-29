"""
Oversight actions and result definitions for the LLM Oversight system.

This module defines the possible actions that the LLM Oversight system can take
when evaluating prompts and responses, as well as the structure for oversight results.
"""

import enum
from typing import Dict, Any, Optional
from datetime import datetime


class OversightAction(enum.Enum):
    """
    Actions that can be taken by the LLM Oversight system.
    
    These actions determine how the system should respond to
    prompts and model outputs.
    """
    
    ALLOW = "allow"              # Allow without modification
    MODIFY = "modify"            # Allow with modifications
    FLAG = "flag"                # Flag for human review but allow
    REJECT = "reject"            # Reject, do not proceed
    LOG_ONLY = "log_only"        # Just log, no action needed


class OversightResult:
    """
    Result of an oversight evaluation.
    
    This class encapsulates the outcome of evaluating a prompt or
    response, including any actions taken and metadata.
    """
    
    def __init__(
        self,
        action: OversightAction = OversightAction.ALLOW,
        reason: Optional[str] = None,
        message: Optional[str] = None,
        modified_prompt: Optional[str] = None,
        modified_system_prompt: Optional[str] = None,
        modified_response: Optional[str] = None,
        risk_score: float = 0.0,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an oversight result.
        
        Args:
            action: Action determined by oversight
            reason: Reason for the action
            message: Message to display to user or system
            modified_prompt: Modified version of the prompt if action is MODIFY
            modified_system_prompt: Modified system prompt if applicable
            modified_response: Modified response if action is MODIFY for response
            risk_score: Risk assessment score (0.0-1.0)
            confidence: Confidence in the assessment (0.0-1.0)
            metadata: Additional metadata about the oversight check
        """
        self.action = action
        self.reason = reason
        self.message = message
        self.modified_prompt = modified_prompt
        self.modified_system_prompt = modified_system_prompt
        self.modified_response = modified_response
        self.risk_score = risk_score
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary representation.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "action": self.action.value,
            "reason": self.reason,
            "message": self.message,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OversightResult':
        """
        Create an OversightResult from a dictionary.
        
        Args:
            data: Dictionary representation of an oversight result
            
        Returns:
            OversightResult instance
        """
        return cls(
            action=OversightAction(data.get("action", OversightAction.ALLOW.value)),
            reason=data.get("reason"),
            message=data.get("message"),
            risk_score=data.get("risk_score", 0.0),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )
