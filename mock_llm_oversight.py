"""
Mock LLM Oversight System for the Autonomous Trading Agent.
"""

import logging
import random
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Use local enum definitions if imported ones are not available
class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    CRITICAL = "CRITICAL"

class MockLLMOversight:
    """Mock LLM-based oversight system for trading decisions."""
    
    def __init__(self):
        self.decisions_reviewed = 0
        self.decisions_approved = 0
        self.decisions_rejected = 0
        self.decisions_modified = 0
        self.approval_rate = 0.9  # 90% approval rate by default
        self.health_status = HealthStatus.HEALTHY
        self.failure_mode = False
        self.running = False
        self.review_thread = None
        self.reasoning_templates = [
            "Decision aligns with current market regime and sentiment signals.",
            "Risk parameters are within acceptable bounds for portfolio allocation.",
            "Technical indicators confirm the trading signal strength.",
            "Position sizing is appropriate for current market volatility.",
            "Trade timing is optimal based on market conditions.",
            "Decision properly accounts for recent market events.",
            "Sentiment analysis supports the trading direction.",
            "Strategy adaptation correctly responds to changing market conditions.",
            "Diversification benefits maintained with this allocation.",
            "Stop-loss and take-profit levels appropriately balance risk/reward ratio."
        ]
        self.rejection_templates = [
            "Risk parameters exceed acceptable bounds for current market conditions.",
            "Technical indicators contradict the proposed trading direction.",
            "Sentiment analysis does not support the confidence level of this trade.",
            "Position sizing is too aggressive given current volatility.",
            "Market regime classification suggests higher uncertainty than accounted for.",
            "Insufficient data to validate the trading signal.",
            "Correlation with existing positions creates excessive portfolio risk.",
            "Decision fails to account for recent significant market events.",
            "Stop-loss placement inadequate for protecting portfolio value.",
            "Timing suboptimal based on intraday volatility patterns."
        ]
        logger.info("Initialized Mock LLM Oversight System")
    
    def review_decision(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review a trading decision and provide feedback."""
        if self.failure_mode:
            if random.random() < 0.7:
                # Simulate timeout or failure
                logger.warning("LLM oversight system timed out or failed to process request")
                return {
                    'status': 'error',
                    'message': 'Oversight system unavailable',
                    'approved': False,
                    'reasoning': 'System error: request timed out'
                }
        
        self.decisions_reviewed += 1
        
        # Determine if decision should be approved
        # Use a combination of random chance and decision data
        confidence = decision_data.get('confidence', 0.5)
        regime = decision_data.get('market_regime', 'unknown')
        sentiment = decision_data.get('sentiment_score', 0.0)
        risk_level = decision_data.get('risk_level', 0.5)
        
        # Calculate approval probability adjusted by decision data
        base_approval_rate = self.approval_rate
        
        # Adjust for confidence
        approval_prob = base_approval_rate * (0.5 + confidence / 2)
        
        # Adjust for alignment between action, regime and sentiment
        action = decision_data.get('action', 'hold').lower()
        regime_factor = 1.0
        sentiment_factor = 1.0
        
        if action == 'buy' and regime in ['bear', 'volatile_bear', 'breakdown']:
            regime_factor = 0.7  # Less likely to approve buying in bear markets
        elif action == 'sell' and regime in ['bull', 'volatile_bull', 'recovery']:
            regime_factor = 0.7  # Less likely to approve selling in bull markets
            
        if action == 'buy' and sentiment < -0.2:
            sentiment_factor = 0.8  # Less likely to approve buying with negative sentiment
        elif action == 'sell' and sentiment > 0.2:
            sentiment_factor = 0.8  # Less likely to approve selling with positive sentiment
            
        # Apply factors
        approval_prob *= regime_factor * sentiment_factor
        
        # Final clamp
        approval_prob = max(0.1, min(0.95, approval_prob))
        
        # Make decision
        approved = random.random() < approval_prob
        
        if approved:
            self.decisions_approved += 1
            reasoning = random.choice(self.reasoning_templates)
        else:
            self.decisions_rejected += 1
            reasoning = random.choice(self.rejection_templates)
            
        # Create detailed response
        response = {
            'status': 'success',
            'decision_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'approved': approved,
            'confidence': random.uniform(0.7, 0.95) if approved else random.uniform(0.4, 0.7),
            'reasoning': reasoning,
            'additional_feedback': self._generate_additional_feedback(decision_data),
            'risk_assessment': {
                'portfolio_impact': random.uniform(0.01, 0.2),
                'market_alignment': random.uniform(0.3, 0.9),
                'recommendation': 'proceed' if approved else 'reconsider'
            }
        }
        
        # Update approval rate after each decision
        if self.decisions_reviewed > 0:
            self.approval_rate = self.decisions_approved / self.decisions_reviewed
            
        logger.info(f"LLM oversight reviewed decision: {approved=}, action={action}, asset={decision_data.get('symbol')}")
        return response
    
    def start(self):
        """Start the LLM oversight system."""
        if self.running:
            logger.warning("LLM oversight system already running")
            return False
        
        self.running = True
        self.review_thread = threading.Thread(target=self._review_loop)
        self.review_thread.daemon = True
        self.review_thread.start()
        
        logger.info("Started LLM oversight system")
        return True
    
    def stop(self):
        """Stop the LLM oversight system."""
        if not self.running:
            logger.warning("LLM oversight system not running")
            return False
        
        self.running = False
        if self.review_thread and self.review_thread.is_alive():
            self.review_thread.join(timeout=3.0)
        
        logger.info("Stopped LLM oversight system")
        return True
    
    def _review_loop(self):
        """Background loop for reviewing any queued decisions."""
        while self.running:
            try:
                # This is a mock loop to simulate background processing
                # In a real system, this would process a queue of decisions to review
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in LLM review loop: {e}")
                time.sleep(1.0)
    
    def _generate_additional_feedback(self, decision_data: Dict[str, Any]) -> str:
        """Generate additional detailed feedback based on decision data."""
        symbol = decision_data.get('symbol', 'unknown')
        action = decision_data.get('action', 'unknown')
        regime = decision_data.get('market_regime', 'unknown')
        
        templates = [
            f"Consider adjusting position size for {symbol} based on {regime} regime volatility.",
            f"Monitor {symbol} closely due to increased correlation with other portfolio assets.",
            f"Recent sentiment shifts for {symbol} may require more frequent re-evaluation of this position.",
            f"The {action} signal for {symbol} is technically valid but timing could be optimized further.",
            f"Consider implementing a trailing stop for this {action} action on {symbol} to manage downside risk."
        ]
        
        return random.choice(templates)
    
    def get_stats(self):
        """Get statistics about the LLM oversight system."""
        # Update approval rate
        if self.decisions_reviewed > 0:
            self.approval_rate = self.decisions_approved / self.decisions_reviewed
        
        return {
            'decisions_reviewed': self.decisions_reviewed,
            'decisions_approved': self.decisions_approved,
            'decisions_rejected': self.decisions_rejected,
            'decisions_modified': self.decisions_modified,
            'approval_rate': self.approval_rate,
            'rejection_rate': self.decisions_rejected / self.decisions_reviewed if self.decisions_reviewed > 0 else 0,
            'modification_rate': self.decisions_modified / self.decisions_reviewed if self.decisions_reviewed > 0 else 0
        }
    
    def check_health(self) -> HealthStatus:
        """Check the health of the LLM oversight system."""
        return self.health_status
    
    def inject_failure(self):
        """Inject a failure into the oversight system for testing resilience."""
        self.failure_mode = True
        self.health_status = HealthStatus.DEGRADED
        logger.warning("Failure injected into LLM oversight system")
    
    def recover(self):
        """Recover from a failure."""
        self.failure_mode = False
        self.health_status = HealthStatus.HEALTHY
        logger.info("LLM oversight system recovered from failure")
