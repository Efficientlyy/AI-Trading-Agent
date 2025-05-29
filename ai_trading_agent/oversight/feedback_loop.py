"""
Feedback loop for LLM oversight decisions.

This module implements a feedback collection and analysis system to track
the outcomes of LLM oversight decisions and improve future performance.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum

from ai_trading_agent.oversight.evaluation import (
    OversightEvaluator, DecisionOutcome, MetricType
)

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback for oversight decisions."""
    TRADE_OUTCOME = "trade_outcome"
    RISK_IMPACT = "risk_impact"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_FEEDBACK = "system_feedback"
    PERFORMANCE_METRICS = "performance_metrics"


class OversightFeedbackLoop:
    """
    Feedback loop for tracking and learning from oversight decision outcomes.
    
    This class collects feedback on oversight decisions, analyzes patterns,
    and provides insights to improve future oversight performance.
    """
    
    def __init__(
        self, 
        evaluator: OversightEvaluator,
        feedback_db_path: str,
        auto_save: bool = True
    ):
        """
        Initialize the oversight feedback loop.
        
        Args:
            evaluator: OversightEvaluator instance to track metrics
            feedback_db_path: Path to store feedback data
            auto_save: Whether to automatically save feedback after recording
        """
        self.evaluator = evaluator
        self.feedback_db_path = feedback_db_path
        self.auto_save = auto_save
        
        # Feedback database
        self.feedback_records: List[Dict[str, Any]] = []
        
        # Decision-feedback mapping
        self.decision_feedback: Dict[str, List[Dict[str, Any]]] = {}
        
        # Feedback statistics
        self.feedback_stats: Dict[str, Any] = {
            "correct_decisions": 0,
            "incorrect_decisions": 0,
            "total_feedback": 0,
            "by_decision_type": {
                "approve": {"correct": 0, "incorrect": 0},
                "reject": {"correct": 0, "incorrect": 0},
                "modify": {"correct": 0, "incorrect": 0}
            },
            "by_symbol": {},
            "by_strategy": {}
        }
        
        # Patterns and insights
        self.insights: List[Dict[str, Any]] = []
        
        # Load existing feedback if available
        self._load_feedback()
    
    def _load_feedback(self) -> None:
        """Load feedback data from disk if available."""
        if os.path.exists(self.feedback_db_path):
            try:
                with open(self.feedback_db_path, 'r') as f:
                    data = json.load(f)
                
                self.feedback_records = data.get("feedback_records", [])
                self.decision_feedback = data.get("decision_feedback", {})
                self.feedback_stats = data.get("feedback_stats", self.feedback_stats)
                self.insights = data.get("insights", [])
                
                logger.info(f"Loaded {len(self.feedback_records)} feedback records from {self.feedback_db_path}")
            except Exception as e:
                logger.error(f"Error loading feedback data: {e}")
    
    def save_feedback(self) -> None:
        """Save feedback data to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.feedback_db_path), exist_ok=True)
            
            # Prepare data for saving
            data = {
                "feedback_records": self.feedback_records,
                "decision_feedback": self.decision_feedback,
                "feedback_stats": self.feedback_stats,
                "insights": self.insights,
                "last_updated": datetime.now().isoformat()
            }
            
            # Save to file
            with open(self.feedback_db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved feedback data to {self.feedback_db_path}")
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
    
    def record_trade_outcome(
        self,
        decision_id: str,
        original_decision: Dict[str, Any],
        oversight_result: Dict[str, Any],
        outcome: DecisionOutcome,
        pnl: float,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        holding_period: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record the outcome of a trade that was subject to oversight.
        
        Args:
            decision_id: Unique identifier for the decision
            original_decision: The original trading decision
            oversight_result: The result from the LLM oversight system
            outcome: Actual outcome of the trade
            pnl: Profit/loss from the trade
            exit_price: Price at which the position was closed
            exit_time: Time when the position was closed
            holding_period: How long the position was held (in minutes)
            metadata: Additional data about the trade
            
        Returns:
            Feedback ID
        """
        # Create feedback record
        timestamp = datetime.now()
        feedback_id = f"feedback_{timestamp.strftime('%Y%m%d%H%M%S')}_{decision_id}"
        
        feedback = {
            "feedback_id": feedback_id,
            "decision_id": decision_id,
            "timestamp": timestamp.isoformat(),
            "feedback_type": FeedbackType.TRADE_OUTCOME.value,
            "original_decision": original_decision,
            "oversight_result": oversight_result,
            "outcome": outcome.value,
            "pnl": pnl,
            "exit_price": exit_price,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "holding_period": holding_period,
            "metadata": metadata or {}
        }
        
        # Add to records
        self.feedback_records.append(feedback)
        
        # Update decision-feedback mapping
        if decision_id not in self.decision_feedback:
            self.decision_feedback[decision_id] = []
        self.decision_feedback[decision_id].append(feedback)
        
        # Update evaluator
        risk_impact = metadata.get("risk_impact", 0.0) if metadata else 0.0
        self.evaluator.record_decision_outcome(
            decision_id=decision_id,
            outcome=outcome,
            pnl_impact=pnl,
            risk_impact=risk_impact,
            metadata={
                "exit_price": exit_price,
                "exit_time": exit_time.isoformat() if exit_time else None,
                "holding_period": holding_period
            }
        )
        
        # Update feedback statistics
        self._update_feedback_stats(feedback)
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_feedback()
        
        return feedback_id
    
    def record_user_feedback(
        self,
        decision_id: str,
        user_rating: int,
        comments: Optional[str] = None,
        is_correct: Optional[bool] = None
    ) -> str:
        """
        Record user feedback on an oversight decision.
        
        Args:
            decision_id: Unique identifier for the decision
            user_rating: Rating from 1-5 (1=poor, 5=excellent)
            comments: User comments about the decision
            is_correct: Whether the user considers the decision correct
            
        Returns:
            Feedback ID
        """
        # Create feedback record
        timestamp = datetime.now()
        feedback_id = f"user_feedback_{timestamp.strftime('%Y%m%d%H%M%S')}_{decision_id}"
        
        feedback = {
            "feedback_id": feedback_id,
            "decision_id": decision_id,
            "timestamp": timestamp.isoformat(),
            "feedback_type": FeedbackType.USER_FEEDBACK.value,
            "user_rating": user_rating,
            "comments": comments,
            "is_correct": is_correct
        }
        
        # Add to records
        self.feedback_records.append(feedback)
        
        # Update decision-feedback mapping
        if decision_id not in self.decision_feedback:
            self.decision_feedback[decision_id] = []
        self.decision_feedback[decision_id].append(feedback)
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_feedback()
        
        return feedback_id
    
    def record_system_feedback(
        self,
        decision_id: str,
        assessment: str,
        improvement_suggestions: List[str],
        confidence: float,
        is_correct: bool
    ) -> str:
        """
        Record system-generated feedback on an oversight decision.
        
        Args:
            decision_id: Unique identifier for the decision
            assessment: System's assessment of the decision
            improvement_suggestions: Suggestions for improvement
            confidence: Confidence in the assessment (0-1)
            is_correct: Whether the system considers the decision correct
            
        Returns:
            Feedback ID
        """
        # Create feedback record
        timestamp = datetime.now()
        feedback_id = f"system_feedback_{timestamp.strftime('%Y%m%d%H%M%S')}_{decision_id}"
        
        feedback = {
            "feedback_id": feedback_id,
            "decision_id": decision_id,
            "timestamp": timestamp.isoformat(),
            "feedback_type": FeedbackType.SYSTEM_FEEDBACK.value,
            "assessment": assessment,
            "improvement_suggestions": improvement_suggestions,
            "confidence": confidence,
            "is_correct": is_correct
        }
        
        # Add to records
        self.feedback_records.append(feedback)
        
        # Update decision-feedback mapping
        if decision_id not in self.decision_feedback:
            self.decision_feedback[decision_id] = []
        self.decision_feedback[decision_id].append(feedback)
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_feedback()
        
        return feedback_id
    
    def _update_feedback_stats(self, feedback: Dict[str, Any]) -> None:
        """
        Update feedback statistics based on new feedback.
        
        Args:
            feedback: The feedback record to process
        """
        self.feedback_stats["total_feedback"] += 1
        
        # Only process trade outcomes for correctness stats
        if feedback["feedback_type"] != FeedbackType.TRADE_OUTCOME.value:
            return
        
        # Get decision and outcome
        oversight_result = feedback["oversight_result"]
        decision = feedback["original_decision"]
        action = oversight_result.get("action", "unknown")
        outcome = feedback["outcome"]
        
        # Determine if the oversight decision was correct
        was_correct = (
            (action == "approve" and outcome == DecisionOutcome.PROFITABLE.value) or
            (action == "reject" and outcome == DecisionOutcome.LOSS.value) or
            (action == "modify" and outcome != DecisionOutcome.LOSS.value)
        )
        
        # Update global stats
        if was_correct:
            self.feedback_stats["correct_decisions"] += 1
        else:
            self.feedback_stats["incorrect_decisions"] += 1
        
        # Update by decision type
        if action in self.feedback_stats["by_decision_type"]:
            if was_correct:
                self.feedback_stats["by_decision_type"][action]["correct"] += 1
            else:
                self.feedback_stats["by_decision_type"][action]["incorrect"] += 1
        
        # Update by symbol
        symbol = decision.get("symbol", "unknown")
        if symbol not in self.feedback_stats["by_symbol"]:
            self.feedback_stats["by_symbol"][symbol] = {
                "correct": 0, "incorrect": 0, "total_pnl": 0.0
            }
        
        if was_correct:
            self.feedback_stats["by_symbol"][symbol]["correct"] += 1
        else:
            self.feedback_stats["by_symbol"][symbol]["incorrect"] += 1
        
        self.feedback_stats["by_symbol"][symbol]["total_pnl"] += feedback["pnl"]
        
        # Update by strategy
        strategy = decision.get("strategy", "unknown")
        if strategy not in self.feedback_stats["by_strategy"]:
            self.feedback_stats["by_strategy"][strategy] = {
                "correct": 0, "incorrect": 0, "total_pnl": 0.0
            }
        
        if was_correct:
            self.feedback_stats["by_strategy"][strategy]["correct"] += 1
        else:
            self.feedback_stats["by_strategy"][strategy]["incorrect"] += 1
            
        self.feedback_stats["by_strategy"][strategy]["total_pnl"] += feedback["pnl"]
    
    def analyze_feedback_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze feedback data to identify patterns and insights.
        
        Returns:
            List of insights derived from feedback
        """
        # Clear previous insights
        self.insights = []
        
        # Need at least 10 feedback records for meaningful analysis
        if len(self.feedback_records) < 10:
            return []
        
        # Convert to DataFrame for analysis
        feedback_df = pd.DataFrame(self.feedback_records)
        
        # Filter to trade outcomes
        trade_outcomes = feedback_df[
            feedback_df["feedback_type"] == FeedbackType.TRADE_OUTCOME.value
        ].copy()
        
        if len(trade_outcomes) < 10:
            return []
        
        # Extract key fields
        trade_outcomes["timestamp"] = pd.to_datetime(trade_outcomes["timestamp"])
        trade_outcomes["action"] = trade_outcomes["oversight_result"].apply(
            lambda x: x.get("action", "unknown") if isinstance(x, dict) else "unknown"
        )
        trade_outcomes["symbol"] = trade_outcomes["original_decision"].apply(
            lambda x: x.get("symbol", "unknown") if isinstance(x, dict) else "unknown"
        )
        trade_outcomes["strategy"] = trade_outcomes["original_decision"].apply(
            lambda x: x.get("strategy", "unknown") if isinstance(x, dict) else "unknown"
        )
        
        # Analyze overall effectiveness
        overall_accuracy = self.feedback_stats["correct_decisions"] / max(
            self.feedback_stats["correct_decisions"] + self.feedback_stats["incorrect_decisions"], 1
        )
        
        # Find patterns by symbol
        symbol_insights = []
        for symbol, stats in self.feedback_stats["by_symbol"].items():
            total = stats["correct"] + stats["incorrect"]
            if total >= 5:  # Need at least 5 trades for meaningful stats
                accuracy = stats["correct"] / total
                # If accuracy is significantly different from overall
                if abs(accuracy - overall_accuracy) > 0.15:
                    symbol_insights.append({
                        "type": "symbol_pattern",
                        "symbol": symbol,
                        "accuracy": accuracy,
                        "sample_size": total,
                        "description": (
                            f"Oversight is {'more' if accuracy > overall_accuracy else 'less'} "
                            f"effective for {symbol} ({accuracy:.1%} accuracy vs {overall_accuracy:.1%} overall)"
                        ),
                        "pnl_impact": stats["total_pnl"]
                    })
        
        # Find patterns by strategy
        strategy_insights = []
        for strategy, stats in self.feedback_stats["by_strategy"].items():
            total = stats["correct"] + stats["incorrect"]
            if total >= 5:  # Need at least 5 trades for meaningful stats
                accuracy = stats["correct"] / total
                # If accuracy is significantly different from overall
                if abs(accuracy - overall_accuracy) > 0.15:
                    strategy_insights.append({
                        "type": "strategy_pattern",
                        "strategy": strategy,
                        "accuracy": accuracy,
                        "sample_size": total,
                        "description": (
                            f"Oversight is {'more' if accuracy > overall_accuracy else 'less'} "
                            f"effective for {strategy} strategy ({accuracy:.1%} accuracy vs {overall_accuracy:.1%} overall)"
                        ),
                        "pnl_impact": stats["total_pnl"]
                    })
        
        # Find time-based patterns
        if len(trade_outcomes) >= 20:
            # Group by week
            trade_outcomes["week"] = trade_outcomes["timestamp"].dt.isocalendar().week
            weekly_stats = trade_outcomes.groupby("week").apply(
                lambda x: {
                    "correct": sum((
                        (x["action"] == "approve") & (x["outcome"] == DecisionOutcome.PROFITABLE.value) |
                        (x["action"] == "reject") & (x["outcome"] == DecisionOutcome.LOSS.value) |
                        (x["action"] == "modify") & (x["outcome"] != DecisionOutcome.LOSS.value)
                    )),
                    "total": len(x),
                    "pnl": x["pnl"].sum()
                }
            ).tolist()
            
            # Check for trend in accuracy
            if len(weekly_stats) >= 3:
                accuracies = [stats["correct"] / max(stats["total"], 1) for stats in weekly_stats]
                trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
                
                if abs(trend) > 0.05:  # Significant trend
                    self.insights.append({
                        "type": "time_trend",
                        "trend": "improving" if trend > 0 else "deteriorating",
                        "magnitude": abs(trend),
                        "description": (
                            f"Oversight effectiveness is {
                                'improving' if trend > 0 else 'deteriorating'
                            } over time ({
                                '+' if trend > 0 else ''
                            }{trend:.1%} accuracy change per week)"
                        )
                    })
        
        # Add insights from symbol and strategy patterns
        self.insights.extend(symbol_insights)
        self.insights.extend(strategy_insights)
        
        # Generate aggregate insights
        aggregated_insights = []
        
        # Problematic action types
        for action, stats in self.feedback_stats["by_decision_type"].items():
            total = stats["correct"] + stats["incorrect"]
            if total >= 10:  # Need sufficient sample
                accuracy = stats["correct"] / total
                if accuracy < 0.6:  # Problematic accuracy
                    aggregated_insights.append({
                        "type": "action_issue",
                        "action": action,
                        "accuracy": accuracy,
                        "sample_size": total,
                        "description": (
                            f"Low accuracy ({accuracy:.1%}) for '{action}' decisions, "
                            f"consider reviewing decision criteria"
                        )
                    })
        
        # Add aggregated insights
        self.insights.extend(aggregated_insights)
        
        # Sort insights by importance (sample size)
        self.insights.sort(key=lambda x: x.get("sample_size", 0) 
                              if "sample_size" in x else 0, 
                              reverse=True)
        
        return self.insights
    
    def generate_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations for improving oversight based on feedback.
        
        Returns:
            List of improvement recommendations
        """
        # Analyze patterns first
        if not self.insights:
            self.analyze_feedback_patterns()
        
        # Convert feedback insights to recommendations
        recommendations = []
        
        for insight in self.insights:
            insight_type = insight["type"]
            
            if insight_type == "symbol_pattern":
                if insight.get("accuracy", 0) < 0.6:
                    recommendations.append({
                        "category": "symbol_specific",
                        "target": insight["symbol"],
                        "recommendation": (
                            f"Develop specialized validation prompts for {insight['symbol']} "
                            f"to address low accuracy ({insight['accuracy']:.1%})"
                        ),
                        "priority": "high" if insight.get("sample_size", 0) > 20 else "medium",
                        "expected_impact": "Improved decision validation for frequently traded asset"
                    })
            
            elif insight_type == "strategy_pattern":
                if insight.get("accuracy", 0) < 0.6:
                    recommendations.append({
                        "category": "strategy_specific",
                        "target": insight["strategy"],
                        "recommendation": (
                            f"Add strategy-specific context for {insight['strategy']} "
                            f"strategy in oversight prompts"
                        ),
                        "priority": "high" if insight.get("sample_size", 0) > 20 else "medium",
                        "expected_impact": "Better oversight decisions for specific strategy types"
                    })
            
            elif insight_type == "time_trend" and insight["trend"] == "deteriorating":
                recommendations.append({
                    "category": "general",
                    "target": "oversight_system",
                    "recommendation": (
                        "Review and update oversight prompts to address declining performance; "
                        "consider adding recent market context"
                    ),
                    "priority": "high",
                    "expected_impact": "Reverse declining effectiveness trend"
                })
                
            elif insight_type == "action_issue":
                recommendations.append({
                    "category": "action_specific",
                    "target": insight["action"],
                    "recommendation": (
                        f"Refine criteria for '{insight['action']}' decisions; "
                        f"consider adding more specific guidance on when to {insight['action']}"
                    ),
                    "priority": "high" if insight["action"] == "approve" else "medium",
                    "expected_impact": f"Improved accuracy for {insight['action']} decisions"
                })
        
        # Add general recommendations based on feedback stats
        total_decisions = (
            self.feedback_stats["correct_decisions"] + 
            self.feedback_stats["incorrect_decisions"]
        )
        
        if total_decisions >= 50:
            # Check overall accuracy
            overall_accuracy = (
                self.feedback_stats["correct_decisions"] / total_decisions
                if total_decisions > 0 else 0
            )
            
            if overall_accuracy < 0.7:
                recommendations.append({
                    "category": "general",
                    "target": "oversight_system",
                    "recommendation": (
                        "Conduct comprehensive review of oversight prompts and validation logic; "
                        "consider enhancing market context and historical results"
                    ),
                    "priority": "critical",
                    "expected_impact": "Significant improvement in overall decision quality"
                })
        
        # Deduplicate and prioritize
        unique_recommendations = {}
        for rec in recommendations:
            key = f"{rec['category']}_{rec['target']}"
            if key not in unique_recommendations or rec['priority'] == 'critical':
                unique_recommendations[key] = rec
        
        return list(unique_recommendations.values())
    
    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get a summary of feedback over the specified period.
        
        Args:
            days: Number of days to include in the summary
            
        Returns:
            Summary of feedback statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent feedback
        recent_feedback = [
            f for f in self.feedback_records 
            if datetime.fromisoformat(f["timestamp"]) >= cutoff_date
        ]
        
        if not recent_feedback:
            return {
                "period_days": days,
                "total_feedback": 0,
                "feedback_types": {},
                "accuracy": 0,
                "avg_pnl": 0,
                "insights": [],
                "recommendations": []
            }
        
        # Count by feedback type
        feedback_types = {}
        for f in recent_feedback:
            feedback_type = f["feedback_type"]
            if feedback_type in feedback_types:
                feedback_types[feedback_type] += 1
            else:
                feedback_types[feedback_type] = 1
        
        # Filter to trade outcomes
        trade_outcomes = [
            f for f in recent_feedback
            if f["feedback_type"] == FeedbackType.TRADE_OUTCOME.value
        ]
        
        # Calculate stats
        correct_count = 0
        total_pnl = 0
        
        for f in trade_outcomes:
            action = f["oversight_result"].get("action", "unknown")
            outcome = f["outcome"]
            pnl = f.get("pnl", 0)
            
            # Count correct decisions
            if (
                (action == "approve" and outcome == DecisionOutcome.PROFITABLE.value) or
                (action == "reject" and outcome == DecisionOutcome.LOSS.value) or
                (action == "modify" and outcome != DecisionOutcome.LOSS.value)
            ):
                correct_count += 1
            
            total_pnl += pnl
        
        # Calculate accuracy and average PnL
        accuracy = correct_count / max(len(trade_outcomes), 1)
        avg_pnl = total_pnl / max(len(trade_outcomes), 1)
        
        # Get insights and recommendations
        insights = self.analyze_feedback_patterns()
        recommendations = self.generate_improvement_recommendations()
        
        return {
            "period_days": days,
            "total_feedback": len(recent_feedback),
            "feedback_types": feedback_types,
            "trade_outcomes": len(trade_outcomes),
            "accuracy": accuracy,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "insights": insights,
            "recommendations": recommendations
        }
    
    def export_feedback_data(self, format: str = "json") -> Any:
        """
        Export feedback data for external analysis.
        
        Args:
            format: Export format ('json' or 'dataframe')
            
        Returns:
            Exported data in the requested format
        """
        if format == "json":
            return {
                "feedback_records": self.feedback_records,
                "feedback_stats": self.feedback_stats,
                "insights": self.insights,
                "exported_at": datetime.now().isoformat()
            }
        elif format == "dataframe":
            return pd.DataFrame(self.feedback_records)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Utility function to create a feedback loop instance
def create_feedback_loop(
    data_dir: str,
    evaluator: Optional[OversightEvaluator] = None
) -> OversightFeedbackLoop:
    """
    Create a feedback loop instance with standard configuration.
    
    Args:
        data_dir: Directory to store feedback data
        evaluator: Optional existing evaluator instance
        
    Returns:
        Configured OversightFeedbackLoop instance
    """
    # Create data directory if needed
    os.makedirs(data_dir, exist_ok=True)
    
    # Create evaluator if not provided
    if evaluator is None:
        evaluator = OversightEvaluator()
    
    # Create feedback loop
    feedback_db_path = os.path.join(data_dir, "oversight_feedback.json")
    feedback_loop = OversightFeedbackLoop(
        evaluator=evaluator,
        feedback_db_path=feedback_db_path,
        auto_save=True
    )
    
    return feedback_loop
