"""
Evaluation metrics for LLM oversight decisions.

This module provides tools for evaluating the quality and usefulness
of LLM oversight decisions in trading contexts.
"""

import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    CONSISTENCY = "consistency"
    LATENCY = "latency"
    RELEVANCE = "relevance"
    PNL_IMPACT = "pnl_impact"
    RISK_REDUCTION = "risk_reduction"
    FALSE_POSITIVES = "false_positives"
    FALSE_NEGATIVES = "false_negatives"


class DecisionOutcome(str, Enum):
    """Possible outcomes of a trading decision."""
    PROFITABLE = "profitable"
    LOSS = "loss"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class OversightEvaluator:
    """
    Evaluates LLM oversight decisions by tracking metrics over time.
    
    This class provides methods to evaluate the performance of the LLM
    oversight system in terms of decision quality, impact on trading
    performance, and system efficiency.
    """
    
    def __init__(
        self, 
        history_size: int = 1000,
        evaluation_window_days: int = 30
    ):
        """
        Initialize the oversight evaluator.
        
        Args:
            history_size: Maximum number of decisions to keep in history
            evaluation_window_days: Number of days to consider for evaluation
        """
        self.history_size = history_size
        self.evaluation_window_days = evaluation_window_days
        
        # Decision history
        self.decision_history: List[Dict[str, Any]] = []
        
        # Metrics history
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {
            metric.value: [] for metric in MetricType
        }
        
        # Reference data for evaluating decision quality
        self.market_outcomes: Dict[str, Dict[str, Any]] = {}
        
        # Performance impact tracking
        self.performance_tracking: Dict[str, List[Dict[str, Any]]] = {
            "approved_decisions": [],
            "rejected_decisions": [],
            "modified_decisions": []
        }
    
    def record_oversight_decision(
        self,
        decision_id: str,
        original_decision: Dict[str, Any],
        oversight_result: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record an oversight decision for later evaluation.
        
        Args:
            decision_id: Unique identifier for the decision
            original_decision: The original trading decision
            oversight_result: The result from the LLM oversight system
            timestamp: When the decision was made (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create a record of this oversight decision
        decision_record = {
            "decision_id": decision_id,
            "timestamp": timestamp,
            "original_decision": original_decision,
            "oversight_result": oversight_result,
            "action_taken": oversight_result.get("action", "unknown"),
            "confidence": oversight_result.get("confidence", 0.0),
            "outcome": DecisionOutcome.UNKNOWN.value,
            "pnl_impact": None,
            "risk_impact": None,
            "evaluation_complete": False
        }
        
        # Add to history, maintaining max size
        self.decision_history.append(decision_record)
        if len(self.decision_history) > self.history_size:
            self.decision_history = self.decision_history[-self.history_size:]
            
        # Track by action type for performance comparison
        action = oversight_result.get("action", "unknown")
        if action == "approve":
            self.performance_tracking["approved_decisions"].append(decision_record)
        elif action == "reject":
            self.performance_tracking["rejected_decisions"].append(decision_record)
        elif action == "modify":
            self.performance_tracking["modified_decisions"].append(decision_record)
    
    def record_decision_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        pnl_impact: Optional[float] = None,
        risk_impact: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record the actual outcome of a previously evaluated decision.
        
        Args:
            decision_id: Unique identifier for the decision
            outcome: Actual outcome of the decision
            pnl_impact: Impact on profit and loss (if applicable)
            risk_impact: Impact on portfolio risk (if applicable)
            metadata: Additional data about the outcome
            
        Returns:
            True if the decision was found and updated, False otherwise
        """
        # Find the decision in history
        for decision in self.decision_history:
            if decision["decision_id"] == decision_id:
                decision["outcome"] = outcome.value
                decision["pnl_impact"] = pnl_impact
                decision["risk_impact"] = risk_impact
                decision["evaluation_complete"] = True
                
                if metadata:
                    decision["outcome_metadata"] = metadata
                    
                # Update performance tracking
                action = decision["action_taken"]
                for action_list in self.performance_tracking.values():
                    for tracked_decision in action_list:
                        if tracked_decision["decision_id"] == decision_id:
                            tracked_decision.update({
                                "outcome": outcome.value,
                                "pnl_impact": pnl_impact,
                                "risk_impact": risk_impact,
                                "evaluation_complete": True
                            })
                            if metadata:
                                tracked_decision["outcome_metadata"] = metadata
                
                return True
                
        return False
    
    def calculate_metrics(
        self, 
        since_days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics based on recorded decisions and outcomes.
        
        Args:
            since_days: Consider only decisions within this many days (default: evaluation_window_days)
            
        Returns:
            Dictionary of calculated metrics
        """
        if since_days is None:
            since_days = self.evaluation_window_days
            
        cutoff_date = datetime.now() - timedelta(days=since_days)
        
        # Filter decisions within the evaluation window
        recent_decisions = [
            d for d in self.decision_history 
            if d["timestamp"] >= cutoff_date and d["evaluation_complete"]
        ]
        
        if not recent_decisions:
            logger.warning("No complete decisions found within evaluation window")
            return {m.value: 0.0 for m in MetricType}
        
        metrics = {}
        
        # Calculate accuracy
        correct_decisions = [
            d for d in recent_decisions
            if (d["action_taken"] == "approve" and d["outcome"] == DecisionOutcome.PROFITABLE.value) or
               (d["action_taken"] == "reject" and d["outcome"] == DecisionOutcome.LOSS.value) or
               (d["action_taken"] == "modify" and d["outcome"] != DecisionOutcome.LOSS.value)
        ]
        metrics[MetricType.ACCURACY.value] = len(correct_decisions) / len(recent_decisions)
        
        # Calculate precision (correct approvals / all approvals)
        approvals = [d for d in recent_decisions if d["action_taken"] == "approve"]
        correct_approvals = [d for d in approvals if d["outcome"] == DecisionOutcome.PROFITABLE.value]
        metrics[MetricType.PRECISION.value] = len(correct_approvals) / max(len(approvals), 1)
        
        # Calculate recall (correct approvals / all profitable opportunities)
        profitable_opportunities = [
            d for d in recent_decisions 
            if d["outcome"] == DecisionOutcome.PROFITABLE.value
        ]
        metrics[MetricType.RECALL.value] = len(correct_approvals) / max(len(profitable_opportunities), 1)
        
        # Calculate consistency (standard deviation of confidence)
        confidences = [d["confidence"] for d in recent_decisions if "confidence" in d]
        metrics[MetricType.CONSISTENCY.value] = 1.0 - min(np.std(confidences) if confidences else 0, 0.5) / 0.5
        
        # Calculate false positives and negatives
        false_positives = [
            d for d in recent_decisions
            if d["action_taken"] == "approve" and d["outcome"] == DecisionOutcome.LOSS.value
        ]
        false_negatives = [
            d for d in recent_decisions
            if d["action_taken"] == "reject" and d["outcome"] == DecisionOutcome.PROFITABLE.value
        ]
        metrics[MetricType.FALSE_POSITIVES.value] = len(false_positives) / max(len(approvals), 1)
        
        rejections = [d for d in recent_decisions if d["action_taken"] == "reject"]
        metrics[MetricType.FALSE_NEGATIVES.value] = len(false_negatives) / max(len(rejections), 1)
        
        # Calculate PnL impact
        approved_pnl = sum(d.get("pnl_impact", 0) or 0 for d in approvals)
        rejected_pnl = sum(d.get("pnl_impact", 0) or 0 for d in false_negatives)
        total_pnl = approved_pnl - rejected_pnl
        metrics[MetricType.PNL_IMPACT.value] = total_pnl
        
        # Calculate risk reduction
        risk_reduction = sum(d.get("risk_impact", 0) or 0 for d in recent_decisions)
        metrics[MetricType.RISK_REDUCTION.value] = risk_reduction
        
        # Store metrics in history
        now = datetime.now()
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append((now, value))
            
            # Trim history if needed
            if len(self.metrics_history[metric_name]) > self.history_size:
                self.metrics_history[metric_name] = self.metrics_history[metric_name][-self.history_size:]
        
        return metrics
    
    def get_metrics_trend(
        self, 
        metric_type: MetricType,
        days: int = 30,
        interval_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get the trend of a specific metric over time.
        
        Args:
            metric_type: The type of metric to analyze
            days: Number of days to include
            interval_hours: Hours per data point
            
        Returns:
            List of metric values over time
        """
        metric_history = self.metrics_history.get(metric_type.value, [])
        if not metric_history:
            return []
            
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [(ts, val) for ts, val in metric_history if ts >= cutoff_date]
        
        # Group by interval
        result = []
        interval_delta = timedelta(hours=interval_hours)
        
        if recent_metrics:
            # Create time buckets
            buckets = {}
            for ts, val in recent_metrics:
                # Round to nearest interval
                bucket_ts = ts.replace(
                    hour=(ts.hour // interval_hours) * interval_hours,
                    minute=0, second=0, microsecond=0
                )
                if bucket_ts in buckets:
                    buckets[bucket_ts].append(val)
                else:
                    buckets[bucket_ts] = [val]
            
            # Calculate average for each bucket
            for ts, values in sorted(buckets.items()):
                result.append({
                    "timestamp": ts.isoformat(),
                    "value": sum(values) / len(values)
                })
                
        return result
    
    def compare_performance(
        self, 
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare performance of different decision types.
        
        Args:
            days: Number of days to include
            
        Returns:
            Performance comparison data
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        results = {}
        for action_type, decisions in self.performance_tracking.items():
            # Filter by date and completion
            filtered_decisions = [
                d for d in decisions 
                if d["timestamp"] >= cutoff_date and d["evaluation_complete"]
            ]
            
            if not filtered_decisions:
                results[action_type] = {
                    "count": 0,
                    "pnl": 0,
                    "avg_pnl": 0,
                    "win_rate": 0,
                    "risk_reduction": 0
                }
                continue
                
            # Calculate metrics
            count = len(filtered_decisions)
            pnl = sum(d.get("pnl_impact", 0) or 0 for d in filtered_decisions)
            avg_pnl = pnl / count if count > 0 else 0
            
            wins = sum(1 for d in filtered_decisions if d["outcome"] == DecisionOutcome.PROFITABLE.value)
            win_rate = wins / count if count > 0 else 0
            
            risk_reduction = sum(d.get("risk_impact", 0) or 0 for d in filtered_decisions)
            
            results[action_type] = {
                "count": count,
                "pnl": pnl,
                "avg_pnl": avg_pnl,
                "win_rate": win_rate,
                "risk_reduction": risk_reduction
            }
            
        return results
    
    def get_decision_history_df(
        self, 
        days: Optional[int] = None,
        include_incomplete: bool = False
    ) -> pd.DataFrame:
        """
        Get decision history as a pandas DataFrame.
        
        Args:
            days: Number of days to include (default: evaluation_window_days)
            include_incomplete: Whether to include decisions without outcomes
            
        Returns:
            DataFrame of decision history
        """
        if days is None:
            days = self.evaluation_window_days
            
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter decisions
        if include_incomplete:
            filtered_decisions = [d for d in self.decision_history if d["timestamp"] >= cutoff_date]
        else:
            filtered_decisions = [
                d for d in self.decision_history 
                if d["timestamp"] >= cutoff_date and d["evaluation_complete"]
            ]
            
        if not filtered_decisions:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "decision_id", "timestamp", "symbol", "action", "oversight_action",
                "confidence", "outcome", "pnl_impact", "risk_impact"
            ])
            
        # Extract relevant fields for DataFrame
        records = []
        for d in filtered_decisions:
            original_decision = d["original_decision"]
            record = {
                "decision_id": d["decision_id"],
                "timestamp": d["timestamp"],
                "symbol": original_decision.get("symbol", "unknown"),
                "action": original_decision.get("action", "unknown"),
                "oversight_action": d["action_taken"],
                "confidence": d["confidence"],
                "outcome": d["outcome"],
                "pnl_impact": d["pnl_impact"],
                "risk_impact": d["risk_impact"]
            }
            records.append(record)
            
        return pd.DataFrame(records)
    
    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics and performance data.
        
        Returns:
            Dictionary containing all metric data
        """
        metrics = self.calculate_metrics()
        
        trends = {}
        for metric_type in MetricType:
            trends[metric_type.value] = self.get_metrics_trend(metric_type)
            
        performance_comparison = self.compare_performance()
        
        # Get summary of recent decisions
        recent_decisions = self.get_decision_history_df(days=7)
        decision_summary = {
            "total": len(recent_decisions),
            "by_action": recent_decisions["oversight_action"].value_counts().to_dict(),
            "by_outcome": recent_decisions["outcome"].value_counts().to_dict(),
            "avg_confidence": recent_decisions["confidence"].mean()
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": metrics,
            "trends": trends,
            "performance_comparison": performance_comparison,
            "recent_decisions": decision_summary
        }
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save evaluator data to a JSON file.
        
        Args:
            filepath: Path to save the evaluation data
        """
        data = {
            "metrics_history": {
                k: [(ts.isoformat(), v) for ts, v in vals]
                for k, vals in self.metrics_history.items()
            },
            "decision_history": self.decision_history,
            "performance_tracking": self.performance_tracking,
            "metadata": {
                "history_size": self.history_size,
                "evaluation_window_days": self.evaluation_window_days,
                "saved_at": datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load_from_file(cls, filepath: str) -> 'OversightEvaluator':
        """
        Load evaluator data from a JSON file.
        
        Args:
            filepath: Path to the evaluation data file
            
        Returns:
            OversightEvaluator instance with loaded data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Create new instance
        evaluator = cls(
            history_size=data["metadata"]["history_size"],
            evaluation_window_days=data["metadata"]["evaluation_window_days"]
        )
        
        # Load metrics history
        evaluator.metrics_history = {
            k: [(datetime.fromisoformat(ts), v) for ts, v in vals]
            for k, vals in data["metrics_history"].items()
        }
        
        # Load decision history and convert timestamps
        evaluator.decision_history = data["decision_history"]
        for decision in evaluator.decision_history:
            if isinstance(decision["timestamp"], str):
                decision["timestamp"] = datetime.fromisoformat(decision["timestamp"])
                
        # Load performance tracking
        evaluator.performance_tracking = data["performance_tracking"]
        for category in evaluator.performance_tracking.values():
            for decision in category:
                if isinstance(decision["timestamp"], str):
                    decision["timestamp"] = datetime.fromisoformat(decision["timestamp"])
        
        return evaluator
