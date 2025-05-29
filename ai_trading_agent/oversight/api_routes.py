"""
API routes for the LLM oversight system.

This module provides REST API endpoints for querying LLM oversight metrics,
decision history, and feedback data for visualization in the frontend dashboard.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ai_trading_agent.oversight.evaluation import OversightEvaluator, DecisionOutcome
from ai_trading_agent.oversight.feedback_loop import OversightFeedbackLoop, create_feedback_loop

# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/oversight", tags=["oversight"])


class DashboardResponse(BaseModel):
    """Response model for dashboard data."""
    metrics: Dict[str, float]
    metrics_trend: List[Dict[str, Union[str, float]]]
    decision_history: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]


# Global evaluator and feedback loop instances
_data_dir = os.environ.get("OVERSIGHT_DATA_DIR", "/app/data/oversight")
_evaluator: Optional[OversightEvaluator] = None
_feedback_loop: Optional[OversightFeedbackLoop] = None


def get_evaluator() -> OversightEvaluator:
    """
    Get the evaluator instance, creating it if necessary.
    
    Returns:
        OversightEvaluator: The evaluator instance
    """
    global _evaluator
    if _evaluator is None:
        evaluator_path = os.path.join(_data_dir, "oversight_evaluator.json")
        if os.path.exists(evaluator_path):
            _evaluator = OversightEvaluator.load_from_file(evaluator_path)
        else:
            _evaluator = OversightEvaluator()
    return _evaluator


def get_feedback_loop() -> OversightFeedbackLoop:
    """
    Get the feedback loop instance, creating it if necessary.
    
    Returns:
        OversightFeedbackLoop: The feedback loop instance
    """
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = create_feedback_loop(_data_dir, get_evaluator())
    return _feedback_loop


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(
    time_range: str = Query("30d", description="Time range (7d, 30d, 90d)"),
):
    """
    Get dashboard data for the LLM oversight system.
    
    Args:
        time_range: Time range for the data (7d, 30d, 90d)
        
    Returns:
        DashboardResponse: Dashboard data for the frontend
    """
    try:
        # Parse time range
        days = int(time_range.rstrip("d"))
        if days not in [7, 30, 90]:
            days = 30  # Default to 30 days
            
        # Get evaluator and feedback loop
        evaluator = get_evaluator()
        feedback_loop = get_feedback_loop()
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(since_days=days)
        
        # Get metrics trends
        metrics_trend = []
        for metric_name in ["accuracy", "precision", "recall"]:
            trend_data = evaluator.get_metrics_trend(
                metric_type=metric_name,
                days=days,
                interval_hours=24 if days <= 7 else 24 * (days // 10)
            )
            
            # Convert to format expected by frontend
            for point in trend_data:
                # Find or create entry for this date
                date_str = datetime.fromisoformat(point["timestamp"]).strftime("%Y-%m-%d")
                existing = next((x for x in metrics_trend if x.get("date") == date_str), None)
                
                if existing:
                    existing[metric_name] = point["value"]
                else:
                    metrics_trend.append({
                        "date": date_str,
                        metric_name: point["value"]
                    })
        
        # Sort by date
        metrics_trend.sort(key=lambda x: x["date"])
        
        # Get decision history
        decision_history = evaluator.get_decision_history_df(days=days).to_dict(orient="records")
        
        # Get insights and recommendations
        insights = feedback_loop.analyze_feedback_patterns()
        recommendations = feedback_loop.generate_improvement_recommendations()
        
        return {
            "metrics": metrics,
            "metrics_trend": metrics_trend,
            "decision_history": decision_history,
            "insights": insights,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard data: {str(e)}")


@router.post("/feedback/trade-outcome")
async def record_trade_outcome(
    decision_id: str,
    outcome: str,
    pnl: float,
    exit_price: Optional[float] = None,
    exit_time: Optional[str] = None,
    holding_period: Optional[int] = None
):
    """
    Record the outcome of a trade that was subject to oversight.
    
    Args:
        decision_id: Unique identifier for the decision
        outcome: Actual outcome of the trade (profitable, loss, neutral)
        pnl: Profit/loss from the trade
        exit_price: Price at which the position was closed
        exit_time: Time when the position was closed (ISO format)
        holding_period: How long the position was held (in minutes)
        
    Returns:
        Dict: Result of recording the outcome
    """
    try:
        # Get feedback loop
        feedback_loop = get_feedback_loop()
        
        # Find the decision in evaluator history
        evaluator = get_evaluator()
        decision_found = False
        
        for decision in evaluator.decision_history:
            if decision["decision_id"] == decision_id:
                decision_found = True
                original_decision = decision["original_decision"]
                oversight_result = decision["oversight_result"]
                
                # Parse outcome
                decision_outcome = DecisionOutcome(outcome)
                
                # Parse exit_time if provided
                parsed_exit_time = None
                if exit_time:
                    parsed_exit_time = datetime.fromisoformat(exit_time)
                
                # Record outcome
                feedback_id = feedback_loop.record_trade_outcome(
                    decision_id=decision_id,
                    original_decision=original_decision,
                    oversight_result=oversight_result,
                    outcome=decision_outcome,
                    pnl=pnl,
                    exit_price=exit_price,
                    exit_time=parsed_exit_time,
                    holding_period=holding_period
                )
                
                return {
                    "success": True,
                    "feedback_id": feedback_id,
                    "message": f"Recorded outcome for decision {decision_id}"
                }
        
        if not decision_found:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
            
    except Exception as e:
        logger.error(f"Error recording trade outcome: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording trade outcome: {str(e)}")


@router.post("/feedback/user")
async def record_user_feedback(
    decision_id: str,
    user_rating: int,
    comments: Optional[str] = None,
    is_correct: Optional[bool] = None
):
    """
    Record user feedback on an oversight decision.
    
    Args:
        decision_id: Unique identifier for the decision
        user_rating: Rating from 1-5 (1=poor, 5=excellent)
        comments: User comments about the decision
        is_correct: Whether the user considers the decision correct
        
    Returns:
        Dict: Result of recording the feedback
    """
    try:
        # Get feedback loop
        feedback_loop = get_feedback_loop()
        
        # Validate rating
        if user_rating < 1 or user_rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Find the decision in evaluator history
        evaluator = get_evaluator()
        decision_found = False
        
        for decision in evaluator.decision_history:
            if decision["decision_id"] == decision_id:
                decision_found = True
                
                # Record feedback
                feedback_id = feedback_loop.record_user_feedback(
                    decision_id=decision_id,
                    user_rating=user_rating,
                    comments=comments,
                    is_correct=is_correct
                )
                
                return {
                    "success": True,
                    "feedback_id": feedback_id,
                    "message": f"Recorded user feedback for decision {decision_id}"
                }
        
        if not decision_found:
            raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
            
    except Exception as e:
        logger.error(f"Error recording user feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording user feedback: {str(e)}")


@router.get("/metrics/summary")
async def get_metrics_summary(
    days: int = Query(30, description="Number of days to include")
):
    """
    Get a summary of oversight metrics.
    
    Args:
        days: Number of days to include
        
    Returns:
        Dict: Summary of oversight metrics
    """
    try:
        # Get evaluator
        evaluator = get_evaluator()
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(since_days=days)
        
        # Get decision history stats
        df = evaluator.get_decision_history_df(days=days)
        
        # Count by action
        action_counts = {}
        if not df.empty:
            action_counts = df["oversight_action"].value_counts().to_dict()
        
        # Count by outcome
        outcome_counts = {}
        if not df.empty:
            outcome_counts = df["outcome"].value_counts().to_dict()
        
        return {
            "metrics": metrics,
            "total_decisions": len(df),
            "action_counts": action_counts,
            "outcome_counts": outcome_counts,
            "time_range_days": days
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics summary: {str(e)}")
