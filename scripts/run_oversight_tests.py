#!/usr/bin/env python
"""
Main script for running LLM oversight tests and evaluation.

This script provides a unified entry point for testing different aspects
of the LLM oversight system integration, generating evaluation metrics,
and displaying performance dashboards.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from ai_trading_agent.oversight.client import OversightClient
from ai_trading_agent.oversight.evaluation import OversightEvaluator, DecisionOutcome
from ai_trading_agent.oversight.feedback_loop import OversightFeedbackLoop, create_feedback_loop
from ai_trading_agent.dashboard.oversight_dashboard import OversightDashboard, create_standalone_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs/oversight_tests.log"), mode='a')
    ]
)
logger = logging.getLogger("oversight_tests")


def run_integration_tests(oversight_url: str) -> bool:
    """
    Run integration tests for the LLM oversight system.
    
    Args:
        oversight_url: URL of the oversight service
        
    Returns:
        True if tests pass, False otherwise
    """
    logger.info("Running LLM oversight integration tests...")
    
    try:
        # Import the test script
        import importlib.util
        test_script_path = os.path.join(project_root, "scripts/test_oversight_integration.py")
        spec = importlib.util.spec_from_file_location("test_oversight_integration", test_script_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Create test arguments
        class TestArgs:
            def __init__(self, url):
                self.url = url
        
        # Run the tests
        args = TestArgs(oversight_url)
        exit_code = test_module.main()
        
        if exit_code == 0:
            logger.info("✅ Integration tests passed")
            return True
        else:
            logger.error("❌ Integration tests failed")
            return False
    except Exception as e:
        logger.error(f"❌ Error running integration tests: {e}")
        return False


def generate_sample_data(
    evaluator: OversightEvaluator,
    feedback_loop: OversightFeedbackLoop,
    num_decisions: int = 50
) -> None:
    """
    Generate sample data for testing the evaluation and feedback components.
    
    Args:
        evaluator: OversightEvaluator instance
        feedback_loop: OversightFeedbackLoop instance
        num_decisions: Number of sample decisions to generate
    """
    logger.info(f"Generating {num_decisions} sample decisions for testing...")
    
    # Sample symbols
    symbols = ["AAPL", "MSFT", "AMZN", "GOOG", "TSLA"]
    
    # Sample strategies
    strategies = ["trend", "momentum", "mean_reversion", "ml_strategy"]
    
    # Sample action types
    actions = ["approve", "reject", "modify"]
    action_weights = [0.6, 0.3, 0.1]  # Probability weights
    
    # Generate decisions
    import random
    import numpy as np
    from datetime import datetime, timedelta
    
    # Start date 30 days ago
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_decisions):
        # Generate random decision
        symbol = random.choice(symbols)
        strategy = random.choice(strategies)
        action = np.random.choice(actions, p=action_weights)
        
        # Random timestamp within the last 30 days
        days_ago = random.randint(0, 29)
        timestamp = start_date + timedelta(days=days_ago)
        
        # Random price and quantity
        base_price = {
            "AAPL": 150, "MSFT": 300, "AMZN": 3000, "GOOG": 2500, "TSLA": 800
        }.get(symbol, 100)
        price = base_price * (1 + random.uniform(-0.05, 0.05))
        quantity = random.randint(1, 20)
        
        # Generate decision ID
        decision_id = f"test_decision_{timestamp.strftime('%Y%m%d%H%M%S')}_{i}"
        
        # Create original decision
        original_decision = {
            "symbol": symbol,
            "action": "buy" if random.random() > 0.4 else "sell",
            "quantity": quantity,
            "price": price,
            "confidence": random.uniform(0.5, 0.95),
            "timestamp": timestamp.isoformat(),
            "strategy": strategy,
            "timeframe": random.choice(["1m", "5m", "15m", "1h", "4h", "1d"]),
            "metadata": {
                "signal_type": random.choice(["price_action", "oscillator", "breakout", "trend"]),
                "parameters": {
                    "lookback_periods": random.choice([10, 20, 50, 100]),
                    "threshold": round(random.uniform(0.3, 0.7), 2)
                }
            }
        }
        
        # Create oversight result
        oversight_result = {
            "action": action,
            "confidence": random.uniform(0.6, 0.95),
            "reason": f"Sample {action} decision for testing",
            "recommendation": "Sample recommendation" if action == "modify" else None
        }
        
        # Record decision in evaluator
        evaluator.record_oversight_decision(
            decision_id=decision_id,
            original_decision=original_decision,
            oversight_result=oversight_result,
            timestamp=timestamp
        )
        
        # Generate outcome (more likely to be correct than incorrect)
        is_correct = random.random() < (0.7 if action == "approve" else 0.6)
        
        if action == "approve":
            outcome = DecisionOutcome.PROFITABLE if is_correct else DecisionOutcome.LOSS
        elif action == "reject":
            outcome = DecisionOutcome.LOSS if is_correct else DecisionOutcome.PROFITABLE
        else:  # modify
            outcome = random.choice([DecisionOutcome.PROFITABLE, DecisionOutcome.NEUTRAL]) if is_correct else DecisionOutcome.LOSS
        
        # Generate PnL (positive for profitable, negative for loss)
        if outcome == DecisionOutcome.PROFITABLE:
            pnl = random.uniform(10, 200)
        elif outcome == DecisionOutcome.LOSS:
            pnl = -random.uniform(10, 150)
        else:  # neutral
            pnl = random.uniform(-10, 20)
        
        # Exit price based on PnL
        exit_price = price * (1 + pnl / (price * quantity))
        
        # Holding period (in minutes)
        holding_period = random.randint(10, 1440)  # 10 mins to 24 hours
        
        exit_time = timestamp + timedelta(minutes=holding_period)
        
        # Record outcome in feedback loop
        feedback_loop.record_trade_outcome(
            decision_id=decision_id,
            original_decision=original_decision,
            oversight_result=oversight_result,
            outcome=outcome,
            pnl=pnl,
            exit_price=exit_price,
            exit_time=exit_time,
            holding_period=holding_period,
            metadata={
                "risk_impact": random.uniform(-0.02, 0.02),
                "market_impact": random.uniform(0, 0.005)
            }
        )
        
        # Add some user feedback randomly
        if random.random() < 0.3:
            user_rating = random.randint(1, 5)
            feedback_loop.record_user_feedback(
                decision_id=decision_id,
                user_rating=user_rating,
                comments=f"Sample user feedback with rating {user_rating}",
                is_correct=user_rating >= 3
            )
    
    logger.info(f"Generated {num_decisions} sample decisions")
    
    # Calculate metrics and analyze feedback
    metrics = evaluator.calculate_metrics()
    insights = feedback_loop.analyze_feedback_patterns()
    recommendations = feedback_loop.generate_improvement_recommendations()
    
    logger.info(f"Generated metrics: Accuracy = {metrics['accuracy']:.2f}, Precision = {metrics['precision']:.2f}")
    logger.info(f"Generated {len(insights)} insights and {len(recommendations)} recommendations")


def run_oversight_evaluation(
    data_dir: str,
    oversight_url: Optional[str] = None,
    generate_samples: bool = False,
    num_samples: int = 50
) -> None:
    """
    Run evaluation of the LLM oversight system.
    
    Args:
        data_dir: Directory to store evaluation data
        oversight_url: URL of the oversight service
        generate_samples: Whether to generate sample data
        num_samples: Number of sample decisions to generate
    """
    logger.info("Running LLM oversight evaluation...")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Create evaluator and feedback loop
    evaluator_path = os.path.join(data_dir, "oversight_evaluator.json")
    
    # Check if evaluator data exists
    if os.path.exists(evaluator_path):
        logger.info(f"Loading existing evaluator data from {evaluator_path}")
        evaluator = OversightEvaluator.load_from_file(evaluator_path)
    else:
        logger.info("Creating new evaluator")
        evaluator = OversightEvaluator()
    
    # Create feedback loop
    feedback_loop = create_feedback_loop(data_dir, evaluator)
    
    # Generate sample data if requested
    if generate_samples:
        generate_sample_data(evaluator, feedback_loop, num_samples)
    
    # Calculate and log metrics
    metrics = evaluator.calculate_metrics()
    logger.info("Oversight Evaluation Metrics:")
    for metric_name, value in metrics.items():
        if metric_name in ["accuracy", "precision", "recall", "consistency"]:
            logger.info(f"  {metric_name.capitalize()}: {value:.2f}")
    
    # Analyze feedback and generate recommendations
    insights = feedback_loop.analyze_feedback_patterns()
    recommendations = feedback_loop.generate_improvement_recommendations()
    
    logger.info(f"Identified {len(insights)} insights from feedback data")
    for i, insight in enumerate(insights[:5]):  # Log top 5 insights
        logger.info(f"  Insight {i+1}: {insight.get('description', 'No description')}")
    
    logger.info(f"Generated {len(recommendations)} improvement recommendations")
    for i, rec in enumerate(recommendations[:5]):  # Log top 5 recommendations
        logger.info(f"  Recommendation {i+1} [{rec['priority']}]: {rec['recommendation']}")
    
    # Save evaluation data
    evaluator.save_to_file(evaluator_path)
    logger.info(f"Saved evaluator data to {evaluator_path}")
    
    # Export metrics summary
    metrics_summary = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "insights_count": len(insights),
        "recommendations_count": len(recommendations),
        "recent_decisions": len(evaluator.get_decision_history_df(days=7))
    }
    
    metrics_path = os.path.join(data_dir, "oversight_metrics_summary.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info(f"Saved metrics summary to {metrics_path}")


def run_oversight_dashboard(data_dir: str, port: int = 8050) -> None:
    """
    Run the oversight dashboard.
    
    Args:
        data_dir: Directory containing evaluation data
        port: Port to run the dashboard on
    """
    logger.info(f"Starting LLM oversight dashboard on port {port}...")
    
    evaluator_path = os.path.join(data_dir, "oversight_evaluator.json")
    
    if not os.path.exists(evaluator_path):
        logger.error(f"Evaluator data not found at {evaluator_path}")
        logger.error("Run evaluation first to generate data for the dashboard")
        return
    
    try:
        # Create and run dashboard
        create_standalone_dashboard(evaluator_path, "0.0.0.0", port)
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")


def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description="LLM Oversight Testing and Evaluation")
    
    # Add common arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(project_root, "data/oversight"),
        help="Directory to store/load evaluation data"
    )
    parser.add_argument(
        "--oversight-url",
        type=str,
        default="http://localhost:8080",
        help="URL of the oversight service"
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")
    
    # Test mode
    test_parser = subparsers.add_parser("test", help="Run integration tests")
    
    # Evaluate mode
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample data for testing"
    )
    eval_parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of sample decisions to generate"
    )
    
    # Dashboard mode
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the dashboard")
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
    
    # Determine mode
    if not args.mode:
        parser.print_help()
        return 1
    
    # Run selected mode
    if args.mode == "test":
        success = run_integration_tests(args.oversight_url)
        return 0 if success else 1
    elif args.mode == "evaluate":
        run_oversight_evaluation(
            args.data_dir,
            args.oversight_url,
            args.generate_samples,
            args.num_samples
        )
    elif args.mode == "dashboard":
        run_oversight_dashboard(args.data_dir, args.port)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
