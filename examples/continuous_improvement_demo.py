"""Continuous Improvement System Demo.

This script demonstrates the Continuous Improvement System with automatic stopping criteria
for optimizing sentiment analysis through controlled experiments.
"""

import asyncio
import logging
import argparse
import json
from datetime import datetime, timedelta
import random
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.logging import setup_logging, get_logger
from src.common.config import config
from src.common.events import event_bus, Event
from src.analysis_agents.sentiment.ab_testing import (
    ab_testing_framework, ExperimentType, ExperimentStatus, 
    TargetingCriteria, VariantAssignmentStrategy
)
from src.analysis_agents.sentiment.continuous_improvement.improvement_manager import continuous_improvement_manager
from src.analysis_agents.sentiment.continuous_improvement.stopping_criteria import (
    stopping_criteria_manager, SampleSizeCriterion, BayesianProbabilityThresholdCriterion,
    ExpectedLossCriterion, ConfidenceIntervalCriterion, TimeLimitCriterion
)
from src.analysis_agents.sentiment.continuous_improvement.bayesian_analysis import bayesian_analyzer

# Setup logging
setup_logging()
logger = get_logger("examples", "continuous_improvement_demo")

def generate_simulation_data():
    """Generate simulated experiment data."""
    # Define experiment variants
    variants = [
        {
            "name": "Control",
            "description": "Current prompt template",
            "weight": 0.5,
            "config": {"template": "Original template"},
            "control": True
        },
        {
            "name": "Enhanced Template",
            "description": "Improved prompt with more context",
            "weight": 0.5,
            "config": {"template": "Enhanced template with more context"},
            "control": False
        }
    ]
    
    # Create the experiment
    experiment = ab_testing_framework.create_experiment(
        name="Simulated Prompt Template Test",
        description="Testing improved prompt templates with automatic stopping criteria",
        experiment_type=ExperimentType.PROMPT_TEMPLATE,
        variants=variants,
        targeting=[TargetingCriteria.ALL_TRAFFIC],
        assignment_strategy=VariantAssignmentStrategy.RANDOM,
        sample_size=200,
        min_confidence=0.95,
        owner="simulation",
        metadata={"simulated": True, "auto_generated": True}
    )
    
    # Start the experiment
    ab_testing_framework.start_experiment(experiment.id)
    
    # Generate simulated metrics
    # Control variant has decent performance
    control_variant = experiment.variants[0]
    treatment_variant = experiment.variants[1]
    
    control_metrics = experiment.variant_metrics[control_variant.id]
    treatment_metrics = experiment.variant_metrics[treatment_variant.id]
    
    # Baseline values
    control_sentiment_accuracy = 0.75
    control_direction_accuracy = 0.70
    control_calibration_error = 0.15
    control_confidence_score = 0.65
    
    # Treatment has better values
    treatment_sentiment_accuracy = 0.82  # ~9% improvement
    treatment_direction_accuracy = 0.78  # ~11% improvement
    treatment_calibration_error = 0.10   # ~33% reduction in error
    treatment_confidence_score = 0.72    # ~11% improvement
    
    return experiment, {
        "control": {
            "variant": control_variant,
            "metrics": control_metrics,
            "values": {
                "sentiment_accuracy": control_sentiment_accuracy,
                "direction_accuracy": control_direction_accuracy,
                "calibration_error": control_calibration_error,
                "confidence_score": control_confidence_score
            }
        },
        "treatment": {
            "variant": treatment_variant,
            "metrics": treatment_metrics,
            "values": {
                "sentiment_accuracy": treatment_sentiment_accuracy,
                "direction_accuracy": treatment_direction_accuracy,
                "calibration_error": treatment_calibration_error,
                "confidence_score": treatment_confidence_score
            }
        }
    }

async def run_simulation(experiment, simulation_data, batches=20, requests_per_batch=10):
    """Run a simulation of the experiment with automatic stopping."""
    logger.info(f"Starting simulation with {batches} batches of {requests_per_batch} requests each")
    
    # Track if experiment is completed
    is_completed = False
    
    # Run batches of requests
    for batch in range(batches):
        if is_completed:
            break
            
        logger.info(f"Processing batch {batch+1}/{batches}")
        
        # Process requests for this batch
        for _ in range(requests_per_batch):
            # Randomly assign to control or treatment
            is_control = random.random() < 0.5
            
            if is_control:
                variant_data = simulation_data["control"]
            else:
                variant_data = simulation_data["treatment"]
            
            variant = variant_data["variant"]
            metrics = variant_data["metrics"]
            values = variant_data["values"]
            
            # Simulate a request
            metrics.requests += 1
            metrics.successes += 1
            
            # Add some noise to the metrics
            noise_factor = 0.05  # 5% noise
            
            # Update sentiment accuracy
            accuracy = values["sentiment_accuracy"] + random.uniform(-noise_factor, noise_factor)
            accuracy = max(0, min(1, accuracy))  # Clamp to [0, 1]
            
            if metrics.sentiment_accuracy == 0:
                metrics.sentiment_accuracy = accuracy
            else:
                metrics.sentiment_accuracy = 0.9 * metrics.sentiment_accuracy + 0.1 * accuracy
            
            # Update direction accuracy
            direction_acc = values["direction_accuracy"] + random.uniform(-noise_factor, noise_factor)
            direction_acc = max(0, min(1, direction_acc))  # Clamp to [0, 1]
            
            if metrics.direction_accuracy == 0:
                metrics.direction_accuracy = direction_acc
            else:
                metrics.direction_accuracy = 0.9 * metrics.direction_accuracy + 0.1 * direction_acc
            
            # Update calibration error
            calib_error = values["calibration_error"] + random.uniform(-noise_factor, noise_factor)
            calib_error = max(0, min(1, calib_error))  # Clamp to [0, 1]
            
            if metrics.calibration_error == 0:
                metrics.calibration_error = calib_error
            else:
                metrics.calibration_error = 0.9 * metrics.calibration_error + 0.1 * calib_error
            
            # Update confidence score
            conf_score = values["confidence_score"] + random.uniform(-noise_factor, noise_factor)
            conf_score = max(0, min(1, conf_score))  # Clamp to [0, 1]
            
            if metrics.confidence_score == 0:
                metrics.confidence_score = conf_score
            else:
                metrics.confidence_score = 0.9 * metrics.confidence_score + 0.1 * conf_score
        
        # Check stopping criteria after each batch
        evaluation = stopping_criteria_manager.evaluate_experiment(experiment)
        
        # Display criteria status
        for criterion_name, result in evaluation["criteria_results"].items():
            logger.info(f"Criterion '{criterion_name}': {result['should_stop']} - {result['reason']}")
        
        if evaluation["should_stop"]:
            logger.info(f"Stopping experiment after {batch+1} batches")
            
            # List reasons for stopping
            for reason in evaluation["stopping_reasons"]:
                logger.info(f"Stopping reason: {reason['criterion']} - {reason['reason']}")
            
            # Complete the experiment
            ab_testing_framework.complete_experiment(experiment.id)
            is_completed = True
            break
        
        # Wait a bit between batches
        await asyncio.sleep(0.5)
    
    # If experiment wasn't stopped by criteria, stop it now
    if not is_completed:
        logger.info(f"Experiment not stopped by criteria after {batches} batches. Stopping now.")
        ab_testing_framework.complete_experiment(experiment.id)
    
    # Run Bayesian analysis
    logger.info("Running Bayesian analysis on the experiment")
    analysis_results = bayesian_analyzer.analyze_experiment(experiment)
    
    # Get an easily readable summary
    summary = analysis_results.get_summary()
    logger.info(f"Analysis summary:\n{summary}")
    
    # Display recommendation
    if analysis_results.has_clear_winner():
        winner = analysis_results.get_winning_variant()
        logger.info(f"Experiment has a clear winner: {winner}")
    else:
        logger.info("Experiment has no clear winner")
    
    return analysis_results

async def main():
    parser = argparse.ArgumentParser(description="Continuous Improvement System Demo")
    parser.add_argument(
        "--batches", type=int, default=15,
        help="Number of simulation batches (default: 15)"
    )
    parser.add_argument(
        "--requests", type=int, default=10,
        help="Requests per batch (default: 10)"
    )
    parser.add_argument(
        "--set-criteria", action="store_true",
        help="Use custom stopping criteria instead of defaults"
    )
    args = parser.parse_args()
    
    # Initialize the AB testing framework
    await ab_testing_framework.initialize()
    
    # Initialize the continuous improvement manager
    await continuous_improvement_manager.initialize()
    
    if args.set_criteria:
        # Configure custom stopping criteria for the demo
        stopping_criteria_manager.clear_criteria()
        
        # Add criteria
        stopping_criteria_manager.add_criterion(
            SampleSizeCriterion(min_samples_per_variant=50)
        )
        stopping_criteria_manager.add_criterion(
            BayesianProbabilityThresholdCriterion(
                probability_threshold=0.90,
                min_samples_per_variant=30
            )
        )
        stopping_criteria_manager.add_criterion(
            ExpectedLossCriterion(
                loss_threshold=0.01,
                min_samples_per_variant=30
            )
        )
        stopping_criteria_manager.add_criterion(
            ConfidenceIntervalCriterion(
                interval_width_threshold=0.10,
                min_samples_per_variant=30
            )
        )
        stopping_criteria_manager.add_criterion(
            TimeLimitCriterion(max_days=7)
        )
        
        logger.info("Configured custom stopping criteria")
    
    # Generate simulated experiment
    experiment, simulation_data = generate_simulation_data()
    
    # Run the simulation
    logger.info("Starting experiment simulation")
    analysis_results = await run_simulation(
        experiment, 
        simulation_data,
        batches=args.batches,
        requests_per_batch=args.requests
    )
    
    # Show experiment status
    experiment_status = experiment.get_status()
    logger.info(f"Experiment final status: {experiment_status}")
    
    # Display final metrics
    control_variant = None
    treatment_variant = None
    
    for variant in experiment.variants:
        if variant.control:
            control_variant = variant
        else:
            treatment_variant = variant
    
    if control_variant and treatment_variant:
        control_metrics = experiment.variant_metrics[control_variant.id].to_dict()
        treatment_metrics = experiment.variant_metrics[treatment_variant.id].to_dict()
        
        logger.info("Final metrics:")
        logger.info(f"Control variant ({control_variant.name}):")
        for metric_name, value in control_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric_name}: {value:.4f}")
        
        logger.info(f"Treatment variant ({treatment_variant.name}):")
        for metric_name, value in treatment_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric_name}: {value:.4f}")
    
    logger.info("Simulation completed successfully")

if __name__ == "__main__":
    asyncio.run(main())