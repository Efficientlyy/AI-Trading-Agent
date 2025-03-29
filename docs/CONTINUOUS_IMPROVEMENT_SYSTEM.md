# Continuous Improvement System

This document provides a comprehensive guide to the Continuous Improvement System for sentiment analysis optimization, including its automatic stopping criteria capabilities.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Experiment Lifecycle](#experiment-lifecycle)
4. [Multi-Variant Testing](#multi-variant-testing)
5. [Bayesian Analysis](#bayesian-analysis)
6. [Automatic Stopping Criteria](#automatic-stopping-criteria)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)
9. [Dashboard Integration](#dashboard-integration)
10. [Troubleshooting](#troubleshooting)

## Overview

The Continuous Improvement System (CIS) is a framework for automated experimentation and optimization of the sentiment analysis system. It automatically identifies improvement opportunities, designs and runs experiments, analyzes results using both frequentist and Bayesian methods, and implements winning variants when clear winners emerge.

Key features:
- Automated experiment generation based on performance metrics
- A/B and multi-variant testing capabilities
- Bayesian analysis for robust evaluation of experiments
- Automatic stopping criteria for efficient resource utilization
- Dashboard integration for monitoring and management

## System Architecture

The Continuous Improvement System consists of several key components:

1. **Improvement Manager**: Coordinates the overall system, identifies improvement opportunities, and manages the experiment lifecycle.

2. **AB Testing Framework**: Provides the foundation for creating, running, and analyzing experiments.

3. **Multi-Variant Testing**: Extends the testing framework to support multiple treatment variants with advanced statistical analysis.

4. **Bayesian Analysis**: Offers probabilistic interpretation of experiment results and handles uncertainty more robustly than traditional methods.

5. **Stopping Criteria Manager**: Evaluates experiments in real-time to determine when sufficient data has been collected for decision-making.

6. **Dashboard Components**: Provides visualizations and controls for monitoring and managing experiments.

## Experiment Lifecycle

Experiments in the Continuous Improvement System follow a defined lifecycle:

1. **Draft**: The experiment is created but not yet active.
2. **Active**: The experiment is running and collecting data.
3. **Paused**: The experiment is temporarily paused.
4. **Completed**: Data collection is finished.
5. **Analyzed**: Results have been analyzed.
6. **Implemented**: The winning variant has been implemented.
7. **Archived**: The experiment is archived for historical reference.

The system can automatically transition experiments through these states based on configured criteria.

## Multi-Variant Testing

While traditional A/B testing compares a control variant to a single treatment variant, multi-variant testing allows simultaneous testing of multiple treatment variants. The system supports this through:

- **ANOVA Analysis**: Determines if there are statistically significant differences between any variants.
- **Tukey HSD**: Performs post-hoc analysis to identify which specific variant pairs differ significantly.
- **Variant Scoring**: Calculates overall scores to determine which variant performs best across all metrics.

## Bayesian Analysis

The system uses Bayesian analysis to provide more robust interpretations of experiment results:

- **Winning Probability**: Calculates the probability each variant is the best.
- **Lift Estimation**: Estimates the improvement relative to the control variant.
- **Credible Intervals**: Provides probability distributions for true effect sizes.
- **Expected Loss**: Quantifies the expected regret for choosing each variant.

Bayesian methods are particularly valuable for making decisions with limited data and accounting for uncertainty.

## Automatic Stopping Criteria

A key feature of the Continuous Improvement System is its ability to automatically determine when experiments have collected sufficient data to make reliable decisions. This optimizes resource usage and accelerates the improvement cycle.

### Available Stopping Criteria

The system includes several stopping criteria that can be configured according to your needs:

1. **Sample Size Criterion**: Stops when each variant has reached a minimum number of samples.

2. **Bayesian Probability Threshold**: Stops when a variant has achieved a specified probability of being the best.

3. **Expected Loss Criterion**: Stops when the expected loss (regret) for choosing the best variant falls below a threshold.

4. **Confidence Interval Width**: Stops when credible intervals become narrow enough to make reliable decisions.

5. **Time Limit**: Stops when an experiment has been running for a maximum duration.

### How Stopping Criteria Work

Each criterion independently evaluates whether an experiment should be stopped. The system:

1. Periodically evaluates all active experiments against all configured criteria
2. If any criterion indicates the experiment should stop, records the reason
3. Automatically completes the experiment and moves it to analysis phase
4. Determines if there's a clear winner that should be implemented

This approach ensures efficient use of resources by avoiding unnecessary data collection while maintaining statistical rigor.

### Customizing Stopping Criteria

You can customize the stopping criteria through configuration or programmatically:

```python
# Configure through configuration file
stopping_criteria:
  min_samples_per_variant: 100
  probability_threshold: 0.95
  loss_threshold: 0.005
  interval_width_threshold: 0.05
  max_experiment_days: 14
```

Or programmatically:

```python
from src.analysis_agents.sentiment.continuous_improvement.stopping_criteria import (
    stopping_criteria_manager, BayesianProbabilityThresholdCriterion
)

# Configure a custom criterion
stopping_criteria_manager.add_criterion(
    BayesianProbabilityThresholdCriterion(
        probability_threshold=0.90,
        min_samples_per_variant=50
    )
)
```

## Configuration

The Continuous Improvement System can be configured through the system configuration file. Main configuration options include:

```yaml
sentiment_analysis:
  continuous_improvement:
    enabled: true
    check_interval: 3600  # 1 hour in seconds
    experiment_generation_interval: 86400  # 1 day in seconds
    max_concurrent_experiments: 3
    auto_implement: false
    significance_threshold: 0.95
    improvement_threshold: 0.05
    results_history_file: "data/continuous_improvement_history.json"
    
    # Stopping criteria configuration
    stopping_criteria:
      min_samples_per_variant: 100
      probability_threshold: 0.95
      loss_threshold: 0.005
      interval_width_threshold: 0.05
      max_experiment_days: 14
      metrics_weight:
        sentiment_accuracy: 0.4
        direction_accuracy: 0.3
        calibration_error: 0.2
        confidence_score: 0.1
      metrics_to_check:
        - sentiment_accuracy
        - direction_accuracy
```

## Usage Examples

### Running a Simple Experiment

```python
# Create an experiment
experiment = ab_testing_framework.create_experiment(
    name="Improved Prompt Template Test",
    description="Testing a new prompt template format",
    experiment_type=ExperimentType.PROMPT_TEMPLATE,
    variants=[
        {
            "name": "Current Template",
            "description": "The existing prompt template",
            "weight": 0.5,
            "config": {"template": current_template},
            "control": True
        },
        {
            "name": "Enhanced Template",
            "description": "Template with improved context",
            "weight": 0.5,
            "config": {"template": enhanced_template},
            "control": False
        }
    ]
)

# Start the experiment
ab_testing_framework.start_experiment(experiment.id)

# The system will automatically:
# 1. Collect data as requests come in
# 2. Periodically evaluate stopping criteria
# 3. Complete the experiment when criteria are met
# 4. Analyze results (frequentist and Bayesian)
# 5. Implement the winning variant if auto_implement is enabled
```

### Manually Checking Stopping Criteria

```python
# Get an experiment
experiment = ab_testing_framework.get_experiment("experiment_id")

# Evaluate stopping criteria
evaluation = stopping_criteria_manager.evaluate_experiment(experiment)

# Check if experiment should be stopped
if evaluation["should_stop"]:
    # List reasons for stopping
    for reason in evaluation["stopping_reasons"]:
        print(f"Stopping reason: {reason['criterion']} - {reason['reason']}")
    
    # Complete the experiment
    ab_testing_framework.complete_experiment(experiment.id)
```

## Dashboard Integration

The Continuous Improvement System integrates with the system dashboard, providing:

- Overview of all experiments and their status
- Detailed metrics and visualizations for each experiment
- Controls for starting, pausing, and managing experiments
- Visualizations of Bayesian analysis results
- Summary of implemented improvements

## Troubleshooting

Common issues and their solutions:

### Experiments Not Starting

- Check if `max_concurrent_experiments` limit has been reached
- Verify the experiment is in `DRAFT` status
- Ensure the system is properly initialized

### Stopping Criteria Not Working

- Check logs for evaluation details
- Verify sample sizes are sufficient for Bayesian analysis
- Ensure criteria thresholds are appropriate for your use case

### No Clear Winner

- The system needs sufficient data to determine a winner
- Adjust the `significance_threshold` or `probability_threshold`
- Consider modifying the variant designs for more distinct differences

### Bayesian Analysis Errors

- Ensure PyMC is properly installed (`pip install pymc arviz`)
- Check for sufficient data in each variant
- Watch for errors in the logs indicating analysis problems

## Advanced Topics

### Creating Custom Stopping Criteria

You can create custom stopping criteria by extending the `StoppingCriterion` base class:

```python
from src.analysis_agents.sentiment.continuous_improvement.stopping_criteria import StoppingCriterion

class CustomCriterion(StoppingCriterion):
    """Custom stopping criterion based on specific business rules."""
    
    def __init__(self, threshold=0.5):
        super().__init__(
            name="custom_criterion",
            description=f"Custom criterion with threshold {threshold}"
        )
        self.threshold = threshold
    
    def should_stop(self, experiment):
        # Custom logic to determine if experiment should stop
        # Returns a tuple: (should_stop, reason)
        return False, "Not implemented yet"

# Add to the manager
stopping_criteria_manager.add_criterion(CustomCriterion(threshold=0.7))
```

### Integrating with External Systems

The Continuous Improvement System can be integrated with external monitoring systems by subscribing to events:

```python
from src.common.events import event_bus

def handle_experiment_stopped(event):
    experiment_id = event.data.get("experiment_id")
    reasons = event.data.get("stopping_reasons", [])
    # Forward to external monitoring system
    
event_bus.subscribe("experiment_stopped", handle_experiment_stopped)
```