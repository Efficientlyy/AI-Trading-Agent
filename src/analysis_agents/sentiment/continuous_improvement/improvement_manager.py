"""Continuous Improvement Manager for sentiment analysis.

This module provides the core manager for the continuous improvement system,
coordinating experiment generation, monitoring, evaluation, and implementation
of improvements to the sentiment analysis system.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import os

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.analysis_agents.sentiment.ab_testing import (
    ab_testing_framework, ExperimentType, ExperimentStatus,
    TargetingCriteria, VariantAssignmentStrategy
)
from src.analysis_agents.sentiment.prompt_tuning import prompt_tuning_system
from src.analysis_agents.sentiment.performance_tracker import performance_tracker
from src.analysis_agents.sentiment.llm_service import LLMService

class ContinuousImprovementManager:
    """Manager for continuous improvement of the sentiment analysis system.
    
    This class coordinates the generation, monitoring, and implementation
    of experiments to continuously improve the sentiment analysis system.
    """
    
    def __init__(self):
        """Initialize the continuous improvement manager."""
        self.logger = get_logger("analysis_agents", "continuous_improvement")
        
        # Load configuration
        self.config = config.get("sentiment_analysis.continuous_improvement", {})
        self.enabled = self.config.get("enabled", False)
        self.check_interval = self.config.get("check_interval", 3600)  # 1 hour default
        self.experiment_generation_interval = self.config.get("experiment_generation_interval", 86400)  # 1 day
        self.max_concurrent_experiments = self.config.get("max_concurrent_experiments", 3)
        self.auto_implement = self.config.get("auto_implement", False)
        self.significance_threshold = self.config.get("significance_threshold", 0.95)
        self.improvement_threshold = self.config.get("improvement_threshold", 0.05)  # 5% improvement
        
        # State tracking
        self.last_experiment_generation = datetime.utcnow() - timedelta(days=1)  # Start by generating soon
        self.last_check = datetime.utcnow()
        self.improvement_history = []
        self.active_task = None
        
        # Track history of experiment results
        self.results_history_file = self.config.get("results_history_file", "data/continuous_improvement_history.json")
        self.results_history = self._load_results_history()
        
        # Map of experiment types to improvement actions
        self.improvement_actions = {
            ExperimentType.PROMPT_TEMPLATE: self._implement_prompt_template,
            ExperimentType.MODEL_SELECTION: self._implement_model_selection,
            ExperimentType.TEMPERATURE: self._implement_temperature,
            ExperimentType.CONTEXT_STRATEGY: self._implement_context_strategy,
            ExperimentType.AGGREGATION_WEIGHTS: self._implement_aggregation_weights,
            ExperimentType.UPDATE_FREQUENCY: self._implement_update_frequency,
            ExperimentType.CONFIDENCE_THRESHOLD: self._implement_confidence_threshold,
        }
        
    async def initialize(self):
        """Initialize the continuous improvement manager."""
        if not self.enabled:
            self.logger.info("Continuous improvement system is disabled.")
            return
        
        self.logger.info("Initializing continuous improvement system...")
        
        # Subscribe to events
        event_bus.subscribe("experiment_analyzed", self.handle_experiment_analyzed)
        event_bus.subscribe("performance_metrics_updated", self.handle_performance_update)
        
        # Make the directory for the results history if it doesn't exist
        os.makedirs(os.path.dirname(self.results_history_file), exist_ok=True)
        
        self.logger.info("Continuous improvement system initialized.")
    
    def _load_results_history(self) -> List[Dict[str, Any]]:
        """Load results history from file.
        
        Returns:
            List of historical experiment results
        """
        try:
            if os.path.exists(self.results_history_file):
                with open(self.results_history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading results history: {e}")
        
        return []
    
    def _save_results_history(self):
        """Save results history to file."""
        try:
            with open(self.results_history_file, 'w') as f:
                json.dump(self.results_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving results history: {e}")
    
    async def run_maintenance(self):
        """Run maintenance tasks for the continuous improvement system."""
        if not self.enabled:
            return
        
        self.logger.info("Running continuous improvement maintenance...")
        
        now = datetime.utcnow()
        
        # Check if it's time to generate experiments
        if (now - self.last_experiment_generation).total_seconds() > self.experiment_generation_interval:
            self.generate_experiments()
            self.last_experiment_generation = now
        
        # Check active experiments for potential implementation
        self.check_experiments()
        
        # Update last check time
        self.last_check = now
        
        self.logger.info("Continuous improvement maintenance completed.")
    
    async def generate_experiments(self):
        """Generate new experiments based on system performance and opportunity detection."""
        self.logger.info("Generating experiments...")
        
        # Get current active experiments
        active_experiments = ab_testing_framework.get_active_experiments_by_type(ExperimentType.PROMPT_TEMPLATE)
        active_experiments.extend(ab_testing_framework.get_active_experiments_by_type(ExperimentType.MODEL_SELECTION))
        active_experiments.extend(ab_testing_framework.get_active_experiments_by_type(ExperimentType.TEMPERATURE))
        
        # If we already have max concurrent experiments, skip generation
        if len(active_experiments) >= self.max_concurrent_experiments:
            self.logger.info(f"Already have {len(active_experiments)} active experiments. Skipping generation.")
            return
        
        # Get performance metrics to identify areas for improvement
        metrics = performance_tracker.get_recent_metrics(days=7)
        
        # Identify opportunities based on metrics
        opportunities = self._identify_improvement_opportunities(metrics)
        
        # Sort opportunities by potential impact
        opportunities.sort(key=lambda x: x["potential_impact"], reverse=True)
        
        # Generate experiments for top opportunities
        num_to_generate = min(
            self.max_concurrent_experiments - len(active_experiments),
            len(opportunities)
        )
        
        for i in range(num_to_generate):
            opportunity = opportunities[i]
            
            try:
                experiment = await self._create_experiment_from_opportunity(opportunity)
                
                if experiment:
                    # Start the experiment
                    ab_testing_framework.start_experiment(experiment.id)
                    self.logger.info(f"Started new experiment: {experiment.name} ({experiment.id})")
                    
                    # Record in history
                    self.improvement_history.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "action": "experiment_generated",
                        "experiment_id": experiment.id,
                        "experiment_name": experiment.name,
                        "opportunity": opportunity
                    })
            
            except Exception as e:
                self.logger.error(f"Error creating experiment for opportunity {opportunity['type']}: {e}")
    
    def _identify_improvement_opportunities(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential improvement opportunities based on metrics.
        
        Args:
            metrics: Recent performance metrics
            
        Returns:
            List of improvement opportunities
        """
        opportunities = []
        
        # Get baseline metrics
        sentiment_accuracy = metrics.get("sentiment_accuracy", 0.8)
        direction_accuracy = metrics.get("direction_accuracy", 0.7)
        confidence_score = metrics.get("confidence_score", 0.7)
        calibration_error = metrics.get("calibration_error", 0.1)
        success_rate = metrics.get("success_rate", 0.95)
        average_latency = metrics.get("average_latency", 500)
        
        # 1. Prompt template improvement opportunity
        if sentiment_accuracy < 0.85 or direction_accuracy < 0.8:
            opportunities.append({
                "type": ExperimentType.PROMPT_TEMPLATE,
                "reason": "Sentiment accuracy or direction accuracy is below target",
                "metrics": {
                    "sentiment_accuracy": sentiment_accuracy,
                    "direction_accuracy": direction_accuracy
                },
                "potential_impact": 0.8 * (1 - min(sentiment_accuracy, direction_accuracy))
            })
        
        # 2. Model selection opportunity
        if calibration_error > 0.08 or confidence_score < 0.75:
            opportunities.append({
                "type": ExperimentType.MODEL_SELECTION,
                "reason": "Calibration error is high or confidence score is low",
                "metrics": {
                    "calibration_error": calibration_error,
                    "confidence_score": confidence_score
                },
                "potential_impact": 0.7 * (calibration_error + (1 - confidence_score))
            })
        
        # 3. Temperature parameter opportunity
        if calibration_error > 0.05:
            opportunities.append({
                "type": ExperimentType.TEMPERATURE,
                "reason": "High calibration error suggests temperature tuning needed",
                "metrics": {
                    "calibration_error": calibration_error
                },
                "potential_impact": 0.6 * calibration_error
            })
        
        # 4. Context strategy opportunity
        by_source_accuracy = metrics.get("by_source", {})
        if by_source_accuracy and len(by_source_accuracy) > 0:
            source_variances = []
            for source, source_metrics in by_source_accuracy.items():
                accuracy = source_metrics.get("sentiment_accuracy", 0)
                source_variances.append((source, abs(accuracy - sentiment_accuracy)))
            
            # If there's high variance between sources, consider context strategy experiment
            avg_variance = sum(v for _, v in source_variances) / len(source_variances)
            if avg_variance > 0.1:
                opportunities.append({
                    "type": ExperimentType.CONTEXT_STRATEGY,
                    "reason": "High variance in accuracy between different sources",
                    "metrics": {
                        "average_variance": avg_variance,
                        "source_variances": dict(source_variances)
                    },
                    "potential_impact": 0.5 * avg_variance
                })
        
        # 5. Aggregation weights opportunity
        by_market_condition = metrics.get("by_market_condition", {})
        if by_market_condition and len(by_market_condition) > 1:
            condition_variances = []
            for condition, condition_metrics in by_market_condition.items():
                accuracy = condition_metrics.get("sentiment_accuracy", 0)
                condition_variances.append((condition, abs(accuracy - sentiment_accuracy)))
            
            # If there's high variance between market conditions, consider aggregation weights
            avg_variance = sum(v for _, v in condition_variances) / len(condition_variances)
            if avg_variance > 0.1:
                opportunities.append({
                    "type": ExperimentType.AGGREGATION_WEIGHTS,
                    "reason": "High variance in accuracy between market conditions",
                    "metrics": {
                        "average_variance": avg_variance,
                        "condition_variances": dict(condition_variances)
                    },
                    "potential_impact": 0.5 * avg_variance
                })
        
        # 6. Update frequency opportunity (if we have enough data)
        frequency_metrics = metrics.get("by_update_frequency", {})
        if frequency_metrics:
            opportunities.append({
                "type": ExperimentType.UPDATE_FREQUENCY,
                "reason": "Testing different update frequencies may improve performance",
                "metrics": frequency_metrics,
                "potential_impact": 0.3  # Lower impact, more of an operational experiment
            })
        
        # 7. Confidence threshold opportunity
        if confidence_score < 0.8:
            opportunities.append({
                "type": ExperimentType.CONFIDENCE_THRESHOLD,
                "reason": "Low confidence scores suggest threshold tuning needed",
                "metrics": {
                    "confidence_score": confidence_score
                },
                "potential_impact": 0.4 * (1 - confidence_score)
            })
        
        return opportunities
    
    async def _create_experiment_from_opportunity(self, opportunity: Dict[str, Any]) -> Optional[Any]:
        """Create an experiment based on an identified opportunity.
        
        Args:
            opportunity: The improvement opportunity
            
        Returns:
            Created experiment or None if creation failed
        """
        exp_type = opportunity["type"]
        
        # Generate a name and description based on the opportunity
        name = f"Auto-generated {exp_type.value.replace('_', ' ').title()} Test"
        description = f"Automatically generated experiment based on: {opportunity['reason']}"
        
        # Create appropriate variants based on experiment type
        if exp_type == ExperimentType.PROMPT_TEMPLATE:
            variants = self._generate_prompt_template_variants()
        elif exp_type == ExperimentType.MODEL_SELECTION:
            variants = self._generate_model_selection_variants()
        elif exp_type == ExperimentType.TEMPERATURE:
            variants = self._generate_temperature_variants()
        elif exp_type == ExperimentType.CONTEXT_STRATEGY:
            variants = self._generate_context_strategy_variants(opportunity)
        elif exp_type == ExperimentType.AGGREGATION_WEIGHTS:
            variants = self._generate_aggregation_weight_variants(opportunity)
        elif exp_type == ExperimentType.UPDATE_FREQUENCY:
            variants = self._generate_update_frequency_variants()
        elif exp_type == ExperimentType.CONFIDENCE_THRESHOLD:
            variants = self._generate_confidence_threshold_variants()
        else:
            self.logger.warning(f"Unknown experiment type: {exp_type}")
            return None
        
        if not variants or len(variants) < 2:
            self.logger.warning(f"Failed to generate variants for {exp_type}")
            return None
        
        # Create the experiment
        try:
            experiment = ab_testing_framework.create_experiment(
                name=name,
                description=description,
                experiment_type=exp_type,
                variants=variants,
                targeting=[TargetingCriteria.ALL_TRAFFIC],
                assignment_strategy=VariantAssignmentStrategy.RANDOM,
                sample_size=200,  # Default sample size, adjust based on experiment type
                min_confidence=self.significance_threshold,
                owner="continuous_improvement",
                metadata={
                    "auto_generated": True,
                    "opportunity": opportunity,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return experiment
        
        except Exception as e:
            self.logger.error(f"Error creating experiment: {e}")
            return None
    
    async def _generate_prompt_template_variants(self) -> List[Dict[str, Any]]:
        """Generate variants for prompt template experiments.
        
        Returns:
            List of variant configurations
        """
        # Get the current template as control
        current_templates = prompt_tuning_system.get_current_templates()
        sentiment_template = current_templates.get("sentiment_analysis", "")
        
        if not sentiment_template:
            self.logger.warning("Could not get current sentiment template")
            return []
        
        control_variant = {
            "name": "Current Template",
            "description": "The currently used template",
            "weight": 0.5,
            "config": {"template": sentiment_template},
            "control": True
        }
        
        # Generate improved templates using an LLM
        llm_service = LLMService()
        llm_service.initialize()
        
        prompt = f"""You are an AI specialist tasked with improving prompt templates for a sentiment analysis system.

The current template is:

```
{sentiment_template}
```

Please create an improved version of this template that might achieve higher accuracy in sentiment analysis for cryptocurrency and blockchain markets. The improvements should aim to:

1. Better capture subtle market sentiment signals
2. Improve directional accuracy (bullish/bearish/neutral)
3. Improve calibration of confidence estimates

The template should:
- Keep the same input variables
- Return the same JSON structure
- Maintain the focus on cryptocurrency markets
- Possibly provide more context or nuanced instructions

Return ONLY the improved template text with no additional commentary or explanations."""
        
        try:
            result = await llm_service.analyze_sentiment(prompt)
            if isinstance(result, dict) and "explanation" in result:
                improved_template = result["explanation"]
            else:
                improved_template = str(result)
            
            # Clean up the template
            improved_template = improved_template.strip()
            if improved_template.startswith("```") and improved_template.endswith("```"):
                improved_template = improved_template[3:-3].strip()
            
            # Create treatment variant
            treatment_variant = {
                "name": "AI-Enhanced Template",
                "description": "Automatically generated improved template",
                "weight": 0.5,
                "config": {"template": improved_template},
                "control": False
            }
            
            llm_service.close()
            
            return [control_variant, treatment_variant]
        
        except Exception as e:
            self.logger.error(f"Error generating prompt template variants: {e}")
            llm_service.close()
            return []
    
    def _generate_model_selection_variants(self) -> List[Dict[str, Any]]:
        """Generate variants for model selection experiments.
        
        Returns:
            List of variant configurations
        """
        # Get current model from config
        current_model = config.get("llm.financial_model", "gpt-4o")
        
        # Define control variant
        control_variant = {
            "name": f"Current Model ({current_model})",
            "description": "The currently used model",
            "weight": 0.5,
            "config": {"model": current_model},
            "control": True
        }
        
        # Define alternative models based on current model
        model_alternatives = {
            "gpt-4o": ["claude-3-opus", "gpt-4-turbo"],
            "gpt-4-turbo": ["claude-3-opus", "gpt-4o"],
            "claude-3-opus": ["gpt-4o", "claude-3-sonnet"],
            "claude-3-sonnet": ["claude-3-opus", "gpt-4o"],
            "gpt-3.5-turbo": ["claude-3-haiku", "gpt-4o"]
        }
        
        # Select an alternative model
        alternative_models = model_alternatives.get(current_model, ["gpt-4o"])
        alternative_model = alternative_models[0] if alternative_models else "gpt-4o"
        
        # Create treatment variant
        treatment_variant = {
            "name": f"Alternative Model ({alternative_model})",
            "description": f"Testing {alternative_model} as an alternative",
            "weight": 0.5,
            "config": {"model": alternative_model},
            "control": False
        }
        
        return [control_variant, treatment_variant]
    
    def _generate_temperature_variants(self) -> List[Dict[str, Any]]:
        """Generate variants for temperature experiments.
        
        Returns:
            List of variant configurations
        """
        # Get current temperature or use default
        current_temp = config.get("llm.temperature", 0.1)
        
        # Define control variant
        control_variant = {
            "name": f"Current Temperature ({current_temp})",
            "description": "The currently used temperature setting",
            "weight": 0.5,
            "config": {"temperature": current_temp},
            "control": True
        }
        
        # Define alternative temperature
        alternative_temp = 0.3 if current_temp <= 0.1 else 0.1
        
        # Create treatment variant
        treatment_variant = {
            "name": f"Alternative Temperature ({alternative_temp})",
            "description": f"Testing temperature={alternative_temp}",
            "weight": 0.5,
            "config": {"temperature": alternative_temp},
            "control": False
        }
        
        return [control_variant, treatment_variant]
    
    def _generate_context_strategy_variants(self, opportunity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate variants for context strategy experiments.
        
        Args:
            opportunity: The improvement opportunity with metrics
            
        Returns:
            List of variant configurations
        """
        # Define control variant (current strategy)
        control_variant = {
            "name": "Standard Context",
            "description": "Current context strategy",
            "weight": 0.5,
            "config": {"context_strategy": "standard"},
            "control": True
        }
        
        # Get source variances from the opportunity
        source_variances = opportunity.get("metrics", {}).get("source_variances", {})
        
        # Determine the source with the lowest accuracy
        lowest_source = None
        lowest_accuracy = 1.0
        
        for source, variance in source_variances.items():
            if variance > lowest_accuracy:
                lowest_source = source
                lowest_accuracy = variance
        
        # Create treatment variant with enhanced context for the weakest source
        strategy_name = f"enhanced_{lowest_source}" if lowest_source else "enhanced"
        treatment_variant = {
            "name": "Enhanced Context",
            "description": f"Enhanced context strategy focusing on {lowest_source}",
            "weight": 0.5,
            "config": {"context_strategy": strategy_name},
            "control": False
        }
        
        return [control_variant, treatment_variant]
    
    def _generate_aggregation_weight_variants(self, opportunity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate variants for aggregation weight experiments.
        
        Args:
            opportunity: The improvement opportunity with metrics
            
        Returns:
            List of variant configurations
        """
        # Define control variant (current weights)
        control_variant = {
            "name": "Current Weights",
            "description": "Current aggregation weights",
            "weight": 0.5,
            "config": {"weight_strategy": "current"},
            "control": True
        }
        
        # Get condition variances from the opportunity
        condition_variances = opportunity.get("metrics", {}).get("condition_variances", {})
        
        # Determine the market condition with the lowest accuracy
        lowest_condition = None
        lowest_accuracy = 1.0
        
        for condition, variance in condition_variances.items():
            if variance > lowest_accuracy:
                lowest_condition = condition
                lowest_accuracy = variance
        
        # Create treatment variant with optimized weights for the weakest condition
        strategy_name = f"optimized_{lowest_condition}" if lowest_condition else "optimized"
        treatment_variant = {
            "name": "Optimized Weights",
            "description": f"Weights optimized for {lowest_condition} market condition",
            "weight": 0.5,
            "config": {"weight_strategy": strategy_name},
            "control": False
        }
        
        return [control_variant, treatment_variant]
    
    def _generate_update_frequency_variants(self) -> List[Dict[str, Any]]:
        """Generate variants for update frequency experiments.
        
        Returns:
            List of variant configurations
        """
        # Get current update frequency
        current_freq = config.get("sentiment_analysis.adaptive_weights.update_frequency", 24)  # Hours
        
        # Define control variant
        control_variant = {
            "name": f"Current Frequency ({current_freq}h)",
            "description": f"Current update frequency of {current_freq} hours",
            "weight": 0.5,
            "config": {"update_frequency": current_freq},
            "control": True
        }
        
        # Define alternative frequency
        alternative_freq = current_freq // 2 if current_freq > 6 else current_freq * 2
        
        # Create treatment variant
        treatment_variant = {
            "name": f"Alternative Frequency ({alternative_freq}h)",
            "description": f"Testing update frequency of {alternative_freq} hours",
            "weight": 0.5,
            "config": {"update_frequency": alternative_freq},
            "control": False
        }
        
        return [control_variant, treatment_variant]
    
    def _generate_confidence_threshold_variants(self) -> List[Dict[str, Any]]:
        """Generate variants for confidence threshold experiments.
        
        Returns:
            List of variant configurations
        """
        # Get current confidence threshold
        current_threshold = config.get("sentiment_analysis.confidence_threshold", 0.6)
        
        # Define control variant
        control_variant = {
            "name": f"Current Threshold ({current_threshold})",
            "description": f"Current confidence threshold of {current_threshold}",
            "weight": 0.5,
            "config": {"confidence_threshold": current_threshold},
            "control": True
        }
        
        # Define alternative threshold (more strict)
        alternative_threshold = min(0.8, current_threshold + 0.1)
        
        # Create treatment variant
        treatment_variant = {
            "name": f"Higher Threshold ({alternative_threshold})",
            "description": f"Testing confidence threshold of {alternative_threshold}",
            "weight": 0.5,
            "config": {"confidence_threshold": alternative_threshold},
            "control": False
        }
        
        return [control_variant, treatment_variant]
    
    async def check_experiments(self):
        """Check active experiments for potential implementation."""
        # Get completed experiments that have been analyzed
        experiments = ab_testing_framework.list_experiments(
            status=[ExperimentStatus.ANALYZED]
        )
        
        for exp_data in experiments:
            # Skip if not auto-generated
            if not exp_data.get("auto_generated", False):
                continue
            
            experiment_id = exp_data["id"]
            experiment = ab_testing_framework.get_experiment(experiment_id)
            
            if not experiment:
                continue
            
            # Check if the experiment has clear results
            if not experiment.results.get("has_clear_winner", False):
                continue
            
            # If auto-implement is enabled, implement the experiment
            if self.auto_implement:
                await self._implement_experiment(experiment)
            else:
                # Otherwise, just log a recommendation
                winning_variant = experiment.results.get("winning_variant")
                self.logger.info(
                    f"Experiment {experiment.name} ({experiment.id}) has a clear winner: "
                    f"{winning_variant}. Auto-implementation is disabled."
                )
    
    async def _implement_experiment(self, experiment):
        """Implement a successful experiment.
        
        Args:
            experiment: The experiment to implement
        """
        # Get the winning variant
        winning_variant_name = experiment.results.get("winning_variant")
        winning_variant = None
        
        for variant in experiment.variants:
            if variant.name == winning_variant_name:
                winning_variant = variant
                break
        
        if not winning_variant:
            self.logger.warning(f"Could not find winning variant {winning_variant_name}")
            return
        
        # Get experiment type
        exp_type = experiment.experiment_type
        
        # Get the implementation function
        implement_func = self.improvement_actions.get(exp_type)
        
        if not implement_func:
            self.logger.warning(f"No implementation function for {exp_type}")
            return
        
        try:
            # Implement the winning variant
            await implement_func(experiment, winning_variant)
            
            # Update experiment status
            experiment.implement()
            
            # Record implementation in history
            improvement_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "experiment_id": experiment.id,
                "experiment_name": experiment.name,
                "experiment_type": exp_type.value,
                "winning_variant": winning_variant_name,
                "variant_config": winning_variant.config,
                "metrics_improvement": experiment.results.get("metrics_improvement", {})
            }
            
            self.improvement_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": "improvement_implemented",
                "details": improvement_record
            })
            
            # Add to results history
            self.results_history.append(improvement_record)
            self._save_results_history()
            
            self.logger.info(
                f"Successfully implemented experiment {experiment.name} ({experiment.id}) "
                f"with winning variant {winning_variant_name}"
            )
            
        except Exception as e:
            self.logger.error(f"Error implementing experiment {experiment.id}: {e}")
    
    async def _implement_prompt_template(self, experiment, winning_variant):
        """Implement a winning prompt template variant.
        
        Args:
            experiment: The experiment
            winning_variant: The winning variant
        """
        template = winning_variant.config.get("template")
        
        if not template:
            self.logger.warning("No template in winning variant")
            return
        
        # Implement through the prompt tuning system
        prompt_type = "sentiment_analysis"
        prompt_tuning_system.add_tuned_prompt(
            prompt_type=prompt_type,
            prompt_template=template,
            name=f"Auto-implemented from experiment {experiment.id}",
            is_default=True,
            metadata={
                "source": "continuous_improvement",
                "experiment_id": experiment.id,
                "experiment_name": experiment.name,
                "implementation_time": datetime.utcnow().isoformat()
            }
        )
        
        # Activate the new template
        prompt_tuning_system.set_active_prompt(prompt_type, template)
        
        self.logger.info(f"Implemented new prompt template from experiment {experiment.id}")
    
    async def _implement_model_selection(self, experiment, winning_variant):
        """Implement a winning model selection variant.
        
        Args:
            experiment: The experiment
            winning_variant: The winning variant
        """
        model = winning_variant.config.get("model")
        
        if not model:
            self.logger.warning("No model in winning variant")
            return
        
        # Update configuration
        config_updates = {
            "llm.financial_model": model
        }
        
        # Update the configuration
        self._update_config(config_updates)
        
        self.logger.info(f"Updated financial model to {model} based on experiment {experiment.id}")
    
    async def _implement_temperature(self, experiment, winning_variant):
        """Implement a winning temperature variant.
        
        Args:
            experiment: The experiment
            winning_variant: The winning variant
        """
        temperature = winning_variant.config.get("temperature")
        
        if temperature is None:
            self.logger.warning("No temperature in winning variant")
            return
        
        # Update configuration
        config_updates = {
            "llm.temperature": temperature
        }
        
        # Update the configuration
        self._update_config(config_updates)
        
        self.logger.info(f"Updated temperature to {temperature} based on experiment {experiment.id}")
    
    async def _implement_context_strategy(self, experiment, winning_variant):
        """Implement a winning context strategy variant.
        
        Args:
            experiment: The experiment
            winning_variant: The winning variant
        """
        strategy = winning_variant.config.get("context_strategy")
        
        if not strategy:
            self.logger.warning("No context strategy in winning variant")
            return
        
        # Update configuration
        config_updates = {
            "sentiment_analysis.context_strategy": strategy
        }
        
        # Update the configuration
        self._update_config(config_updates)
        
        self.logger.info(f"Updated context strategy to {strategy} based on experiment {experiment.id}")
    
    async def _implement_aggregation_weights(self, experiment, winning_variant):
        """Implement a winning aggregation weights variant.
        
        Args:
            experiment: The experiment
            winning_variant: The winning variant
        """
        strategy = winning_variant.config.get("weight_strategy")
        
        if not strategy:
            self.logger.warning("No weight strategy in winning variant")
            return
        
        # Update configuration
        config_updates = {
            "sentiment_analysis.weight_strategy": strategy
        }
        
        # Update the configuration
        self._update_config(config_updates)
        
        self.logger.info(f"Updated weight strategy to {strategy} based on experiment {experiment.id}")
    
    async def _implement_update_frequency(self, experiment, winning_variant):
        """Implement a winning update frequency variant.
        
        Args:
            experiment: The experiment
            winning_variant: The winning variant
        """
        frequency = winning_variant.config.get("update_frequency")
        
        if frequency is None:
            self.logger.warning("No update frequency in winning variant")
            return
        
        # Update configuration
        config_updates = {
            "sentiment_analysis.adaptive_weights.update_frequency": frequency
        }
        
        # Update the configuration
        self._update_config(config_updates)
        
        self.logger.info(f"Updated update frequency to {frequency} hours based on experiment {experiment.id}")
    
    async def _implement_confidence_threshold(self, experiment, winning_variant):
        """Implement a winning confidence threshold variant.
        
        Args:
            experiment: The experiment
            winning_variant: The winning variant
        """
        threshold = winning_variant.config.get("confidence_threshold")
        
        if threshold is None:
            self.logger.warning("No confidence threshold in winning variant")
            return
        
        # Update configuration
        config_updates = {
            "sentiment_analysis.confidence_threshold": threshold
        }
        
        # Update the configuration
        self._update_config(config_updates)
        
        self.logger.info(f"Updated confidence threshold to {threshold} based on experiment {experiment.id}")
    
    def _update_config(self, updates: Dict[str, Any]):
        """Update configuration with the specified values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        # For each update, set it in the config
        for key, value in updates.items():
            config.set(key, value)
        
        # Also try to persist changes to config file if supported
        try:
            config.save()
        except Exception as e:
            self.logger.warning(f"Could not save configuration changes: {e}")
    
    async def handle_experiment_analyzed(self, event: Event):
        """Handle an experiment analyzed event.
        
        Args:
            event: The event data
        """
        data = event.data
        experiment_id = data.get("experiment_id")
        
        if not experiment_id:
            return
        
        # If auto_implement is enabled, check if we should implement
        if self.auto_implement:
            experiment = ab_testing_framework.get_experiment(experiment_id)
            if experiment and experiment.results.get("has_clear_winner", False):
                await self._implement_experiment(experiment)
    
    async def handle_performance_update(self, event: Event):
        """Handle a performance metrics update event.
        
        Args:
            event: The event data
        """
        # Performance updates may trigger new experiment generation
        # This is handled during maintenance runs
        pass
    
    def get_improvement_history(self) -> List[Dict[str, Any]]:
        """Get the history of improvements.
        
        Returns:
            List of improvement records
        """
        return self.improvement_history
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the continuous improvement system.
        
        Returns:
            Status information
        """
        return {
            "enabled": self.enabled,
            "last_check": self.last_check.isoformat(),
            "last_experiment_generation": self.last_experiment_generation.isoformat(),
            "improvements_count": len(self.results_history),
            "auto_implement": self.auto_implement,
            "active_experiments": len(ab_testing_framework.active_experiment_ids)
        }


# Singleton instance
continuous_improvement_manager = ContinuousImprovementManager()